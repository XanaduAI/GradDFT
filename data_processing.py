import os
from warnings import warn
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
from pyscf import gto
from pyscf.data.elements import ELEMENTS, CONFIGURATION

from interface import loader as load 
from interface import saver as save
from interface import molecule_from_pyscf
from molecule import make_reaction
from interface.pyscf import process_mol
from utils.types import Hartree2kcalmol, Picometers2Angstroms, Bohr2Angstroms

dirpath = os.path.dirname(__file__)
data_dir = 'data/'
data_path = os.path.join(dirpath, data_dir)

# Select the configuration here
basis = 'def2-tzvp' # This basis is available for all elements up to atomic number 86
grid_level = 2
omegas = [0.] # This indicates the values of omega in the range-separted exact-exchange.
            # It is relatively memory intensive. omega = 0 is the usual Coulomb kernel.
            # Leave empty if no Coulomb kernel is expected.


def process_dimers(training = True, combine = False, max_cycle = None):

    # We first read the excel file
    dataset_file = os.path.join(dirpath, data_dir, 'raw/XND_dataset.xlsx')
    dimers_df = pd.read_excel(dataset_file, header=0, index_col=None, sheet_name="Dimers")
    atoms_df = pd.read_excel(dataset_file, header=0, index_col=0, sheet_name="Atoms")

    # We will save three files: one with one transition metal (tm) dimers, another with non-tm dimers
    # and finally one with none
    molecules = []
    tm_molecules = []
    non_tm_molecules = []

    tms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    for i in tqdm(range(len(dimers_df.index)), desc = 'Processing dimers'):

        # Extract the molecule information

        atom1 = dimers_df['Atom1'][i]
        atom2 = dimers_df['Atom2'][i]

        if np.isnan(dimers_df['Energy (Hartrees) experimental from dissociation, D0'][i]) \
            or np.isnan(dimers_df['Zero-point energy correction'][i]):
            print(f'The dissociation energy or zero point correction of dimer {atom1}-{atom2} is not available')
            continue
        else:
            dissociation_e =  float(dimers_df['Energy (Hartrees) experimental from dissociation, D0'][i])
            zero_e = - float(dimers_df['Zero-point energy correction'][i])
            atom1_e = float(atoms_df['ccsd(t)/cbs energy 3-point'][atom1]) 
            atom2_e = float(atoms_df['ccsd(t)/cbs energy 3-point'][atom2])
            energy = dissociation_e - zero_e + atom1_e + atom2_e

        spin = int(dimers_df['Multiplicity'][i]) - 1
        charge = 0
        bond_length = float(dimers_df['Bond distance (A)'][i])

        geometry = [[atom1, [0,0,0]], [atom2, [bond_length,0,0]]]

        # Create a mol and a molecule
        mol = gto.M(atom = geometry,
            basis=basis, charge = charge, spin = spin)
        _, mf = process_mol(mol, compute_energy=False, grid_level = grid_level,
                            training=training, max_cycle=max_cycle)
        if max_cycle: energy = mf.e_tot
        molecule = molecule_from_pyscf(mf, name = atom1+atom2, energy=energy, scf_iteration = max_cycle, omegas = omegas)

        # Attach the molecule to the list of molecules
        if atom1 in tms or atom2 in tms:
            tm_molecules.append(molecule)
        else:
            non_tm_molecules.append(molecule)
        molecules.append(molecule)

    # If combine, then we return the lists
    if not combine:
        if training: data_folder = os.path.join(data_path, 'training/')
        else: data_folder = os.path.join(data_path, 'evaluation/')

        data_file = os.path.join(data_folder, 'dimers/tm_dimers.h5')
        save(molecules = tm_molecules, fname = data_file, training = training)
        
        data_file = os.path.join(data_folder, 'dimers/non_tm_dimers.h5')
        save(molecules = non_tm_molecules, fname = data_file, training = training)

        data_file = os.path.join(data_folder, 'dimers/dimers.h5')
        save(molecules = molecules, fname = data_file, training = training)
    
    else:
        return molecules, tm_molecules, non_tm_molecules

def process_atoms(training = True, combine = False, max_cycle = None, charge = 0, noise = 0):

    # We first read the excel file of the dataset
    dataset_file = os.path.join(dirpath, data_dir, 'raw/XND_dataset.xlsx')
    atoms_df = pd.read_excel(dataset_file, header=0, index_col=0, sheet_name="Atoms")

    charge = 0
    molecules = []

    for i in tqdm(range(1,37), desc = 'Processing atoms'):
        atom = ELEMENTS[i]
        energy = float(atoms_df['ccsd(t)/cbs energy 3-point'][atom]) + np.random.normal(0, noise)

        if ELEMENTS.index(atom) <= charge: continue
        ghost_atom = ELEMENTS[i-charge]
        spin = compute_spin_element(ghost_atom) # In case the atom is charged, the configuration changes
        charge = charge
        geometry = [[atom, [0,0,0]]]

        # Create the Molecule object
        mol = gto.M(atom = geometry,
                basis=basis, charge = charge, spin = spin)
        _, mf = process_mol(mol, compute_energy=False, grid_level = grid_level, 
                            training=training, max_cycle = max_cycle)
        if max_cycle: energy = mf.e_tot
        molecule = molecule_from_pyscf(mf, name = atom, energy=energy, scf_iteration = max_cycle, omegas = omegas)
        molecules.append(molecule)

    # Save or return the list of molecules
    if not combine:
        if training: data_folder = os.path.join(data_path, 'training/')
        else: data_folder = os.path.join(data_path, 'evaluation/')

        data_file = os.path.join(data_folder, 'atoms/atoms.h5')
        save(molecules = molecules, fname = data_file, training = training)

    else:
        return molecules

def process_dissociation(atom1 = 'H', atom2 = 'H', charge = 0, spin = 0, file = 'H2_dissociation.xlsx', training = True, combine = False, max_cycle = None, training_distances = None, noise = 0):
    
    # read file data/raw/dissociation/H2_dissociation.xlsx
    dissociation_file = os.path.join(dirpath, data_dir, 'raw/dissociation/', file)
    dissociation_df = pd.read_excel(dissociation_file, header=0, index_col=0)

    molecules = []
    if training_distances is None: distances = dissociation_df.index
    else: distances = training_distances

    for i in tqdm(distances):
        d = [dis for dis in dissociation_df.index if np.isclose(i, dis)][0]
        try:
            energy = dissociation_df.loc[d,'energy (Ha)'] + np.random.normal(loc = 0, scale = noise)
        except:
            warnings.warn(f"No dissociation energy data for distance {d}")
        geometry = [[atom1,[0,0,0]],[atom2,[0,0,d]]]
        mol = gto.M(atom = geometry, basis = basis, charge = charge, spin = spin)
        
        _, mf = process_mol(mol, compute_energy=False, grid_level = grid_level,
                training=training, max_cycle=max_cycle)
        if max_cycle: energy = mf.e_tot
        name= '_'.join([atom1, atom2, str(charge), str(d)])
        molecule = molecule_from_pyscf(mf, name = name, energy = energy, scf_iteration = max_cycle, omegas = omegas)
        molecules.append(molecule)

    # Save or return the list of molecules
    if not combine:
        if training: data_folder = os.path.join(data_path, 'training/')
        else: data_folder = os.path.join(data_path, 'evaluation/')
        data_file = os.path.join(data_folder, 'dissociation/', file[:-5]+'.h5') 
        save(molecules = molecules, fname = data_file, training = training)
    else:
        return molecules


########### Auxiliary functions #############
def compute_spin_element(atom):

    i = ELEMENTS.index(atom)
    configuration = CONFIGURATION[i] # This indicates the electronic configuration of a given atom

    spin = configuration[0] % 2
    spin += min(configuration[1] % 6, 6 - configuration[1] % 6)
    spin += min(configuration[2] % 10, 10 - configuration[2] % 10)
    spin += min(configuration[3] % 14, 14 - configuration[3] % 14)

    return spin



########### Execute them #############
process_dimers()
process_atoms()
process_dissociation(atom1 = 'H', atom2 = 'H', charge = 0, spin = 0, file = 'H2_dissociation.xlsx')