import os
from warnings import warn
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import random
np.random.seed(0) 

from openfermion.chem import geometry_from_pubchem
from pyscf.data.elements import ELEMENTS, CONFIGURATION

dirpath = os.path.dirname(__file__)
config_path = os.path.normpath(dirpath + "/config/config.json")

from utils import Utils

tools = Utils(config_path)

config_variables = tools.get_config_variables()
import sys

sys.path.append(dirpath)

data_path = os.path.normpath(dirpath + config_variables['data_dir'])
#args = tools.parse_arguments()

from pyscf import gto

from interface import loader as load 
from interface import saver as save
from interface import molecule_from_pyscf
from molecule import make_reaction
from interface.pyscf import process_mol
from utils.types import Hartree2kcalmol, Picometers2Angstroms, Bohr2Angstroms

molecule_names = config_variables["molecule_list"]

def generate_from_name(molecule_names):
    """
    Generates the data for the molecules in molecule_names.
    Can be used specifically to generate data from transition metals.

    Parameters
    ----------
    molecule_names : list[str]
        List of molecule names to generate data for.
    """

    molecules = []
    reactions = []

    for molecule_name in molecule_names: #,"methane","benzene","titanium dioxide","dititanium oxide"
        # Generate molecule
        geometry = geometry_from_pubchem(molecule_name)

        mol = gto.M(atom = geometry,
            basis=config_variables["basis"], symmetry=True)
        energy, mf = process_mol(mol, compute_energy=True, grid_level = config_variables["grid_level"], training=True)

        molecule = molecule_from_pyscf(mf, name = molecule_name, energy = energy)

        # Saving molecule
        molecules.append(molecule)

        # Saving reaction
        r = make_reaction(reactants=(molecule), products=(molecule), energy=0)
        #reactions.append(r)

    # Saving data
    data_file = os.path.normpath(data_path+'/'+config_variables["data_fname"])
    save(omegas = [1e20, 0.4], molecules = molecules, reactions = reactions,
            fname = data_file)

    # Checks
    for typ, system in load(fpath = data_file):
        print(typ, system.name, system.energy)
        if typ == "molecule":
            print(system.spin)

def process_S66x8(training = True, number_of_files = None):
    """
    Processes the S66x8 dataset.
    CCSD(T)/CBS interaction energies in organic noncovalent complexes - dissociation curves
    """

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/S66x8/')

    for file in os.listdir(raw_dir)[:number_of_files]:

        reactants = []
        products = []
        reactant_numbers = []
        product_numbers = []
        total_electrons = 0

        with open(os.path.normpath(raw_dir +'/'+ file), 'r') as xyz:
            N = int(xyz.readline())
            header = xyz.readline()
            geometry = []
            for line in xyz:
                atom,x,y,z = line.split()
                geometry.append([atom, float(x),float(y),float(z)])
                total_electrons += ELEMENTS.index(atom)
        if total_electrons > config_variables["max_electrons"]: continue

        # Product
        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], symmetry=True)
        product_numbers.append(1)
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=True)
        reactname = file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[-1][:-4]
        product = molecule_from_pyscf(mf, name = reactname)

        # Energy
        reaction_energy = float(header.split()[2].split('=')[1])
        reaction_energy_unit = header.split()[1].split('=')[1]
        if reaction_energy_unit == 'kcalmol':
            reaction_energy /= Hartree2kcalmol 
        else: raise ValueError("Unknown unit for reaction energy")

        charges = np.array([int(header.split()[3+i].split('=')[1]) for i in range(3)])
        #todo: what to do if charges are not 0?
        assert np.allclose(charges, np.zeros(charges.shape)), "Charges are not 0 in file {}".format(file)

        # Reactants
        molecule_name1, molecule_name2 = file.split('_')[1].split('-')

        for name in [molecule_name1, molecule_name2]:

            if name == 'Peptide': name = 'N-Methylacetamide'
            elif name == 'AcNH2': name == 'Acetamide'
            #elif name == 'AcOH': name == 'Acetic acid'
            #elif name == 'MeOH': name = 'Methanol'
            #elif name == 'MeNH2': name = 'Methylamine'

            geometry = geometry_from_pubchem(name)
            if geometry is None: 
                warn("Could not find molecule {0} from file {1}".format(name,file))
                break

            mol = gto.M(atom = geometry,
                basis=config_variables["basis"], symmetry=True)

            _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training = training)

            reactants.append(molecule_from_pyscf(mf, name = name))

            if molecule_name1 == molecule_name2:
                reactant_numbers.append(2)
                break
            else:
                reactant_numbers.append(1)

        if geometry is None: continue

        products.append(product)

        # Make reaction
        reactions = []

        reaction = make_reaction(reactants, products, reactant_numbers, product_numbers, reaction_energy, name = reactname)
        reactions.append(reaction)

        # Save
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/S66x8/'+reactname+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/S66x8/'+reactname+'.hdf5')
        save(omegas = [1e20, 0.4], molecules = [], reactions = reactions,
                chunk_size=config_variables["chunk_size"], fname = data_file)

def process_w4x17(training = True):
    """
    Processes the W4-17 dataset.
    A diverse and high-confidence dataset of atomization energies for benchmarking high-level electronic structure methods
    CCSD(T)/cc-pV(Q+d)Z optimized geometries
    """

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/W4-17/')
    reactions = []

    it = 0

    for file in [f for f in os.listdir(raw_dir) if 'W4-17' not in f]:

        print('Processing file', file)

        product_numbers = []
        products = []
        reactant_numbers = []
        reactants = []

        with open(os.path.normpath(raw_dir +'/'+ file), 'r') as xyz:
            N = int(xyz.readline())
            #if N > config_variables["max_atoms"]: continue
            header = xyz.readline()
            charge = int(header.split()[0][-1])
            if charge != 0:
                warnings.warn("Charge is not 0 in file {}".format(file))
                continue
            multiplicity = int(header.split()[1][-1])
            geometry = []
            total_electrons = 0
            for line in xyz:
                atom,x,y,z = line.split()
                total_electrons += ELEMENTS.index(atom)
                geometry.append([atom, float(x),float(y),float(z)])
            if total_electrons > config_variables["max_electrons"]: continue
            # 1 Bohr = 0.52917721092 Angstrom => x Angstrom = (x/0.52917721092) Bohr

        # Product
        reactname = file[:-4]
        mol = gto.M(atom = geometry, unit = 'Angstrom',
                basis=config_variables["basis"], charge = charge, spin = multiplicity-1)
        product_numbers.append(1)
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=training)

        product = molecule_from_pyscf(mf, name = reactname)

        products.append(product)

        # Energy
        reaction_energy = 0
        with open(os.path.normpath(raw_dir +'/W4-17_Ref_TAE0.txt'), 'r') as txt:
            lines = txt.readlines()
            for line in lines[5:206]:
                molecule_name = file.replace('.xyz','').replace('mr_','')
                if molecule_name in line:
                    reaction_energy = -float(line.split()[1])
                    reaction_energy /= Hartree2kcalmol
                    break
        assert reaction_energy != 0

        # Reactants
        atom_symbols = {}
        for atom in mol.atom:
            if atom[0] in atom_symbols.keys():
                atom_symbols[atom[0]] += 1
            else:
                atom_symbols[atom[0]] = 1

        n_electrons = 0
        for atom in atom_symbols.keys():
            geometry = [[atom, 0.,0.,0.]]
            mol = gto.M(atom = geometry, basis=config_variables["basis"],
                symmetry = 1, spin = compute_spin_element(atom))
            _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=training)

            n_electrons += mf.mol.nelectron * atom_symbols[atom]

            reactants.append(molecule_from_pyscf(mf, name = atom))
            reactant_numbers.append(atom_symbols[atom])

        # Make reaction
        reaction = make_reaction(reactants, products, reactant_numbers, product_numbers, reaction_energy, name = reactname)
        reactname = reactname + '_'+str(n_electrons)
        reactions.append(reaction)
    
        # Save
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/W4-17/'+str((it+1)//config_variables["reactions_per_file"])+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/W4-17/'+str((it+1)//config_variables["reactions_per_file"])+'.hdf5')

        if (it+1) % config_variables["reactions_per_file"] == 0:
            save(omegas = [1e20, 0.4], molecules = [], reactions = reactions,
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

            reactions = []

        it += 1

def process_hait(main_atoms = ['C', 'Cl', 'F', 'H', 'N', 'O', 'S'], training = True, combine = False):
    # Dataset from J. Chem. Theory Comput. 2019, 15, 5370−5385
    # Download it from https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00674

    # Read the energy files
    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/Hait19what/')

    df = pd.read_excel(io = os.path.normpath(raw_dir +'/Hait19what_energies.xlsx'), sheet_name = 0)
    table_names = ["Atoms", "Hydrides", "Oxides", "Carbides", "Nitrides", "Sulphides", "Chlorides", "Fluorides"]
    df = df.rename(columns=lambda x: x.strip() if type(x) is str else x)

    # Rename columns using the 20th row, because that is complete
    df.columns = df.iloc[18]
    df.rename(columns = {'Hydrides':'Atom'}, inplace = True)

    def split_df(df, table_names):
        "split the dataframe wherever there is a all-nan row"
        dfs = {}
        nan_indices = df[df.isnull().all(axis=1)].index
        dfs[table_names[0]] = df.iloc[0:nan_indices[0]]
        table_index = 1
        for r in range(len(nan_indices)):
            i = nan_indices[r]
            if r != len(nan_indices)-1:
                j = nan_indices[r+1]
            else: j = len(df)+1
            if j != i+1:
                dfs[table_names[table_index]] = pd.DataFrame(df.iloc[i+2:j])
                table_index += 1
        return dfs

    dfs = split_df(df, table_names)

    for k in dfs.keys():
        dfs[k].set_index('Atom', inplace = True)

    dfs['Atoms']['Best ASCI+PT2 energy (hartree)'] =  dfs['Atoms']['Best ASCI+PT2 energy (hartree)'].fillna(dfs['Atoms']['ROHF reference energy (hartree)'])
    dfs['Atoms']['Extrapolated ASCI+PT2 (hartree)'] =  dfs['Atoms']['Extrapolated ASCI+PT2 (hartree)'].fillna(dfs['Atoms']['Best ASCI+PT2 energy (hartree)'])

    atom_energies = dict(zip(dfs['Atoms'].index, dfs['Atoms']['Extrapolated ASCI+PT2 (hartree)']))

    def read_hait_qchem(dfs):
        "Read a qchem file with only two atoms"
        new_dfs = {}
        for folder in os.listdir(geometries_path):
            group = folder +'s'
            charges = {}
            multiplicities = {}
            bond_lengths = {}

            for file in os.listdir(os.path.normpath(geometries_path +'/'+folder)):
                with open(os.path.normpath(geometries_path +'/'+folder+'/'+file), 'r') as xyz:
                    N = xyz.readline()
                    header = xyz.readline()
                    charge = int(header.split()[0])
                    multiplicity = int(header.split()[1])

                    atom = xyz.readline()[:-1]
                    bond_length = float(xyz.readline().split()[2])

                    # Add charge, multiplicity and bond length to the dfs[group]
                    for row in dfs[group].index:
                        atom_name = row # dfs[group].loc[row]
                        if atom in atom_name:
                            charges[atom_name] = charge
                            multiplicities[atom_name] = multiplicity
                            bond_lengths[atom_name] = bond_length


            #energies = dfs[group]['Extrapolated ASCI+PT2 (hartree)']
            energies = dict(zip(dfs[group].index, dfs[group]['Extrapolated ASCI+PT2 (hartree)']))
            new_df = pd.DataFrame(
                {'Energies (hartrees)': energies,
                'Charge': charges,
                'Multiplicity': multiplicities,
                'Bond length (angstrom)': bond_lengths}
            )
            for index in new_df.index:
                if 'HS' in index and '*' not in index:
                    new_df.drop(index, inplace = True)
            new_df['Transition Metal'] = [i[:2] for i in new_df.index]
            new_df['Main Atom'] = folder[0] if folder != 'Chloride' else 'Cl'

            new_df['TM Energy (hartrees)'] = [atom_energies[t] for t in new_df['Transition Metal']]
            new_df['Main Atom Energy (hartrees)'] = [atom_energies[a] for a in new_df['Main Atom']]

            new_df['Compound'] = [a+t for a,t in zip(new_df['Main Atom'], new_df['Transition Metal'])]
            new_df.set_index('Compound', inplace = True)
            print(new_df.head(10))
            new_dfs[group] = new_df

        return new_dfs

    geometries_path = os.path.normpath(raw_dir +'/geometries/')
    dfs = read_hait_qchem(dfs)

    # Save diatomic molecules
    if main_atoms == 'all':
        molecules = generate_molecules_hait(dfs, training = training)
        if not combine:
            if training:
                data_file = os.path.normpath(data_path+config_variables['training_dir']+ '/hait19what'+'/hait19what.hdf5')
            else:
                data_file = os.path.normpath(data_path+config_variables['evaluation_dir']+ '/hait19what'+'/hait19what.hdf5')
                save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                    chunk_size=config_variables["chunk_size"],fname = data_file, training=training)
        else: return molecules
    else:
        for atom in tqdm(main_atoms):
            if training:
                data_file = os.path.normpath(data_path+config_variables['training_dir']+ '/hait19what'+'/hait19what_'+ atom + '.hdf5')
            else:
                data_file = os.path.normpath(data_path+config_variables['evaluation_dir']+ '/hait19what'+'/hait19what_'+ atom + '.hdf5')
            molecules = generate_molecules_hait(dfs, main_atoms = [atom], training = training)
            save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"],fname = data_file, training = training)

    if training:
        data_file = os.path.normpath(data_path+config_variables['training_dir']+ '/hait19what'+'/hait19what_elements.hdf5')  
    else:
        data_file = os.path.normpath(data_path+config_variables['evaluation_dir']+ '/hait19what'+'/hait19what_elements.hdf5')
    molecules = generate_atomic_molecules_hait(atom_energies, training = training)
    save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
            chunk_size=config_variables["chunk_size"], fname = data_file, training=training)

def process_furche(training = True):

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/Furche06/')

    furche_file = os.path.normpath(raw_dir+ '/Furche06_experimental.csv')
    furche_df = pd.read_csv(furche_file, sep = ';', header=0, index_col=False, na_values = ["None"])
    print(furche_df.head())

    reactions = []

    group = 'double_dimers'

    for i, row in enumerate(furche_df.index):

        print('Row: ', i, ' of ', len(furche_df.index))

        # Geometry
        atom0 = furche_df.loc[row, 'Dimer_element1']
        atom1 = furche_df.loc[row, 'Dimer_element2']

        #total_electrons = ELEMENTS.index(atom0) + ELEMENTS.index(atom1)
        #if total_electrons > config_variables["max_electrons"]: continue
        if pd.isna(furche_df.loc[row, 'Bond_length(pm)']): continue

        bond_length = float(furche_df.loc[row, 'Bond_length(pm)'].replace(",", ".")) * Picometers2Angstroms
        geometry = [[atom0, (0,0,0)],[atom1, (bond_length,0,0)]]
        multiplicity = furche_df.loc[row, 'Multiplicity']

        reaction_energy = furche_df.loc[row, 'Energy(kcal/mol)'] / Hartree2kcalmol

        mol = gto.M(atom = geometry, basis=config_variables["basis"], charge = 0, spin = multiplicity-1, unit = 'angstrom')
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=training)
        products = [molecule_from_pyscf(mf, name = atom0+atom1)]
        reactants =[]

        for atom in [atom0, atom1]:
            mol = gto.M(atom = [[atom, (0,0,0)]], basis=config_variables["basis"], charge = 0, spin = compute_spin_element(atom), unit = 'angstrom')
            _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=training)
            reactants.append(molecule_from_pyscf(mf, name = atom))

        product_numberes = [1]
        reactant_numbers = [1,1]

        name = atom0+atom1

        reaction = make_reaction(reactants, products, reactant_numbers, product_numberes, reaction_energy, name = name)

        if i == len(furche_df.index)-1:
            reactions.append(reaction)
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/Furche06/'+group+'.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/Furche06/'+group+'.hdf5')

            save(omegas = [1e20, 0.4], molecules = [], reactions = reactions,
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
    
        if (group == 'double_dimers' and atom0 == atom1) or (group == atom1):
            reactions.append(reaction)
        else:
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/Furche06/'+group+'.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/Furche06/'+group+'.hdf5')

            save(omegas = [1e20, 0.4], molecules = [], reactions = reactions,
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
            reactions = [reaction]
            group = atom1

def process_libe(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation']):

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/')
    file = 'libe.json'
    with open(os.path.normpath(raw_dir +'/'+ file), 'r') as xyz:
        molecule_list = json.load(xyz)

    molecules = []
    it = 0
    molecules_processed = []

    pbar = tqdm(total = config_variables["max_molecules"])
    for molecule in molecule_list:
        geometry = []
        n_electrons = 0
        for i in range(len(molecule['molecule']['sites'])):
            geometry.append([molecule['molecule']['sites'][i]['species'][0]['element'],molecule['molecule']['sites'][i]['xyz']])
            n_electrons += ELEMENTS.index(molecule['molecule']['sites'][i]['species'][0]['element'])

        multiplicity = molecule['molecule']['spin_multiplicity']
        charge = molecule['molecule']['charge']

        if charge != 0: continue
        if n_electrons > config_variables["max_electrons"]: continue
        if molecule['formula_alphabetical'] in molecules_processed: continue
        else: molecules_processed.append(molecule['formula_alphabetical'])

        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], charge = charge, spin = multiplicity-1)
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                            training=training, max_cycle=max_cycle)

        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot
        else: energy = molecule['thermo']['raw']['electronic_energy_Ha']

        molecule = molecule_from_pyscf(mf, name = molecule['molecule_id'], energy=energy, training = training, scf_iteration = max_cycle)
        molecules.append(molecule)

        if not combine:
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/libe/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/libe/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')

            if (it+1) % config_variables["molecules_per_file"] == 0:
                save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                        chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

                molecules = []

        it += 1
        pbar.update(1)
        if it >= config_variables["max_molecules"]:
            pbar.close() 
            break

    if combine: 
        return molecules
    elif len(molecules) > 0:
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/libe/'+str(1+(it+1)//config_variables["molecules_per_file"])+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/libe/'+str(1+(it+1)//config_variables["molecules_per_file"])+'.hdf5')

        save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

def process_dimers(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation'], tms_percentages = None, max_molecules = 175):

    # We first read the elements.csv file
    elements_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    elements_df = pd.read_excel(elements_file, header=0, index_col=None, sheet_name="Dimers")
    #elements_df = elements_df.iloc[:177]

    molecules = []
    tm_molecules = []
    non_tm_molecules = []
    print(elements_df.columns)
    print(len(elements_df.index))

    tms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    for i in tqdm(range(len(elements_df.index)), desc = 'Processing dimers'):
        atom1 = elements_df['Atom1'][i]
        atom2 = elements_df['Atom2'][i]

        if np.isnan(elements_df['Energy (Hartrees) experimental from dissociation'][i]) \
            or np.isnan(elements_df['Zero-point energy correction'][i]):
            continue
        else:
            energy = float(elements_df['Energy (Hartrees) experimental from dissociation'][i]) \
            - float(elements_df['Zero-point energy correction'][i])

        spin = int(elements_df['Multiplicity'][i]) - 1
        charge = 0
        bond_length = elements_df['Bond distance (A)'][i]

        geometry = [[atom1, [0,0,0]], [atom2, [bond_length,0,0]]]

        mol = gto.M(atom = geometry,
            basis=config_variables["basis"], charge = charge, spin = spin)
            
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                            training=training, max_cycle=max_cycle)

        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = atom1+atom2, energy=energy, training = training, scf_iteration = max_cycle)

        if tms_percentages is not None:
            if atom1 in tms or atom2 in tms:
                tm_molecules.append(molecule)
            else:
                non_tm_molecules.append(molecule)
        else:
            molecules.append(molecule)

        if len(molecules) > config_variables["molecules_per_file"] and not combine:
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dimers/dimers_' + str(tms_percentages)+'_' + str(max_molecules) +  '.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dimers/dimers_'  + str(tms_percentages)+'_' + str(max_molecules) +  '.hdf5')

            save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

            molecules = []

    random.seed(config_variables['seed'])
    random.shuffle(tm_molecules)
    random.shuffle(non_tm_molecules)

    if max_molecules is None:
        max_molecules = len(tm_molecules) + len(non_tm_molecules)


    #for ma_mol in max_molecules:
        #data_file = os.path.normpath(data_path + config_variables['training_dir']+'/dimers/dimers_transition_metals_'+str(ma_mol)+'.hdf5')
        #save(omegas = [], molecules = tm_molecules[:ma_mol], reactions = [],
        #            chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
    data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dimers/dimers_transition_metals.hdf5')
    save(omegas = [], molecules = tm_molecules, reactions = [],
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
        
        #data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dimers/dimers_non_transition_metals.hdf5')
        #save(omegas = [], molecules = non_tm_molecules, reactions = [],
        #            chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
    data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dimers/dimers_non_transition_metals.hdf5')
    save(omegas = [], molecules = non_tm_molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
    #except: pass

    '''if tms_percentages is None:
        train_molecules = molecules
    
    else:
        for tms_p in tms_percentages:
            n_molecules_tm = int(max_molecules*tms_p)
            n_molecules_non_tm = int(max_molecules*(1-tms_p))
            train_molecules = tm_molecules[:n_molecules_tm] + non_tm_molecules[:n_molecules_non_tm]
            eval_molecules = tm_molecules[n_molecules_tm:] + non_tm_molecules[n_molecules_non_tm:]

            if len(train_molecules) > 0:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dimers/dimers_' + str(max_molecules) + '_'+ str(tms_p) + '.hdf5')
                save(omegas = [], molecules = train_molecules, reactions = [],
                        chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dimers/dimers_' + str(max_molecules) + '_'+ str(tms_p) + '.hdf5')
                save(omegas = [], molecules = eval_molecules, reactions = [],
                        chunk_size=config_variables["chunk_size"], fname = data_file, training = not training)

    if combine:
        return molecules'''


def process_dimers_2tms(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation']):

    # We first read the elements.csv file
    elements_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    elements_df = pd.read_excel(elements_file, header=0, index_col=None, sheet_name="Dimers")
    #elements_df = elements_df.iloc[:177]

    molecules = []

    print(elements_df.columns)
    print(len(elements_df.index))

    tms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    for i in tqdm(range(len(elements_df.index)), desc = 'Processing dimers'):
        atom1 = elements_df['Atom1'][i]
        atom2 = elements_df['Atom2'][i]

        if atom1 not in tms or atom2 not in tms: continue

        energy = float(elements_df['Alternative Energy (Hartrees) experimental/theoretical from dissociation'][i])
        if np.isnan(energy) or energy == 0.: 
            print(i, atom1, atom2, 'not found energy')
            continue

        spin = int(elements_df['Multiplicity'][i]) - 1
        charge = 0
        bond_length = elements_df['Bond distance (A)'][i]

        geometry = [[atom1, [0,0,0]], [atom2, [bond_length,0,0]]]

        mol = gto.M(atom = geometry,
            basis=config_variables["basis"], charge = charge, spin = spin)
            
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                            training=training, max_cycle=max_cycle)

        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = atom1+atom2, energy=energy, training = training, scf_iteration = max_cycle)


        molecules.append(molecule)

        if len(molecules) > config_variables["molecules_per_file"] and not combine:
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dimers/dimers_2tms.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dimers/dimers_2tms.hdf5')

            save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

            molecules = []


    if combine:
        return molecules
    else :
        data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dimers/dimers_2tms.hdf5')
        save(omegas = [], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)


def process_atoms(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation'], charge = 0, exclude_tms = False, xc_functional = config_variables['xc_functional'], noise = 0, difficulty = 0):

    # We first read the elements.csv file
    elements_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    if charge == 0:
        sheet_name = "Atoms"
    elif charge == 1:
        sheet_name = "Cations"
    elif charge == -1:
        sheet_name = "Anions"
    elif charge == 2:
        sheet_name = "Cations+2"
    elif charge == -2:
        sheet_name = "Anions-2"
    else:
        raise ValueError("Charge must be -2, -1, 0, 1 or 2")
    elements_df = pd.read_excel(elements_file, header=0, index_col=0, sheet_name=sheet_name)

    tms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    if difficulty == 0:
        training_molecules = ELEMENTS[1:37]
        test_molecules = ELEMENTS[1:37]
    elif difficulty == 1:
        training_molecules = ELEMENTS[1:19] + ELEMENTS[20:37:2]
        test_molecules = ELEMENTS[19:37:2]
    elif difficulty == 2:
        training_molecules = ELEMENTS[1:21] + ELEMENTS[31:37]
        test_molecules = ELEMENTS[21:31] # TRANSITION METALS

    molecules = []

    for i in tqdm(range(1,37), desc = 'Processing atoms'):
        atom = ELEMENTS[i]
        if exclude_tms and atom in tms: continue
        if training and atom not in training_molecules: continue
        elif not training and atom not in test_molecules: continue
        energy = elements_df['Energy [Ha] def2-tzvp - DFT'][atom]
        try:
            energy = float(energy) + np.random.normal(0, noise)
        except: continue
        if np.isnan(energy): continue
        if ELEMENTS.index(atom) <= charge: continue
        ghost_atom = ELEMENTS[i-charge]

        spin = compute_spin_element(ghost_atom)
        charge = charge

        geometry = [[atom, [0,0,0]]]

        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], charge = charge, spin = spin)
        
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], 
                            training=training, max_cycle = max_cycle, xc_functional = xc_functional)

        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = atom, energy=energy, training = training, scf_iteration = max_cycle)

        molecules.append(molecule)

        if not combine and len(molecules) > config_variables["molecules_per_file"]:
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/atoms/' + sheet_name+ '_' + str((i+1)//config_variables["molecules_per_file"])+ '.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/atoms/' + sheet_name + '_' + str((i+1)//config_variables["molecules_per_file"])+ '.hdf5')

            save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

            molecules = []

    if not combine:
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/atoms/' + sheet_name+'_' + str(noise)+'_' + str(difficulty)+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/atoms/' + sheet_name +'_' + str(noise)+'_' + str(difficulty)+ '.hdf5')

        save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
    else:
        return molecules

def process_QM9(training = True, combine = False):
    """
    Processes the QM9 dataset.
    QM9 dataset
    """

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/QM9.xyz/')

    # We first read the elements.csv file
    molecules_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    molecules_df = pd.read_excel(molecules_file, header=0, index_col=0, sheet_name="QM9")

    it = 0

    molecules = []

    for file in tqdm(molecules_df.index, desc = 'Processing QM9'):

        total_energy = float(molecules_df['energy def2-infinite'][file])
        spin = 0
        charge = 0

        geometry = []

        with open (os.path.normpath(raw_dir + '/' +  file), "r") as myfile:
            lines=myfile.readlines()

            natoms = int(lines[0])
            if natoms > config_variables['max_atoms']: continue
            geometry = []
            total_electrons = 0
            for i in range(2, 2+natoms):
                line = lines[i].split()
                geometry.append([line[0], [float(line[1].replace('*^', 'e')), float(line[2].replace('*^', 'e')), float(line[3].replace('*^', 'e'))]])

                # check the number of electrons
                total_electrons += ELEMENTS.index(line[0])

        if total_electrons > config_variables['max_electrons']: continue

        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], charge = charge, spin = spin)
        
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], 
                            training=training, max_cycle = config_variables['max_DFT_cycles_data_generation'])

        if config_variables['max_DFT_cycles_data_generation']: total_energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = file[:-4], energy=total_energy, training = training, scf_iteration = config_variables['max_DFT_cycles_data_generation'])

        molecules.append(molecule)

        if len(molecules) > config_variables["molecules_per_file"] and not combine:
            if training:
                data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/QM9/QM9_' + str((it+1)//config_variables["molecules_per_file"])+ '.hdf5')
            else:
                data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/QM9/QM9_' + str((it+1)//config_variables["molecules_per_file"])+ '.hdf5')

            save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                    chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
            
            molecules = []

    return molecules

#################### Modified datasets ####################

def process_S66x8_molecules(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation']):
    """
    Processes the S66x8 dataset.
    CCSD(T)/CBS interaction energies in organic noncovalent complexes - dissociation curves
    """

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/S66x8/')

    # We first read the elements.csv file
    molecules_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    molecules_df = pd.read_excel(molecules_file, header=0, index_col=0, sheet_name="S66x8_molecules")

    it = 0

    pbar = tqdm(total = config_variables["max_molecules"])
    molecules = []

    for file in os.listdir(raw_dir):

        total_electrons = 0

        with open(os.path.normpath(raw_dir +'/'+ file), 'r') as xyz:
            N = int(xyz.readline())
            header = xyz.readline()
            geometry = []
            for line in xyz:
                atom,x,y,z = line.split()
                geometry.append([atom, float(x),float(y),float(z)])
                total_electrons += ELEMENTS.index(atom)
        if total_electrons > config_variables["max_electrons"]: continue


        # Energy
        reaction_energy = float(header.split()[2].split('=')[1])
        reaction_energy_unit = header.split()[1].split('=')[1]
        if reaction_energy_unit == 'kcalmol':
            reaction_energy /= Hartree2kcalmol 
        else: raise ValueError("Unknown unit for reaction energy")

        charges = np.array([int(header.split()[3+i].split('=')[1]) for i in range(3)])

        assert np.allclose(charges, np.zeros(charges.shape)), "Charges are not 0 in file {}".format(file)

        # Reactants
        molecule_name1, molecule_name2 = file.split('_')[1].split('-')
        distance = float(file.split('_')[-1][:-4])
        if distance == 0.9: continue


        for name in [molecule_name1, molecule_name2]:

            if name == 'Peptide': name = 'N-Methylacetamide (Peptide)'
            elif name == 'AcNH2': name == 'Acetamide (AcNH2)'
            elif name == 'AcOH': name == 'Acetic acid (AcOH)'
            elif name == 'MeOH': name = 'Methanol (MeOH)'
            elif name == 'MeNH2': name = 'Methylamine (MeNH2)'

            reaction_energy += float(molecules_df['Energy (Hartrees) def2-infinite'][name])

        if np.isnan(reaction_energy): continue

        # Product
        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], symmetry=True)

        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=True)
        reactname = file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[-1][:-4]
        molecule = molecule_from_pyscf(mf, name = reactname, energy = reaction_energy, training = training, scf_iteration = max_cycle)

        molecules.append(molecule)
        pbar.update(1)

        # Save
        if not combine:
            if (it+1) % config_variables["molecules_per_file"] == 0:
                if training:
                    data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/S66x8_molecules/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
                else:
                    data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/S66x8_molecules/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
                save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                        chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

                molecules = []

        it += 1
        if it >= config_variables["max_molecules"]: 
            break

    if combine: return molecules

def process_X40x10_molecules(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation']):
    """
    Processes the S66x8 dataset.
    CCSD(T)/CBS interaction energies in organic noncovalent complexes - dissociation curves
    """

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/X40x10/')

    # We first read the elements.csv file
    molecules_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    molecules_df = pd.read_excel(molecules_file, header=0, index_col=0, sheet_name="X40x10_molecules")

    it = 0

    pbar = tqdm(total = config_variables["max_molecules"])
    molecules = []

    with open(os.path.normpath( dirpath + config_variables['data_dir']+ '/raw/X40x10/X40x10.yaml'), 'r') as yaml_file:
        yaml_text = yaml_file.read()

    for file in os.listdir(raw_dir):
        if file == "x40x10.yaml": continue

        total_electrons = 0

        with open(os.path.normpath(raw_dir +'/'+ file), 'r') as xyz:

            N = int(xyz.readline())
            header = xyz.readline()
            geometry = []
            for line in xyz:
                atom,x,y,z = line.split()
                geometry.append([atom, float(x),float(y),float(z)])
                total_electrons += ELEMENTS.index(atom)
        if total_electrons > config_variables["max_electrons"]: continue


        # Energy: read from file raw/X40x10/X40x10.yaml, which is 
        # split yaml_text by lines and make a list
        yaml_lines = yaml_text.splitlines()
        for i in range(len(yaml_lines)):
            if file[:-4] in yaml_lines[i]:
                reaction_energy = float(yaml_lines[i+2].split()[1])/Hartree2kcalmol
                break

        # Reactants
        molecule_name1, molecule_name2 = file.split('_')[1].split('-')

        for name in [molecule_name1, molecule_name2]:

            if name == 'methanol' or name == 'mOH': name = 'methanol (mOH)'
            elif name == 'methylamine' or name == 'mNH2': name = 'methylamine (mNH2)'

            reaction_energy += float(molecules_df['Energy (Hartrees) def2-infinite'][name])

        if np.isnan(reaction_energy): continue

        # Product
        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], symmetry=True)

        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=True)
        reactname = file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[-1][:-4]
        molecule = molecule_from_pyscf(mf, name = reactname, energy = reaction_energy, training = training, scf_iteration = max_cycle)

        molecules.append(molecule)
        pbar.update(1)

        # Save
        if not combine:
            if (it+1) % config_variables["molecules_per_file"] == 0:
                if training:
                    data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/S66x8_molecules/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
                else:
                    data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/S66x8_molecules/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
                save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                        chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

                molecules = []

        it += 1
        if it >= config_variables["max_molecules"]: 
            break

    if combine: return molecules

def process_w4x17_molecules(training = True, combine = False, max_cycle = config_variables['max_DFT_cycles_data_generation']):
    """
    Processes the W4-17 dataset.
    A diverse and high-confidence dataset of atomization energies for benchmarking high-level electronic structure methods
    CCSD(T)/cc-pV(Q+d)Z optimized geometries
    """

    raw_dir = os.path.normpath(dirpath + config_variables['data_dir'] + '/raw/W4-17/')

    elements_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
    elements_df = pd.read_excel(elements_file, header=0, index_col=0, sheet_name = 'Atoms')

    molecules = []

    it = 0

    pbar = tqdm(total = config_variables["max_molecules"])

    for file in [f for f in os.listdir(raw_dir) if 'W4-17' not in f]:

        if '.DS_Store' in file: continue

        with open(os.path.normpath(raw_dir +'/'+ file), 'r') as xyz:
            print('Processing file', file)
            N = int(xyz.readline())
            #if N > config_variables["max_atoms"]: continue
            header = xyz.readline()
            charge = int(header.split()[0][-1])
            if charge != 0:
                warnings.warn("Charge is not 0 in file {}".format(file))
                continue
            multiplicity = int(header.split()[1][-1])
            geometry = []
            total_electrons = 0
            for line in xyz:
                atom,x,y,z = line.split()
                total_electrons += ELEMENTS.index(atom)
                geometry.append([atom, float(x),float(y),float(z)])
            if total_electrons > config_variables["max_electrons"]: continue

        # Molecule
        reactname = file[:-4]

        mol = gto.M(atom = geometry,
                basis=config_variables["basis"], charge = charge, spin = multiplicity-1)

        # Energy
        reaction_energy = 0
        with open(os.path.normpath(raw_dir +'/W4-17_Ref_TAE0.txt'), 'r') as txt:
            lines = txt.readlines()
            for line in lines[5:206]:
                molecule_name = file.replace('.xyz','').replace('mr_','')
                if molecule_name in line:
                    reaction_energy = -float(line.split()[1])
                    reaction_energy /= Hartree2kcalmol
                    break
        assert reaction_energy != 0

        # Reactants -> energy
        atom_symbols = {}
        for atom in mol.atom:
            if atom[0] in atom_symbols.keys():
                atom_symbols[atom[0]] += 1
            else:
                atom_symbols[atom[0]] = 1

        not_found = False
        for k, v in atom_symbols.items():
            if k not in elements_df.index:
                warnings.warn("Element {} not in elements.csv".format(k))
                not_found = True
                continue
            atom_energy = float(elements_df.loc[k,'Energy [Ha] cc-vVXZ'])
            reaction_energy += atom_energy*v

        # Add molecule
        if not_found: 
            continue

        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                training=training, max_cycle=max_cycle)

        if config_variables['max_DFT_cycles_data_generation']: reaction_energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = reactname, energy = reaction_energy, training = training, scf_iteration = max_cycle)
        molecules.append(molecule)
        pbar.update(1)

        # Save
        if not combine:
            if (it+1) % config_variables["molecules_per_file"] == 0:
                if training:
                    data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/W4-17_molecules/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
                else:
                    data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/W4-17_molecules/'+str((it+1)//config_variables["molecules_per_file"])+'.hdf5')
                save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                        chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

                molecules = []

        it += 1
        if it >= config_variables["max_molecules"]: 
            break
    if combine:
        return molecules
    elif len(molecules) > 0:
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/W4-17_molecules/'+str(1+(it+1)//config_variables["molecules_per_file"]+1)+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/W4-17_molecules/'+str(1+(it+1)//config_variables["molecules_per_file"]+1)+'.hdf5')
        save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

def process_dissociation(atom = 'H', training = True, combine = False, max_cycle = None, training_distances = None, noise = 0):

    # Read the dissociation energy data from `H2_dissociation.xlsx`
    dissociation_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/'+atom+'2_dissociation.xlsx')
    dissociation_df = pd.read_excel(dissociation_file, header=0, index_col=0)

    molecules = []
    if training:
        geometry = [[atom,[0,0,0]]]
        mol = gto.M(atom = geometry,
            basis=config_variables["basis"], charge = 0, spin = 1)
        
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                training=training)
        
        # Load energy from the elements.xlsx file
        elements_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/elements.xlsx')
        elements_df = pd.read_excel(elements_file, header=0, index_col=0, sheet_name="Atoms")

        energy = float(elements_df.loc[atom,'Energy [Ha] cc-pVXZ']) + np.random.normal(loc = 0, scale = noise)

        molecule = molecule_from_pyscf(mf, name = atom+'2', energy = energy, training = training, scf_iteration = max_cycle)

        molecules.append(molecule)

    if training_distances is None: distances = dissociation_df.index
    elif training: distances = training_distances
    else: distances = dissociation_df.index.difference(training_distances)

    for i in tqdm(range(len(distances))):

        d = float(distances[i])
        try:
            energy = dissociation_df.loc[d,'energy'] + np.random.normal(loc = 0, scale = noise)
        except:
            warnings.warn("No dissociation energy data for distance {}".format(d))

        geometry = [[atom,[0,0,0]],[atom,[0,0,d]]]

        mol = gto.M(atom = geometry,
            basis=config_variables["basis"], charge = 0, spin = 0)
        
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                training=training, max_cycle=max_cycle)
        
        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = atom+'2_'+str(d), energy = energy, training = training, scf_iteration = max_cycle)

        molecules.append(molecule)

    if not combine:
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/'+atom+'2_molecules_'+str(noise)+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dissociation/'+atom+'2_molecules_'+str(noise)+'.hdf5')
        save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
        
    else:
        return molecules
    
def process_experimental_dissociation(atom = 'H', training = True, combine = False, max_cycle = None, training_distances = None, noise = 0):
    
    # read file data/raw/dissociation/H2_dissociation.xlsx
    dissociation_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/dissociation/'+atom+'2_dissociation.xlsx')
    dissociation_df = pd.read_excel(dissociation_file, header=0, index_col=0)

    molecules = []
    if training_distances is None: distances = dissociation_df.index
    elif training: distances = training_distances
    else: distances = dissociation_df.index.difference(training_distances)

    for i in tqdm(distances):
        d = [dis for dis in dissociation_df.index if np.isclose(i, dis)][0]  #round(float(distances[i]), ndigits=3)
        try:
            energy = dissociation_df.loc[d,'energy (Ha)'] + np.random.normal(loc = 0, scale = noise)
        except:
            try:
                energy = dissociation_df.loc[d,'Energy (Ha)'] + np.random.normal(loc = 0, scale = noise)
            except:
                warnings.warn("No dissociation energy data for distance {}".format(d))
        geometry = [[atom,[0,0,0]],[atom,[0,0,d]]]
        mol = gto.M(atom = geometry,
            basis=config_variables["basis"], charge = 0, spin = 0)
        
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                training=training, max_cycle=max_cycle)
        
        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot

        molecule = molecule_from_pyscf(mf, name = atom+'2_'+str(d), energy = energy, scf_iteration = max_cycle)

        molecules.append(molecule)

    if not combine:
        if training:
            data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/'+atom+'2_molecules_'+str(noise)+'.hdf5')
        else:
            data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/dissociation/'+atom+'2_molecules_'+str(noise)+'.hdf5')
        save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)

    else:
        return molecules
    

def process_water(noise = 0., training = True, max_cycle = None, combine = False):
    
    # read the file data/raw/water/water.xlsx
    water_file = os.path.normpath(dirpath + config_variables['data_dir']+ '/raw/water.xlsx')
    water_df = pd.read_excel(water_file, header=0, index_col=None)

    molecules = []
    for i in tqdm(range(len(water_df))):

        r1 = water_df.loc[i,'r1']
        x = water_df.loc[i,'x']
        y = water_df.loc[i,'y']

        mol = gto.Mole()
        mol.atom = f'''O 0 0 0;
                    H 0 {r1} 0;
                    H {x} {y} 0'''
        mol.basis = 'def2-tzvp'
        mol.build()

        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"],
                training=training, max_cycle=max_cycle)
        
        if config_variables['max_DFT_cycles_data_generation']: energy = mf.e_tot
        else: energy = water_df.loc[i,'energy'] + np.random.normal(loc = 0, scale = noise)

        molecule = molecule_from_pyscf(mf, name = 'water_'+str(i), energy = energy, training = training, scf_iteration = max_cycle)

        molecules.append(molecule)

    if training:
        data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/water/water_molecules_'+str(noise)+'.hdf5')
    else:
        data_file = os.path.normpath(data_path+ config_variables['evaluation_dir']+'/water/water_molecules_'+str(noise)+'.hdf5')

    if not combine:
        save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
                chunk_size=config_variables["chunk_size"], fname = data_file, training = training)
    else:
        return molecules




#################### Auxiliary functions ####################

def generate_molecules_hait(dfs, main_atoms = ['C', 'Cl', 'F', 'H', 'N', 'O', 'S'], training = True):
    molecules = []
    for k,v in dfs.items():
        # Product
        for compound in v.index:
            geometry = []
            ma = v.loc[compound,'Main Atom']
            if ma not in main_atoms: continue
            tm = v.loc[compound,'Transition Metal']
            bond_length = v.loc[compound,'Bond length (angstrom)']
            energy = v.loc[compound,'Energies (hartrees)']
            charge = v.loc[compound,'Charge']
            spin = v.loc[compound,'Multiplicity'] - 1
            geometry.append([ma,[0,0,0]])
            geometry.append([tm,[0,0,bond_length]])
            mol = gto.M(atom = geometry,
                        basis='def2-TZVP',# Basis indicated in the publication J. Chem. Theory Comput. 2019, 15, 3610−3622
                        charge = charge,
                        spin = spin,
                        unit = 'angstrom')
            _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training = training)
            molecule = molecule_from_pyscf(mf, name = compound, training=training, energy=energy)
            molecules.append(molecule)

    return molecules

def generate_atomic_molecules_hait(atom_energies, training = True):
    molecules = []

    for k,v in atom_energies.items():
        spin = compute_spin_element(k)
        mol = gto.M(atom = [[k,[0,0,0]]],
                    basis='def2-TZVP',# Basis indicated in the publication J. Chem. Theory Comput. 2019, 15, 3610−3622
                    spin = spin)
        _, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training = training)
        molecule = molecule_from_pyscf(mf, name = k, training = training, energy = v)

        molecules.append(molecule)

    return molecules

def compute_spin_element(atom):

    i = ELEMENTS.index(atom)
    configuration = CONFIGURATION[i]

    spin = configuration[0] % 2
    spin += min(configuration[1] % 6, 6 - configuration[1] % 6)
    spin += min(configuration[2] % 10, 10 - configuration[2] % 10)
    spin += min(configuration[3] % 14, 14 - configuration[3] % 14)

    return spin






###################################################################################################

#generate_from_name(['water_test'])
#process_S66x8('training' == args.mode))
#process_w4x17(training = ('training' == args.mode))
#process_hait(training = ('training' == args.mode), main_atoms="all")
#process_furche(training = ('training' == args.mode))
#process_w4x17_molecules(training = ('training' == args.mode), combine = False)
#process_libe(training = ('training' == args.mode))

#atoms = process_atoms(training = ('training' == args.mode), combine=False)

#w4x17_molecules = process_w4x17_molecules(training = ('training' == args.mode), combine = True)
#QM9_molecules = process_QM9(training = ('training' == args.mode), combine = True)
#config_variables['max_electrons'] = 25
#X40x10_molecules = process_X40x10_molecules(training = ('training' == args.mode), combine = True)
#S66x8_molecules = process_S66x8_molecules(training = ('training' == args.mode), combine = True)


#atoms = []
#for max_cycle in [0, 2, 5, 10, None]:
#for xc_functional in ['TPSSh', 'PBE0','SCAN', 'PW6B95', 'MN15', 'M06', 'B3LYP', 'wB97M_V', 'wB97X_V']:
#atoms = process_atoms(training = ('training' == args.mode), combine=True, charge = 0, exclude_tms = True, max_cycle = None)

#atoms = process_atoms(training = ('training' == args.mode), combine=True, charge = 0)
#cations = process_atoms(training = ('training' == args.mode), combine=True, charge = 1)
#anions = process_atoms(training = ('training' == args.mode), combine=True, charge = -1)

#libe_molecules = process_libe(training = ('training' == args.mode), combine = False)

#molecules = libe_molecules #QM9_molecules + S66x8_molecules + X40x10_molecules + dimers

'''data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/atoms/atoms_ano_test.hdf5')
save(omegas = [1e20, 0.4], molecules = atoms, reactions = [],
        chunk_size=config_variables["chunk_size"], 
        fname = data_file,
        training = ('training' == args.mode))'''

############# Dissociation curves #############

#training_distances = np.linspace(0.5, 5, 37, endpoint = True)

geometry = [['H',[0,0,0]]]
mol = gto.M(atom = geometry, basis=config_variables["basis"], charge = 0, spin = compute_spin_element('H'))
_, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=True, max_cycle=None)
H_molecule = molecule_from_pyscf(mf, name = 'H', energy = -0.5, scf_iteration = None)

H_dis_extra = [0.5, 0.75, 1, 1.25, 1.5]
H_extra = process_experimental_dissociation(atom = 'H', training_distances = H_dis_extra, combine = True, noise = 0)
data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/H2_extrapolation_molecules.hdf5')
save(omegas = [], molecules = H_extra + [H_molecule], reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
H_dis_inter = [0.5, 0.75, 1, 1.5, 1.75, 2, 3.5, 3.75, 4]
H_inter = process_experimental_dissociation(atom = 'H', training_distances = H_dis_inter, combine = True, noise = 0)
data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/H2_interpolation_molecules.hdf5')
save(omegas = [], molecules = H_inter + [H_molecule], reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
H_molecules = process_experimental_dissociation(atom = 'H', training_distances = None, combine = True, noise = 0)
data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/H2_dissociation.hdf5')
save(omegas = [], molecules = H_molecules, reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
H_molecules = H_molecules + [H_molecule]

##

#geometry = [['N',[0,0,0]]]
#mol = gto.M(atom = geometry, basis=config_variables["basis"], charge = 0, spin = compute_spin_element('N'))
#_, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=True, max_cycle=None)
#N_molecule = molecule_from_pyscf(mf, name = 'N', energy = -54.5800522356946, training = True, scf_iteration = None)

#N_dis_extra = [0.9, 1.1, 1.3, 1.5, 1.7]
#N_extra = process_experimental_dissociation(atom = 'N', training = True, training_distances = N_dis_extra, combine = True, noise = 0)
#data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/N2_extrapolation_molecules.hdf5')
#save(omegas = [], molecules = H_molecules + N_extra + [N_molecule], reactions = [],chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
#N_dis_inter = [0.9, 1.1, 1.3,  1.6, 1.8, 2, 2.4, 2.6, 2.8]
#N_inter = process_experimental_dissociation(atom = 'N', training = True, training_distances = N_dis_inter, combine = True, noise = 0)
#data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/N2_interpolation_molecules.hdf5')
#save(omegas = [], molecules = H_molecules + N_inter + [N_molecule], reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
#N_molecules = process_experimental_dissociation(atom = 'N', training = True, training_distances = None, combine = True, noise = 0)
#data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/N2_dissociation.hdf5')
#save(omegas = [], molecules = N_molecules, reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
#N_molecules = N_molecules + [N_molecule]
##

#geometry = [['Cr',[0,0,0]]]
#mol = gto.M(atom = geometry, basis=config_variables["basis"], charge = 0, spin = compute_spin_element('Cr'))
#_, mf = process_mol(mol, compute_energy=False, grid_level = config_variables["grid_level"], training=True, max_cycle=None)
#Cr_molecule = molecule_from_pyscf(mf, name = 'Cr', energy = -1044.11160310329, training = True, scf_iteration = None)

#Cr_dis_extra = [1.577, 1.6788, 1.792, 1.878, 1.964, 2.112]
#Cr_extra = process_experimental_dissociation(atom = 'Cr', training = True, training_distances = Cr_dis_extra, combine = True, noise = 0)
#data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/Cr2_extrapolation_molecules.hdf5')
#save(omegas = [], molecules = H_molecules + N_molecules + Cr_extra + [Cr_molecule], reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
#Cr_dis_inter = [1.577, 1.6788, 1.792, 2.429, 2.574, 2.777, 3.004, 3.09, 3.201]
#Cr_inter = process_experimental_dissociation(atom = 'Cr', training = True, training_distances = Cr_dis_inter, combine = True, noise = 0)
#data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/Cr2_interpolation_molecules.hdf5')
#save(omegas = [], molecules = H_molecules + N_molecules + Cr_inter + [Cr_molecule], reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
#Cr_molecules = process_experimental_dissociation(atom = 'Cr', training = True, training_distances = None, combine = True, noise = 0)
#data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/Cr2_dissociation.hdf5')
#save(omegas = [], molecules = Cr_molecules, reactions = [], chunk_size=config_variables["chunk_size"], fname = data_file, training = True)
'''
#training_distances = [0.375, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.5, 3.625, 3.75]

training_distances = np.linspace(0.25, 5, 39, endpoint = True)
molecules = process_experimental_dissociation(atom = 'H', training = True, training_distances = training_distances, combine = True)
training_distances = [1.05, 1.1, 1.15, 1.2, 1.25, 1.75, 2, 2.5, 3] #np.linspace(0.8, 6, 53, endpoint = True)
molecules += process_experimental_dissociation(atom = 'N', training = True, training_distances = training_distances, combine = True)

data_file = os.path.normpath(data_path+ config_variables['training_dir']+'/dissociation/H2N2_generalization.hdf5')
save(omegas = [1e20, 0.4], molecules = molecules, reactions = [],
        chunk_size=config_variables["chunk_size"], 
        fname = data_file,
        training = ('training' == args.mode))'''

#process_water(training = True)

############# Atoms #############
#process_atoms(training = True, combine=False, charge = 0, noise = 0, difficulty=1)
#process_atoms(training = True, combine=False, charge = 0, noise = 1e-3, difficulty=1)
#process_atoms(training = True, combine=False, charge = 0, noise = 1e-2, difficulty=1)
#process_atoms(training = True, combine=False, charge = 0, noise = 1e-1, difficulty=1)
#process_atoms(training = True, combine=False, charge = 0, noise = 1, difficulty=1)

#############
#process_dimers(training = ('training' == args.mode), combine = False, tms_percentages = [0.999])
#process_dimers_2tms(training = True, combine = False)