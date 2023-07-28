import os
from jax import numpy as jnp
from tqdm import tqdm

from interface import molecule_from_pyscf, loader
from interface import saver as save
from molecule import Molecule, make_reaction

# In this basic tutorial we want to introduce the concept of a molecule, which is a class that contains
# all the information about a molecule that we need to compute its energy and its gradient.

# To prepare a molecule, we need to compute many properties of such system. We will use PySCF to do so,
# though we could use any other software. For example:
from pyscf import gto, dft
# Define the geometry of the molecule
geometry = [['H', (0, 0, 0)], ['F', (0, 0, 1.1)]]
mol = gto.M(atom = geometry, basis = 'def2-tzvp', charge = 0, spin = 0)

# To perform DFT we also need a grid
grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

# And we will also need a mean-field object
mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# If we want to use the molecule to compute HF exact-exchange components we will need to decide which values of
# omega we want to use in the range separated Coulomb kernel erfc(omega*r)/r.
# omega = 0. indicates no range separation: the kernel will be 1/r.
omegas = [0., 0.4]

# Then we can use the following function to generate the molecule object
name = 'HF'
HF_molecule = molecule_from_pyscf(mf, grad_order=2, name = name, 
                            energy=ground_truth_energy, omegas = omegas)

# Alternatively we may compute and pass each of the properties of the molecule separately:
HF_molecule = Molecule(
        HF_molecule.grid, HF_molecule.atom_index, HF_molecule.nuclear_pos, HF_molecule.ao, HF_molecule.grad_ao, 
        HF_molecule.grad_n_ao, HF_molecule.rdm1, HF_molecule.nuclear_repulsion, HF_molecule.h1e, HF_molecule.vj, 
        HF_molecule.mo_coeff, HF_molecule.mo_occ, HF_molecule.mo_energy,
        HF_molecule.mf_energy, HF_molecule.s1e, HF_molecule.omegas, HF_molecule.chi, HF_molecule.rep_tensor, 
        HF_molecule.energy, HF_molecule.basis, HF_molecule.name, HF_molecule.spin, HF_molecule.charge, 
        HF_molecule.unit_Angstrom, HF_molecule.grid_level, HF_molecule.scf_iteration, HF_molecule.fock
    )

# Most of these properties are Arrays, others are floats or integers.
# molecule.grad_ao is a dictionary of arrays, indicating the n-th order gradients
# of the atomic orbitals, \nabla^n ao = \sum_i (\partial^n f / \partial x_i^n)

# Also worth mentioning that to avoid type errors, we convert strings (the basis, the name of the molecule)
# into integers
name_ints = jnp.array([ord(char) for char in name])
name = ''.join(chr(num) for num in name_ints)
print(name, name_ints)

# Now let's talk about how to save and load a molecule (or a list of Molecules).
dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.join(dirpath, "data/examples/")
data_file = os.path.join(training_data_dirpath, "HF.hdf5")
save(molecules = [HF_molecule], fname = data_file)

# We can load the molecule from the file
# Select if we are going to train, as well as the omegas we will finally use
load = loader(fname = data_file, randomize=True, training = False, config_omegas = [])
for _, system in tqdm(load, 'Molecules/reactions per file'):
    HF_molecule = system
    print('Molecule name', ''.join(chr(num) for num in HF_molecule.name)) # We use training = False so molecule.name is a string


# We can also create reactions, save and load them. For example, let us emulate the formation reaction of HF
# from H and F atoms:
products = [HF_molecule]

reaction_energy = ground_truth_energy

reactants = []
for atom in ['H', 'F']:
    # Define the geometry of the molecule
    mol = gto.M(atom = [[atom, (0, 0, 0)]], basis = 'def2-tzvp', charge = 0, spin = 1)

    # To perform DFT we also need a grid
    grids = dft.gen_grid.Grids(mol)
    grids.level = 2
    grids.build()

    # And we will also need a mean-field object
    mf = dft.UKS(mol)
    mf.grids = grids
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, grad_order=2, name = atom,
                                energy=ground_truth_energy, omegas = omegas)

    reactants.append(molecule)
    reaction_energy -= ground_truth_energy

reaction = make_reaction(reactants, products, [1,1], [1], reaction_energy, name = 'HF_formation')

# Saving them
data_file = os.path.join(training_data_dirpath, "HF_formation.hdf5")
save(molecules = [HF_molecule], reactions = [reaction], fname = data_file)

# Loading them
load = loader(fname = data_file, randomize=True, training = False, config_omegas = [])
for _, system in tqdm(load, 'Molecules/reactions per file'):
    print(type(system), ''.join(chr(num) for num in system.name)) # We use training = False so system.name is a string