# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import os
from jax import grad, vmap, numpy as jnp
import jax
from tqdm import tqdm

from grad_dft.interface import molecule_from_pyscf, loader
from grad_dft.interface import saver as save
from grad_dft.molecule import Molecule, make_reaction

# In this basic tutorial we want to introduce the concept of a molecule, which is a class that contains
# all the information about a molecule that we need to compute its energy and its gradient.

########################### Initializing a Molecule object #########################################

# To prepare a molecule, we need to compute many properties of such system. We will use PySCF to do so,
# though we could use any other software. For example:
from pyscf import gto, dft
from grad_dft.utils.types import Array

# Define the geometry of the molecule
geometry = [["H", (0, 0, 0)], ["F", (0, 0, 1.1)]]
mol = gto.M(atom=geometry, basis="def2-tzvp", charge=0, spin=0)

# And we will also need a mean-field object
mf = dft.UKS(mol, xc="b3lyp")
mf.max_cycle = 0  # WE can select whether we want to converge the SCF or not
ground_truth_energy = mf.kernel()

# If we want to use the molecule to compute HF exact-exchange components we will need to decide which values of
# omega we want to use in the range separated Coulomb kernel erfc(omega*r)/r.
# omega = 0. indicates no range separation: the kernel will be 1/r.
omegas = [0.0, 0.4]

# Then we can use the following function to generate the molecule object
name = "HF"
HF_molecule = molecule_from_pyscf(
    mf, grad_order=2, name=name, energy=ground_truth_energy, omegas=omegas
)

# Alternatively we may compute and pass each of the properties of the molecule separately:
HF_molecule = Molecule(
    HF_molecule.grid,
    HF_molecule.atom_index,
    HF_molecule.nuclear_pos,
    HF_molecule.ao,
    HF_molecule.grad_ao,
    HF_molecule.grad_n_ao,
    HF_molecule.rdm1,
    HF_molecule.nuclear_repulsion,
    HF_molecule.h1e,
    HF_molecule.vj,
    HF_molecule.mo_coeff,
    HF_molecule.mo_occ,
    HF_molecule.mo_energy,
    HF_molecule.mf_energy,
    HF_molecule.s1e,
    HF_molecule.omegas,
    HF_molecule.chi,
    HF_molecule.rep_tensor,
    HF_molecule.energy,
    HF_molecule.basis,
    HF_molecule.name,
    HF_molecule.spin,
    HF_molecule.charge,
    HF_molecule.unit_Angstrom,
    HF_molecule.grid_level,
    HF_molecule.scf_iteration,
    HF_molecule.fock,
)

# Most of these properties are Arrays, others are floats or integers.
# molecule.grad_ao is a dictionary of arrays, indicating the n-th order gradients
# of the atomic orbitals, \nabla^n ao = \sum_i (\partial^n f / \partial x_i^n)

# Also worth mentioning that to avoid type errors, we convert strings (the basis, the name of the molecule)
# into integers
name_ints = jnp.array([ord(char) for char in name])
name = "".join(chr(num) for num in name_ints)
print(name, name_ints)

########################### Computing gradients #########################################

# Now that we have a molecule we can compute gradients with respect to some of the properties of the molecule.
# For example, we can compute the gradient of the electronic density with respect to the atomic orbitals.

# Let us compute |\nabla \rho|. In molecule.py we have defined the following function:


def grad_density(rdm1: Array, ao: Array, grad_ao: Array) -> Array:
    return 2 * jnp.einsum("...ab,ra,rbj->r...j", rdm1, ao, grad_ao)


grad_density_0 = grad_density(HF_molecule.rdm1, HF_molecule.ao, HF_molecule.grad_ao)


# Alternatively, we can compute grad_density as follows:
# Parallelizing over the spin (first vmap) and the atomic orbitals (second vmap) axes
def parallelized_density(rdm1: Array, ao: Array) -> Array:
    return jnp.einsum("ab,a,b->", rdm1, ao, ao)


grad_density_ao = vmap(
    vmap(grad(parallelized_density, argnums=1), in_axes=[None, 0]), in_axes=[0, None]
)(HF_molecule.rdm1, HF_molecule.ao)
grad_density_1 = jnp.einsum("...rb,rbj->r...j", grad_density_ao, HF_molecule.grad_ao)

# We can check we get the same result
print(
    "Are the two forms of computing the gradient of the density the same?",
    jnp.allclose(grad_density_0, grad_density_1),
)

# We can now compute one of the finite-range adimensional variables in the article
grad_density_norm = jnp.linalg.norm(grad_density_0, axis=-1)
density = HF_molecule.density()
# We need to avoid dividing by zero
x = jnp.where(
    density > 1e-27,
    grad_density_norm / (2 * (3 * jnp.pi**2) ** (1 / 3) * density ** (4 / 3)),
    0.0,
)
u = x**2 / (1 + x**2)
print("We can check the range is bounded between", jnp.min(u), jnp.max(u))

########################### Saving and loading #########################################

# Now let's talk about how to save and load a molecule (or a list of Molecules).
dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.join(dirpath, "data/examples/")
data_file = os.path.join(training_data_dirpath, "HF.hdf5")
save(molecules=[HF_molecule], fname=data_file)

# We can load the molecule from the file
# Select if we are going to train, as well as the omegas we will finally use
load = loader(fname=data_file, randomize=True, training=False, config_omegas=[])
for _, system in tqdm(load, "Molecules/reactions per file"):
    HF_molecule = system
    print(
        "Molecule name", "".join(chr(num) for num in HF_molecule.name)
    )  # We use training = False so molecule.name is a string


# We can also create reactions, save and load them. For example, let us emulate the formation reaction of HF
# from H and F atoms:
products = [HF_molecule]

reaction_energy = ground_truth_energy

reactants = []
for atom in ["H", "F"]:
    # Define the geometry of the molecule
    mol = gto.M(atom=[[atom, (0, 0, 0)]], basis="def2-tzvp", charge=0, spin=1)

    # To perform DFT we also need a grid
    grids = dft.gen_grid.Grids(mol)
    grids.level = 2
    grids.build()

    # And we will also need a mean-field object
    mf = dft.UKS(mol)
    mf.grids = grids
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(
        mf, grad_order=2, name=atom, energy=ground_truth_energy, omegas=omegas
    )

    reactants.append(molecule)
    reaction_energy -= ground_truth_energy

reaction = make_reaction(reactants, products, [1, 1], [1], reaction_energy, name="HF_formation")

# Saving them
data_file = os.path.join(training_data_dirpath, "HF_formation.hdf5")
save(molecules=[HF_molecule], reactions=[reaction], fname=data_file)

# Loading them
load = loader(fname=data_file, randomize=True, training=False, config_omegas=[])
for _, system in tqdm(load, "Molecules/reactions per file"):
    print(
        type(system), "".join(chr(num) for num in system.name)
    )  # We use training = False so system.name is a string
