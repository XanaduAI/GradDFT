from flax.core import freeze
from jax import numpy as jnp
from popular_functionals import B3LYP, B88, LSDA, LYP, VWN, PW92

from interface.pyscf import molecule_from_pyscf

# This file aims to test, given some electronic density, whether our
# implementation of classical functionals closely matches libxc (pyscf default).

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

# First we define a molecule:
from pyscf import gto, dft
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1')

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = 'b3lyp'
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas = [0.])

params = freeze({'params': {}})

from constraints import constraint_x1, constraint_x2, constraint_x3

#### Constraint x1 ####
constraint_satisfaction = constraint_x1(B3LYP, params, molecule)
print('Does the functional satisfy constraint x1?', all(constraint_satisfaction))

#### Constraint x2 ####
constraint_satisfaction = constraint_x2(B3LYP, params, molecule)
print('Does the functional satisfy constraint x2?', constraint_satisfaction)

#### Constraint x3 ####
constraint_satisfaction = constraint_x3(B3LYP, params, molecule)
print('Does the functional satisfy constraint x3?', constraint_satisfaction)