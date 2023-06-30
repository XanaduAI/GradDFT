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

from constraints import constraint_x1, constraint_x2, constraints_x3_c3_c4

#### Constraint x1 ####
x1 = constraint_x1(B3LYP, params, molecule)
print('Does the functional satisfy constraint x1?', all(x1))

#### Constraint x2 ####
x2 = constraint_x2(B3LYP, params, molecule)
print('Does the functional satisfy constraint x2?', x2)

#### Constraint x3, c3, c4 ####
x3, (c3, c4) = constraints_x3_c3_c4(B3LYP, params, molecule, gamma = 2.)
print('Does the functional satisfy constraint x3?', all(x3))
print('Does the functional satisfy constraint c3?', c3)
print('Does the functional satisfy constraint c4?', c4)