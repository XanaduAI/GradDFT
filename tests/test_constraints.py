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

# H atom:
molH = gto.M(atom = 'H 0 0 0', spin = 1, basis = 'cc-pvqz')
grids = dft.gen_grid.Grids(molH)
grids.level = 3
grids.build()
mf = dft.UKS(molH)
mf.grids = grids
mf.xc = 'b3lyp'
ground_truth_energy = mf.kernel()
molecule1 = molecule_from_pyscf(mf, omegas = [0.])

molHp = gto.M(atom = 'H 0 0 0', charge = -1, spin = 0, basis = 'cc-pvqz')
grids = dft.gen_grid.Grids(molHp)
grids.level = 3
grids.build()
mf = dft.UKS(molHp)
mf.grids = grids
mf.xc = 'b3lyp'
ground_truth_energy = mf.kernel()
molecule2 = molecule_from_pyscf(mf, omegas = [0.])

params = freeze({'params': {}})

from constraints import constraint_x4, constraint_x6, constraint_x7, constraints_x1_c1, constraint_x2, constraint_c2, constraints_fractional_charge_spin, constraints_x3_c3_c4, constraint_x5

#### Constraint x1 ####
x1, c1 = constraints_x1_c1(B3LYP, params, molecule)
print(f'Does the functional B3LYP satisfy constraint x1?', all(x1))
print(f'Does the functional B3LYP satisfy constraint c1?', all(c1))

#### Constraint x2 ####
x2 = constraint_x2(B3LYP, params, molecule)
print(f'Does the functional B3LYP satisfy constraint x2?', x2)

#### Constraint c2 ####
c2 = constraint_c2(LYP, params, molecule)
print(f'Does the functional LYP satisfy constraint c2?', c2)

#### Constraint x3, c3, c4 ####
x3, (c3, c4) = constraints_x3_c3_c4(B3LYP, params, molecule, gamma = 2.)
print(f'Does the functional B3LYP satisfy constraint x3?', all(x3))
print(f'Does the functional B3LYP satisfy constraint c3?', c3)
print(f'Does the functional B3LYP satisfy constraint c4?', c4)

#### Constraint x4 ####
#x4s2, x4q2, x4qs2, x4s4  = constraint_x4(B3LYP, params, molecule)
#print(f'Does the functional B3LYP satisfy constraints x4?', x4s2, x4q2, x4qs2, x4s4)

#### Constraint x5 ####
x5inf, x50 = constraint_x5(B3LYP, params, molecule)
print(f'Does the functional B3LYP satisfy constraint x5?', x5inf, x50)

#### Constraint x6 ####
x61, x62 = constraint_x6(B3LYP, params, molecule)
print(f'Does the functional B3LYP satisfy constraint x6?', x61, x62)

#### Constraint x7 ####
x7 = constraint_x7(B3LYP, params, molecule)
print(f'Does the functional B3LYP satisfy constraint x7?', x7)

#### Constraint fractional charge & spin ####
fcs = constraints_fractional_charge_spin(B3LYP, params, molecule1, molecule2, gamma = 0.5, mol = molH)
print(f'Does the functional B3LYP satisfy the fractional charge & spin constrain?', fcs)