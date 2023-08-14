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

from flax.core import freeze
from grad_dft.popular_functionals import B3LYP, LYP

from grad_dft.interface.pyscf import molecule_from_pyscf

# This file aims to test some of the constraints implemented in constraints.py.

from jax import config

config.update("jax_enable_x64", True)

# First we define a molecule:
from pyscf import gto, dft

mol = gto.M(atom="H 0 0 0; F 0 0 1.1")

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "b3lyp"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[0.0])

# H atom:
molH = gto.M(atom="H 0 0 0", spin=1, basis="cc-pvqz")
grids = dft.gen_grid.Grids(molH)
grids.level = 3
grids.build()
mf = dft.UKS(molH)
mf.grids = grids
mf.xc = "b3lyp"
ground_truth_energy = mf.kernel()
molecule1e = molecule_from_pyscf(mf, omegas=[0.0])

# Negatively charged H atom
molHp = gto.M(atom="H 0 0 0", charge=-1, spin=0, basis="cc-pvqz")
grids = dft.gen_grid.Grids(molHp)
grids.level = 3
grids.build()
mf = dft.UKS(molHp)
mf.grids = grids
mf.xc = "b3lyp"
ground_truth_energy = mf.kernel()
molecule2e = molecule_from_pyscf(mf, omegas=[0.0])

params = freeze({"params": {}})

from grad_dft.constraints import (
    constraint_c6,
    constraint_x4,
    constraint_x6,
    constraint_x7,
    constraint_xc2,
    constraint_xc4,
    constraints_x1_c1,
    constraint_x2,
    constraint_c2,
    constraints_fractional_charge_spin,
    constraints_x3_c3_c4,
    constraint_x5,
)

#### Constraint x1 ####
x1, c1 = constraints_x1_c1(B3LYP, params, molecule)
print(f"Quadratic loss of the functional B3LYP from constraint x1?", x1)
print(f"Quadratic loss of the functional B3LYP from constraint c1?", c1)

#### Constraint x2 ####
x2 = constraint_x2(B3LYP, params, molecule)
print(f"Quadratic loss of the functional B3LYP from constraint x2?", x2)

#### Constraint c2 ####
c2 = constraint_c2(LYP, params, molecule)
print(f"Quadratic loss of the functional LYP from constraint c2?", c2)

#### Constraint x3, c3, c4 ####
x3, (c3, c4) = constraints_x3_c3_c4(B3LYP, params, molecule, gamma=2.0)
print(f"Quadratic loss of the functional B3LYP from constraint x3?", x3)
print(f"Quadratic loss of the functional B3LYP from constraint c3?", c3)
print(f"Quadratic loss of the functional B3LYP from constraint c4?", c4)

#### Constraint x4 #### This requires masks for the appropriate functional
# x4s2, x4q2, x4qs2, x4s4  = constraint_x4(B3LYP, params, molecule, s2_mask, q2_mask, qs2_mask, s4_mask)
# print(f'Quadratic loss of the functional B3LYP from constraints x4?', x4s2, x4q2, x4qs2, x4s4)

#### Constraint x5 ####
x5inf, x50 = constraint_x5(B3LYP, params, molecule)
print(f"Quadratic loss of the functional B3LYP from constraint x5?", x5inf, x50)

#### Constraint x6 ####
x61, x62 = constraint_x6(B3LYP, params, molecule)
print(f"Quadratic loss of the functional B3LYP from constraint x6?", x61, x62)

#### Constraint x7 ####
x7 = constraint_x7(B3LYP, params, molecule2e)
print(f"Quadratic loss of the functional B3LYP from constraint x7?", x7)

#### Constraint c6 ####
c6 = constraint_c6(B3LYP, params, molecule)
print(f"Quadratic loss of the functional B3LYP from constraint c6?", c6)

#### Constraint xc2 ####
xc2 = constraint_xc2(B3LYP, params, molecule)
print(f"Quadratic loss of the functional B3LYP from constraint xc2?", xc2)

#### Constraint xc4 ####
xc4 = constraint_xc4(B3LYP, params, molecule2e)
print(f"Quadratic loss of the functional B3LYP from constraint xc4?", xc4)

#### Constraint fractional charge & spin ####
fcs = constraints_fractional_charge_spin(B3LYP, params, molecule1e, molecule2e, gamma=0.5, mol=molH)
print(
    f"Quadratic loss of the functional B3LYP from the fractional charge & spin constrain (xc1)?",
    fcs,
)
