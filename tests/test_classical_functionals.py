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
from jax import numpy as jnp
from popular_functionals import B3LYP, B88, LSDA, LYP, VWN, PW92

from interface.pyscf import molecule_from_pyscf

# This file aims to test, given some electronic density, whether our
# implementation of classical functionals closely matches libxc (pyscf default).

# again, this only works on startup!
from jax import config

from utils.types import Hartree2kcalmol

config.update("jax_enable_x64", True)

# First we define a molecule:
from pyscf import gto, dft
from train import molecule_predictor

mol = gto.M(atom="H 0 0 0; F 0 0 1.1")

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

params = freeze({"params": {}})

#### LSDA ####

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "LDA"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[])
predict_molecule = molecule_predictor(LSDA)
predicted_e, fock = predict_molecule(params, molecule)

lsdadiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
assert jnp.allclose(lsdadiff, 0, atol=1e1)

##### B88 ####

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "B88"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[])
predict_molecule = molecule_predictor(B88)
predicted_e, fock = predict_molecule(params, molecule)

b88diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
assert jnp.allclose(b88diff, 0, atol=1e1)

##### VWN ####

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "LDA_C_VWN"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[])
predict_molecule = molecule_predictor(VWN)
predicted_e, fock = predict_molecule(params, molecule)

vwndiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
assert jnp.allclose(vwndiff, 0, atol=1e1)

##### LYP ####

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "GGA_C_LYP"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[])
predict_molecule = molecule_predictor(LYP)
predicted_e, fock = predict_molecule(params, molecule)

lypdiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
assert jnp.allclose(lypdiff, 0, atol=1e1)

#### B3LYP ####

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "b3lyp"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[0.0])
predict_molecule = molecule_predictor(B3LYP)
predicted_e, fock = predict_molecule(params, molecule)

b3lypdiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
assert jnp.allclose(b3lypdiff, 0, atol=1e1)


#### PW92 ####

mf = dft.UKS(mol)
mf.grids = grids
mf.xc = "LDA_C_PW"
ground_truth_energy = mf.kernel()

molecule = molecule_from_pyscf(mf, omegas=[])
predict_molecule = molecule_predictor(PW92)
predicted_e, fock = predict_molecule(params, molecule)

pw92diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
assert jnp.allclose(pw92diff, 0, atol=1e1)
