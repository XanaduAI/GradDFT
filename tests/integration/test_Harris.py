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
import pytest
from grad_dft.popular_functionals import B3LYP, B88, LSDA, LYP, VWN, PW92

from grad_dft.interface.pyscf import molecule_from_pyscf

# This file aims to test, given some electronic density, whether our
# implementation of classical functionals closely matches libxc (pyscf default).

# again, this only works on startup!
from jax import config

from grad_dft.utils.types import Hartree2kcalmol
from openfermion import geometry_from_pubchem
from grad_dft.train import molecule_predictor, Harris_energy_predictor

config.update("jax_enable_x64", True)

# First we define a molecule:
from pyscf import gto, dft

params = freeze({"params": {}})

mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis = 'def2-tzvp')
mol = gto.M(atom="Li 0 0 0", spin = 1, basis = 'def2-tzvp')

mols = [
    gto.M(atom="H 0 0 0; F 0 0 1.1", basis = 'def2-tzvp'),
    gto.M(atom="Li 0 0 0", spin = 1, basis = 'def2-tzvp'),
]


@pytest.mark.parametrize("mol", mols)
def test_Harries_LDA(mol):
    mf = dft.UKS(mol)
    mf.xc = "LDA" # LDA is the same as LDA_X.
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(LSDA)
    predicted_e, fock = predict_molecule(params, molecule)

    Harris_predict = Harris_energy_predictor(LSDA)
    Harries_e = Harris_predict(params, molecule)

    diff = (Harries_e - predicted_e) * Hartree2kcalmol
    assert jnp.allclose(diff, 0, atol=1e2) and jnp.less_equal(diff, 0)


@pytest.mark.parametrize("mol", mols)
def test_Harries_B88(mol):
    mf = dft.UKS(mol)
    mf.xc = "B88" # LDA is the same as LDA_X.
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(B88)
    predicted_e, fock = predict_molecule(params, molecule)

    Harris_predict = Harris_energy_predictor(B88)
    Harries_e = Harris_predict(params, molecule)

    diff = (Harries_e - predicted_e) * Hartree2kcalmol
    assert jnp.allclose(diff, 0, atol=1e2) and jnp.less_equal(diff, 0)

