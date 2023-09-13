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
from grad_dft.functional import DM21 # A class, needs to be instanciated!
from grad_dft.popular_functionals import B3LYP, B88, LSDA, LYP, VWN, PW92

from grad_dft.interface.pyscf import molecule_from_pyscf
from grad_dft.external import NeuralNumInt
from grad_dft.external import Functional

# This file aims to test, given some electronic density, whether our
# implementation of classical functionals closely matches libxc (pyscf default).

# again, this only works on startup!
from jax import config

from grad_dft.utils.types import Hartree2kcalmol
from openfermion import geometry_from_pubchem

config.update("jax_enable_x64", True)

# First we define a molecule:
from pyscf import gto, dft
from grad_dft.train import molecule_predictor

mol = gto.M(atom="H 0 0 0; F 0 0 1.1")

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

params = freeze({"params": {}})

mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis = "def2-tzvp")
mol = gto.M(atom="Li 0 0 0", spin = 1, basis = "def2-tzvp")

mols = [
    gto.M(atom="H 0 0 0; F 0 0 1.1", basis = "def2-tzvp"),
    gto.M(atom="Li 0 0 0", spin = 1, basis = "def2-tzvp"),
]

#### LSDA ####
@pytest.mark.parametrize("mol", mols)
def test_lda(mol):
    mf = dft.UKS(mol)
    mf.grids = grids
    mf.xc = "LDA" # LDA is the same as LDA_X.
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(LSDA)
    predicted_e, fock = predict_molecule(params, molecule)

    lsdadiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(lsdadiff, 0, atol=1e-3)

##### B88 ####
@pytest.mark.parametrize("mol", mols)
def test_b88(mol):
    mf = dft.UKS(mol)
    mf.grids = grids
    mf.xc = "B88"
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(B88)
    predicted_e, fock = predict_molecule(params, molecule)

    b88diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(b88diff, 0, atol=1e-3)

##### VWN ####
@pytest.mark.parametrize("mol", mols)
def test_vwn(mol):
    mf = dft.UKS(mol)
    mf.grids = grids
    mf.xc = "LDA_C_VWN"
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(VWN)
    predicted_e, fock = predict_molecule(params, molecule)

    vwndiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
    
    assert not jnp.isnan(fock).any()
    assert jnp.allclose(vwndiff, 0, atol=1)

##### LYP ####
# This test differs slightly due to the use of the original LYP functional definition
# in C. Lee, W. Yang, and R. G. Parr., Phys. Rev. B 37, 785 (1988) (doi: 10.1103/PhysRevB.37.785)
# instead of the one in libxc: B. Miehlich, A. Savin, H. Stoll, and H. Preuss., Chem. Phys. Lett. 157, 200 (1989) (doi: 10.1016/0009-2614(89)87234-3)
@pytest.mark.parametrize("mol", mols)
def test_lyp(mol):
    mf = dft.UKS(mol)
    mf.grids = grids
    mf.xc = "GGA_C_LYP"
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(LYP)
    predicted_e, fock = predict_molecule(params, molecule)

    lypdiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(lypdiff, 0, atol=1)

#### B3LYP ####
# This test will only pass if you set B3LYP_WITH_VWN5 = True in pyscf_conf.py.
# See pyscf_conf.py in .github/workflows
# This test differs slightly due to the use of the original LYP functional definition
# in C. Lee, W. Yang, and R. G. Parr., Phys. Rev. B 37, 785 (1988) (doi: 10.1103/PhysRevB.37.785)
# instead of the one in libxc: B. Miehlich, A. Savin, H. Stoll, and H. Preuss., Chem. Phys. Lett. 157, 200 (1989) (doi: 10.1016/0009-2614(89)87234-3)
@pytest.mark.parametrize("mol", mols)
def test_b3lyp(mol):
    mf = dft.UKS(mol)
    mf.grids = grids
    mf.xc = "b3lyp"
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[0.0])
    predict_molecule = molecule_predictor(B3LYP)
    predicted_e, fock = predict_molecule(params, molecule)

    b3lypdiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(b3lypdiff, 0, atol=1)


#### PW92 ####
@pytest.mark.parametrize("mol", mols)
def test_pw92(mol):
    mf = dft.UKS(mol)
    mf.grids = grids
    mf.xc = "LDA_C_PW"
    ground_truth_energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, omegas=[])
    predict_molecule = molecule_predictor(PW92)
    predicted_e, fock = predict_molecule(params, molecule)

    pw92diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(pw92diff, 0, atol=1e-3)


#### DM21 ####
@pytest.mark.parametrize("mol", mols)
def test_dm21(mol):
    mf = dft.UKS(mol)
    mf._numint = NeuralNumInt(Functional.DM21)
    ground_truth_energy = mf.kernel()

    functional = DM21() # Note that DM21 is a class, that needs to be instantiated.
    params = functional.generate_DM21_weights() 

    molecule = molecule_from_pyscf(mf, omegas=[0.0, 0.4])
    predict_molecule = molecule_predictor(functional)
    predicted_e, fock = predict_molecule(params, molecule)

    dm21diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(dm21diff, 0, atol=1)