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

import os
import warnings

import pytest

from grad_dft.interface import molecule_from_pyscf
from grad_dft.external import NeuralNumInt
from grad_dft.external import Functional

# This only works on startup!
from jax.config import config

from grad_dft.train import molecule_predictor

config.update("jax_enable_x64", True)

dirpath = os.path.dirname(os.path.dirname(__file__))
import sys

sys.path.append(dirpath)
config_path = os.path.normpath(dirpath + "/config/config.json")

data_path = os.path.normpath(dirpath + "/data")
model_path = os.path.normpath(dirpath + "/DM21_model")

learning_rate = 1e-3

from grad_dft.interface import molecule_from_pyscf
from grad_dft.evaluate import make_scf_loop, make_orbital_optimizer
from grad_dft.external.density_functional_approximation_dm21.density_functional_approximation_dm21.compute_hfx_density import (
    get_hf_density,
)

from pyscf import gto, dft, cc, scf
import jax.numpy as jnp
from grad_dft.functional import DM21
from grad_dft.utils.types import Hartree2kcalmol

functional = DM21()
params = functional.generate_DM21_weights()


###################### Closed shell ############################

MOL_WATER = gto.Mole()
MOL_WATER.atom = "O 0.0 0.0 0.0; H 0.2774 0.8929 0.2544; H 0.6068 -0.2383 -0.7169"
MOL_WATER.basis = "def2-tzvp"
MOL_WATER.build()
mf2 = scf.RHF(MOL_WATER)
mf2.kernel()
mycc = cc.CCSD(mf2).run()
ccsd_energy = mycc.e_tot


@pytest.mark.parametrize("mol", [MOL_WATER])
def test_predict(mol):
    mf = dft.UKS(mol)
    mf.max_cycle = 0
    energy = mf.kernel()
    ## Load the molecule, RKS
    warnings.warn("Remember to set the grid level to 3 in the config file!")

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0.0, 0.4], scf_iteration=0)

    # tx = adam(learning_rate = learning_rate)
    # iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    # e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional, verbose=2, max_cycles=10)
    e_XND = iterator(params, molecule)

    mf = dft.RKS(mol)
    mf._numint = NeuralNumInt(Functional.DM21)
    mf.max_cycle = 10
    e_DM = mf.kernel()

    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1)


##################
# test_predict(mol)


###################### Open shell ############################

mol = gto.Mole()
mol.atom = "Li 0 0 0"
mol.basis = "def2-tzvp"  # alternatively basis_set_exchange.api.get_basis(name='cc-pvdz', fmt='nwchem', elements='Co')
mol.spin = 1
mol.build()
mf = dft.UKS(mol)
mf.max_cycle = 0
energy = mf.kernel()

grid = mf.grids


@pytest.mark.parametrize("mol", [mol])
def test_predict(mol):
    mf = dft.UKS(mol)
    mf.max_cycle = 0
    energy = mf.kernel()
    ## Load the molecule, UKS
    warnings.warn("Remember to set the grid level to 3 in the config file!")

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0, 0.4], scf_iteration=0)

    # tx = adam(learning_rate = learning_rate)
    # iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    # e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional, verbose=2, max_cycles=1)
    e_XND = iterator(params, molecule)

    mf = dft.UKS(mol)
    mf._numint = NeuralNumInt(Functional.DM21)
    mf.max_cycle = 1
    e_DM = mf.kernel()

    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1)


##################
# test_predict(mol)


###################### Check against DM21 tests ############################


def test_rks():
    warnings.warn(
        "Remember to set the basis to sto-3g in the config file before running this test, and the grid level to 3."
    )
    ni = NeuralNumInt(Functional.DM21)

    mol = gto.Mole()
    mol.atom = [["Ne", 0.0, 0.0, 0.0]]
    mol.basis = "sto-3g"
    mol.build()

    mf = dft.RKS(mol)
    mf.small_rho_cutoff = 1.0e-20
    mf._numint = ni
    e_DM = mf.run().e_tot  # Expected -126.898521

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0.0, 0.4])

    # tx = adam(learning_rate = learning_rate)
    # iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    # e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional, verbose=2, functional_type="DM21")
    e_XND = iterator(params, molecule)

    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1)


def test_uks():
    warnings.warn(
        "Remember to set the basis to sto-3g in the config file before running this test, and the grid level to 3."
    )
    ni = NeuralNumInt(Functional.DM21)

    mol = gto.Mole()
    mol.atom = [["C", 0.0, 0.0, 0.0]]
    mol.spin = 2
    mol.basis = "sto-3g"
    mol.build()

    mf = dft.UKS(mol)
    mf.small_rho_cutoff = 1.0e-20
    mf._numint = ni
    e_DM = mf.run().e_tot  # Expected -37.34184876

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0.0, 0.4])

    # tx = adam(learning_rate = learning_rate)
    # iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    # e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional, verbose=2, functional_type="DM21")
    e_XND = iterator(params, molecule)

    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1)


##################
# test_rks()
# test_uks()


molecule_name = "CoC"
mol = gto.Mole()
mol.atom = [["Co", [0, 0, 0]], ["C", [1.56, 0, 0]]]  # def2-tzvp
mol.basis = (
    "def2-tzvp"  # basis_set_exchange.api.get_basis(name='cc-pvdz', fmt='nwchem', elements='Co')
)
mol.spin = 1
mol.unit = "angstrom"
mol.build()
mf = dft.UKS(mol)
energy = mf.kernel()

grid = mf.grids

# test_predict(mol)
