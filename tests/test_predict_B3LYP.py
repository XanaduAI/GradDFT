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

from interface import molecule_from_pyscf

# This only works on startup!
from jax.config import config

config.update("jax_enable_x64", True)

dirpath = os.path.dirname(os.path.dirname(__file__))
import sys

sys.path.append(dirpath)
config_path = os.path.normpath(dirpath + "/config/config.json")

data_path = os.path.normpath(dirpath + "/data")
model_path = os.path.normpath(dirpath + "/DM21_model")

learning_rate = 1e-3

from interface import molecule_from_pyscf
from evaluate import make_scf_loop, make_orbital_optimizer
from openfermion import geometry_from_pubchem

from pyscf import gto, dft, cc, scf
import numpy as np
from utils.types import Hartree2kcalmol

from popular_functionals import B3LYP


params = {"params": {}}


###################### Closed shell ############################

molecule_name = "water"
geometry = geometry_from_pubchem(molecule_name)
mol = gto.M(atom=geometry, basis="def2-tzvp")
mol.build()
mf2 = scf.RHF(mol)
mf2.kernel()
mycc = cc.CCSD(mf2).run()
ccsd_energy = mycc.e_tot
mf = dft.UKS(mol)
# mf.xc = 'B3LYP'
mf.max_cycle = 0
mf.kernel()

functional = B3LYP
grid = mf.grids


def test_predict(mf, energy):
    ## Load the molecule, RKS
    warnings.warn("Remember to set the grid level to 3 in the config file!")

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0.0], scf_iteration=0)

    # tx = adam(learning_rate = learning_rate)
    # iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    # e_XND_DF4T = iterator(params, molecule)

    mf = dft.UKS(mol)
    mf.xc = "B3LYP"
    mf.max_cycle = 10
    e_DM = mf.kernel()

    iterator = make_scf_loop(functional, verbose=2, max_cycles=10)
    e_XND = iterator(params, molecule)

    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol=1e1)


##################
test_predict(mf, energy=ccsd_energy)


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


def test_predict(mf, energy):
    ## Load the molecule, UKS
    warnings.warn("Remember to set the grid level to 3 in the config file!")

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0.0], scf_iteration=0)

    # tx = adam(learning_rate = learning_rate)
    # iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    # e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional, verbose=2, max_cycles=10)
    e_XND = iterator(params, molecule)

    mf = dft.UKS(mol)
    mf.xc = "B3LYP"
    mf.max_cycle = 10
    e_DM = mf.kernel()

    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol=1e1)


##################
test_predict(mf, energy=ccsd_energy)
