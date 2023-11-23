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
from grad_dft import (
    solid_from_pyscf, 
    energy_predictor, # A class, needs to be instanciated!
    B88, LSDA, VWN, PW92
)
from grad_dft.utils.types import Hartree2kcalmol
import numpy as np


# This file aims to test, given some electronic density, whether our
# implementation of popular functionals closely matches libxc (pyscf default).

# These tests are specifically for solids. We test only LDAs and GGGas here.

# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)

# First we define a molecule:
from pyscf.pbc import gto, dft

PARAMS = freeze({"params": {}})
DIFF_TOL = 1e-3 # in KCal/Mol so is quite small
KPTS = [2, 1, 1]

# Look at ficticious solid Hydrogen and Lithium 

# Bond lengths in Angstroms. Taken from https://cccbdb.nist.gov/diatomicexpbondx.asp.
# This is for the molecule obviously, but we will use it as the solid lattice constant.
H2_EXP_BOND_LENGTH = 1.3984
LI2_EXP_BOND_LENGTH = 5.0512

LAT_VEC_H = 2 * np.array(
    [
        [H2_EXP_BOND_LENGTH, 0.0, 0.0],
        [0.0, H2_EXP_BOND_LENGTH, 0.0],
        [0.0, 0.0, H2_EXP_BOND_LENGTH]
    ]
)

LAT_VEC_LI = 2 * np.array(
    [
        [LI2_EXP_BOND_LENGTH, 0.0, 0.0],
        [0.0, LI2_EXP_BOND_LENGTH, 0.0],
        [0.0, 0.0, LI2_EXP_BOND_LENGTH]
    ]
)

GEOM_H = "H 0.0 0.0 0.0; H %.5f 0.0 0.0" % (H2_EXP_BOND_LENGTH)
GEOM_LI = "H 0.0 0.0 0.0; H %.5f 0.0 0.0" % (H2_EXP_BOND_LENGTH)

sols = [
    gto.M(
        a = LAT_VEC_H,
        atom=GEOM_H,
        basis="sto-3g",
    ),
    gto.M(
        a = LAT_VEC_LI,
        atom=GEOM_LI,
        basis="sto-3g",
    )
]

#### LSDA ####
@pytest.mark.parametrize("sol", sols)
def test_lda(sol):
    kmf = dft.KRKS(sol, kpts=sol.make_kpts(KPTS))
    kmf.xc = "LDA" # LDA is the same as LDA_X.
    ground_truth_energy = kmf.kernel()

    gd_sol = solid_from_pyscf(kmf)
    compute_energy = energy_predictor(LSDA)
    predicted_e, fock = compute_energy(PARAMS, gd_sol)

    lsdadiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(lsdadiff, 0, atol=DIFF_TOL)

##### B88 ####
@pytest.mark.parametrize("sol", sols)
def test_b88(sol):
    kmf = dft.KRKS(sol, kpts=sol.make_kpts(KPTS))
    kmf.xc = "B88"
    ground_truth_energy = kmf.kernel()

    gd_sol = solid_from_pyscf(kmf)
    compute_energy = energy_predictor(B88)
    predicted_e, fock = compute_energy(PARAMS, gd_sol)

    b88diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(b88diff, 0, atol=DIFF_TOL)


##### VWN ####
@pytest.mark.parametrize("sol", sols)
def test_vwn(sol):
    kmf = dft.KRKS(sol, kpts=sol.make_kpts(KPTS))
    kmf.xc = "LDA_C_VWN"
    ground_truth_energy = kmf.kernel()

    gd_sol = solid_from_pyscf(kmf)
    compute_energy = energy_predictor(VWN)
    predicted_e, fock = compute_energy(PARAMS, gd_sol)

    vwndiff = (ground_truth_energy - predicted_e) * Hartree2kcalmol
    
    assert not jnp.isnan(fock).any()
    assert jnp.allclose(vwndiff, 0, atol=DIFF_TOL)


#### PW92 ####
@pytest.mark.parametrize("sol", sols)
def test_pw92(sol):
    kmf = dft.KRKS(sol, kpts=sol.make_kpts(KPTS))
    kmf.xc = "LDA_C_PW"
    ground_truth_energy = kmf.kernel()

    gd_sol = solid_from_pyscf(kmf)
    compute_energy = energy_predictor(PW92)
    predicted_e, fock = compute_energy(PARAMS, gd_sol)

    pw92diff = (ground_truth_energy - predicted_e) * Hartree2kcalmol

    assert not jnp.isnan(fock).any()
    assert jnp.allclose(pw92diff, 0, atol=DIFF_TOL)
