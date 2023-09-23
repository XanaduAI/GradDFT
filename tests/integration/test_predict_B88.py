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

""" These tests check the implementation of the B88 functional for two 
    molecule and RKS vs UKS DFT calcualtions.
"""

import pytest

# This only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

from grad_dft import (
    molecule_from_pyscf,
    scf_loop, 
    diff_scf_loop
)
from grad_dft.utils.types import Hartree2kcalmol
from grad_dft.popular_functionals import B88

from pyscf import gto, dft
import jax.numpy as jnp


FUNCTIONAL = B88
PARAMS = {"params": {}}

MOL_WATER = gto.Mole()
MOL_WATER.atom = "O 0.0 0.0 0.0; H 0.2774 0.8929 0.2544; H 0.6068 -0.2383 -0.7169"
MOL_WATER.basis = "def2-tzvp"
MOL_WATER.build()

MOL_LI = gto.Mole()
MOL_LI.atom = "Li 0 0 0"
MOL_LI.basis = "def2-tzvp"
MOL_LI.spin = 1
MOL_LI.build()

SCF_ITERS = 10

@pytest.mark.parametrize("mol_and_name", [(MOL_WATER, "water"), (MOL_LI, "Li")])
def test_predict(mol_and_name: tuple[gto.Mole, str]) -> None:
    r"""Compare the total energy predicted by Grad-DFT for the B88 functional versus PySCF.
    The function is hard-coded for water and atoimic Li.

    Args:
        mol_and_name (tuple[gto.Mole, str]): PySCF molecule object and the name of the molecule.
    """
    mol, name = mol_and_name
    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    
    mf.xc = "B88"
    mf.max_cycle = 10
    e_DM = mf.kernel()
    
    # Start from Non-SCF density
    molecule = molecule_from_pyscf(mf, energy=e_DM, omegas=[], scf_iteration=0)

    iterator = scf_loop(FUNCTIONAL, verbose=2, cycles=10)
    molecule_out = iterator(PARAMS, molecule)
    e_XND = molecule_out.energy
    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1e-6), f"Energy difference with PySCF for B88 on {name} exceeds the threshold."


@pytest.mark.parametrize("mol_and_name", [(MOL_WATER, "water"), (MOL_LI, "Li")])
def test_jit(mol_and_name: tuple[gto.Mole, str]) -> None:
    r"""Compare the total energy predicted by Grad-DFT for the B88 functional versus PySCF.
    The function is hard-coded for water and atoimic Li.

    Args:
        mol_and_name (tuple[gto.Mole, str]): PySCF molecule object and the name of the molecule.
    """
    mol, name = mol_and_name
    if mol.spin == 0: mf = dft.RKS(mol)
    else: mf = dft.UKS(mol)
    
    mf.xc = "B88"
    mf.max_cycle = 0
    mf.kernel()
    
    # Start from Non-SCF density
    molecule = molecule_from_pyscf(mf, omegas=[], scf_iteration=0)

    iterator = scf_loop(FUNCTIONAL, verbose=2, cycles=10)
    molecule_out = iterator(PARAMS, molecule)
    e_XND = molecule_out.energy
    
    iterator = diff_scf_loop(FUNCTIONAL, cycles=10)
    molecule_out = iterator(PARAMS, molecule)
    e_XND_jit = molecule_out.energy

    kcalmoldiff = (e_XND - e_XND_jit) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1e-6), f"Energy difference with between jitted and non-jitted SCF for B88 on {name} exceeds the threshold."