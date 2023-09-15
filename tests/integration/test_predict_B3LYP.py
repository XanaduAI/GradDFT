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

""" These tests check the implementation of the B3LYP functional for two 
    molecule and RKS vs UKS DFT calcualtions.
"""

import pytest

# This only works on startup!
from jax.config import config

config.update("jax_enable_x64", True)

from grad_dft.interface import molecule_from_pyscf
from grad_dft.evaluate import make_scf_loop, make_jitted_scf_loop
from grad_dft.utils.types import Hartree2kcalmol
from grad_dft.popular_functionals import B3LYP

from openfermion import geometry_from_pubchem

from pyscf import gto, dft

import numpy as np


FUNCTIONAL = B3LYP
PARAMS = {"params": {}}

GEOMETRY = geometry_from_pubchem("water")
MOL_WATER = gto.M(atom=GEOMETRY, basis="def2-tzvp")

MOL_LI = gto.Mole()
MOL_LI.atom = "Li 0 0 0"
MOL_LI.basis = "def2-tzvp"
MOL_LI.spin = 1
MOL_LI.build()

@pytest.mark.parametrize("mol_and_name", [(MOL_WATER, "water"), (MOL_LI, "Li")])
def test_predict(mol_and_name: tuple[gto.Mole, str]) -> None:
    r"""Compare the total energy predicted by Grad-DFT for the B3LYP functional versus PySCF.
    The function is hard-coded for water and atoimic Li.

    Args:
        mol_and_name (tuple[gto.Mole, str]): PySCF molecule object and the name of the molecule.
    """
    mol, name = mol_and_name
    mf = dft.UKS(mol)
    mf.max_cycle = 0
    energy = mf.kernel()

    molecule = molecule_from_pyscf(mf, energy=energy, omegas=[0.0], scf_iteration=0)


    iterator = make_scf_loop(FUNCTIONAL, verbose=2, max_cycles=10)
    molecule_out = iterator(PARAMS, molecule)
    e_XND = molecule_out.energy

    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = "B3LYP"
    mf.max_cycle = 10
    e_DM = mf.kernel()
    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol=1)