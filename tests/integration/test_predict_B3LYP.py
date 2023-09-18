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

from grad_dft import molecule_from_pyscf, make_scf_loop
from grad_dft.utils.types import Hartree2kcalmol
from grad_dft.popular_functionals import B3LYP

from pyscf import gto, dft

import jax.numpy as jnp


FUNCTIONAL = B3LYP
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


# This test will only pass if you set B3LYP_WITH_VWN5 = True in pyscf_conf.py.
# See pyscf_conf.py in .github/workflows
# This test differs slightly due to the use of the original LYP functional definition
# in C. Lee, W. Yang, and R. G. Parr., Phys. Rev. B 37, 785 (1988) (doi: 10.1103/PhysRevB.37.785)
# instead of the one in libxc: B. Miehlich, A. Savin, H. Stoll, and H. Preuss., Chem. Phys. Lett. 157, 200 (1989) (doi: 10.1016/0009-2614(89)87234-3)
@pytest.mark.parametrize("mol_and_name", [(MOL_WATER, "water"), (MOL_LI, "Li")])
def test_predict(mol_and_name: tuple[gto.Mole, str]) -> None:
    r"""Compare the total energy predicted by Grad-DFT for the B3LYP functional versus PySCF.
    The function is hard-coded for water and atoimic Li.

    Args:
        mol_and_name (tuple[gto.Mole, str]): PySCF molecule object and the name of the molecule.
    """
    mol, name = mol_and_name
    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = "B3LYP"
    mf.max_cycle = 10
    e_DM = mf.kernel()

    molecule = molecule_from_pyscf(mf, energy=e_DM, omegas=[0.0], scf_iteration=0)

    iterator = make_scf_loop(FUNCTIONAL, verbose=2, max_cycles=10)
    molecule_out = iterator(PARAMS, molecule)
    e_XND = molecule_out.energy
    kcalmoldiff = (e_XND - e_DM) * Hartree2kcalmol
    assert jnp.allclose(kcalmoldiff, 0, atol=1), f"Energy difference with PySCF for B3LYP on {name} exceeds the threshold."