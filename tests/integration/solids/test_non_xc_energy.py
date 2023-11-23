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

"""The goal of this module is to test that the implementation of the total
energy is correct in Grad-DFT (for solids) ignoring the exchange and correlation functional.
We do so by comparing the energy volume curve of [ficticious] solid Hydrogren to the energy
calculated by PySCF.
"""

from grad_dft import solid_from_pyscf

import numpy as np

from pyscf.pbc import gto, dft

from jax import config

import pytest

config.update("jax_enable_x64", True)

# Bond lengths in Angstroms. Taken from https://cccbdb.nist.gov/diatomicexpbondx.asp.
# This is for the molecule obviously, but we will use it as the solid lattice constant.
H2_EXP_BOND_LENGTH = 1.3984

SCF_ITERS = 200
NUM_POINTS_CURVE = 10
LAT_PARAM_FRAC_CHANGE = 0.1
ERR_TOL = 1e-8
KPTS = [2, 1, 1]

H2_LAT_VECS = [
    np.array(
        [
            [p, 0.0, 0.0],
            [0.0, p, 0.0],
            [0.0, 0.0, p]
        ]
    ) for p in np.linspace(
        2 * (1 - LAT_PARAM_FRAC_CHANGE) * H2_EXP_BOND_LENGTH,
        2 * (1 + LAT_PARAM_FRAC_CHANGE) * H2_EXP_BOND_LENGTH,
        NUM_POINTS_CURVE,
    )
]

H2_GEOMS = [
    """
    H  0.0   0.0   0.0
    H  %.5f   0.0    0.0 
    """
    % (bl)
    for bl in np.linspace(
        (1 - LAT_PARAM_FRAC_CHANGE) * H2_EXP_BOND_LENGTH,
        (1 + LAT_PARAM_FRAC_CHANGE) * H2_EXP_BOND_LENGTH,
        NUM_POINTS_CURVE,
    )
]

H2_TRAJ = [
    gto.M(
        a = lat_vec,
        atom=geom,
        basis="sto-3g",
    )
    for geom, lat_vec in zip(H2_GEOMS, H2_LAT_VECS)
]



def solid_and_energies(geom) -> tuple[float, float]:
    r"""Calculate the total energy of crystal geometry with PySCF and Grad-DFT with no XC component in the electronic energy

    Args:
        geom (gto.M): The periodicic gaussian orbital object from PySCF. Contains atomic positions, basis set and lattice vectors
    Returns:
        tuple[float, float]: the energy predicts by PySCF and Grad-DFT
    """
    kmf = dft.KRKS(geom, kpts=geom.make_kpts(KPTS))
    kmf.xc = "0.00*LDA"  # quick way of having no XC energy in PySCF
    E_pyscf = kmf.kernel(max_cycle=SCF_ITERS)
    sol = solid_from_pyscf(kmf)
    E_gdft = sol.nonXC()
    return E_pyscf, E_gdft


@pytest.mark.parametrize(
    "geom",
    H2_TRAJ,
)
def test_diatomic_molecule_energy(geom) -> None:
    """Compare the total energies as a function of solid lattice parameter predicted by PySCF and Grad-DFT with no XC component in the electronic energy

    Args:
        geom (gto.M): The periodicic gaussian orbital object from PySCF. Contains atomic positions, basis set and lattice vectors
    """
    E_pyscf, E_gdft = solid_and_energies(geom)
    tot_energy_error = np.abs(E_pyscf - E_gdft)
    assert (
        tot_energy_error < ERR_TOL
    ), f"Total energy difference exceeds threshold: {tot_energy_error}"
