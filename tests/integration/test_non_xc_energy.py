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
energy is correct in Grad-DFT ignoring the exchange and correlation functional.
We do so by comparing the dissociation energies of three molecules
(H_2, LiF and CaO) to the dissociation energy calculated by PySCF.
"""

from grad_dft.interface import molecule_from_pyscf

import numpy as np

from pyscf import gto
from pyscf import dft

from jax import config

import pytest

config.update("jax_enable_x64", True)

# Bond lengths in Angstroms. Taken from https://cccbdb.nist.gov/diatomicexpbondx.asp
H2_EXP_BOND_LENGTH = 0.741
LIF_EXP_BOND_LENGTH = 1.564
CAO_EXP_BOND_LENGTH = 1.822

SCF_ITERS = 200
NUM_POINTS_CURVE = 10
BOND_LENGTH_FRAC_CHANGE = 0.1
ERR_TOL = 1e-8

ATOMIC_SPECIES = ["H", "Li", "F", "Ca", "O"]
SPINS = [1, 1, 1, 0, 0]

# geometries (also test constants)
FREE_ATOMS_PYSCF = [
    gto.M(atom="%s  0.0   0.0    0.0" % (s), basis="sto-3g", spin=sigma)
    for s, sigma in zip(ATOMIC_SPECIES, SPINS)
]

H2_GEOMS = [
    """
    H  0.0   -%.5f   0.0
    H  0.0   %.5f    0.0 
    """
    % (bl / 2, bl / 2)
    for bl in np.linspace(
        (1 - BOND_LENGTH_FRAC_CHANGE) * H2_EXP_BOND_LENGTH,
        (1 + BOND_LENGTH_FRAC_CHANGE) * H2_EXP_BOND_LENGTH,
        NUM_POINTS_CURVE,
    )
]

H2_TRAJ = [
    gto.M(
        atom=geom,
        basis="sto-3g",
    )
    for geom in H2_GEOMS
]

LIF_GEOMS = [
    """
    LI  0.0   -%.5f   0.0
    F  0.0   %.5f    0.0 
    """
    % (bl / 2, bl / 2)
    for bl in np.linspace(
        (1 - BOND_LENGTH_FRAC_CHANGE) * LIF_EXP_BOND_LENGTH,
        (1 + BOND_LENGTH_FRAC_CHANGE) * LIF_EXP_BOND_LENGTH,
        NUM_POINTS_CURVE,
    )
]

LIF_TRAJ = [
    gto.M(
        atom=geom,
        basis="sto-3g",
    )
    for geom in LIF_GEOMS
]

LIF_GEOMS = [
    """
    LI  0.0   -%.5f   0.0
    F  0.0   %.5f    0.0 
    """
    % (bl / 2, bl / 2)
    for bl in np.linspace(
        (1 - BOND_LENGTH_FRAC_CHANGE) * LIF_EXP_BOND_LENGTH,
        (1 + BOND_LENGTH_FRAC_CHANGE) * LIF_EXP_BOND_LENGTH,
        NUM_POINTS_CURVE,
    )
]

LIF_TRAJ = [
    gto.M(
        atom=geom,
        basis="sto-3g",
    )
    for geom in LIF_GEOMS
]

CAO_GEOMS = [
    """
    CA  0.0   -%.5f   0.0
    O  0.0   %.5f    0.0 
    """
    % (bl / 2, bl / 2)
    for bl in np.linspace(
        (1 - BOND_LENGTH_FRAC_CHANGE) * LIF_EXP_BOND_LENGTH,
        (1 + BOND_LENGTH_FRAC_CHANGE) * LIF_EXP_BOND_LENGTH,
        NUM_POINTS_CURVE,
    )
]

CAO_TRAJ = [
    gto.M(
        atom=geom,
        basis="sto-3g",
    )
    for geom in LIF_GEOMS
]


def molecule_and_energies(geom: str) -> tuple[float, float]:
    r"""Calculate the total energy of an atomic geometry with PySCF and Grad-DFT with no XC component in the electronic energy

    Args:
        geom (str): the cartesian coordinates of an atomic geometry in the format: SPECIES1 X Y Z, SPECIES2 X Y Z ...

    Returns:
        tuple[float, float]: the energy predicts by PySCF and Grad-DFT
    """
    mf = dft.RKS(geom)
    mf.xc = "0.00*LDA"  # quick way of having no XC energy in PySCF
    E_pyscf = mf.kernel(max_cycle=SCF_ITERS)
    molecule = molecule_from_pyscf(mf, scf_iteration=SCF_ITERS)
    E_gdft = molecule.nonXC()
    return E_pyscf, E_gdft


@pytest.fixture
def free_atom_energies() -> dict[str, dict[str, float]]:
    r"""For all of the ATOMIC_SPECIES, calculate the isolated atom total energy for PySCF and Grad-DFT with no XC component in the electronic energy.

    Returns:
        dict[str, dict[str, float]]: total energies for PySCF and Grad-DFT
    """
    energies_pyscf = {}
    energies_gdft = {}

    for i, species in enumerate(ATOMIC_SPECIES):
        E_pyscf, E_gdft = molecule_and_energies(FREE_ATOMS_PYSCF[i])
        energies_pyscf[species] = E_pyscf
        energies_gdft[species] = E_gdft
        free_atom_error = np.abs(E_gdft - E_pyscf)
        assert (
            free_atom_error < ERR_TOL
        ), f"Total energy difference for {species} exceeds threshold: {free_atom_error}"

    return {"pyscf": energies_pyscf, "gdft": energies_gdft}


@pytest.mark.parametrize("species", ATOMIC_SPECIES)
def test_free_atom_energies(species: str, free_atom_energies: dict[str, dict[str, float]]) -> None:
    r"""Compare the total energies of the free atoms between PySCF and Grad-DFT with no XC component in the electronic energy

    Args:
        species (str): an atomic species, e.g He.
        free_atom_energies (dict[str, dict[str, float]]): the energies of the free atoms predicted by PySCF and Grad-DFT
    """
    E_pyscf = free_atom_energies["pyscf"][species]
    E_gdft = free_atom_energies["gdft"][species]
    error = np.abs(E_gdft - E_pyscf)
    assert error < ERR_TOL, f"Total energy difference for {species} exceeds threshold: {error}"


@pytest.mark.parametrize(
    "geom,species1,species2",
    [(geom, "H", "H") for geom in H2_TRAJ]
    + [(geom, "Li", "F") for geom in LIF_TRAJ]
    + [(geom, "Ca", "O") for geom in CAO_TRAJ],
)
def test_diatomic_molecule_energy(
    geom: str, species1: str, species2: str, free_atom_energies: dict[str, dict[str, float]]
) -> None:
    """Compare the total energies and dissociation energies predicted by PySCF and Grad-DFT with no XC component in the electronic energy

    Args:
        geom (str): the cartesian coordinates of an atomic geometry in the format: SPECIES1 X Y Z, SPECIES2 X Y Z ...
        species1 (str): an atomic species for one atom in the diatomic molecule, e.g Li.
        species2 (str): an atomic species for the other atom in the diatomic molecule, e.g F.
        free_atom_energies (dict[str, dict[str, float]]): total energies predicted by PySCF and Grad-DFT for the free atoms
    """
    E_pyscf, E_gdft = molecule_and_energies(geom)
    dis_energy_pyscf = E_pyscf - (
        free_atom_energies["pyscf"][species1] + free_atom_energies["pyscf"][species2]
    )
    dis_energy_gdft = E_gdft - (
        free_atom_energies["gdft"][species1] + free_atom_energies["gdft"][species2]
    )
    tot_energy_error = np.abs(E_pyscf - E_gdft)
    dis_energy_error = np.abs(dis_energy_pyscf - dis_energy_gdft)
    assert (
        tot_energy_error < ERR_TOL
    ), f"Total energy difference exceeds threshold: {tot_energy_error}"
    assert (
        dis_energy_error < ERR_TOL
    ), f"Dissociation energy difference exceeds threshold: {dis_energy_error}"
