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
from typing import Union

from pyscf import gto
from pyscf import dft

from jax import config

config.update("jax_enable_x64", True)

# Bond lengths in Angstroms. Taken from https://cccbdb.nist.gov/diatomicexpbondx.asp
H2_EXP_BOND_LENGTH = 0.741
LIF_EXP_BOND_LENGTH = 1.564
CAO_EXP_BOND_LENGTH = 1.822

SCF_ITERS = 200
NUM_POINTS_CURVE = 10
BOND_LENGTH_FRAC_CHANGE = 0.05

# Calculate free atom energies first
atomic_species = ["H", "Li", "F", "Ca", "O"]
spins = [1, 1, 1, 0, 0]
free_atoms_pyscf = [
    gto.M(
        atom = '%s  0.0   0.0    0.0' % (s), 
        basis = 'sto-3g', 
        spin=sigma
    ) for s, sigma in zip(atomic_species, spins)
]

free_atom_energies_pyscf = {}
free_atom_energies_gdft = {}
free_atom_errs = []

for i, species in enumerate(atomic_species):
    mf = dft.RKS(free_atoms_pyscf[i])
    mf.xc = "0.00*LDA" # Quick way to get zero XC contrib
    E_pyscf = mf.kernel(max_cycle=SCF_ITERS)
    free_atom_energies_pyscf[species] = E_pyscf
    molecule = molecule_from_pyscf(mf, scf_iteration=SCF_ITERS)
    E_gdft = molecule.nonXC()
    free_atom_energies_gdft[species] = E_gdft
    free_atom_errs.append(np.abs(E_pyscf - E_gdft))

print(free_atom_errs)
    
    
# PySCF
