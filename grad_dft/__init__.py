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

from .molecule import (
    Grid,
    Molecule, 
    Reaction, 
    make_reaction,
    abs_clip, 
    make_rdm1, 
    orbital_grad,
    density,
    grad_density,
    coulomb_energy
)
from .functional import (
    DispersionFunctional,
    Functional, 
    NeuralFunctional, 
    DM21, 
    correlation_polarization_correction, 
    exchange_polarization_correction,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    dm21_densities,
    densities,
    dm21_combine_cinputs,
    dm21_combine_densities,
    dm21_hfgrads_cinputs,
    dm21_hfgrads_densities,
)
from .train import (
    make_train_kernel,
    molecule_predictor, 
    Harris_energy_predictor,
    simple_energy_loss,
    mse_energy_loss, 
    mse_density_loss, 
    mse_energy_and_density_loss
)
from .evaluate import (
    make_orbital_optimizer,
    make_jitted_orbital_optimizer,
    make_non_scf_predictor,
    make_simple_scf_loop,
    make_jitted_simple_scf_loop,
    make_scf_loop,
    make_jitted_scf_loop
)
from .interface import (
    molecule_from_pyscf, 
    loader, 
    saver
)
from .popular_functionals import (
    LSDA,
    B88,
    VWN,
    LYP,
    B3LYP,
    PW92
)
