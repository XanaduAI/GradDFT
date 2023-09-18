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

from jax.random import PRNGKey
from optax import adam, apply_updates
from jax.lax import stop_gradient

from grad_dft import (
    molecule_from_pyscf,
    make_non_scf_predictor,
    simple_energy_loss,
    DM21
)

from jax.config import config
config.update("jax_enable_x64", True)

# In this example we aim to explain how we can train the DM21 functional.

# First we define a molecule, using pyscf:
from pyscf import gto, dft

mol = gto.M(atom="H 0 0 0; F 0 0 1.1")

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# Then we compute quite a few properties which we pack into a class called Molecule.
# omegas will indicate the values of w in the range-separated Coulomb kernel
#  erf(w|r-r'|)/|r-r'|.
# Note that w = 0 indicates the usual Coulomb kernel 1/|r-r'|.
molecule = molecule_from_pyscf(mf, omegas=[0.0, 0.4])

# We create the functional and load its weights.
functional = DM21()
params = functional.generate_DM21_weights()

key = PRNGKey(42)  # Jax-style random seed

# We generate the input densities from the molecule we created before
grad_densities = functional.energy_densities(molecule)
nograd_densities = stop_gradient(functional.nograd_densities(molecule))
densities = functional.combine_densities(grad_densities, nograd_densities)

# We now generated the inputs to the coefficients nn
grad_cinputs = functional.coefficient_inputs(molecule)
nograd_cinputs = stop_gradient(functional.nograd_coefficient_inputs(molecule))
coefficient_inputs = functional.combine_inputs(grad_cinputs, nograd_cinputs)

# We can use this features to compute the energy by parts
energy = functional.xc_energy(params, molecule.grid, coefficient_inputs, densities)
energy += molecule.nonXC()

# Or alternatively, we can use an already prepared function that does everything for us
predicted_energy = functional.energy(params, molecule)
print("Predicted_energy:", predicted_energy)

# Then, we define the optimizer
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 15
predict = make_non_scf_predictor(functional)
for iteration in range(n_epochs):
    # Here we use a default loss without regularizer, but it is easy to adapt it.
    (cost_value, predicted_energy), grads = simple_energy_loss(
        params, predict, molecule, ground_truth_energy
    )
    print("Iteration", iteration, "Predicted energy:", predicted_energy)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

# We can save the checkpoint back.
functional.save_checkpoints(params, tx, step=n_epochs)
