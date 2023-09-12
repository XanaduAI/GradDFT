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

import warnings
from jax.random import PRNGKey
from jax.lax import stop_gradient
from optax import adam
from grad_dft.evaluate import make_scf_loop

from grad_dft.interface.pyscf import molecule_from_pyscf
from grad_dft.functional import DM21

# In this example we aim to explain how we can implement the self-consistent loop
# with the DM21 functional.

# First we define a molecule, using pyscf:
from pyscf import gto, dft

warnings.warn('--- This example should be executed after intermediate_examples/example_HF_training.py ---')

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

functional = DM21()
params = functional.generate_DM21_weights()

# We can also load the params from the previous example
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
step = 15
train_state = functional.load_checkpoint(
    tx, ckpt_dir="ckpts/checkpoint_" + str(step) + "/", step=step
)
params = train_state.params
tx = train_state.tx
opt_state = tx.init(params)

key = PRNGKey(42)  # Jax-style random seed

# We generate the input densities from the molecule we created before
grad_densities = functional.energy_densities(molecule)
nograd_densities = stop_gradient(functional.nograd_densities(molecule))
densities = functional.combine_densities(grad_densities, nograd_densities)

# We now generated the inputs to the coefficients nn
grad_cinputs = functional.coefficient_inputs(molecule)
nograd_cinputs = stop_gradient(functional.nograd_coefficient_inputs(molecule))
coefficient_inputs = functional.combine_inputs(grad_cinputs, nograd_cinputs)

# And then we compute the energy
predicted_energy = functional.xc_energy(
    params, molecule.grid, coefficient_inputs, densities
)
predicted_energy += molecule.nonXC()
print("Predicted_energy (detailed code):", predicted_energy)

# Alternatively, we can use an already prepared function that does everything
predicted_energy = functional.energy(params, molecule)
print("Predicted_energy:", predicted_energy)

# Finally, we create and implement the self-consistent loop.
scf_iterator = make_scf_loop(functional, verbose=2, max_cycles=5)
predicted_e = scf_iterator(params, molecule)

print(f"The predicted energy is {predicted_e}")
