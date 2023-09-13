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

from functools import partial
from jax import numpy as jnp, value_and_grad
from jax.random import split, PRNGKey
from jax.nn import sigmoid, gelu
from flax import linen as nn
from jax import config
from optax import adam, apply_updates
from tqdm import tqdm

from jaxtyping import install_import_hook


from grad_dft.molecule import Molecule
from grad_dft.evaluate import (
    make_jitted_orbital_optimizer,
    make_orbital_optimizer,
    make_scf_loop,
    make_jitted_scf_loop,
)
from grad_dft.train import molecule_predictor
from grad_dft.functional import NeuralFunctional, default_loss
from grad_dft.interface import molecule_from_pyscf

config.update("jax_enable_x64", True)
config.update('jax_debug_nans', True)

from optax import clip_by_global_norm

clip_by_global_norm(1e-5)

# In this basic tutorial we want to introduce the concept of a functional.

# We we will prepare a molecule, following the previous tutorial:
from pyscf import gto, dft

# Define the geometry of the molecule
mol = gto.M(atom=[["F", (0, 0, 0)], ["H", (0, 0, 1)]], basis="def2-tzvp", charge=0, spin=0)
mf = dft.UKS(mol)
ground_truth_energy = mf.kernel()

# Then we can use the following function to generate the molecule object
HH_molecule = molecule_from_pyscf(mf)

########################### Creating a neural functional #########################################


# Now we create our neuralfunctional. We need to define at least the following methods: densities and coefficients
# which compute the two vectors that get dot-multiplied and then integrated over space. If the functional is a
# neural functional we also need to define coefficient_inputs, for which in this case we will reuse the densities function.
def coefficient_inputs(molecule: Molecule, *_, **__):
    rho = jnp.clip(molecule.density(), a_min = 1e-20)
    return jnp.concatenate((rho, ), axis = 1)

def energy_densities(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = molecule.density()
    # To avoid numerical issues in JAX we limit too small numbers.
    rho = jnp.clip(rho, a_min = clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = -3/2 * (3/(4*jnp.pi)) ** (1/3) * (rho**(4/3)).sum(axis = 1, keepdims = True)
    # For simplicity we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    return lda_e

out_features = 1
def coefficients(instance, rhoinputs):
    r"""
    Instance is an instance of the class Functional or NeuralFunctional.
    rhoinputs is the input to the neural network, in the form of an array.
    localfeatures represents the potentials e_\theta(r).

    The output of this function is the energy density of the system.
    """

    x = nn.Dense(features=out_features)(rhoinputs)
    x = nn.LayerNorm()(x)
    x = gelu(x)
    return sigmoid(x)

neuralfunctional = NeuralFunctional(coefficients, energy_densities, coefficient_inputs)

# Now we can initialize the parameters of the neural network
key = PRNGKey(42)
cinputs = coefficient_inputs(HH_molecule)
params = neuralfunctional.init(key, cinputs)

########################### Training the functional #########################################

@partial(value_and_grad, has_aux=True)
def loss(params, molecule_predict, molecule, trueenergy):
    r"""
    Computes the loss function, here MSE, between predicted and true energy

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    molecule_predict: Callable.
        Use molecule_predict = molecule_predictor(functional) to generate it.
    molecule: Molecule
    trueenergy: float

    Returns
    ----------
    Tuple[float, float]
    The loss and predicted energy.

    Note
    ----------
    Since it has the decorator @partial(value_and_grad, has_aux = True)
    it will compute the gradients with respect to params.
    """

    predictedenergy, _, _ = molecule_predict(params, molecule)
    cost_value = (predictedenergy - trueenergy) ** 2

    return cost_value, predictedenergy

learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 20
scf_iterator = make_jitted_scf_loop(neuralfunctional, cycles=3)

for iteration in tqdm(range(n_epochs), desc="Training epoch"):
    (cost_value, predicted_energy), grads = loss(
        params, scf_iterator, HH_molecule, ground_truth_energy
    )
    print("Iteration", iteration, "Predicted energy:", predicted_energy, "Cost value:", cost_value)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

neuralfunctional.save_checkpoints(params, tx, step=n_epochs)


########################### Loading the functional and scf loop #########################################

# We can load the functional checkpoint with the following code
step = 20
train_state = neuralfunctional.load_checkpoint(
    tx, ckpt_dir="ckpts/checkpoint_" + str(step) + "/", step=step
)
params = train_state.params
tx = train_state.tx
opt_state = tx.init(params)

# The following code takes some execution time

# Create the scf iterator
HH_molecule = molecule_from_pyscf(mf)
scf_iterator = make_scf_loop(neuralfunctional, verbose=2, max_cycles=1)
energy = scf_iterator(params, HH_molecule)
print("Energy from the scf loop:", energy)

# We can alternatively use the jit-ed version of the scf loop
HH_molecule = molecule_from_pyscf(mf)
scf_iterator = make_jitted_scf_loop(neuralfunctional, cycles=1)
jitted_energy, _, _ = scf_iterator(params, HH_molecule)
print("Energy from the jitted scf loop:", jitted_energy)

# We can even use a direct optimizer of the orbitals
HH_molecule = molecule_from_pyscf(mf)
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
orbital_optimizer = make_orbital_optimizer(neuralfunctional, tx, max_cycles=20)
optimized_energy = orbital_optimizer(params, HH_molecule)
print("Energy from the orbital optimizer:", optimized_energy)


# We can even use a direct optimizer of the orbitals
HH_molecule = molecule_from_pyscf(mf)
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
orbital_optimizer = make_jitted_orbital_optimizer(neuralfunctional, tx, max_cycles=20)
jitted_optimized_energy = orbital_optimizer(params, HH_molecule)
print("Energy from the orbital optimizer:", jitted_optimized_energy)