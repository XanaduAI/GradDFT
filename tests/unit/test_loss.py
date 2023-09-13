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

"""The goal of this module is to test that the loss functions in ~/grad_dft/train.py produce:

(1) Values which make sense given dummy innput

(2) Gradients free of NaN's when the used with a dummy molecule_predictor
"""

from jaxtyping import Array, PyTree, Scalar, Float, Int
from jax.random import PRNGKey, normal, randint
import jax.numpy as jnp
from jax import grad, jit
import pytest

from flax import struct


from grad_dft.train import mse_energy_loss, mse_density_loss, mse_energy_and_density_loss

SEEDS = [1984, 1993]
RANDOM_KEYS = [PRNGKey(seed) for seed in SEEDS]

TRUTH_ENERGIES = [12.8, 10.1]
TRUTH_DENSITIES = [normal(rand_key, (1000, 2)) for rand_key in RANDOM_KEYS]

PARAMS = jnp.array([0.11, 0.80, 0.24])
GRID_WEIGHTS = jnp.ones(shape=(1000,))

LOSS_FUNCTIONS = [mse_energy_loss, mse_density_loss, mse_energy_and_density_loss]


@struct.dataclass
class dummy_grid:
    r"""A dummy Grid object used only to access the weights attribute used in
    the density loss functions
    """
    weights: Array


GRID = dummy_grid(GRID_WEIGHTS)


@struct.dataclass
class dummy_molecule:
    r"""A dummy Molecule object used only to access the num_elec attribute used in
    loss functions
    """
    num_elec: Scalar
    grid: dummy_grid


MOLECULES = [dummy_molecule(1.0, GRID), dummy_molecule(2.0, GRID)]


def dummy_predictor(params: PyTree, molecule: dummy_molecule):
    r"""A dummy function matching the signature of the predictor functions in Grad-DFT

    Args:
        params (PyTree): params used to make predictions
        molecule (Molecule): a dummy Grad-DFT Molecule object

    Returns:
        tuple[Scalar, Array, Array]: The total energy, density and 1RDM
    """
    total_energy = 10.0 + params[0]
    density = jnp.ones(shape=(1000, 2)) + params[1]
    # we don't presently implement loss functions using an RDM1, but we could in the future
    rdm1 = jnp.ones(shape=(10, 10)) + params[2]
    return (total_energy, density, rdm1)


LOSS_ARGS = [
    [PARAMS, dummy_predictor, MOLECULES, TRUTH_ENERGIES, True],
    [PARAMS, dummy_predictor, MOLECULES, TRUTH_DENSITIES, True],
    [PARAMS, dummy_predictor, MOLECULES, TRUTH_DENSITIES, TRUTH_ENERGIES, True],
]

LOSS_INFO = [(LOSS_FUNCTIONS[i], LOSS_ARGS[i]) for i in range(3)]


@pytest.mark.parametrize("loss_info", LOSS_INFO)
def test_loss_functions(loss_info: tuple) -> None:
    loss_func, loss_args = loss_info
    loss = loss_func(*loss_args)
    # Pure loss test
    assert not jnp.isnan(
        loss
    ).any(), f"Loss for loss function {loss_func.__name__} contains a NaN. It should not."

    assert (
        loss >= 0
    ), f"Loss for loss function {loss_func.__name__} is less than 0 which shouldn't be possible"

    # Gradient tests
    grad_fn = grad(loss_func)
    gradient = grad_fn(*loss_args)
    assert not jnp.isnan(
        gradient
    ).any(), f"Loss gradients for loss function {loss_func.__name__} contains a NaN. It should not."
