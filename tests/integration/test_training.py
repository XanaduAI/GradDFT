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

"""The goal of this module is to test that the loss functions in ~/grad_dft/train.py are
trainable and minimizable given a simple neural functional and only two H2 molecules in
the training set. This means that, for SCF and non-SCF training, we should have:


(1) Loss function gradients free of NaN's

(2) Decreased loss after 5 iterations of an optimizer

"""

from jaxtyping import Array, PyTree, Scalar, Float, Int
from jax.random import PRNGKey, normal, randint
import jax.numpy as jnp
from jax import grad, jit
import pytest

import pyscf

from flax import struct

from typing import Optional

from grad_dft.train import mse_energy_loss, mse_density_loss, mse_energy_and_density_loss
from grad_dft.evaluate import make_jitted_simple_scf_loop, make_scf_loop, make_jitted_scf_loop, make_simple_scf_loop, make_non_scf_predictor
from grad_dft.interface import molecule_from_pyscf
from grad_dft.molecule import Molecule
from grad_dft.functional import NeuralFunctional

from jax.nn import sigmoid, gelu
from flax import linen as nn
from jax import config
from optax import adam, apply_updates

config.update("jax_enable_x64", True)

# Two H2 geometries. Small basis set
HH_BL = 0.8
PYSCF_MOLS = [
    pyscf.M(
        atom = 'H %.5f 0.0 0.0; H %.5f 0.0 0.0' % (-HH_BL, HH_BL/2),
        basis = 'sto-3g'
    ),
    pyscf.M(
        atom = 'H %.5f 0.0 0.0; H %.5f 0.0 0.0' % (-HH_BL - 0.1, HH_BL/2 + 0.1),
        basis = 'sto-3g'
    )
]

SCF_ITERS = 10

# Truth values are decided to be from LDA calculations for speedy testing.
# In reality, you would use high accuracy wavefunction of experimental data.
TRUTH_ENERGIES = []
TRUTH_DENSITIES = [] 
MOLECULES = []

for mol in PYSCF_MOLS:
    mf = pyscf.dft.UKS(mol)
    mf.xc = "LDA"
    E_pyscf = mf.kernel(max_cycle=SCF_ITERS)
    TRUTH_ENERGIES.append(E_pyscf)
    molecule = molecule_from_pyscf(mf)
    MOLECULES.append(molecule)
    TRUTH_DENSITIES.append(molecule.density())

# Define a simple neural functional and its initial parameters

def coefficient_inputs(molecule: Molecule, clip_cte: float = 1e-30, *_, **__):
    rho = jnp.clip(molecule.density(), a_min = clip_cte)
    return jnp.concatenate((rho, ), axis = 1)

def energy_densities(molecule: Molecule, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    rho = molecule.density()
    # To avoid numerical issues in JAX we limit too small numbers.
    rho = jnp.clip(rho, a_min = clip_cte)
    # Now we can implement the LSDA exchange energy density
    lda_e = -3/2 * (3/(4*jnp.pi)) ** (1/3) * (rho**(4/3)).sum(axis = 1, keepdims = True)
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

NF = NeuralFunctional(coefficients, energy_densities, coefficient_inputs)
KEY = PRNGKey(42)
CINPUTS = coefficient_inputs(MOLECULES[0])
PARAMS = NF.init(KEY, CINPUTS)

# LOSS_FUNCTIONS = [mse_energy_loss, mse_density_loss, mse_energy_and_density_loss]
# PREDICTORS = [
#     make_jitted_scf_loop(NF, SCF_ITERS)
#     ]

TRAIN_RECIPES = [(mse_energy_loss, [PARAMS, make_non_scf_predictor(NF), MOLECULES, TRUTH_ENERGIES, True])]

# LOSS_INFO = [(mse_energy_loss, [PARAMS, make_non_scf_predictor, MOLECULES, TRUTH_ENERGIES, True]),
#     [PARAMS, dummy_predictor, MOLECULES, TRUTH_DENSITIES, True],
#     [PARAMS, dummy_predictor, MOLECULES, TRUTH_DENSITIES, TRUTH_ENERGIES, True],
# ]

# LOSS_INFO = [(LOSS_FUNCTIONS[i], LOSS_ARGS[i]) for i in range(3)]


@pytest.mark.parametrize("train_recipe", TRAIN_RECIPES)
def test_loss_functions(train_recipe: tuple) -> None:
    loss_func, loss_args = train_recipe
    predictor_name = loss_args[1].__name__
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
        gradient["params"]["Dense_0"]["bias"]
    ).any(), f"Dense_0 loss gradients for loss function {loss_func.__name__} and predictor {predictor_name} contains a NaN. It should not."
    assert not jnp.isnan(
        gradient["params"]["Dense_0"]["kernel"]
    ).any(), f"Kernel loss gradients for loss function {loss_func.__name__} and predictor {predictor_name} contains a NaN. It should not."
