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
from jax import config, value_and_grad
from optax import adam, apply_updates

import sys

config.update("jax_enable_x64", True)

# Two H2 geometries. Small basis set
HH_BL = 0.8
PYSCF_MOLS = [
    pyscf.M(
        atom = 'H %.5f 0.0 0.0; H %.5f 0.0 0.0' % (-HH_BL, HH_BL/2),
        basis = 'ccpvdz'
    ),
    pyscf.M(
        atom = 'H %.5f 0.0 0.0; H %.5f 0.0 0.0' % (-HH_BL - 0.1, HH_BL/2 + 0.1),
        basis = 'ccpvdz'
    )
]

SCF_ITERS = 5

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
    
    hf = mol.UHF().run()
    ci = pyscf.ci.CISD(hf).run()
    
    # DFT calculations used for initial guesses for densities
    mf_dft = mol.UKS().run()
    grad_dft_mol = molecule_from_pyscf(mf_dft)
    ci_rdm1 = ci.make_rdm1(ao_repr=True)
    dft_ci_rdm1 = grad_dft_mol.replace(rdm1=jnp.asarray(ci_rdm1))
    # Works because we use the same AOs for DFT and CI
    den_ci = dft_ci_rdm1.density()
    
    TRUTH_ENERGIES.append(ci.e_tot)
    TRUTH_DENSITIES.append(dft_ci_rdm1.density())

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


TRAIN_RECIPES = [
    # Non-SCF training on the energy only
    (mse_energy_loss, [PARAMS, make_non_scf_predictor(NF), MOLECULES, TRUTH_ENERGIES, True]),
    
    # DIIS-SCF training on the energy only
    (mse_energy_loss, [PARAMS, make_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_ENERGIES, True]),
    # DIIS-SCF training on the density only
    (mse_density_loss, [PARAMS, make_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, True]),
    # DIIS-SCF training on energy and density
    (mse_energy_and_density_loss, [PARAMS, make_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, TRUTH_ENERGIES, 1.0, 1.0, True]),
    
    # Jitted DIIS-SCF training on the energy only
    (mse_energy_loss, [PARAMS, make_jitted_scf_loop(NF, cycles=SCF_ITERS), MOLECULES, TRUTH_ENERGIES, True]),
    # Jitted DIIS-SCF training on the density only
    (mse_density_loss, [PARAMS, make_jitted_scf_loop(NF, cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, True]),
    # Jitted DIIS-SCF training on energy and density
    (mse_energy_and_density_loss, [PARAMS, make_jitted_scf_loop(NF, cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, TRUTH_ENERGIES, 1.0, 1.0, True]),
    
    # Linear mixing SCF training on the energy only
    (mse_energy_loss, [PARAMS, make_simple_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_ENERGIES, True]),
    # Linear mixing SCF training on the density only
    (mse_density_loss, [PARAMS, make_simple_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, True]),
    # Linear SCF training on energy and density
    (mse_energy_and_density_loss, [PARAMS, make_simple_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, TRUTH_ENERGIES, 1.0, 1.0, True]),
    
    # Jitted Linear mixing SCF training on the energy only
    (mse_energy_loss, [PARAMS, make_jitted_simple_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_ENERGIES, True]),
    # Jitted Linear mixing SCF training on the density only
    (mse_density_loss, [PARAMS, make_jitted_simple_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, True]),
    # Jitted Linear SCF training on energy and density
    (mse_energy_and_density_loss, [PARAMS, make_jitted_simple_scf_loop(NF, max_cycles=SCF_ITERS), MOLECULES, TRUTH_DENSITIES, TRUTH_ENERGIES, 1.0, 1.0, True]),
]


@pytest.mark.parametrize("train_recipe", TRAIN_RECIPES)
def test_loss_functions(train_recipe: tuple) -> None:
    r"""Same objectives as the unit test: test_loss.py but the predictors are now real DFT calculations
    with Neural functionals.

    Args:
        train_recipe (tuple): information regarding the loss, its arguments and the predictor. See TRAIN_RECIPES variable above.
    """
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
    ).any(), f"Bias loss gradients for loss function {loss_func.__name__} and predictor {predictor_name} contains a NaN. It should not."
    assert not jnp.isnan(
        gradient["params"]["Dense_0"]["kernel"]
    ).any(), f"Kernel loss gradients for loss function {loss_func.__name__} and predictor {predictor_name} contains a NaN. It should not."

LR = 0.001
MOMENTUM = 0.9

# and implement the optimization loop
N_EPOCHS = 5

@pytest.mark.parametrize("train_recipe", TRAIN_RECIPES)
def test_minimize(train_recipe: tuple) -> None:
    r"""Check that the loss functions with different predictords are minimizable in 5 iterations.

    Args:
        train_recipe (tuple):train_recipe (tuple): information regarding the loss, its arguments and the predictor. See TRAIN_RECIPES variable above.
    """
    
    loss_func, loss_args = train_recipe
    predictor_name = loss_args[1].__name__
    
    tr_params = NF.init(KEY, CINPUTS)
    loss_args[0] = tr_params
    
    tx = adam(learning_rate=LR, b1=MOMENTUM)
    opt_state = tx.init(PARAMS)
    loss_and_grad = value_and_grad(loss_func)
    cost_history = []
    for i in range(N_EPOCHS):
        cost_value, grads = loss_and_grad(*loss_args)
        # print(grads)
        cost_history.append(cost_value)
        updates, opt_state = tx.update(grads, opt_state, tr_params)
        tr_params = apply_updates(tr_params, updates)
        loss_args[0] = tr_params
    assert (
        cost_history[-1] <= cost_history[0]
    ), f"Training recipe for loss function {loss_func.__name__} and {predictor_name} did not reduce the cost in 5 iterations"