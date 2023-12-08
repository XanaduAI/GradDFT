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
trainable and minimizable given a simple neural functional and only two H Solid objects in
the training set. These tests do the same as in ~/tests/molecules/test_training.py but we are testing for Solids
instead.

For SCF and non-SCF training, we should have:


(1) Loss function gradients free of NaN's

(2) Decreased loss after 5 iterations of an optimizer

"""

from jax.random import PRNGKey
import jax.numpy as jnp
import numpy as np
from jax import grad
import pytest
from pyscf.pbc import gto, scf, cc, ci

from grad_dft import (
    solid_from_pyscf,
    mse_energy_loss, 
    mse_density_loss, 
    mse_energy_and_density_loss,
    diff_simple_scf_loop, 
    simple_scf_loop, 
    non_scf_predictor,
    Solid,
    NeuralFunctional
)

from jax.nn import sigmoid, gelu
from flax import linen as nn
from optax import adam, apply_updates
from jax import config, value_and_grad
config.update("jax_enable_x64", True)

# Two H solid geometries. Small basis set
LAT_VEC = np.array(
    [[3.6, 0.0, 0.0], 
     [0.0, 3.6, 0.0], 
     [0.0, 0.0, 3.6]]
)

PYSCF_SOLS = [
    gto.M(
        a = LAT_VEC,
        atom = """H     0.0  0.0  0.0
                  H     1.4  0.0   0.0""",
        basis = 'sto-3g',
        space_group_symmetry=False,
        symmorphic=False,
    ),
    
    gto.M(
        a = LAT_VEC,
        atom = """H     0.0  0.0  0.
                  H     1.4  0.0   0.0""",
        basis = 'sto-3g',
        space_group_symmetry=False,
        symmorphic=False,
    ),
    
]

SCF_ITERS = 5

# Truth values are decided to be from MP2 calculations.
TRUTH_ENERGIES = []
TRUTH_DENSITIES = [] 
SOLIDS = []
KPTS = [2, 1, 1]

for sol in PYSCF_SOLS:
    kmf = scf.KRKS(sol, kpts=sol.make_kpts(KPTS))
    kmf.xc = "LDA"
    E_pyscf = kmf.kernel(max_cycle=SCF_ITERS)
    solid = solid_from_pyscf(kmf)
    SOLIDS.append(solid)
    
    khf = scf.KRHF(sol, kpts=sol.make_kpts(KPTS))
    khf = khf.run()
    mp2 = khf.MP2().run()
    mp2_rdm1 = np.asarray(mp2.make_rdm1())
    E_tr = mp2.e_tot
        
    # DFT calculations used for their grids to calculate MP2 densities
    kmf_dft_dummy = scf.KRKS(sol, kpts=sol.make_kpts(KPTS))
    kmf_dft_dummy.kernel(max_cycle=1)
    grad_dft_sol_dummy = solid_from_pyscf(kmf_dft_dummy)
    dft_kccsd_rdm1 = grad_dft_sol_dummy.replace(rdm1=jnp.asarray(mp2_rdm1))
    # Works because we use the same AOs for DFT and MP2
    den_tr = grad_dft_sol_dummy.density()
    
    TRUTH_ENERGIES.append(E_tr)
    TRUTH_DENSITIES.append(den_tr)

# Define a simple neural functional and its initial parameters

def coefficient_inputs(solid: Solid, clip_cte: float = 1e-30, *_, **__):
    rho = jnp.clip(solid.density(), a_min = clip_cte)
    return jnp.concatenate((rho, ), axis = 1)

def energy_densities(solid: Solid, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    rho = solid.density()
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
CINPUTS = coefficient_inputs(SOLIDS[0])
PARAMS = NF.init(KEY, CINPUTS)


# Only linear mixing SCF and non SCF training implemented for Solid objects at present
TRAIN_RECIPES = [
    # Non-SCF training on the energy only
    (mse_energy_loss, [PARAMS, non_scf_predictor(NF), SOLIDS, TRUTH_ENERGIES, True]),
    
    # Linear mixing SCF training on the energy only
    (mse_energy_loss, [PARAMS, simple_scf_loop(NF, cycles=SCF_ITERS), SOLIDS, TRUTH_ENERGIES, True]),
    # Linear mixing SCF training on the density only
    (mse_density_loss, [PARAMS, simple_scf_loop(NF, cycles=SCF_ITERS), SOLIDS, TRUTH_DENSITIES, True]),
    # Linear SCF training on energy and density
    (mse_energy_and_density_loss, [PARAMS, simple_scf_loop(NF, cycles=SCF_ITERS), SOLIDS, TRUTH_DENSITIES, TRUTH_ENERGIES, 1.0, 1.0, True]),
    
    # Jitted Linear mixing SCF training on the energy only
    (mse_energy_loss, [PARAMS, diff_simple_scf_loop(NF, cycles=SCF_ITERS), SOLIDS, TRUTH_ENERGIES, True]),
    # Jitted Linear mixing SCF training on the density only
    (mse_density_loss, [PARAMS, diff_simple_scf_loop(NF, cycles=SCF_ITERS), SOLIDS, TRUTH_DENSITIES, True]),
    # Jitted Linear SCF training on energy and density
    (mse_energy_and_density_loss, [PARAMS, diff_simple_scf_loop(NF, cycles=SCF_ITERS), SOLIDS, TRUTH_DENSITIES, TRUTH_ENERGIES, 1.0, 1.0, True]),
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