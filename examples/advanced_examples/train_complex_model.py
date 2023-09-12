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

from ast import Return
from functools import partial
from flax.core import unfreeze, freeze
from typing import Dict, Optional, Union
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
from jax.nn import gelu
from flax import linen as nn
import numpy as np
from optax import adam
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer
from grad_dft.molecule import Molecule

from grad_dft.train import make_train_kernel, molecule_predictor
from grad_dft.functional import (
    DispersionFunctional,
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    densities,
    dm21_combine_cinputs,
    dm21_combine_densities,
    dm21_hfgrads_cinputs,
    dm21_hfgrads_densities,
)
from grad_dft.interface.pyscf import loader

from torch.utils.tensorboard import SummaryWriter
import jax

from jax.config import config

config.update("jax_disable_jit", True)

# In this example we explain how to replicate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dissociation/")
training_files = ["H2_dissociation_omegas.h5"]
# alternatively, use "H2plus_extrapolation.h5". You will have needed to execute in data_processing.py
# distances = [0.5, 0.75, 1, 1.25, 1.5]
# process_dissociation(atom1 = 'H', atom2 = 'H', charge = 0, spin = 0, file = 'H2_dissociation.xlsx', energy_column_name='cc-pV5Z', training_distances=distances)
# process_dissociation(atom1 = 'H', atom2 = 'H', charge = 1, spin = 1, file = 'H2plus_dissociation.xlsx', energy_column_name='cc-pV5Z', training_distances=distances)


####### Model definition #######

# Then we define the Functional, via an function whose output we will integrate.
n_layers = 10
width_layers = 512
squash_offset = 1e-4
layer_widths = [width_layers] * n_layers
nlc_layer_widths = [width_layers // 4] * (n_layers // 2)
out_features = 20  # 2 for each spin x 2 for exchange/correlation x 4 for MGGA + 4 for HF
sigmoid_scale_factor = 2.0
activation = gelu
loadcheckpoint = False


def nn_coefficients(instance, rhoinputs, *_, **__):
    x = canonicalize_inputs(rhoinputs)  # Making sure dimensions are correct

    # Initial layer: log -> dense -> tanh
    x = jnp.log(jnp.abs(x) + squash_offset)  # squash_offset = 1e-4
    instance.sow("intermediates", "log", x)
    x = instance.dense(features=layer_widths[0])(x)  # features = 256
    instance.sow("intermediates", "initial_dense", x)
    x = jnp.tanh(x)
    instance.sow("intermediates", "tanh", x)

    # 6 Residual blocks with 256-features dense layer and layer norm
    for features, i in zip(layer_widths, range(len(layer_widths))):  # layer_widths = [256]*6
        res = x
        x = instance.dense(features=features)(x)
        instance.sow("intermediates", "residual_dense_" + str(i), x)
        x = x + res  # nn.Dense + Residual connection
        instance.sow("intermediates", "residual_residual_" + str(i), x)
        x = instance.layer_norm()(x)  # + res # nn.LayerNorm
        instance.sow("intermediates", "residual_layernorm_" + str(i), x)
        x = activation(x)  # activation = jax.nn.gelu
        instance.sow("intermediates", "residual_elu_" + str(i), x)

    return instance.head(x, out_features, sigmoid_scale_factor)


def nn_dispersion(instance: nn.Module, x, *_, **__):
    Rab = x[:, 0]  # The distance
    n0 = x[:, 1:]  #

    ####### First we model RAB0 ########

    # Implement a neural network that takes x as input
    x = instance.dense(features=nlc_layer_widths[0])(n0)  # features = 256
    instance.sow("intermediates", "initial_dense", x)
    x = jnp.tanh(x)
    instance.sow("intermediates", "tanh", x)

    # 6 Residual blocks with 256-features dense layer and layer norm
    for features, i in zip(
        nlc_layer_widths, range(len(nlc_layer_widths))
    ):  # layer_widths = [256]*6
        res = x
        x = instance.dense(features=features)(x)
        instance.sow("intermediates", "residual_dense_" + str(i), x)
        x = x + res  # nn.Dense + Residual connection
        instance.sow("intermediates", "residual_residual_" + str(i), x)
        x = instance.layer_norm()(x)  # + res # nn.LayerNorm
        instance.sow("intermediates", "residual_layernorm_" + str(i), x)
        x = activation(x)  # activation = jax.nn.gelu
        instance.sow("intermediates", "residual_elu_" + str(i), x)

    Rab0 = instance.head(x, 1, sigmoid_scale_factor)

    ####### Then we model CAB ########

    # Implement a neural network that takes x as input
    x = instance.dense(features=nlc_layer_widths[0])(n0)  # features = 256
    instance.sow("intermediates", "initial_dense", x)
    x = jnp.tanh(x)
    instance.sow("intermediates", "tanh", x)

    # 6 Residual blocks with 256-features dense layer and layer norm
    for features, i in zip(
        nlc_layer_widths, range(len(nlc_layer_widths))
    ):  # layer_widths = [256]*6
        res = x
        x = instance.dense(features=features)(x)
        instance.sow("intermediates", "residual_dense_" + str(i), x)
        x = x + res  # nn.Dense + Residual connection
        instance.sow("intermediates", "residual_residual_" + str(i), x)
        x = instance.layer_norm()(x)  # + res # nn.LayerNorm
        instance.sow("intermediates", "residual_layernorm_" + str(i), x)
        x = activation(x)  # activation = jax.nn.gelu
        instance.sow("intermediates", "residual_elu_" + str(i), x)

    Cab = instance.head(x, 1, sigmoid_scale_factor)

    return Cab / (1 + jnp.exp(-(Rab / Rab0 - 1)))


def combine_densities(densities, ehf):
    ehf = jnp.reshape(ehf, (ehf.shape[2], ehf.shape[0] * ehf.shape[1]))
    return jnp.concatenate((densities, ehf), axis=1)


omegas = jnp.array([0.0, 0.4])
functional = NeuralFunctional(
    coefficients=nn_coefficients,
    energy_densities=partial(densities, functional_type="MGGA"),
    nograd_densities=lambda molecule, *_, **__: molecule.HF_energy_density(omegas),
    densitygrads=lambda self, params, molecule, nograd_densities, cinputs, grad_densities, *_, **__: dm21_hfgrads_densities(
        self, params, molecule, nograd_densities, cinputs, grad_densities, omegas
    ),
    combine_densities=combine_densities,
    coefficient_inputs=dm21_coefficient_inputs,
    nograd_coefficient_inputs=lambda molecule, *_, **__: molecule.HF_energy_density(omegas),
    coefficient_input_grads=lambda self, params, molecule, nograd_cinputs, grad_cinputs, densities, *_, **__: dm21_hfgrads_cinputs(
        self, params, molecule, nograd_cinputs, grad_cinputs, densities, omegas
    ),
    combine_inputs=dm21_combine_cinputs,
)

DispersionNN = DispersionFunctional(dispersion=nn_dispersion)

####### Initializing the functional and some parameters #######

key = PRNGKey(42)  # Jax-style random seed

# We generate the features from the molecule we created before, to initialize the parameters
(key,) = split(key, 1)
rhoinputs = jax.random.normal(key, shape=[2, 11])
params = functional.init(key, rhoinputs)

dispersioninputs = jax.random.normal(key, shape=[2, 4])
dparams = DispersionNN.init(key, dispersioninputs)

params = unfreeze(params)
params["dispersion"] = dparams["params"]
params = freeze(params)

loadcheckpoint = False
checkpoint_step = 0
momentum = 0.9
initepoch = checkpoint_step
num_epochs = 101
lr = 1e-4
tx = adam(learning_rate=lr, b1=momentum)
opt_state = tx.init(params)
cost_val = jnp.inf

orbax_checkpointer = PyTreeCheckpointer()

ckpt_dir = os.path.join(dirpath, "ckpts/", "checkpoint_" + str(checkpoint_step) + "/")
if loadcheckpoint:
    train_state = functional.load_checkpoint(
        tx=tx, ckpt_dir=ckpt_dir, step=checkpoint_step, orbax_checkpointer=orbax_checkpointer
    )
    params = train_state.params
    tx = train_state.tx
    opt_state = tx.init(params)
    initepoch = train_state.step

########### Definition of the loss function #####################

# Here we use one of the following. We will use the second here.
molecule_predict = molecule_predictor(functional, nlc_functional=DispersionNN)


@partial(value_and_grad, has_aux=True)
def loss(params, molecule, true_energy):
    # In general the loss function should be able to accept [params, system (eg, molecule or reaction), true_energy]

    predicted_energy, fock = molecule_predict(params, molecule)
    cost_value = (predicted_energy - true_energy) ** 2

    # We may want to add a regularization term to the cost, be it one of the
    # fock_grad_regularization, dm21_grad_regularization, or orbital_grad_regularization in train.py;
    # or even the satisfaction of the constraints in constraints.py.

    metrics = {
        "predicted_energy": predicted_energy,
        "ground_truth_energy": true_energy,
        "mean_abs_error": jnp.mean(jnp.abs(predicted_energy - true_energy)),
        "mean_sq_error": jnp.mean((predicted_energy - true_energy) ** 2),
        "cost_value": cost_value,
        #'regularization': regularization_logs
    }

    return cost_value, metrics


kernel = jax.jit(make_train_kernel(tx, loss))

######## Training epoch ########


def train_epoch(state, training_files, training_data_dirpath):
    r"""Train for a single epoch."""

    batch_metrics = []
    params, opt_state, cost_val = state
    for file in tqdm(training_files, "Files"):
        fpath = os.path.join(training_data_dirpath, file)
        print("Training on file: ", fpath, "\n")

        load = loader(fname=fpath, randomize=True, training=True, config_omegas=omegas)
        for _, system in tqdm(load, "Molecules/reactions per file"):
            params, opt_state, cost_val, metrics = kernel(params, opt_state, system, system.energy)
            del system

            # Logging the resulting metrics
            for k in metrics.keys():
                print(k, metrics[k])
            batch_metrics.append(metrics)

    epoch_metrics = {
        k: np.mean([jax.device_get(metrics[k]) for metrics in batch_metrics])
        for k in batch_metrics[0]
    }
    state = (params, opt_state, cost_val)
    return state, metrics, epoch_metrics


######## Training loop ########

writer = SummaryWriter()
for epoch in range(initepoch, num_epochs + initepoch):
    # Use a separate PRNG key to permute input data during shuffling
    # rng, input_rng = jax.random.split(rng)

    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state

    # Save metrics and checkpoint
    print(f"Epoch {epoch} metrics:")
    for k in epoch_metrics:
        print(f"-> {k}: {epoch_metrics[k]:.5f}")
    for metric in epoch_metrics.keys():
        writer.add_scalar(f"/{metric}/train", epoch_metrics[metric], epoch)
    writer.flush()
    functional.save_checkpoints(params, tx, step=epoch, orbax_checkpointer=orbax_checkpointer)
    print(f"-------------\n")
    print(f"\n")
