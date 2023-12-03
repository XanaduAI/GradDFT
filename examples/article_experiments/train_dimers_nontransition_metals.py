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
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
from jax.nn import gelu
import numpy as np
from optax import adam
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer

from grad_dft import (
    train_kernel, 
    energy_predictor,
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    dm21_densities,
    loader
)

from torch.utils.tensorboard import SummaryWriter
import jax

import json
def convert(o):
    if isinstance(o, np.float32):
        return float(o)  
    return o

# In this example we explain how to replicate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dimers/")
training_files = ["non_tm_dimers.h5"]

####### Model definition #######

# Then we define the Functional, via an function whose output we will integrate.
n_layers = 10
width_layers = 512
squash_offset = 1e-4
layer_widths = [width_layers] * n_layers
out_features = 4
sigmoid_scale_factor = 2.0
activation = gelu


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


functional = NeuralFunctional(
    coefficients=nn_coefficients,
    coefficient_inputs=dm21_coefficient_inputs,
    energy_densities=partial(dm21_densities, functional_type="MGGA"),
)

####### Initializing the functional and some parameters #######

key = PRNGKey(1)  # Jax-style random seed #todo: select this

# We generate the features from the molecule we created before, to initialize the parameters
(key,) = split(key, 1)
rhoinputs = jax.random.normal(key, shape=[2, 7])
params = functional.init(key, rhoinputs)

#todo: change this
loadcheckpoint = True # If true, do not forget to switch of the recreation of tx
checkpoint_step = 323
learning_rate = 1e-8
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
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
    epoch = train_state.step

########### Definition of the loss function #####################

# Here we use one of the following. We will use the second here.
compute_energy = energy_predictor(functional)


@partial(value_and_grad, has_aux=True)
def loss(params, molecule, true_energy):
    # In general the loss function should be able to accept [params, system (eg, molecule or reaction), true_energy]

    predicted_energy, fock = compute_energy(params, molecule)
    cost_value = (predicted_energy - true_energy) ** 2

    # We may want to add a regularization term to the cost, be it one of the
    # fock_grad_regularization, dm21_grad_regularization, or orbital_grad_regularization in train.py;
    # or even the satisfaction of the constraints in constraints.py.

    metrics = {
        #"predicted_energy": predicted_energy,
        #"ground_truth_energy": true_energy,
        "mean_abs_error": jnp.mean(jnp.abs(predicted_energy - true_energy)),
        "mean_sq_error": jnp.mean((predicted_energy - true_energy) ** 2),
        "cost_value": cost_value,
        #'regularization': regularization_logs
    }

    return cost_value, metrics

######## Training epoch ########


def train_epoch(state, training_files, training_data_dirpath):
    r"""Train for a single epoch."""

    batch_metrics = []
    params, opt_state, cost_val = state
    for file in tqdm(training_files, "Files"):
        fpath = os.path.join(training_data_dirpath, file)
        print("Training on file: ", fpath, "\n")

        load = loader(fname=fpath, randomize=True, training=True, config_omegas=[])
        for _, system in tqdm(load, "Molecules/reactions per file"):
            params, opt_state, cost_val, metrics = kernel(params, opt_state, system, system.energy)
            del system

            # Logging the resulting metrics
            # for k in metrics.keys():
            #    print(k, metrics[k])
            batch_metrics.append(metrics)

    epoch_metrics = {
        k: np.mean([jax.device_get(metrics[k]) for metrics in batch_metrics])
        for k in batch_metrics[0]
    }
    state = (params, opt_state, cost_val)
    return state, metrics, epoch_metrics


######## Training loop ########

writer = SummaryWriter()
epoch_results = {}

results_path_json = os.path.normpath(dirpath + f'/checkpoints/epoch_results_{checkpoint_step}.json')


initepoch = checkpoint_step
num_epochs = 101-initepoch
lr = 3e-6
tx = adam(learning_rate=lr, b1=momentum)
kernel = jax.jit(train_kernel(tx, loss))
opt_state = tx.init(params)
cost_val = jnp.inf
# save epoch results to json
for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    # Use a separate PRNG key to permute input data during shuffling
    # rng, input_rng = jax.random.split(rng)
    print(f"Epoch {epoch}")

    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state

    # Save metrics and checkpoint
    epoch_results[epoch] = epoch_metrics
    print(f"Epoch {epoch} metrics:")
    for k in epoch_metrics:
        print(f"-> {k}: {epoch_metrics[k]:.5f}")
    for metric in epoch_metrics.keys():
        writer.add_scalar(f"/{metric}/train", epoch_metrics[metric], epoch)
    writer.flush()
    functional.save_checkpoints(params, tx, step=epoch, orbax_checkpointer=orbax_checkpointer)
    # print(f"-------------\n")
    print(f"\n")

    with open(results_path_json, 'w') as fp:
        json.dump(epoch_results, fp, default=convert)




initepoch = checkpoint_step
num_epochs = 201-initepoch
lr = 1e-6
#tx = adam(learning_rate=lr, b1=momentum)
kernel = jax.jit(train_kernel(tx, loss))
opt_state = tx.init(params)
for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    # Use a separate PRNG key to permute input data during shuffling
    # rng, input_rng = jax.random.split(rng)

    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state

    # Save metrics and checkpoint
    epoch_results[epoch] = epoch_metrics
    print(f"Epoch {epoch} metrics:")
    for k in epoch_metrics:
        print(f"-> {k}: {epoch_metrics[k]:.5f}")
    for metric in epoch_metrics.keys():
        writer.add_scalar(f"/{metric}/train", epoch_metrics[metric], epoch)
    writer.flush()
    functional.save_checkpoints(params, tx, step=epoch, orbax_checkpointer=orbax_checkpointer)
    # print(f"-------------\n")
    print(f"\n")

    with open(results_path_json, 'w') as fp:
        json.dump(epoch_results, fp, default=convert)


initepoch = 201
num_epochs = 100
lr = 1e-7
tx = adam(learning_rate=lr, b1=momentum) #todo: change this
kernel = jax.jit(train_kernel(tx, loss))
opt_state = tx.init(params)
cost_val = jnp.inf

for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    # Use a separate PRNG key to permute input data during shuffling
    # rng, input_rng = jax.random.split(rng)

    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state

    # Save metrics and checkpoint
    epoch_results[epoch] = epoch_metrics
    print(f"Epoch {epoch} metrics:")
    for k in epoch_metrics:
        print(f"-> {k}: {epoch_metrics[k]:.5f}")
    for metric in epoch_metrics.keys():
        writer.add_scalar(f"/{metric}/train", epoch_metrics[metric], epoch)
    writer.flush()
    functional.save_checkpoints(params, tx, step=epoch, orbax_checkpointer=orbax_checkpointer)
    # print(f"-------------\n")
    print(f"\n")

    with open(results_path_json, 'w') as fp:
        json.dump(epoch_results, fp, default=convert)


initepoch = checkpoint_step
num_epochs = 351-initepoch
lr = 1e-8
#tx = adam(learning_rate=lr, b1=momentum) #todo: change this
kernel = jax.jit(train_kernel(tx, loss))
opt_state = tx.init(params)
cost_val = jnp.inf

for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    # Use a separate PRNG key to permute input data during shuffling
    # rng, input_rng = jax.random.split(rng)

    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state

    # Save metrics and checkpoint
    epoch_results[epoch] = epoch_metrics
    print(f"Epoch {epoch} metrics:")
    for k in epoch_metrics:
        print(f"-> {k}: {epoch_metrics[k]:.5f}")
    for metric in epoch_metrics.keys():
        writer.add_scalar(f"/{metric}/train", epoch_metrics[metric], epoch)
    writer.flush()
    functional.save_checkpoints(params, tx, step=epoch, orbax_checkpointer=orbax_checkpointer)
    # print(f"-------------\n")
    print(f"\n")

    with open(results_path_json, 'w') as fp:
        json.dump(epoch_results, fp, default=convert)

