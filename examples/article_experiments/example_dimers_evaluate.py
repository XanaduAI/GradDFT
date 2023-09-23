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
import json
import os
from optax import adam
from tqdm import tqdm
from grad_dft import (
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    dm21_densities,
    loader,
    energy_predictor
)

from orbax.checkpoint import PyTreeCheckpointer
from jax import numpy as jnp
from jax.nn import gelu

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dimers/")

# Select here the folder where the checkpoints are stored
ckpt_folder = "ckpt_dimers"  # This folder should be located in dirpath
# Select here the file you would like to evaluate your model on
test_files = ["dimers_SCAN.hdf5"]  # this file should be located in the training_data_dirpath above
# Select the name of the file to save the results in the checkpoint folder ckpt_folder
json_results = "prediction_dict.json"

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


# Load the model
epoch = 2
tx = adam(learning_rate=1e-6, b1=0.9)
orbax_checkpointer = PyTreeCheckpointer()
ckpt_dir = os.path.join(dirpath, ckpt_folder, "checkpoint_" + str(epoch) + "/")
train_state = functional.load_checkpoint(
    tx=tx, ckpt_dir=ckpt_dir, step=epoch, orbax_checkpointer=orbax_checkpointer
)
params = train_state.params
tx = train_state.tx
opt_state = tx.init(params)
epoch = train_state.step


# Evaluate the model
# Select here the file you would like to evaluate your model on
test_files = ["dimers_SCAN.hdf5"]
prediction_dict = {}

compute_energy = energy_predictor(functional)

for file in tqdm(test_files, "Files"):
    fpath = os.path.join(training_data_dirpath, file)
    print("Evaluating file: ", fpath, "\n")

    load = loader(fname=fpath, randomize=True, training=True, config_omegas=[])
    for _, system in tqdm(load, "Molecules/reactions per file"):
        predicted_energy, fock = compute_energy(params, system)
        name = "".join(chr(num) for num in list(system.name))
        name = name.replace("b'", "").replace("'", "")
        prediction_dict[name] = float(predicted_energy)
        break

# save to json
jsonname = os.path.join(dirpath, ckpt_folder, json_results)
with open(jsonname, "w") as fp:
    json.dump(prediction_dict, fp)
