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

import collections
from functools import partial
from typing import Dict, OrderedDict
from jax.random import split, PRNGKey
from jax import numpy as jnp
from jax.nn import gelu
from matplotlib import pyplot as plt

plt.rcParams["text.usetex"] = True
import numpy as np
from optax import adam
import pandas as pd
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer
import h5py

from train import molecule_predictor
from functional import (
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    dm21_densities,
)
from interface.pyscf import loader

import jax
from jax import config

config.update("jax_enable_x64", True)

# In this example we explain how to evaluate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# todo: Select here the file to evaluate

# Select here the file you would like to evaluate your model on
test_files = ["H2plus_dissociation.h5"]

# Select here the folder where the checkpoints are stored
ckpt_folder = "checkpoints/ckpts_H2plus_extrapolation/"
training_data_dirpath = os.path.join(dirpath, ckpt_folder) #os.path.normpath(dirpath + "/data/training/dissociation/")


# Just for plotting, indicate the name of the file this model was trained on
train_file = 'H2plus_extrapolation_train.hdf5'
control_files = [train_file]
# alternatively, use "H2_extrapolation.h5". You will have needed to execute in data_processing.py
#process_dissociation(atom1 = 'H', atom2 = 'H', charge = 0, spin = 0, file = 'H2_dissociation.xlsx', energy_column_name='cc-pV5Z')
#process_dissociation(atom1 = 'H', atom2 = 'H', charge = 1, spin = 1, file = 'H2plus_dissociation.xlsx', energy_column_name='cc-pV5Z')



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


cinputs = dm21_coefficient_inputs
functional = NeuralFunctional(
    coefficients=nn_coefficients,
    coefficient_inputs=dm21_coefficient_inputs,
    energy_densities=partial(dm21_densities, functional_type="MGGA"),
)

####### Initializing the functional and some parameters #######

key = PRNGKey(42)  # Jax-style random seed

# We generate the features from the molecule we created before, to initialize the parameters
(key,) = split(key, 1)
rhoinputs = jax.random.normal(key, shape=[2, 7])
params = functional.init(key, rhoinputs)

# Select the checkpoint to load and more parameters
loadcheckpoint = True
checkpoint_step = 301
learning_rate = 1e-4
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)
cost_val = jnp.inf

orbax_checkpointer = PyTreeCheckpointer()

ckpt_dir = os.path.join(dirpath, ckpt_folder, "checkpoint_" + str(checkpoint_step) + "/")
if loadcheckpoint:
    train_state = functional.load_checkpoint(
        tx=tx, ckpt_dir=ckpt_dir, step=checkpoint_step, orbax_checkpointer=orbax_checkpointer
    )
    params = train_state.params
    tx = train_state.tx
    opt_state = tx.init(params)
    epoch = train_state.step

########### Definition of the molecule energy prediction function #####################

# Here we use one of the following. We will use the second here.
molecule_predict = jax.jit(molecule_predictor(functional))

######## Predict function ########


def predict(state, training_files, training_data_dirpath):
    """Predict molecules in file."""
    energies = {}
    params, _, _ = state
    for file in tqdm(training_files, "Files"):
        fpath = os.path.join(training_data_dirpath, file)
        print("Training on file: ", fpath, "\n")
        load = loader(fname=fpath, randomize=True, training=True, config_omegas=[])
        for _, system in tqdm(load, "Molecules/reactions per file"):
            predicted_energy, _ = molecule_predict(params, system)
            energies["".join(chr(num) for num in list(system.name))] = float(predicted_energy)
            del system
    return energies


######## Plotting the evaluation results ########

# Predictions
state = params, opt_state, cost_val
predictions_dict = predict(state, test_files, training_data_dirpath)
control_dict = predict(state, control_files, training_data_dirpath)
for k in control_dict.keys():
    print(k, control_dict[k], predictions_dict[k])

main_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Reading the dissociation curves from files
raw_data_folder = os.path.join(main_folder, "data/raw/dissociation/")

dissociation_H2_file = os.path.join(raw_data_folder, "H2_dissociation.xlsx")
dissociation_H2_df = pd.read_excel(dissociation_H2_file, header=0, index_col=0)

dissociation_H2plus_file = os.path.join(raw_data_folder, "H2plus_dissociation.xlsx")
dissociation_H2plus_df = pd.read_excel(dissociation_H2plus_file, header=0, index_col=0)

dissociation_N2_file = os.path.join(raw_data_folder, "N2_dissociation.xlsx")
dissociation_N2_df = pd.read_excel(dissociation_N2_file, header=0, index_col=0)

#todo: Select here the file where original data is stored
df = dissociation_H2plus_df
image_file_name = 'dissociation_H2plus_extrapolation.pdf'
column = 'cc-pVQZ'

def MAE(predictions: Dict, dissociation_df: pd.DataFrame):
    dissociation = dissociation_df[column].to_dict()
    MAE = 0
    for k in predictions.keys():
        MAE += abs(predictions[k] - dissociation[k])
    MAE /= len(predictions.keys())
    return MAE


predictions = OrderedDict()
for key in predictions_dict.keys():
    d = float(key.split("_")[-1][:-1])
    predictions[d] = predictions_dict[key]

data_file = os.path.join(training_data_dirpath, train_file)
with h5py.File(data_file, "r") as f:
    molecules = list(f.keys())

trained_dict = {}
for m in molecules:
    d = float(m.split("_")[-2])
    d = [dis for dis in df.index if np.isclose(d, dis)][0]
    trained_dict[d] = df.loc[d, column]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

predictions = collections.OrderedDict(sorted(predictions.items()))
x, y = [], []
for k, v in predictions.items():
    x.append(k)
    y.append(v)
ax.plot(x, y, '--', label='Predictions', color='#00a8ff', linewidth=2.5)

true_energies = {k: v for (k, v) in zip(df.index, df[column])} #todo
ax.plot(df.index, df[column], '-', color = '#192a56', label='FCI', linewidth=2.5)
ax.plot(trained_dict.keys(), trained_dict.values(), 'o', label='Training set', color='black')

ax.set_ylabel(r'Energy (Ha)', fontsize=24)
ax.set_xlabel(r'Interatomic distance ($\mathring{A}$)', fontsize=24)
finaly = list(df[column])[-1]
miny = min(df[column])
maxy = finaly + (finaly - miny)/5
miny -= (finaly - miny)/10
#todo: change range as appropriate
ax.set_ybound(-0.63, -0.4) 
#ax.set_ybound(-1.2, -0.85)
#ax.set_ybound(-109.63, -109.)
t = ax.text(0.15, 0.9, r'(a) $H_2^+$ extrapolation', transform=ax.transAxes, fontsize=24)
t.set_bbox(dict(facecolor='white', alpha=1, linewidth=0))
mae = MAE(predictions, df)

from matplotlib.ticker import MultipleLocator
xminorLocator = MultipleLocator(0.1)
ax.xaxis.set_minor_locator(xminorLocator)
yminorLocator = MultipleLocator(0.025)
ax.yaxis.set_minor_locator(yminorLocator)

#ax.text(0.9, 0.3, r'MAE: '+  f"{mae:.1e}"+ r' Ha', ha='right', va='bottom', transform=ax.transAxes, fontsize=24)
ax.legend(fontsize="24", loc = 'lower right')
ax.tick_params(axis='both', which='major', labelsize=24, direction = 'in')
ax.tick_params(axis='both', which='minor', direction = 'in')


file = os.path.join(dirpath, ckpt_folder, image_file_name)
plt.tight_layout()
#plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
fig.savefig(file, dpi = 200, bbox_inches="tight") #bbox_inches='tight'
plt.show()
plt.close()

for k in predictions.keys():
    print(k, predictions[k], df.loc[k,column])

print('The MAE over all predictions in H2 dissociation extrapolation is '+  str(MAE(predictions, df)))


print(
    "The MAE over all predictions in H2 dissociation interpolation is " + str(MAE(predictions, df))
)
