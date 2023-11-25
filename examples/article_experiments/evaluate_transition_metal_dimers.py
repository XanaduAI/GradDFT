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
import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp
from jax.nn import gelu
import numpy as np
from optax import adam
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer
from pyscf.data.elements import ELEMENTS

import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import legend
import numpy as np
import os
import pandas as pd
import h5py
import re
import seaborn as sns

from grad_dft import (
    train_kernel, 
    energy_predictor,
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    dm21_densities,
    loader
)

def convert(o):
    if isinstance(o, np.float32):
        return float(o)  
    return o


dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Select here the file you would like to evaluate your model on
test_files = ["tm_dimers.h5", "non_tm_dimers.h5"]

# Select here the folder where the checkpoints are stored
ckpt_folder = "checkpoints/ckpts_tms/"
data_dirpath = os.path.join(dirpath, ckpt_folder) 

tms = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

####### Model definition #######

# Then we define the Functional, via an function whose output we will integrate.
n_layers = 10
width_layers = 512
squash_offset = 1e-4
layer_widths = [width_layers] * n_layers
out_features = 4
sigmoid_scale_factor = 2.0
activation = gelu
loadcheckpoint = True #todo: change this


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

key = PRNGKey(2)  # Jax-style random seed #todo: select this

# We generate the features from the molecule we created before, to initialize the parameters
(key,) = split(key, 1)
rhoinputs = jax.random.normal(key, shape=[2, 7])
params = functional.init(key, rhoinputs)

checkpoint_step = 351
learning_rate = 1e-7
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)
cost_val = jnp.inf

orbax_checkpointer = PyTreeCheckpointer()

ckpt_dir = os.path.join(dirpath, "checkpoints/ckpts_tms/", "checkpoint_" + str(checkpoint_step) + "/")
if loadcheckpoint:
    train_state = functional.load_checkpoint(
        tx=tx, step=checkpoint_step, orbax_checkpointer=orbax_checkpointer, ckpt_dir = ckpt_dir
    )
    params = train_state.params
    tx = train_state.tx
    opt_state = tx.init(params)
    epoch = train_state.step

molecules = []

compute_energy = jax.jit(energy_predictor(functional))

def predict(state, test_files, data_dirpath):
    """Predict molecules in file."""
    energies = {}
    true_energies = {}
    params, _, _ = state
    for file in tqdm(test_files, "Files"):
        fpath = os.path.join(data_dirpath, file)
        print("Training on file: ", fpath, "\n")
        load = loader(fname=fpath, randomize=True, training=True, config_omegas=[])
        for _, system in tqdm(load, "Molecules/reactions per file"):
            true_energies["".join(chr(num) for num in list(system.name))] = float(system.energy)
            predicted_energy, _ = compute_energy(params, system)
            energies["".join(chr(num) for num in list(system.name))] = float(predicted_energy)
            del system
    return energies, true_energies

def load_energies(test_files, data_dirpath):
    """Predict molecules in file."""
    true_energies = {}
    for file in tqdm(test_files, "Files"):
        fpath = os.path.join(data_dirpath, file)
        print("Training on file: ", fpath, "\n")
        load = loader(fname=fpath, randomize=True, training=True, config_omegas=[])
        for _, system in tqdm(load, "Molecules/reactions per file"):
            true_energies["".join(chr(num) for num in list(system.name))] = float(system.energy)
            del system
    return true_energies

targets = load_energies(test_files, data_dirpath)
#state = params, opt_state, cost_val
#predictions, targets = predict(state, test_files, data_dirpath)

# Load predictions from json file
with open('checkpoints/ckpts_tms/dimers_predictions.json') as json_file:
    predictions = json.load(json_file)

# change the keys from f"b'{atom}'" to f"{atom}".
#predictions = {k[2:-1]: v for k, v in predictions.items()}
targets = {k[2:-1]: v for k, v in targets.items()}

# save the predictions
#fpath = os.path.join(ckpt_folder, "dimers_predictions.json")
#with open(fpath, 'w') as fp:
#    json.dump(predictions, fp, default=convert)


######################### Plotting functionality ##########################################

def heatmap(data, row_labels, col_labels, molecules, unit = 'Ha', colbar = True):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    from matplotlib import pyplot as plt

    plt.rcParams['savefig.pad_inches'] = 0

    if colbar: fig = plt.figure(figsize=(9, 7))
    else: fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()

    cmap = matplotlib.colormaps['cividis']
    #cmap = plt.get_cmap('cmr.cosmic')
    cmap.set_bad(color='#f5f6fa')
    if not colbar: yticklabels = row_labels
    else: yticklabels=['']*len(row_labels)
    ax = sns.heatmap(data, cmap=cmap, annot=False, linewidths = 1, vmax=2,
                    xticklabels=col_labels, yticklabels=yticklabels, cbar = colbar)

    for molecule in molecules:
        a1, a2 = re.findall('[A-Z][a-z]*', molecule)
        ax.add_patch(Rectangle((atoms.index(a1), atoms.index(a2)), 1, 1, fill=False, edgecolor='gold', lw=1))
        ax.add_patch(Rectangle((atoms.index(a2), atoms.index(a1)), 1, 1, fill=False, edgecolor='gold', lw=1))

    plt.yticks(rotation=0, fontsize=24)
    plt.xticks(rotation=90, fontsize=24)

    if colbar:
        cbar = ax.collections[0].colorbar
        cbar.set_label(unit, rotation=0, labelpad=15, fontsize=18, loc = 'top')

        cbar.ax.yaxis.set_label_coords(2.55, 0.05)

        # Adjust the fontsize of colorbar labels
        cbar.ax.yaxis.set_tick_params(labelsize=18)  # Adjust the fontsize here

    plt.autoscale(tight=True)
    plt.tight_layout()

    return fig, ax
# From a dict of dicts, create a dataframe
atoms = ELEMENTS[1:37]

atom_indices = list(range(1, 37))
for i in range(1, 37, 2):
    atom_indices[i] = ''

#atoms.remove('He') #todo
#atoms.remove('Ne')
#atoms.remove('Ar')
#atoms.remove('Kr')
def dict_to_df(dictionary):
    result_dict = {}
    for a1 in atoms:
        a1_list = []
        for a2 in atoms:
            found = False
            for k, v in dictionary.items():
                ka1, ka2 = re.findall('[A-Z][^A-Z]*', k)
                if (ka1 == a1 and ka2 == a2) or (ka1 == a2 and ka2 == a1):
                    a1_list.append(v)
                    found = True
            if not found:
                a1_list.append(np.nan)
        result_dict[a1] = a1_list
    dataframe = pd.DataFrame(result_dict)
    #print(dataframe.head())
    return dataframe

def average_error(prediction_dict, target_dict):
    ''' Computes the average error between the predictions and the targets in dimers containing/not containing transition metals '''
    mae_tms = []
    mae_no_tms = []
    for k in target_dict.keys():
        a1, a2 = re.findall('[A-Z][^A-Z]*', k)
        if a1 in tms or a2 in tms:
            mae_tms.append(abs(target_dict[k] - prediction_dict[k]))
        else:
            mae_no_tms.append(abs(target_dict[k] - prediction_dict[k]))
    return np.mean(mae_tms), np.mean(mae_no_tms)


########################### Predictions from model trained on transition metals ###################################

diff_dict = {}
for k, v in targets.items():
    diff_dict[k] = abs(v - predictions[k])

print('When trained on transition metals, the MAE (Ha) for dimers containing / not-containing transition metals is: ', np.round(average_error(predictions, targets)[0],4), np.round(average_error(predictions, targets)[1],4))


results_df = dict_to_df(diff_dict)
results_array = results_df.to_numpy()

molecules = list(targets.keys())

# For each molecule in molecules, if the key contains a transition metal, then add a rectangle to the heatmap
tms_molecules = []
for molecule in molecules:
    a1, a2 = re.findall('[A-Z][a-z]*', molecule)
    if a1 in tms or a2 in tms:
        tms_molecules.append(molecule)

# make heatmap
fig, ax = heatmap(data = results_array, row_labels = atom_indices, col_labels = atom_indices, molecules=tms_molecules, colbar=True)
plt.title('(b) Training on dimers with transition metals', fontsize = 18, y = -0.17)

file = os.path.join(data_dirpath, "dimer_tm_heatmap.pdf")
#plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
fig.savefig(file, dpi = 300)
plt.close()

print('When trained on transition metals, the MAE (Ha) for dimers containing / not-containing transition metals is: ', np.round(average_error(predictions, targets)[0],4), np.round(average_error(predictions, targets)[1],4))
