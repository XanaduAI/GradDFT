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
import itertools
import re
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
from jax.nn import gelu
import numpy as np
from optax import adam
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer
import json
from matplotlib.ticker import MultipleLocator


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
from jax import config
config.update("jax_enable_x64", True)

# In this example we explain how to evaluate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Select here the folder where the checkpoints are stored
ckpt_folder = "checkpoints/ckpts_noise/"

import json
def convert(o):
    if isinstance(o, np.float32):
        return float(o)  
    return o

####### Model definition #######

# Then we define the Functional, via an function whose output we will integrate.
n_layers = 10
width_layers = 1024
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

loadcheckpoint = True #todo: change this
checkpoint_step = 441 #todo: change this
learning_rate = 1e-8
momentum = 0.9

orbax_checkpointer = PyTreeCheckpointer()

def average_error(prediction_dict, noisy_target_dict, clean_target_dict):
    ''' Computes the average error between the predictions and the targets in dimers containing/not containing transition metals '''
    noisy_mae = []
    clean_mae = []
    for k in prediction_dict.keys():
        noisy_mae.append(abs(noisy_target_dict[k] - prediction_dict[k]))
        clean_mae.append(abs(clean_target_dict[k] - prediction_dict[k]))
    return np.mean(noisy_mae), np.mean(clean_mae)

########### Definition of the molecule energy prediction function #####################

# Here we use one of the following. We will use the second here.
compute_energy = jax.jit(energy_predictor(functional))

######## Predict function ########

def predict(state, test_files, training_data_dirpath):
    """Predict molecules in file."""
    energies = {}
    true_energies = {}
    params, _, _ = state
    for file in tqdm(test_files, "Files"):
        fpath = os.path.join(training_data_dirpath, file)
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


####### Initializing the functional and some parameters #######

# todo: Select here the file to evaluate

noise_list = [0.0001] #, 0.001, 0.01, 0.1, 1]
seed_list = [0, 1, 2, 3, 4, 5]

data_noise = []
resulting_error = []
resulting_error_noisy = []

for noise, seed in itertools.product(noise_list, seed_list):

    key = PRNGKey(seed)  # Jax-style random seed #todo: select this
    np.random.seed(seed)  # Numpy-style random seed

    # Select here the file you would like to evaluate your model on
    test_files = [f"noise_{seed}_{noise}.h5"]
    training_data_dirpath = os.path.join(dirpath, ckpt_folder, f"seed_{seed}_noise_{noise}/") 

    # We generate the features from the molecule we created before, to initialize the parameters
    (key,) = split(key, 1)
    rhoinputs = jax.random.normal(key, shape=[2, 7])
    params = functional.init(key, rhoinputs)

    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(params)
    cost_val = jnp.inf

    ckpt_dir = os.path.join(dirpath, f"checkpoints/ckpts_noise/seed_{seed}_noise_{noise}/", "checkpoint_" + str(checkpoint_step) + "/")
    if loadcheckpoint:
        train_state = functional.load_checkpoint(
            tx=tx, step=checkpoint_step, orbax_checkpointer=orbax_checkpointer, ckpt_dir = ckpt_dir
        )
        params = train_state.params
        tx = train_state.tx
        opt_state = tx.init(params)
        epoch = train_state.step
    
    state = params, opt_state, cost_val

    predictions_path = os.path.join(dirpath, f"checkpoints/ckpts_noise/seed_{seed}_noise_{noise}/")

    # Check if "predictions.json" in predictions_path
    if os.path.exists(os.path.join(predictions_path, "predictions.json")):
        with open(os.path.join(predictions_path, "predictions.json"), "r") as f:
            predictions = json.load(f)
        noisy_targets = load_energies(test_files, training_data_dirpath)
    else:
        predictions, noisy_targets = predict(state, test_files, training_data_dirpath)
        for k in predictions.keys():
            print(k, predictions[k], noisy_targets[k])
            # Save predictions
        with open(os.path.join(predictions_path, "predictions.json"), "w") as f:
            json.dump(predictions, f, default=convert)

    # clean data targets
    clean_data_dirpath = os.path.join(dirpath, "data/training/noise/")
    clean_targets = load_energies(["noise_0_0.hdf5"], clean_data_dirpath)

    mae_clean, mae_noisy = average_error(predictions, noisy_targets, clean_targets)

    data_noise.append(noise)
    resulting_error.append(mae_clean)
    resulting_error_noisy.append(mae_noisy)


# Set up the figure and axis
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import pandas as pd


fig, ax = plt.subplots(figsize=(10, 6))

df = pd.DataFrame({
    'Column1 (Data_noise)': [str(d) for d in data_noise  + data_noise],
    'Column2 (resulting_error)':  resulting_error_noisy + resulting_error,
    'Column3 (source)': ['resulting_error_noisy'] * len(resulting_error_noisy) + ['ground_truth'] * len(data_noise)
})

palette = {'ground_truth': '#e1b12c', 'resulting_error_noisy': '#273c75'}

ax = sns.boxplot(x="Column1 (Data_noise)", y="Column2 (resulting_error)", hue="Column3 (source)", data=df, boxprops=dict(alpha=.3), palette=palette)  # RUN PLOT   
ax = sns.stripplot(x="Column1 (Data_noise)", y="Column2 (resulting_error)", hue="Column3 (source)", data=df, dodge=True,palette=palette)  # RUN PLOT   

# Set labels and title
ax.set_xlabel('Standard deviation of the Gaussian noise (Ha)', fontsize=16)
ax.set_ylabel('Error (Ha)', fontsize=16)

# Set y-axis to log scale
ax.set_yscale('log')

# Set tick font sizes
ax.tick_params(axis='both', which='major', labelsize=14, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=14, direction='in')

# Set custom x-axis tick positions and labels
custom_xticks = [0, 1, 2, 3, 4]
custom_xtick_labels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']
plt.xticks(custom_xticks, custom_xtick_labels)
ax.set_xticks(custom_xticks)
ax.set_xticklabels(custom_xtick_labels)

# Add more y-axis ticks
ax.set_yticks([1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1])  # Specify the locations of the tick marks
ax.set_yticklabels([r'$1\cdot 10^{-2}$',r'$2\cdot 10^{-2}$',r'$3\cdot 10^{-2}$',r'$4\cdot 10^{-2}$', '', r'$6\cdot 10^{-2}$', '',  r'$8\cdot 10^{-2}$', '', r'$1\cdot 10^{-1}$',
                    r'$2\cdot 10^{-1}$', r'$3\cdot 10^{-1}$', r'$4\cdot 10^{-1}$', r'$5\cdot 10^{-1}$', '',
                    r'$7\cdot 10^{-1}$'])  # Set the labels for the tick marks


# Set legend
category_colors = {'MAE on ground-truth data': '#e1b12c', 'MAE on noisy training data': '#273c75'}
legend_elements = [Patch(facecolor=color, label=category) for category, color in category_colors.items()]
plt.legend(handles=legend_elements)

plt.tight_layout()


# Save the figure
plotpath = os.path.join(dirpath, "checkpoints/noise/")
#fig.savefig('checkpoints/ckpts_noise/noise_vs_error_boxplot_horizontal.pdf', dpi=300)


################################# Plot the MAE vs epoch #####################################

noise_list = [1, 0.1, 0.01, 0.001, 0.0001]
seed_list = [0, 1, 2, 3, 4, 5]

mae = {}

for noise in noise_list:
    mae_noise = []
    for seed in seed_list:

        file_name = f"epoch_results_0_{seed}_{noise}.json"
        file = os.path.join(dirpath, f"checkpoints/ckpts_noise/seed_{seed}_noise_{noise}/", file_name)

        # for each noise, create an array with the mae at each epoch for all seeds, of size (n_seeds, n_epochs)
        with open(file, "r") as f:
            results = json.load(f)

        mae_noise_seed = []
        for epoch in results.keys():
            mae_noise_seed.append(results[epoch]["mean_abs_error"])
        mae_noise.append(mae_noise_seed)
    mae[noise] = np.array(mae_noise)

# Plot the mean MAE accross seeds vs epoch
fig, ax = plt.subplots(figsize=(10, 6))

for noise, color in zip(noise_list, ["#c23616", "#e1b12c", "#44bd32", "#0097e6", "#8c7ae6"]):
    ax.plot(np.arange(441), np.mean(mae[noise], axis=0), color=color, label=f"noise = {noise}")

# Plot shaded area for standard deviation
for noise, color in zip(noise_list, ["#c23616", "#e1b12c", "#44bd32", "#0097e6", "#8c7ae6"]):
    ax.fill_between(np.arange(441), np.mean(mae[noise], axis=0) - np.std(mae[noise], axis=0), np.mean(mae[noise], axis=0) + np.std(mae[noise], axis=0), color=color, alpha=0.1)

# Set labels and title
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Mean absolute error (Ha)', fontsize=16)

# Set y-axis to log scale
ax.set_yscale('log')

# Set tick font sizes
ax.tick_params(axis='both', which='major', labelsize=14, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=14, direction='in')

minor_locator = MultipleLocator(10)
ax.xaxis.set_minor_locator(minor_locator)

# Set plot limits
ax.set_ylim([1e-2, 1e1])
ax.set_xlim([0, 450])

# add legend with text size = 14
ax.legend(fontsize=14, loc = 'lower left')
plt.tight_layout()

# add verticle lines at 101, 201, 301, 391
ax.axvline(x=100, color='lightgray', linestyle='-', linewidth=1)
ax.axvline(x=200, color='lightgray', linestyle='-', linewidth=1)
ax.axvline(x=300, color='lightgray', linestyle='-', linewidth=1)
ax.axvline(x=390, color='lightgray', linestyle='-', linewidth=1)

# add text at the middle of the space between the verticle lines saying "lr = 1e-4"
ax.text(51, 8, r'lr = $10^{-4}$', ha='center', va='top', transform=ax.transData, fontsize=14, color='gray')
ax.text(151, 8, r'lr = $10^{-5}$', ha='center', va='top', transform=ax.transData, fontsize=14, color='gray')
ax.text(251, 8, r'lr = $10^{-6}$', ha='center', va='top', transform=ax.transData, fontsize=14, color='gray')
ax.text(351, 8, r'lr = $10^{-7}$', ha='center', va='top', transform=ax.transData, fontsize=14, color='gray')
ax.text(421, 8, r'lr = $10^{-8}$', ha='center', va='top', transform=ax.transData, fontsize=14, color='gray')


# Save the figure
fig.savefig('checkpoints/ckpts_noise/MAE_vs_epoch.pdf', dpi=300)