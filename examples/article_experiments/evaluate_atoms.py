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
from jax import config
config.update("jax_enable_x64", True)

# In this example we explain how to evaluate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# todo: Select here the file to evaluate

# Select here the file you would like to evaluate your model on
test_files = ["atoms.h5"]

# Select here the folder where the checkpoints are stored
ckpt_folder = "checkpoints/atoms/"
training_data_dirpath = os.path.join(dirpath, ckpt_folder) 
import json
def convert(o):
    if isinstance(o, np.float32):
        return float(o)  
    return o

# In this example we explain how to replicate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/atoms/")
training_files = ["atoms_training.h5"]

####### Model definition #######

# Then we define the Functional, via an function whose output we will integrate.
n_layers = 10
width_layers = 1024
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

key = PRNGKey(1)  # Jax-style random seed #todo: select this

# We generate the features from the molecule we created before, to initialize the parameters
(key,) = split(key, 1)
rhoinputs = jax.random.normal(key, shape=[2, 7])
params = functional.init(key, rhoinputs)

checkpoint_step = 441 #todo: change this
learning_rate = 3e-6
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)
cost_val = jnp.inf

orbax_checkpointer = PyTreeCheckpointer()

ckpt_dir = os.path.join(dirpath, "ckpts/atoms/", "checkpoint_" + str(checkpoint_step) + "/")
if loadcheckpoint:
    train_state = functional.load_checkpoint(
        tx=tx, step=checkpoint_step, orbax_checkpointer=orbax_checkpointer, ckpt_dir = "ckpts/atoms/"
    )
    params = train_state.params
    tx = train_state.tx
    opt_state = tx.init(params)
    epoch = train_state.step

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


######## Plotting the evaluation results ########

# Predictions
state = params, opt_state, cost_val
predictions, targets = predict(state, test_files, training_data_dirpath)
for k in predictions.keys():
    print(k, predictions[k], targets[k])

from pyscf.data.elements import ELEMENTS, CONFIGURATION

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
training_atoms = ELEMENTS[1:19] + ["Ca", "Ge", "Se", "Kr"] + transition_metals[::2]
test_atoms =  ["K", "Ga", "As", "Br"] + transition_metals[1::2]

atoms = np.array(ELEMENTS[1:37])
training_mask = np.array([atom in training_atoms for atom in atoms])
test_mask = np.array([atom in test_atoms for atom in atoms])
print(atoms[training_mask])

#targets = {'Al': -242.34716796875, 'Ar': -527.5089111328125, 'As': -2235.6865234375, 'B': -24.649311065673828, 'Be': -14.661041259765625, 'Br': -2573.966552734375, 'C': -37.84539031982422, 'Ca': -677.5281372070312, 'Cl': -460.1239929199219, 'Co': -1382.6351318359375, 'Cr': -1044.35595703125, 'Cu': -1640.3936767578125, 'F': -99.74347686767578, 'Fe': -1262.55810546875, 'Ga': -1924.7037353515625, 'Ge': -2076.804931640625, 'H': -0.4941110610961914, 'He': -2.896589994430542, 'K': -599.8792724609375, 'Kr': -2753.6103515625, 'Li': -7.472015380859375, 'Mg': -200.05953979492188, 'Mn': -1149.224365234375, 'N': -54.591041564941406, 'Na': -162.2571258544922, 'Ne': -128.9423065185547, 'Ni': -1508.1883544921875, 'O': -75.0745849609375, 'P': -341.2413330078125, 'S': -398.0894775390625, 'Sc': -760.5699462890625, 'Se': -2401.3564453125, 'Si': -289.3520202636719, 'Ti': -849.2960205078125, 'V': -943.83154296875, 'Zn': -1779.2974853515625}
#predictions = {'Al': -242.34051513671875, 'Ar': -527.5103149414062, 'As': -2235.405029296875, 'B': -24.811548233032227, 'Be': -14.622066497802734, 'Br': -2574.293212890625, 'C': -38.076690673828125, 'Ca': -677.5549926757812, 'Cl': -460.1126708984375, 'Co': -1382.62158203125, 'Cr': -1040.7052001953125, 'Cu': -1640.4189453125, 'F': -99.81681823730469, 'Fe': -1263.396484375, 'Ga': -1924.8819580078125, 'Ge': -2076.779052734375, 'H': -0.406159371137619, 'He': -2.6437671184539795, 'K': -599.7616577148438, 'Kr': -2753.63232421875, 'Li': -7.312269687652588, 'Mg': -200.1741943359375, 'Mn': -1149.2525634765625, 'N': -53.720760345458984, 'Na': -162.272705078125, 'Ne': -128.9036407470703, 'Ni': -1508.4461669921875, 'O': -75.10851287841797, 'P': -341.3023681640625, 'S': -398.0742492675781, 'Sc': -760.5703125, 'Se': -2401.338623046875, 'Si': -289.32244873046875, 'Ti': -848.8484497070312, 'V': -943.79736328125, 'Zn': -1779.6044921875}


# Two subplots
fig = plt.figure(figsize=(12, 7))
ax, ax2 = fig.subplots(2, 1, sharex=True)

# Plot 1
#ax.set_xlabel('Atoms', fontsize=14)
ax.tick_params(axis='y', which='major', labelsize=14, direction = 'in')
ax.tick_params(axis='y', which='minor', labelsize=14, direction = 'in')
ax.tick_params(axis='x', which='major', labelsize=14)

# Plot difference between predictions and targets, ordered according to atoms, in log y scale
diffs = (np.array([predictions[atom] for atom in atoms]) - np.array([targets[atom] for atom in atoms]))

std = np.std(diffs)
training_mean = np.mean(abs(diffs[training_mask]))
test_mean = np.mean(abs(diffs[test_mask]))

# Now we print them also in the plot
for a, d in zip(atoms, diffs):
    if a in training_atoms: label, color = 'Training', '#192a56'
    elif a in test_atoms: label, color = 'Test', '#00a8ff'
    ax.scatter(a, d, label = label, color = color)
#ax.scatter(atoms[training_mask], diffs[training_mask], label='Training MAE')
#ax.scatter(atoms[test_mask], diffs[test_mask], label='Test MAE')
ax.set_ylabel('Error (Ha)', fontsize=14)

ax.text(0.03, 0.9, '(a) Error', transform=ax.transAxes, fontsize=14)

#plot line at 0
ax.plot(atoms, np.zeros(len(atoms)), 'k--')


handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend1 = ax.legend(by_label.values(), by_label.keys(), fontsize=14, loc ='center left' )
ax.add_artist(legend1)


# Plot 2
# Plot absolute errors as 
for a, d in zip(atoms, diffs):
    if a in training_atoms: label, color = 'Training', '#192a56'
    elif a in test_atoms: label, color = 'Test', '#00a8ff'
    ax2.bar(a, abs(d), label = label, color = color)
#ax2.bar(atoms[training_mask], abs(diffs[training_mask]), label='Training MAE')
#ax2.bar(atoms[test_mask], abs(diffs[test_mask]), label='Test MAE')
ax2.plot(atoms, np.ones(len(atoms))*training_mean, '-', color = '#192a56', label='Training MAE')
ax2.plot(atoms, np.ones(len(atoms))*test_mean, '--', color = '#00a8ff', label='Test MAE')
# Now we print them also in the plot
ax.text(0.6, 2., 'Training MAE: {:.1e} Ha'.format(training_mean), horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, color = '#192a56', fontsize=14)
ax.text(0.6, 1.9, 'Test MAE: {:.1e} Ha'.format(test_mean), horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, color = '#00a8ff', fontsize=14)
ax2.text(0.03, 0.9, '(b) Absolute error', transform=ax2.transAxes, fontsize = 14)
#ax2.text(0.5, 0.85, 'Std: {:.4f} kcal/mol'.format(std), horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes)

handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend2 = ax.legend(by_label.values(), by_label.keys(), fontsize=14)
ax.add_artist(legend2)

ax2.set_ylabel('Absolute error (Ha)', fontsize=14)
ax2.tick_params(axis='x', which='major', labelsize=14, rotation=90)
ax2.tick_params(axis='x', which='minor', labelsize=14, rotation=90)
ax2.tick_params(axis='y', which='major', labelsize=14, direction = 'in')
ax2.tick_params(axis='y', which='minor', labelsize=14, direction = 'in')

#fig.suptitle('Absolute errors in electronic energies, no noise', fontsize=16)

ref_mean_training = training_mean
ref_mean_test = test_mean
ref_std = std

# set log scale
ax2.set_yscale('log')

from matplotlib.ticker import MultipleLocator
#xminorLocator = MultipleLocator(0.1)
#ax.xaxis.set_minor_locator(xminorLocator)
yminorLocator = MultipleLocator(0.25)
ax.yaxis.set_minor_locator(yminorLocator)
yminorLocator = MultipleLocator(0.25)
#ax2.yaxis.set_minor_locator(yminorLocator)

plt.show()


#save
#tight layout
plt.tight_layout()
fig.savefig('atoms_generalization.pdf', dpi=100)

