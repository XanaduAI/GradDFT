import collections
from functools import partial
from typing import Dict, OrderedDict
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
from jax.nn import gelu
from matplotlib import pyplot as plt
#plt.rcParams['text.usetex'] = True
import numpy as np
from optax import adam
import pandas as pd
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer
from evaluate import make_test_kernel
import h5py

from train import make_train_kernel, molecule_predictor
from functional import NeuralFunctional, canonicalize_inputs, dm21_features
from interface.pyscf import loader

from torch.utils.tensorboard import SummaryWriter
import jax

# In this example we explain how to replicate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.

dirpath = os.path.dirname(os.path.dirname(__file__))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dissociation/")
#todo: change here the file to evaluate
test_files = ["H2_extrapolation.h5"]
ckpt_folder = "ckpts_H2_extrapolation/"
train_file = 'H2_extrapolation_train.hdf5'
control_files = [train_file]
# alternatively, use "H2plus_extrapolation.h5". You will have needed to execute in data_processing.py
#process_dissociation(atom1 = 'H', atom2 = 'H', charge = 0, spin = 0, file = 'H2_dissociation.xlsx', energy_column_name='cc-pV5Z')
#process_dissociation(atom1 = 'H', atom2 = 'H', charge = 1, spin = 1, file = 'H2plus_dissociation.xlsx', energy_column_name='cc-pV5Z')



####### Model definition #######

# Then we define the Functional, via an function whose output we will integrate.
n_layers = 10
width_layers = 512
squash_offset = 1e-4
layer_widths = [width_layers]*n_layers
out_features = 4
sigmoid_scale_factor = 2.
activation = gelu
loadcheckpoint = True

def function(instance, rhoinputs, localfeatures, *_, **__):
    x = canonicalize_inputs(rhoinputs) # Making sure dimensions are correct

    # Initial layer: log -> dense -> tanh
    x = jnp.log(jnp.abs(x) + squash_offset) # squash_offset = 1e-4
    instance.sow('intermediates', 'log', x)
    x = instance.dense(features=layer_widths[0])(x) # features = 256
    instance.sow('intermediates', 'initial_dense', x)
    x = jnp.tanh(x)
    instance.sow('intermediates', 'tanh', x)

    # 6 Residual blocks with 256-features dense layer and layer norm
    for features,i in zip(layer_widths,range(len(layer_widths))): # layer_widths = [256]*6
        res = x
        x = instance.dense(features=features)(x)
        instance.sow('intermediates', 'residual_dense_'+str(i), x)
        x = x + res # nn.Dense + Residual connection
        instance.sow('intermediates', 'residual_residual_'+str(i), x)
        x = instance.layer_norm()(x) #+ res # nn.LayerNorm
        instance.sow('intermediates', 'residual_layernorm_'+str(i), x) 
        x = activation(x) # activation = jax.nn.gelu
        instance.sow('intermediates', 'residual_elu_'+str(i), x)

    x = instance.head(x, out_features, sigmoid_scale_factor)

    return jnp.einsum('ri,ri->r', x, localfeatures)

features = partial(dm21_features, functional_type = 'MGGA')
functional = NeuralFunctional(function = function, features = features)

####### Initializing the functional and some parameters #######

key = PRNGKey(42) # Jax-style random seed

# We generate the features from the molecule we created before, to initialize the parameters
key, = split(key, 1)
rhoinputs = jax.random.normal(key, shape = [2, 7])
localfeatures = jax.random.normal(key, shape = [2, out_features])
params = functional.init(key, rhoinputs, localfeatures)

checkpoint_step = 301
learning_rate = 1e-4
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)
cost_val = jnp.inf

orbax_checkpointer = PyTreeCheckpointer()

ckpt_dir = os.path.join(dirpath, ckpt_folder, 'checkpoint_' + str(checkpoint_step) +'/')
if loadcheckpoint:
    train_state = functional.load_checkpoint(tx = tx, ckpt_dir = ckpt_dir, step = checkpoint_step, orbax_checkpointer=orbax_checkpointer)
    params = train_state.params
    tx = train_state.tx
    opt_state = tx.init(params)
    epoch = train_state.step

########### Definition of the loss function ##################### 

# Here we use one of the following. We will use the second here.
molecule_predict = jax.jit(molecule_predictor(functional))

######## Test epoch ########


def predict(state, training_files, training_data_dirpath):
    """Predict molecules in file."""
    energies = {}
    params, _, _ = state
    for file in tqdm(training_files, 'Files'):
        fpath = os.path.join(training_data_dirpath, file)
        print('Training on file: ', fpath, '\n')
        load = loader(fpath = fpath, randomize=True, training = True, config_omegas = [])
        for _, system in tqdm(load, 'Molecules/reactions per file'):
            predicted_energy, _ = molecule_predict(params, system)      
            energies[''.join(chr(num) for num in list(system.name))] = float(predicted_energy)
            del system
    return energies

######## Plotting the evaluation results ########

# Predictions
state = params, opt_state, cost_val
predictions_dict = predict(state, test_files, training_data_dirpath)
control_dict = predict(state, control_files, training_data_dirpath)
for k in control_dict.keys():
    print(k, control_dict[k], predictions_dict[k])

main_folder = os.path.dirname(os.path.dirname(__file__))

# Reading the dissociation curves from files
raw_data_folder = os.path.join(main_folder, 'data/raw/dissociation/')

dissociation_H2_file = os.path.join(raw_data_folder, 'H2_dissociation.xlsx')
dissociation_H2_df = pd.read_excel(dissociation_H2_file, header=0, index_col=0)

dissociation_H2plus_file = os.path.join(raw_data_folder, 'H2plus_dissociation.xlsx')
dissociation_H2plus_df = pd.read_excel(dissociation_H2plus_file, header=0, index_col=0)

dissociation_N2_file = os.path.join(raw_data_folder, 'N2_dissociation.xlsx')
dissociation_N2_df = pd.read_excel(dissociation_N2_file, header=0, index_col=0)

#todo: change here too the file
df = dissociation_H2_df
column = 'cc-pV5Z'
image_file_name = 'dissociation_H2_extra.pdf'

def MAE(predictions: Dict, dissociation_df: pd.DataFrame):
    dissociation = dissociation_df[column].to_dict()
    MAE = 0
    for k in predictions.keys():
        MAE += abs(predictions[k] - dissociation[k])
    MAE /= len(predictions.keys())
    return MAE


predictions = OrderedDict()
for k in predictions_dict.keys():
    d = float(k.split('_')[-1][:-1])
    predictions[d] = predictions_dict[k]


train_data_folder = os.path.join(main_folder, 'data/training/dissociation/')

data_file = os.path.join(train_data_folder, train_file)
with h5py.File(data_file, 'r') as f:
    molecules = list(f.keys())

trained_dict = {}
for m in molecules:
    d = float(m.split('_')[-2])
    d = [dis for dis in df.index if np.isclose(d, dis)][0]
    trained_dict[d] = df.loc[d,column]


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

predictions = collections.OrderedDict(sorted(predictions.items()))
x, y = [], []
for k, v in predictions.items():
    x.append(k)
    y.append(v)
ax.plot(x, y, '-', label='Model predictions', color='red')
#if system is 'H2' or 'H2plus':

#else:
#    column = 'energy (Ha)'
true_energies = {k: v for (k, v) in zip(df.index, df[column])}
ax.plot(df.index, df[column], 'b-', label='Ground Truth')
ax.plot(trained_dict.keys(), trained_dict.values(), 'o', label='Trained points', color='black')

ax.set_ylabel('Energy (Ha)')
ax.set_xlabel('Distance (A)')
finaly = list(df[column])[-1]
miny = min(df[column])
maxy = finaly + (finaly - miny)/5
miny -= (finaly - miny)/10
ax.set_ybound(-0.65, -0.4)
mae = MAE(predictions, df)
# Write title inside the figure
#todo: change title
#ax.set_title('H2 dissociation', loc='center')
#ax.text(0.5, 0.9, r'$H_2$ dissociation', ha='center', va='center', transform=ax.transAxes, fontsize=16)
ax.text(0.9, 0.5, 'Evaluation MAE: '+  f"{mae:.4e}"+ ' Ha', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)
ax.legend()


file = os.path.join(dirpath, ckpt_folder, image_file_name)
fig.savefig(file, dpi = 300)
plt.show()
plt.close()

for k in predictions.keys():
    print(k, predictions[k], df.loc[k,column])

print('The MAE over all predictions in H2 dissociation extrapolation is '+  str(MAE(predictions, df)))


