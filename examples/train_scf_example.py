from functools import partial
import os
import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
import numpy as np
from optax import adam
import tqdm

from interface.pyscf import molecule_from_pyscf
from interface.pyscf import loader
from functional import NeuralFunctional, canonicalize_inputs, dm21_molecule_features, local_features
from jax.nn import gelu
from orbax.checkpoint import PyTreeCheckpointer
from torch.utils.tensorboard import SummaryWriter

from train import make_scf_training_loop, make_train_kernel, molecule_predictor

from jax.config import config
config.update('jax_disable_jit', True)


orbax_checkpointer = PyTreeCheckpointer()

# In this example we aim to train a model on the result of a self-consistent loop.

####### Model definition #######

# First we define a molecule, using pyscf:
from pyscf import gto, dft
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1')

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# Then we compute quite a few properties which we pack into a class called Molecule
molecule = molecule_from_pyscf(mf)

# Then we define the Functional, via an function whose output we will integrate.
squash_offset = 1e-4
layer_widths = [128]*5
out_features = 16 # 2 spin channels, 2 for exchange/correlation, 4 for MGGA
sigmoid_scale_factor = 2.
activation = gelu

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

def features(molecule, functional_type='MGGA', *_, **__):
    localfeatures = local_features(molecule, functional_type)
    rhoinputs = dm21_molecule_features(molecule)
    return [rhoinputs, localfeatures]

functional = NeuralFunctional(function = function, features = features)

####### Initializing the functional and some parameters #######

key = PRNGKey(42) # Jax-style random seed

# We generate the features from the molecule we created before, to initialize the parameters
localfeatures = local_features(molecule, functional_type='MGGA')
rhoinputs = dm21_molecule_features(molecule)
key, = split(key, 1)
params = functional.init(key, rhoinputs, localfeatures)

learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)

num_epochs = 50
cost_val = jnp.inf

dirpath = os.path.dirname(os.path.dirname(__file__))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/")
training_files = '/dissociation/H2_extrapolation_molecules.hdf5'

####### Loss function and train kernel #######

# Here we use one of the following. We will use the second here.
molecule_predict = molecule_predictor(functional)
scf_train_loop = make_scf_training_loop(functional, max_cycles=1)

@partial(value_and_grad, has_aux = True)
def loss(params, molecule, ground_truth_energy):

    #predicted_energy, fock = molecule_predict(params, molecule)
    predicted_energy, fock, rdm1 = scf_train_loop(params, molecule)
    cost_value = (predicted_energy - ground_truth_energy) ** 2

    # We may want to add a regularization term to the cost, be it one of the
    # fock_grad_regularization, dm21_grad_regularization, or orbital_grad_regularization in train.py;
    # or even the satisfaction of the constraints in constraints.py.

    metrics = {'predicted_energy': predicted_energy,
                'ground_truth_energy': ground_truth_energy,
                'mean_abs_error': jnp.mean(jnp.abs(predicted_energy - ground_truth_energy)),
                'mean_sq_error': jnp.mean((predicted_energy - ground_truth_energy)**2),
                'cost_value': cost_value,
                #'regularization': regularization_logs
                }

    return cost_value, metrics

kernel = jax.jit(make_train_kernel(tx, loss))

######## Training epoch ########

def train_epoch(state, training_files, training_data_dirpath):
    r"""Train for a single epoch."""

    batch_metrics = []
    params, opt_state, cost_val = state
    fpath = os.path.join(training_data_dirpath + training_files)
    print('Training on file: ', fpath, '\n')

    load = loader(fname = fpath, randomize=True, training = True, config_omegas = [])
    for _, system in tqdm.tqdm(load):
        params, opt_state, cost_val, metrics = kernel(params, opt_state, system, system.energy)
        del system

        for k in metrics.keys():
            print(k, metrics[k])
        batch_metrics.append(metrics)

    epoch_metrics = {
        k: np.mean([jax.device_get(metrics[k]) for metrics in batch_metrics])
        for k in batch_metrics[0]}
    state = (params, opt_state, cost_val)
    return state, metrics, epoch_metrics



######## Training loop ########

writer = SummaryWriter()
for epoch in range(1, num_epochs + 1):

    # Use a separate PRNG key to permute input data during shuffling
    #rng, input_rng = jax.random.split(rng)

    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state

    # Save metrics and checkpoint
    print(f"Epoch {epoch} metrics:")
    for k in epoch_metrics:
        print(f"-> {k}: {epoch_metrics[k]:.5f}")
    for metric in epoch_metrics.keys():
        writer.add_scalar(f'/{metric}/train', epoch_metrics[metric], epoch)
    writer.flush()
    functional.save_checkpoints(params, tx, step = epoch, orbax_checkpointer = orbax_checkpointer)
    print(f"-------------\n")
    print(f"\n")

writer.flush()

functional.save_checkpoints(params, tx, step = num_epochs)


