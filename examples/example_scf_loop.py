from jax import numpy as jnp
from optax import adam
from evaluate import make_molecule_scf_loop

from interface.pyscf import molecule_from_pyscf
from molecule import dm21_combine, dm21_features
from functional import NeuralFunctional, canonicalize_inputs
from jax.nn import gelu

# First we define a molecule:
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

# Then we define the (local) Functional, via an (arbitrary) function f whose output we will integrate (or not, if decide so)
squash_offset = 1e-4
layer_widths = [256]*6
out_features = 4
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
        x = instance.activation(x) # activation = jax.nn.gelu
        instance.sow('intermediates', 'residual_elu_'+str(i), x)

    x = instance.head(x, out_features, sigmoid_scale_factor)

    return jnp.einsum('ri,ri->r', x, localfeatures)

functional = NeuralFunctional(function)

# Load the saved checkpoints
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
step = 50
train_state = functional.load_checkpoint(tx, ckpt_dir='ckpts/checkpoint_' + str(step) +'/', step = step)
params = train_state.params
tx = train_state.tx
opt_state = tx.init(params)

# Create the scf iterator
scf_iterator = make_molecule_scf_loop(functional, feature_fn=dm21_features, combine_features_hf = dm21_combine, verbose = 2)
predicted_e = scf_iterator(params, molecule)
