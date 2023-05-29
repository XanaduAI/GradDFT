from jax.random import split, PRNGKey
from jax import numpy as jnp
from optax import adam, apply_updates
from evaluate import make_molecule_scf_loop

from interface.pyscf import molecule_from_pyscf
from molecule import default_features
from functional import NeuralFunctional, canonicalize_inputs, default_loss
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

def f(instance, rhoinputs, localfeatures, *_, **__):
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

functional = NeuralFunctional(f)

# We generate the parameters
key = PRNGKey(42) # Jax-style random seed
rhoinputs, localfeatures = default_features(molecule = molecule, functional_type='MGGA')
key, = split(key, 1)
params = functional.init(key, rhoinputs, localfeatures)

# Create the scf iterator
scf_iterator = make_molecule_scf_loop(functional, feature_fn=default_features, verbose = 2)
predicted_e = scf_iterator(params, molecule)
