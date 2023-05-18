from functional import LocalFunctional, Functional
from typing import Sequence, Callable, List
from functools import partial
from optax import GradientTransformation
from flax.training.train_state import TrainState
from orbax.checkpoint import Checkpointer, SaveArgs
import os

import jax
from jax.nn.initializers import lecun_normal, zeros, he_normal
from jax import numpy as jnp
from jax.nn import gelu

from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze
from interface.pyscf import molecule_from_pyscf

from utils import Array, PyTree, DType, default_dtype
from molecule import default_features_ex_hf



from pyscf import gto, dft
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'augpc3', symmetry = True)

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.grids = grids
mf.kernel()

molecule = molecule_from_pyscf(mf)

gridcoords, gridweights = grids.coords, grids.weights

squash_offset = 1e-4
layer_widths = [256]*6
activation = gelu
out_features = 4
sigmoid_scale_factor = 2.

def canonicalize_inputs(x):

    x = jnp.asarray(x)

    if x.ndim == 1:
        return jnp.expand_dims(x, axis=1)
    elif x.ndim == 0:
        raise ValueError("`features` has to be at least 1D array!")
    else:
        return x

def f(instance, rhoinputs, localfeatures):
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
        x = activation(x) # activation = jax.nn.elu
        instance.sow('intermediates', 'residual_elu_'+str(i), x)

    x = instance.head(x, out_features, sigmoid_scale_factor)

    return jnp.einsum('ri,ri->r', x, localfeatures)


functional = Functional(f)

key = jax.random.PRNGKey(42) # Jax-style random seed
rhoinputs = jax.random.normal(key, shape=(*gridweights.shape, 7))
key, = jax.random.split(key, 1)
localfeatures = jax.random.normal(key, shape=(*gridweights.shape, out_features))

rhoinputs, localfeatures, _ = default_features_ex_hf(molecule, 'MGGA')

key, = jax.random.split(key, 1)

input_dict = {'rhoinputs': rhoinputs.T, 'localfeatures': localfeatures.T}

params = functional.init(key, input_dict)

localweights = functional.apply(params, input_dict)

energy = functional.energy(params, gridweights, input_dict)
print(energy)