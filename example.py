from functional import LocalFunctional
from typing import Sequence, Callable, List
from functools import partial
from optax import GradientTransformation
from flax.training.train_state import TrainState
from orbax.checkpoint import Checkpointer, SaveArgs
import os

import jax
from jax.nn.initializers import lecun_normal, zeros, he_normal
from jax import numpy as jnp

from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze

from utils import Array, PyTree, DType, default_dtype


def canonicalize_inputs(x):

    x = jnp.asarray(x)

    if x.ndim == 1:
        return jnp.expand_dims(x, axis=1)
    elif x.ndim == 0:
        raise ValueError("`features` has to be at least 1D array!")
    else:
        return x

# First we define what will be our function f to integrate
class FeedForwardFunctional(nn.Module):

    layer_widths: Sequence[int]
    out_features: int = 3
    activation: Callable[[Array], Array] = jax.nn.gelu
    squash_offset: float = 1e-4
    sigmoid_scale_factor: float = 2.0
    kernel_init: Callable = he_normal()
    bias_init: Callable = zeros
    param_dtype: DType = default_dtype()

    """A feed-forward parameterization of a DFT functional a'la DM21.

    Parameters
    ----------
    layer_widths : Sequence[int]
        A sequence of layer widths for each `nn.Dense` layer in the network.
        For example,`layer_widths = [256, 256]` will generate an embedding/input
        layer of width 256 and an MLP with additional 2 layers of width 256 each.
    out_features : int, optional
        The number of output features at each lattice site.
    activation : Callable[[Array],Array], optional
        The (differentiable) activation function for the MLP part.
    squash_offset : float, optional
        (TO DO)
    sigmoid_scale_factor : float, optional
        (TO DO)
    kernel_init : Callable, optional
        (TO DO)
    bias_init : Callable, optional
        (TO DO)
    param_dtype : DType, optional
        The parameter data type. Defaults to jax.numpy.float32
        or jax.numpy.float64 if it is enabled.

    Notes
    -----
    This class inherits from `flax.linen.Module` which means that an `apply` method
    is generated as the main way of evaluating the network. There is also an
    `apply_and_integrate` convenience method provided to for immediate
    integration over a quadrature grid.

    Examples
    --------
    Example `flax` workflow:
    >>> Fxc = FeedForwardFunctional(layer_widths=[256, 256, 256]) # 256-dim embedding + 3-layer MLP
    >>> key = jax.random.PRNGKey(42)
    >>> example_input = jax.random.normal(key, shape=(10000, 11)) # (lattice_size, n_features_in)
    >>> key, = jax.random.split(key, 1)
    >>> params = Fxc.init(key, example_input) # Lazily initialize parameters with input
    >>> local_weights = Fxc.apply(params, example_input) # Network evaluation syntax
    >>> print(local_weights.shape) # (lattice_size, n_features_out)
    (10000, 3)
    """

    def setup(self):

        self.dense = partial(
            nn.Dense,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.layer_norm = partial(
            nn.LayerNorm,
            param_dtype=self.param_dtype
        )

    def head(self, x: Array):

        # Final layer: dense -> sigmoid -> scale (x2)
        x = self.dense(features=self.out_features)(x) # out_features = 3
        self.sow('intermediates', 'head_dense', x)
        x = jax.nn.sigmoid(x / self.sigmoid_scale_factor)
        self.sow('intermediates', 'sigmoid', x)
        out = self.sigmoid_scale_factor * x # sigmoid_scale_factor = 2.0
        self.sow('intermediates', 'sigmoid_product', out)

        return jnp.squeeze(out) # Eliminating unnecessary dimensions

    @nn.compact
    def __call__(self, x: Array):

        # Expected: x.shape = (*batch_size, n_grid_points, n_in_features = 11)

        x = canonicalize_inputs(x) # Making sure dimensions are correct

        # Initial layer: log -> dense -> tanh
        x = jnp.log(jnp.abs(x) + self.squash_offset) # squash_offset = 1e-4
        self.sow('intermediates', 'log', x)
        x = self.dense(features=self.layer_widths[0])(x) # features = 256
        self.sow('intermediates', 'initial_dense', x)
        x = jnp.tanh(x)
        self.sow('intermediates', 'tanh', x)

        # 6 Residual blocks with 256-features dense layer and layer norm
        for features,i in zip(self.layer_widths,range(len(self.layer_widths))): # layer_widths = [256]*6
            res = x
            x = self.dense(features=features)(x)
            self.sow('intermediates', 'residual_dense_'+str(i), x)
            x = x + res # nn.Dense + Residual connection
            self.sow('intermediates', 'residual_residual_'+str(i), x)
            x = self.layer_norm()(x) #+ res # nn.LayerNorm
            self.sow('intermediates', 'residual_layernorm_'+str(i), x) 
            x = self.activation(x) # activation = jax.nn.elu
            self.sow('intermediates', 'residual_elu_'+str(i), x)

        return self.head(x)

FFFunctional = FeedForwardFunctional(layer_widths = [256, 256, 256])

def f(inputs, params, localfeatures):
    localweights = FFFunctional.apply(params, inputs)
    return jnp.einsum('...r,...r->...r', localweights, localfeatures)


functional = LocalFunctional(f)
# Given rhoinputs, gridweights, params, localweights

from pyscf import gto, dft
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'augpc3', symmetry = True)

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

gridcoords, gridweights = grids.coords, grids.weights

key = jax.random.PRNGKey(42) # Jax-style random seed
rhoinput = jax.random.normal(key, shape=(*gridweights.shape, 7))

key, = jax.random.split(key, 1)
params = functional.init(key, rhoinput)

localweights = functional.apply(params, rhoinput)

key, = jax.random.split(key, 1)
localfeatures = jax.random.normal(
    key,
    shape=(*gridweights.shape, localweights.shape[-1])
)


energy = functional.energy(rhoinput, gridweights, params, localfeatures)