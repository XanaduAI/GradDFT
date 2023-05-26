from jax import numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jax import vmap
from jax.nn import sigmoid
from typing import Callable, Optional
from functools import partial
from jax.nn.initializers import zeros, he_normal
from jax import value_and_grad
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze

from utils import Scalar, Array, PyTree, DType, default_dtype
from molecule import Molecule

def external_f(instance, x):
    x = instance.dense(x)
    x = 0.5*jnp.tanh(x)
    return x

class Functional(nn.Module):
    ''' A base class of local functionals.
    F[n(r)] = \int f(n(r)) d^3 r
    F[n(r)] = f( \int n(r) d^3 r)

    Parameters
    ----------
    function: Callable
        Implements the function f above.
        Example:
        ```
        def external_f(instance, x):
            x = instance.dense(x)
            x = 0.5*jnp.tanh(x)
            return x
        ```
    '''

    f: staticmethod
    is_xc: bool = True
    is_local: bool = True
    kernel_init: Callable = he_normal()
    bias_init: Callable = zeros
    param_dtype: DType = default_dtype()

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

    def head(self, x: Array, out_features, sigmoid_scale_factor):

        # Final layer: dense -> sigmoid -> scale (x2)
        x = self.dense(features=out_features)(x) # out_features = 3
        self.sow('intermediates', 'head_dense', x)
        x = sigmoid(x / sigmoid_scale_factor)
        self.sow('intermediates', 'sigmoid', x)
        out = sigmoid_scale_factor * x # sigmoid_scale_factor = 2.0
        self.sow('intermediates', 'sigmoid_product', out)

        return jnp.squeeze(out) # Eliminating unnecessary dimensions

    @nn.compact
    def __call__(self, *inputs) -> Scalar:
        '''Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Paramters
        ---------
        inputs: inputs to the function f

        Returns
        -------
        Union[Array, Scalar]
        '''

        return self.f(self, *inputs)
    
    def apply_and_integrate(self, params: PyTree, molecule: Molecule, *args):
        '''
        Total energy of local functional
        
        Paramters
        ---------
        params: PyTree
            params of the neural network if there is one in self.f
        molecule: Molecule
        args: inputs to the function self.f

        Returns
        -------
        Union[Array, Scalar]
        '''

        localfeatures = self.apply(params, *args)
        return self._integrate(localfeatures, molecule.grid.weights)
    
    def energy(self, params: PyTree, molecule: Molecule, *args):
        '''
        Total energy of local functional
        
        Paramters
        ---------
        params: PyTree
            params of the neural network if there is one in self.f
        molecule: Molecule
        args: inputs to the function self.f

        Returns
        -------
        Union[Array, Scalar]
        '''

        if self.is_local: 
            energy = self.apply_and_integrate(params, molecule, *args)
        else: 
            energy = self.apply(params, molecule, *args)

        if self.is_xc:
            energy += molecule.nonXC()

        return energy

    def _integrate(
        self, features: Array, gridweights: Array, precision: Optional[Precision] = Precision.HIGHEST
    ) -> Array:

        """Helper function that performs grid quadrature (integration) 
				in a differentiable way (using jax.numpy).

        Parameters
        ----------
        features : Array
            features to integrate.
            Expected shape: (...,n_grid)
        gridweights: Array
            gridweights.
            Expected shape: (n_grid)
        precision : Precision, optional
            The precision to use for the computation, by default Precision.HIGHEST

        Returns
        -------
        Array
        """

        return jnp.einsum("r,r...->...", gridweights, features, precision = precision)


######################### Helper functions #########################


def canonicalize_inputs(x):

    x = jnp.asarray(x)

    if x.ndim == 1:
        return jnp.expand_dims(x, axis=1)
    elif x.ndim == 0:
        raise ValueError("`features` has to be at least 1D array!")
    else:
        return x
    

@partial(value_and_grad, has_aux = True)
def defaultloss(params, functional, molecule, trueenergy, *functioninputs):
    ''' Computes the loss function, here MSE, between predicted and true energy'''

    predictedenergy = functional.energy(params, molecule, *functioninputs)
    cost_value = (predictedenergy - trueenergy) ** 2

    return cost_value, predictedenergy

def _canonicalize_fxc(fxc: Functional) -> Callable:

    if hasattr(fxc, "energy"):
        return fxc.energy
    if hasattr(fxc, "apply"):
        return fxc.apply
    elif callable(fxc):
        return fxc
    else:
        raise RuntimeError(
            f"`fxc` should be a flax `Module` with a `predict_exc` method or a callable, got {type(fxc)}"
        )