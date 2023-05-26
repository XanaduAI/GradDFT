from dataclasses import dataclass
from typing import Callable, List, Optional
from functools import partial

from jax import value_and_grad
from jax import numpy as jnp
from jax.lax import Precision
from jax.nn import sigmoid, gelu, elu
from jax.nn.initializers import zeros, he_normal
from jax.random import normal, PRNGKey

from jax.experimental import checkify

from flax import struct
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints
from flax.training.train_state import TrainState
from optax import GradientTransformation
from orbax.checkpoint import Checkpointer, PyTreeCheckpointer

from utils import Scalar, Array, PyTree, DType, default_dtype
from molecule import Molecule

def external_f(instance, x):
    x = instance.dense(x)
    x = 0.5*jnp.tanh(x)
    return x

@dataclass
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
    is_xc: bool
    is_local: bool

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

        checkify.check(self.is_local, "This function should only be used with local functionals")

        return jnp.einsum("r,r...->...", gridweights, features, precision = precision)

@dataclass
class NeuralFunctional(Functional):

    f: staticmethod
    is_xc: bool = True
    is_local: bool = True
    kernel_init: Callable = he_normal()
    bias_init: Callable = zeros
    activation: Callable = gelu
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

    def save_checkpoints(self, params: PyTree, tx: GradientTransformation, step:int, orbax_checkpointer: Checkpointer = None, ckpt_dir: str = 'ckpts'):

        """A convenience function to save the network parameters to disk.

        Parameters
        ----------
        params : PyTree
            Neural network parameters, usually a `flax.core.FrozenDict`.
        tx : optax.GradientTransformation
            The optimizer used to train the network.
        """

        state = train_state.TrainState.create(apply_fn=self.apply,
                                            params=params,
                                            tx=tx)

        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=step, overwrite=True, 
                                    orbax_checkpointer=orbax_checkpointer, keep_every_n_steps = 50)

    def load_checkpoint(self, tx: GradientTransformation = None, ckpt_dir: str = 'ckpts', step: int = None, orbax_checkpointer: Checkpointer = None) -> PyTree:

        """A convenience function to load the network parameters from disk.

        Parameters
        ----------
        ckpt_dir : str, optional
            The directory where the checkpoint is saved.
            Defaults to 'ckpts'.
        """

        state_dict = orbax_checkpointer.restore(ckpt_dir)
        state = TrainState(params = freeze(state_dict['params']), tx = tx, step = step, 
                                opt_state = tx.init(freeze(state_dict['params'])), apply_fn=self.apply)

        return state
    
@dataclass
class DM21(NeuralFunctional):

    activation = elu
    squash_offset = 1e-4
    layer_widths = [256]*6
    out_features = 4
    sigmoid_scale_factor = 2.

    def f(instance, rhoinputs, localfeatures, *_, **__):
        x = canonicalize_inputs(rhoinputs) # Making sure dimensions are correct

        # Initial layer: log -> dense -> tanh
        x = jnp.log(jnp.abs(x) + instance.squash_offset) # squash_offset = 1e-4
        instance.sow('intermediates', 'log', x)
        x = instance.dense(features=instance.layer_widths[0])(x) # features = 256
        instance.sow('intermediates', 'initial_dense', x)
        x = jnp.tanh(x)
        instance.sow('intermediates', 'tanh', x)

        # 6 Residual blocks with 256-features dense layer and layer norm
        for features,i in zip(instance.layer_widths,range(len(instance.layer_widths))): # layer_widths = [256]*6
            res = x
            x = instance.dense(features=features)(x)
            instance.sow('intermediates', 'residual_dense_'+str(i), x)
            x = x + res # nn.Dense + Residual connection
            instance.sow('intermediates', 'residual_residual_'+str(i), x)
            x = instance.layer_norm()(x) #+ res # nn.LayerNorm
            instance.sow('intermediates', 'residual_layernorm_'+str(i), x) 
            x = instance.activation(x) # activation = jax.nn.gelu
            instance.sow('intermediates', 'residual_elu_'+str(i), x)

        x = instance.head(x, instance.out_features, instance.sigmoid_scale_factor)

        return jnp.einsum('ri,ri->r', x, localfeatures)

def generate_DM21_weights(self, folder: str = 'DM21_model', num_layers_with_dm_parameters: int = 7, n_input_features: int = 11, rng = PRNGKey(0)):

    """A convenience function to generate the DM21 weights and biases.

    Parameters
    ----------
    folder : str, optional
        The folder to the DM21 weights.
        Defaults to 'DM21_model'. Download the DM21 weights from
        https://github.com/deepmind/deepmind-research/tree/72c72d530f7de050451014895c1068b588f94733/density_functional_approximation_dm21/density_functional_approximation_dm21/checkpoints/DM21
    
    Returns
    -------
    params: FrozenDict
        The DM21 weights and biases.
    """

    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

    variables = tf.saved_model.load(folder).variables

    def tf_tensor_to_jax(tf_tensor: tf.Tensor) -> Array:
        return jnp.asarray(tf_tensor.numpy())

    def vars_to_params(variables: List[tf.Variable]) -> PyTree:
        import re
        params = {}
        for var in variables:

            if 'ResidualBlock_' in var.name:
                number = int(re.findall("ResidualBlock_[0-9]", var.name)[0][-1])+1
            elif 'ResidualBlock/' in var.name:
                number = 1
            elif 'Squash' in var.name:
                number = 0
            elif 'Output' in var.name:
                number = 7
            else:
                raise ValueError('Unknown variable name.')

            if '/linear/' in var.name:
                if 'Dense_'+str(number) not in params.keys(): params['Dense_'+str(number)] = {}
                if '/w:' in var.name:
                    params['Dense_'+str(number)]['kernel'] = tf_tensor_to_jax(var.value())
                elif '/b:' in var.name:
                    params['Dense_'+str(number)]['bias'] = tf_tensor_to_jax(var.value())
            elif '/layer_norm/' in var.name:
                if 'LayerNorm_'+str(number-1) not in params.keys(): params['LayerNorm_'+str(number-1)] = {}
                if 'gamma:' in var.name:
                    params['LayerNorm_'+str(number-1)]['scale'] = tf_tensor_to_jax(var.value())
                elif 'beta:' in var.name:
                    params['LayerNorm_'+str(number-1)]['bias'] = tf_tensor_to_jax(var.value())
        return params

    example_input = normal(rng, shape=(1, n_input_features))
    params = self.init(rng, example_input)

    dm_params = vars_to_params(variables)

    new_params = {}
    for key in params['params'].keys():
        check_same_params = []
        for k in params['params'][key].keys():
            if key in dm_params.keys(): check_same_params.append(params['params'][key][k].shape != dm_params[key][k].shape)
            else: check_same_params.append(True)
        if int(key.split('_')[1]) > num_layers_with_dm_parameters or any(check_same_params):
            new_params[key] = params['params'][key]
            if 'Dense' in key and new_params[key]['kernel'].shape[0] == new_params[key]['kernel'].shape[1]: # DM21 suggests initializing the kernel matrices close to the identity matrix
                new_params[key] = unfreeze(new_params[key])
                new_params[key]['kernel'] = new_params[key]['kernel'] + jnp.identity(new_params[key]['kernel'].shape[0])
                new_params[key] = freeze(new_params[key])
        else:
            new_params[key] = dm_params[key]

    params = unfreeze(params)
    params['params'] = new_params
    params = freeze(params)
    return params



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
def default_loss(params, functional, molecule, trueenergy, *functioninputs):
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