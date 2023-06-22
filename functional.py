from dataclasses import dataclass
import itertools
from typing import Callable, Optional, List, Dict, Union
from functools import partial

from jax import grad, value_and_grad
from jax import numpy as jnp
from jax.lax import Precision, stop_gradient
from jax.nn import sigmoid, gelu, elu
from jax.nn.initializers import zeros, he_normal
from jax.random import normal, PRNGKey

from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints
from flax.training.train_state import TrainState
from optax import GradientTransformation
from orbax.checkpoint import Checkpointer, PyTreeCheckpointer

from utils import Scalar, Array, PyTree, DType, default_dtype
from molecule import Molecule

@dataclass
class Functional(nn.Module):
    r""" A base class of local functionals.
    .. math::
        F[n(r)] = \int f(n(r)) d^3 r

    or
    .. math::
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

    features : Callable, optional
        A function that calculates and/or loads the molecule features where gradient is
        computed via auto differentiation.

        If given, it must be a callable with the following signature:

        feature_fn(molecule: Molecule, *args, **kwargs) -> Union[Array, Sequence[Arrays]]

    nograd_features : Callable, optional
        A function that calculates and/or loads the molecule features where gradient is
        computed via in featuregrads.

        nograd_features(molecule: Molecule, *args, **kwargs) -> Union[Array, Sequence[Arrays]]

    featuregrads: Callable, optional
        A function to compute the Fock matrix using gradients of with respect to those features 
        where autodifferentiation is not used.

        If given has signature

        featuregrads(functional: nn.Module, params: Dict, molecule: Molecule, 
            features: List[Array], nogradfeatures: Array, *args) - > Fock matrix: Array of shape (2, nao, nao)

    combine : Callable, optional
        A function that joins the features computed with and without autodifferentiation.

    is_xc: bool
        Whether the functional models only the exchange-correlation energy

    """

    function: staticmethod
    features: staticmethod
    nograd_features: staticmethod = None
    featuregrads: staticmethod = None
    combine: staticmethod = None
    is_xc: bool = True

    @nn.compact
    def __call__(self, *inputs) -> Scalar:
        r"""Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Paramters
        ---------
        inputs: inputs to the function f

        Returns
        -------
        Union[Array, Scalar]
        """

        return self.function(self, *inputs)
    
    def apply_and_integrate(self, params: PyTree, molecule: Molecule, *inputs):
        r"""
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
        """

        localfeatures = self.apply(params, *inputs)
        return self._integrate(localfeatures, molecule.grid.weights)
    
    def energy(self, params: PyTree, molecule: Molecule, *args):
        r"""
        Total energy of local functional
        
        Paramters
        ---------
        params: PyTree
            params of the neural network if there is one in self.f
        molecule: Molecule
        args: inputs to the function self.f

        Returns
        -------
        Scalar

        Note
        -------
        Integrates the energy over the grid.
        If the function is_xc, it will add the rest of the energy components
        computed with function molecule.nonXC()
        """

        energy = self.apply_and_integrate(params, molecule, *args)

        if self.is_xc:
            energy += stop_gradient(molecule.nonXC())

        return energy

    def _integrate(
        self, features: Array, gridweights: Array, precision: Optional[Precision] = Precision.HIGHEST
    ) -> Array:
        r"""
        Helper function that performs grid quadrature (integration) 
				in a differentiable way (using jax.numpy).

        Parameters
        ----------
        features : Array
            features to integrate.
            Expected shape: (n_grid, ...)
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

@dataclass
class NeuralFunctional(Functional):
    r"""
    Neural functional, subclass of Functional
    """

    function: staticmethod
    features: staticmethod
    nograd_features: staticmethod = None
    featuregrads: staticmethod = None
    combine: staticmethod = None
    is_xc: bool = True
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

    def head(self, x: Array, local_features, sigmoid_scale_factor):

        # Final layer: dense -> sigmoid -> scale (x2)
        x = self.dense(features=local_features)(x) # eg local_features = 3 in DM21
        self.sow('intermediates', 'head_dense', x)
        x = sigmoid(x / sigmoid_scale_factor)
        self.sow('intermediates', 'sigmoid', x)
        out = sigmoid_scale_factor * x # sigmoid_scale_factor = 2.0 in DM21
        self.sow('intermediates', 'sigmoid_product', out)

        return jnp.squeeze(out) # Eliminating unnecessary dimensions

    def save_checkpoints(self, params: PyTree, tx: GradientTransformation, step: Optional[int], 
                        orbax_checkpointer: Checkpointer = PyTreeCheckpointer(), ckpt_dir: str = 'ckpts'):

        r"""
        A convenience function to save the network parameters to disk.

        Parameters
        ----------
        params : PyTree
            Neural network parameters, usually a `flax.core.FrozenDict`.
        tx : optax.GradientTransformation
            The optimizer used to train the network.
        step: int
            The epoch of the optimizer.
        orbax_checkpointer: Optional[Checkpointer] = PyTreeCheckpointer()

        Returns
        ----------
        None
        """

        state = train_state.TrainState.create(apply_fn=self.apply,
                                            params=params,
                                            tx=tx)

        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=step, overwrite=True, 
                                    orbax_checkpointer=orbax_checkpointer, keep_every_n_steps = 50)

    def load_checkpoint(self, tx: GradientTransformation = None, ckpt_dir: str = 'ckpts', step: Optional[int] = None, 
                        orbax_checkpointer: Checkpointer = PyTreeCheckpointer()) -> PyTree:

        r"""
        A convenience function to load the network parameters from disk.

        Parameters
        ----------
        tx : Optional[optax.GradientTransformation] = None
            The optimizer used to train the network.
        ckpt_dir : Optional[str]
            The directory where the checkpoint is saved.
            Defaults to 'ckpts'.
        step: Optional[int] = None
        orbax_checkpointer: Optional[Checkpointer] = PyTreeCheckpointer()

        Returns
        ----------
        TrainState

        Note
        ----------
        If the checkpoints are saved with the function `save_checkpoints`,
        this usually creates a folder called `checkpoint_[step]` where the checkpoint is saved
        and which should be added to ckpt_dir for this funtion to work
        """

        state_dict = orbax_checkpointer.restore(ckpt_dir)
        state = TrainState(params = freeze(state_dict['params']), tx = tx, step = step, 
                                opt_state = tx.init(freeze(state_dict['params'])), apply_fn=self.apply)

        return state
    


######################## DM21 ########################

def dm21_molecule_features(
        molecule: Molecule, 
        clip_cte: Optional[float] = 1e-27,
        *_, **__
    ):
    r"""
    Computes the electronic density and derivatives

    Parameters
    ----------
    molecule:
        class Molecule
    clip_cte: Optional[float]
        default 1e-27 (chosen carefully, take care if decrease)
    
    Returns
    -------
        Array: shape (n_grid, 7) where 7 is the number of features
    """

    rho = molecule.density()
    # We need to clip rho away from 0 to obtain good gradients.
    rho = jnp.maximum(abs(rho) , clip_cte)*jnp.sign(rho)
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()

    grad_rho_norm = jnp.sum(grad_rho**2, axis=-1)
    grad_rho_norm_sumspin = jnp.sum(grad_rho.sum(axis=0, keepdims=True) ** 2, axis=-1)

    features = jnp.concatenate((rho, grad_rho_norm_sumspin, grad_rho_norm, tau), axis=0)

    return features.T

def dm21_local_features(molecule: Molecule, functional_type: Optional[Union[str, Dict[str, int]]] = 'LDA', clip_cte: float = 1e-27, *_, **__):
    r"""
    Generates and concatenates different functional levels

    Parameters:
    ----------
    molecule:
        class Molecule

    functional_type:
        Either one of 'LDA', 'GGA', 'MGGA' or Dictionary
        {'u_range': range(), 'w_range': range()} that generates 
        a functional

        .. math::
            \sum_{i\in \text{u_range}} \sum_{j\in \text{w_range}} c_{ij} u^i w^j

        where

        .. math::
            x = \frac{|\grad \rho|^{1/2}}{\rho^{4/3}}
            u = \frac{\beta x}{1 + \beta x}

        and

        .. math::
            t = \frac{3(6\pi^2)^{2/3}}{5}\frac{\rho^{5/3}}{\tau}
            w = \frac{\beta t^{-1}}{1+ \beta t^{-1}}

    Returns:
    --------
        Array: shape (n_grid, n_features)
    """

    beta = 1/1024.

    if isinstance(functional_type, str):
        if functional_type == 'LDA' or functional_type == 'DM21':  
            u_range, w_range = range(0,1), range(0,1)
        elif functional_type == 'GGA':
            u_range, w_range = range(0,2), range(0,1)
        elif functional_type == 'MGGA': 
            u_range, w_range = range(0,2), range(0,2)
        else: raise ValueError(f'Functional type {functional_type} not recognized, must be one of LDA, GGA, MGGA.')

    # Molecule preprocessing data
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    # LDA preprocessing data
    log_rho = jnp.log2(jnp.clip(rho, a_min = clip_cte))

    # GGA preprocessing data
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min = clip_cte))/2
    log_x_sigma = log_grad_rho_norm - 4/3.*log_rho
    log_u_sigma = jnp.where(jnp.greater(log_rho,jnp.log2(clip_cte)), log_x_sigma - jnp.log2(1 + beta*(2**log_x_sigma)) + jnp.log2(beta), 0)

    # MGGA preprocessing data
    log_tau = jnp.log2(jnp.clip(tau, a_min = clip_cte))
    log_1t_sigma = -(5/3.*log_rho - log_tau + 2/3.*jnp.log2(6*jnp.pi**2) + jnp.log2(3/5.))
    log_w_sigma = jnp.where(jnp.greater(log_rho, jnp.log2(clip_cte)), log_1t_sigma - jnp.log2(1 + beta*(2**log_1t_sigma)) + jnp.log2(beta), 0)

    # Compute the local features
    localfeatures = jnp.empty((0, log_rho.shape[-1]))
    for i, j in itertools.product(u_range, w_range):
        mgga_term = jnp.expand_dims((2**(4/3.*log_rho + i * log_u_sigma + j * log_w_sigma)).sum(axis=0), axis = 0) \
                    * jnp.where(jnp.logical_and(i==0, j==0), -2 * jnp.pi * (3 / (4 * jnp.pi)) ** (4 / 3), 1) # to match DM21
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=0)

    return localfeatures.T

def dm21_features(molecule: Molecule, functional_type: Optional[Union[str, Dict]] = 'LDA', clip_cte: float = 1e-27, *args, **kwargs):

    r"""
    Generates all features except the HF energy features.
    
    Parameters
    ----------
    molecule: Molecule
    
    functional_type: Optional[Union[str, Dict]]
        Either one of 'LDA', 'GGA', 'MGGA' or Dictionary
        {'u_range': range(), 'w_range': range()} that generates 
        a functional. See `default_functionals` function.

    clip_cte: float
        A numerical threshold to avoid numerical precision issues
        Default: 1e-27

    Returns
    ----------
    Tuple[Array, Array]
        The features and local features, similar to those used by DM21
    """
    
    features = dm21_molecule_features(molecule, *args, **kwargs)
    localfeatures = dm21_local_features(molecule, functional_type, clip_cte)

    # We return them with the first index being the position r and the second the feature.
    return features, localfeatures

def dm21_combine(features, ehf):

    r"""
    Default way to combine Hartree-Fock and the rest of the input default features.

    Parameters
    ----------
    ehf: Array
        The Hartree-Fock features
        shape: (n_omega, n_spin, n_grid_points)
    features: Array
        The rest of input features that constitute the input to the neural network in a 
        functional of the form similar to DM21.
        shape: (n_grid, n_input_features)
    local features: Array
        The rest of local features that get dot-multiplied by the output of a neural network
        in a functional of the form similar to DM21.
        shape: (n_grid, n_local_features)

    Returns
    ----------
    Tuple[Array, Array]
        features, shape: (n_grid, n_features + n_spin * n_omegas)
        local_features, shape: (n_grid, n_local_features + n_omegas)
    """
    features, local_features = features

    # Remember that DM concatenates the hf density in the x features by spin...
    features = jnp.concatenate([features, ehf[:,0].T, ehf[:,1].T], axis=1)

    # ... and in the y features by omega.
    local_features = jnp.concatenate([local_features] + [ehf[i].sum(axis=0, keepdims=True).T for i in range(len(ehf))], axis=1)
    return features,local_features

def dm21_hfgrads(functional: nn.Module, params: Dict, molecule: Molecule, features: List[Array], ehf: Array, omegas = jnp.array([0., 0.4])):
    vxc_hf = molecule.HF_density_grad_2_Fock(functional, params, omegas, ehf, features)
    return vxc_hf.sum(axis=0) # Sum over omega
    
@dataclass
class DM21(NeuralFunctional):

    r"""
    Creates the architecture of the DM21 functional.
    Contains a function to generate the weights, called `generate_DM21_weights`
    """

    function: Callable = lambda self, *inputs: self.default_nn(*inputs)
    features: Callable = dm21_features
    nograd_features: Callable = lambda molecule, *_, **__: molecule.HF_energy_density([0., 0.4])
    featuregrads: Callable = lambda self, params, molecule, features, nograd_features, *_, **__: dm21_hfgrads(self, params, molecule, features, nograd_features)
    combine: Callable = dm21_combine
    activation: Callable = elu
    squash_offset: float = 1e-4
    layer_widths: Array = jnp.array([256,256,256,256,256,256])
    local_features: int = 3
    sigmoid_scale_factor: float = 2.

    def default_nn(instance, rhoinputs, localfeatures, *_, **__):
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

        x = instance.head(x, instance.local_features, instance.sigmoid_scale_factor)

        return jnp.einsum('ri,ri->r', x, localfeatures)

    def generate_DM21_weights(self, folder: str = 'DM21_model', num_layers_with_dm_parameters: int = 7, n_input_features: int = 11, rng = PRNGKey(0)):

        r"""
        A convenience function to generate the DM21 weights and biases.

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

        example_features = normal(rng, shape=(2, n_input_features))
        example_local_features = normal(rng, shape=(2, self.local_features))
        params = self.init(rng, example_features, example_local_features)

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
def default_loss(params: PyTree, functional: Functional, molecule: Molecule, trueenergy: float, *functionalinputs):
    
    r"""
    Computes the default loss function, here MSE, between predicted and true energy

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    molecule: Molecule
    trueenergy: float
    *functionalinputs: Sequence
        inputs to be passed to evaluate functional.energy(params, molecule, *functionalinputs)

    Returns
    ----------
    Tuple[float, float]
    The loss and predicted energy.

    Note
    ----------
    Since it has the decorator @partial(value_and_grad, has_aux = True)
    it will compute the gradients with respect to params.
    """

    predictedenergy = functional.energy(params, molecule, *functionalinputs)
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

################ Spin polarization correction functions ################


def exchange_polarization_correction(e_PF, rho):
    r"""Spin polarization correction to an exchange functional using eq 2.71 from 
    Carsten A. Ullrich, "Time-Dependent Density-Functional Theory".

    Parameters
    ----------
    e_PF: 
        Array, shape (2, n_grid)
        The paramagnetic/ferromagnetic energy contributions on the grid, to be combined.

    rho:
        Array, shape (2, n_grid)
        The electronic density of each spin polarization at each grid point.

    Returns
    ----------
    e_tilde
        Array, shape (n_grid)
        The ready to be integrated electronic energy density.
    """
    zeta = (rho[0] - rho[1])/ rho.sum(axis = 0)
    def fzeta(z): return ((1-z)**(4/3) + (1+z)**(4/3) - 2) / (2*(2**(1/3) - 1))
    # Eq 2.71 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    return e_PF[0] + (e_PF[1]-e_PF[0])*fzeta(zeta)


def correlation_polarization_correction(e_PF: Array, rho: Array, clip_cte: float = 1e-27):
    r"""Spin polarization correction to a correlation functional using eq 2.75 from 
    Carsten A. Ullrich, "Time-Dependent Density-Functional Theory".

    Parameters
    ----------
    e_PF: 
        Array, shape (2, n_grid)
        The paramagnetic/ferromagnetic energy contributions on the grid, to be combined.

    rho:
        Array, shape (2, n_grid)
        The electronic density of each spin polarization at each grid point.

    clip_cte:
        float, defaults to 1e-27
        Small constant to avoid numerical issues when dividing by rho.

    Returns
    ----------
    e_tilde
        Array, shape (n_grid)
        The ready to be integrated electronic energy density.
    """

    e_tilde_PF = jnp.einsum('sr,r->sr', e_PF, rho.sum(axis = 0))

    log_rho = jnp.log2(jnp.clip(rho.sum(axis = 0), a_min = clip_cte))
    assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()
    log_rs =  jnp.log2((3/(4*jnp.pi))**(1/3)) - log_rho/3.

    zeta = jnp.where(rho.sum(axis = 0) > clip_cte, (rho[0] - rho[1]) / (rho.sum(axis = 0)), 0.)
    def fzeta(z): 
        zm = 2**(4*jnp.log2(1-z)/3)
        zp = 2**(4*jnp.log2(1+z)/3)
        return (zm + zp - 2) / (2*(2**(1/3) - 1))

    A_ = 0.016887
    alpha1 = 0.11125
    beta1 = 10.357
    beta2 = 3.6231
    beta3 = 0.88026
    beta4 = 0.49671

    ars = 2**(jnp.log2(alpha1) + log_rs)
    brs_1_2 = 2**(jnp.log2(beta1) +  log_rs/2)
    brs = 2**(jnp.log2(beta2)+log_rs)
    brs_3_2 = 2**(jnp.log2(beta3)+3*log_rs/2)
    brs2 = 2**(jnp.log2(beta4)+2*log_rs)

    alphac = 2*A_*(1+ars)*jnp.log(1+(1/(2*A_))/(brs_1_2 + brs + brs_3_2 + brs2))
    assert not jnp.isnan(alphac).any() and not jnp.isinf(alphac).any()

    fz = jnp.round(fzeta(zeta), int(jnp.log10(clip_cte)))
    z4 = jnp.round(2**(4*jnp.log2(jnp.clip(zeta, a_min = clip_cte))), int(jnp.log10(clip_cte)))

    e_tilde = e_tilde_PF[0] + alphac * (fz/(grad(grad(fzeta))(0.)))* (1-z4) + (e_tilde_PF[1]-e_tilde_PF[0]) * fz*z4
    assert not jnp.isnan(e_tilde).any() and not jnp.isinf(e_tilde).any()

    return e_tilde




def local_features(molecule: Molecule, functional_type: Optional[Union[str, Dict[str, int]]] = 'LDA', clip_cte: float = 1e-27, *_, **__):
    r"""
    Generates and concatenates different functional levels

    Parameters:
    ----------
    molecule:
        class Molecule

    functional_type:
        Either one of 'LDA', 'GGA', 'MGGA' or Dictionary
        {'u_range': range(), 'w_range': range()} that generates 
        a functional

        .. math::
            \sum_{i\in \text{u_range}} \sum_{j\in \text{w_range}} c_{ij} u^i w^j

        where

        .. math::
            x = \frac{|\grad \rho|^{1/2}}{\rho^{4/3}}
            u = \frac{\beta x}{1 + \beta x}

        and

        .. math::
            t = \frac{3(6\pi^2)^{2/3}}{5}\frac{\rho^{5/3}}{\tau}
            w = \frac{\beta t^{-1}}{1+ \beta t^{-1}}

    Returns:
    --------
        Array: shape (n_grid, n_features)
    """

    beta = 1/1024.

    if isinstance(functional_type, str):
        if functional_type == 'LDA' or functional_type == 'DM21':  
            u_range, w_range = range(0,1), range(0,1)
        elif functional_type == 'GGA':
            u_range, w_range = range(0,2), range(0,1)
        elif functional_type == 'MGGA': 
            u_range, w_range = range(0,2), range(0,2)
        else: raise ValueError(f'Functional type {functional_type} not recognized, must be one of LDA, GGA, MGGA.')

    # Molecule preprocessing data
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    # LDA preprocessing data
    log_rho = jnp.log2(jnp.clip(rho, a_min = clip_cte))

    # GGA preprocessing data
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min = clip_cte))/2
    log_x_sigma = log_grad_rho_norm - 4/3.*log_rho
    log_u_sigma = jnp.where(jnp.greater(log_rho,jnp.log2(clip_cte)), 
                            log_x_sigma - jnp.log2(1 + beta*(2**log_x_sigma)) + jnp.log2(beta), 0)

    # MGGA preprocessing data
    log_tau = jnp.log2(jnp.clip(tau, a_min = clip_cte))
    log_1t_sigma = log_tau -5/3.*log_rho
    log_w_sigma = jnp.where(jnp.greater(log_rho, jnp.log2(clip_cte)), 
                            log_1t_sigma - jnp.log2(1 + beta*(2**log_1t_sigma)) + jnp.log2(beta), 0)

    # Compute the local features
    localfeatures = jnp.empty((0, log_rho.shape[-1]))
    for i, j in itertools.product(u_range, w_range):
        mgga_term = 2**(4/3.*log_rho + i * log_u_sigma + j * log_w_sigma)

        # First we concatenate the exchange terms
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=0)

    ######### Correlation features ###############

    grad_rho_norm_sq_ss = jnp.sum((grad_rho.sum(axis = 0))**2, axis=-1)
    log_grad_rho_norm_ss = jnp.log2(jnp.clip(grad_rho_norm_sq_ss, a_min = clip_cte))/2
    log_rho_ss = jnp.log2(jnp.clip(rho.sum(axis = 0), a_min = clip_cte))
    log_x_ss = log_grad_rho_norm_ss - 4/3.*log_rho_ss

    log_u_ss = jnp.where(jnp.greater(log_rho_ss,jnp.log2(clip_cte)), 
                            log_x_ss - jnp.log2(1 + beta*(2**log_x_ss)) + jnp.log2(beta), 0)
    
    log_u_ab = jnp.where(jnp.greater(log_rho_ss,jnp.log2(clip_cte)), 
                            log_x_ss - 1 - jnp.log2(1 + beta*(2**(log_x_ss-1))) + jnp.log2(beta), 0)

    log_u_c = jnp.concat((log_u_ss, log_u_ab), axis = 0)


    log_tau_ss = jnp.log2(jnp.clip(tau.sum(axis = 0), a_min = clip_cte))
    log_1t_ss = log_tau_ss - 5/3.*log_rho_ss
    log_w_ss = jnp.where(jnp.greater(log_rho, jnp.log2(clip_cte)), 
                            log_1t_sigma - jnp.log2(1 + beta*(2**log_1t_sigma)) + jnp.log2(beta), 0)
    
    log_w_ab = jnp.where(jnp.greater(log_rho, jnp.log2(clip_cte)), 
                            log_1t_sigma - 1 - jnp.log2(1 + beta*(2**(log_1t_sigma-1))) + jnp.log2(beta), 0)
    
    log_w_c = jnp.concat((log_w_ss, log_w_ab), axis = 0)


    A_ = jnp.array([[0.031091],
                    [0.015545]])
    alpha1 = jnp.array([[0.21370],
                    [0.20548]])
    beta1 = jnp.array([[7.5957],
                    [14.1189]])
    beta2 = jnp.array([[3.5876],
                    [6.1977]])
    beta3 = jnp.array([[1.6382],
                    [3.3662]])
    beta4 = jnp.array([[0.49294],
                    [0.62517]])
    
    log_rho = jnp.log2(jnp.clip(rho.sum(axis = 0), a_min = clip_cte))
    log_rs = jnp.log2((3/(4*jnp.pi))**(1/3)) - log_rho/3.
    rs = 2**log_rs

    e_PW92 = -2*A_*(1+alpha1*rs)*jnp.log(1+(1/(2*A_))/(beta1*jnp.sqrt(rs) + beta2*rs + beta3*rs**(3/2) + beta4*rs**2))

    # Compute the local features
    localfeatures = jnp.empty((0, log_rho.shape[-1]))
    for i, j in itertools.product(u_range, w_range):
        mgga_term = 2**(jnp.log2(e_PW92) + i * log_u_c + j * log_w_c)

        # First we concatenate the exchange terms
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=0)

    return localfeatures.T