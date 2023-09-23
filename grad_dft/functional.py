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

from dataclasses import dataclass
import itertools
import os
from typing import Callable, Optional, List, Dict, Union
from functools import partial
import math

from jax import grad
from jax import numpy as jnp
from jax.lax import Precision, stop_gradient
from jax.nn import sigmoid, gelu, elu
from jax.nn.initializers import zeros, he_normal
from jax.random import normal, PRNGKey

from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar, Float, jaxtyped

from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints
from flax.training.train_state import TrainState
from optax import GradientTransformation
from orbax.checkpoint import Checkpointer, PyTreeCheckpointer
from typeguard import typechecked

from grad_dft import (
    abs_clip,
    Grid,
    Molecule
)
from grad_dft.utils.types import DType, default_dtype


@dataclass
class Functional(nn.Module):
    r"""A base class of local functionals.
    .. math::
        F[n(r)] = \int f(n(r)) d^3 r

    or
    .. math::
        F[n(r)] = f( \int n(r) d^3 r)

    Parameters
    ----------
    function: Callable
        Implements the function f above.


    coefficients : Callable
        A function that computes and returns the weights c_\theta. If it requires
        some inputs, these may be computed via function compute_coefficient_inputs below.

    energy_densities : Callable
        A function that computes and returns the energy densities e_\theta that can be autodifferentiated
        with respect to the reduced density matrix.

        densities(molecule: Molecule, *args, **kwargs) -> Array

    nograd_densities : Callable, optional
        A function that calculates the molecule energy densities e_\theta where gradient with respect to the
        reduced density matrix is computed via in densitygrads.

        nograd_densities(molecule: Molecule, *args, **kwargs) -> Array

    featuregrads: Callable, optional
        A function to compute contributions to the Fock matrix for energy densities
        where autodifferentiation is not used.

        If given has signature

        featuregrads(functional: nn.Module, params: PyTree, molecule: Molecule,
            nograd_densities: Array, coefficient_inputs: Array, grad_densities, *args) - > Fock matrix: Array of shape (2, nao, nao)

    combine_densities : Callable, optional
        A function that joins the densities computed with and without autodifferentiation.
        combine_densities(grad_densities: Array, nograd_densities: Array, *args, **kwargs) -> Array

    coefficient_inputs : Callable, optional
        A function that computes the inputs to the coefficients function, that can be autodifferentiated
        with respect to the reduced density matrix.

        coefficient_inputs(molecule: Molecule, *args, **kwargs) -> Array

    nograd_coefficient_inputs : Callable, optional
        A function that computes the inputs to the coefficients function, where gradient with respect to the
        reduced density matrix is computed via in coefficient_input_grads.

        nograd_coefficient_inputs(molecule: Molecule, *args, **kwargs) -> Array

    coefficient_inputs_grads: Callable, optional
        A function to compute contributions to the Fock matrix for coefficient inputs
        where autodifferentiation is not used.

        If given has signature

        coefficient_inputs_grads(functional: nn.Module, params: PyTree, molecule: Molecule,
            nograd_coefficient_inputs: Array, grad_coefficient_inputs: Array, densities, *args) - > Fock matrix: Array of shape (2, nao, nao)

    combine_coefficient_inputs : Callable, optional
        A function that joins the coefficient inputs computed with and without autodifferentiation.
        combine_densities(grad_coefficient_inputs: Array, nograd_coefficient_inputs: Array, *args, **kwargs) -> Array

    is_xc: bool
        Whether the functional models only the exchange-correlation energy

    exchange_mask: Array[bool]
        A mask to indicate which elements of the Fock matrix are exchange
        contributions. If defined, must be of shape (n_features).
        The correlation mask would be 1 - exchange_mask.

    """

    coefficients: staticmethod
    energy_densities: staticmethod
    coefficient_inputs: staticmethod = None

    nograd_densities: staticmethod = None
    densitygrads: staticmethod = None
    combine_densities: staticmethod = None

    nograd_coefficient_inputs: staticmethod = None
    coefficient_input_grads: staticmethod = None
    combine_inputs: staticmethod = None

    is_xc: bool = True
    exchange_mask: Array = None

    @nn.compact
    def __call__(self, coefficient_inputs) -> Scalar:
        r"""Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Parameters
        ---------
        inputs: inputs to the function f

        Returns
        -------
        Union[Array, Scalar]
        """

        return self.coefficients(self, coefficient_inputs)

    def compute_densities(self, molecule: Molecule,  clip_cte: float = 1e-30, *args, **kwargs):
        r"""
        Computes the densities for the functional, both with and without autodifferentiation.

        Parameters
        ----------
        molecule: Molecule
            The molecule to compute the densities for

        Returns
        -------
        densities: Array
        """

        if self.nograd_densities and self.energy_densities:
            densities = self.energy_densities(molecule, *args, **kwargs)
            nograd_densities = stop_gradient(self.nograd_densities(molecule, *args, **kwargs))
            densities = self.combine_densities(densities, nograd_densities)

        elif self.energy_densities:
            densities = self.energy_densities(molecule, *args, **kwargs)

        elif self.nograd_densities:
            densities = stop_gradient(self.nograd_densities(molecule, *args, **kwargs))
        densities = abs_clip(densities, clip_cte) #todo: investigate if we can lower this
        return densities

    def compute_coefficient_inputs(self, molecule: Molecule, *args, **kwargs):
        r"""
        Computes the inputs to the coefficients method in the functional

        Parameters
        ----------
        molecule: Molecule
            The molecule to compute the inputs for the coefficients

        Returns
        -------
        coefficient_inputs: Array
        """

        if self.nograd_coefficient_inputs and self.coefficient_inputs:
            cinputs = self.coefficient_inputs(molecule, *args, **kwargs)
            nograd_cinputs = stop_gradient(
                self.nograd_coefficient_inputs(molecule, *args, **kwargs)
            )
            cinputs = self.combine_inputs(cinputs, nograd_cinputs)

        elif self.coefficient_inputs:
            cinputs = self.coefficient_inputs(molecule, *args, **kwargs)

        elif self.nograd_coefficient_inputs:
            cinputs = stop_gradient(self.nograd_coefficient_inputs(molecule, *args, **kwargs))

        else:
            cinputs = None

        return cinputs

    def xc_energy(
        self, 
        params: PyTree, 
        grid: Grid, 
        coefficient_inputs: Float[Array, "grid cinputs"],
        densities: Float[Array, "grid densities"],
        clip_cte: float = 1e-30,
        **kwargs
    ) -> Scalar:
        r"""
        Total energy of local functional

        Parameters
        ---------
        params: PyTree
            params of the neural network if there is one in self.f
        grid: Grid
            grid to integrate over.
        coefficient_inputs: Array
            inputs to the coefficients method in Functional.
        densities: Array
            energy densities used to compute the energy.

        **kwargs:
            key word arguments for the coefficients function

        Returns
        -------
        Scalar
        """

        coefficients = self.apply(params, coefficient_inputs, **kwargs)
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, densities)
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, grid.weights)

    def energy(self, params: PyTree, molecule: Molecule, *args, **kwargs) -> Scalar:
        r"""
        Total energy of local functional

        Parameters
        ---------
        params: PyTree
            params of the neural network if there is one in self.f
        molecule: Molecule

        *args: other arguments to compute_densities or compute_coefficient_inputs
        **kwargs: other key word arguments to densities and self.xc_energy

        Returns
        -------
        Scalar

        Note
        -------
        Integrates the energy over the grid.
        If the function is_xc, it will add the rest of the energy components
        computed with function molecule.nonXC()
        """

        densities = self.compute_densities(molecule, *args, **kwargs)
        # sys.exit()
        cinputs = self.compute_coefficient_inputs(molecule, *args)

        energy = self.xc_energy(params, molecule.grid, cinputs, densities, **kwargs)

        if self.is_xc:
            energy += molecule.nonXC()

        return energy

    def _integrate(
        self,
        energy_density: Float[Array, "grid"],
        gridweights: Float[Array, "grid"],
        precision: Optional[Precision] = Precision.HIGHEST,
        clip_cte: float = 1e-30,
    ) -> Scalar:
        r"""
        Helper function that performs grid quadrature (integration)
                                in a differentiable way (using jax.numpy).

        Parameters
        ----------
        energy_density : Float[Array, "grid"]
            energy_density to integrate.
        gridweights: Float[Array, "grid"]
            gridweights.
        precision : Precision, optional
            The precision to use for the computation, by default Precision.HIGHEST

        Returns
        -------
        Scalar
        """

        #todo: study if we can lower this clipping constants
        return jnp.einsum("r,r->", abs_clip(gridweights, clip_cte), abs_clip(energy_density, clip_cte), precision=precision)


@dataclass
class NeuralFunctional(Functional):
    r"""
    Neural functional, subclass of Functional.

    Parameters
    ----------
    The methods of Functional

    kernel_init: Callable, optional
        kernel initializer for the neural network, by default he_normal()
    bias_init: Callable, optional
        bias initializer for the neural network, by default zeros
    activation: Callable, optional
        activation function for the neural network, by default gelu
    param_dtype: DType, optional

    Notes
    ----------
    coefficients is expected to implement a neural network. For example
        ```
        def externally_defined_coefficients(instance, x):
            x = instance.dense(x)
            x = 0.5*jnp.tanh(x)
            return x
        ```

    NeuralFunctional contains some additional methods, such as implementation of dense
    layers, and saving/loading checkpoints
    """

    coefficients: staticmethod
    energy_densities: staticmethod
    coefficient_inputs: staticmethod = None

    nograd_densities: staticmethod = None
    densitygrads: staticmethod = None
    combine_densities: staticmethod = None

    nograd_coefficient_inputs: staticmethod = None
    coefficient_input_grads: staticmethod = None
    combine_inputs: staticmethod = None

    is_xc: bool = True
    exchange_mask: Array = None

    kernel_init: Callable = he_normal()
    bias_init: Callable = zeros
    activation: Callable = gelu
    param_dtype: DType = default_dtype()

    def setup(self):
        r"""Sets up the neural network layers. """
        self.dense = partial(
            nn.Dense,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.layer_norm = partial(nn.LayerNorm, param_dtype=self.param_dtype)

    def head(self, x: Array, local_features, sigmoid_scale_factor):
        r"""
        Final layer of the neural network.
        """
        # Final layer: dense -> sigmoid -> scale (x2)
        x = self.dense(features=local_features)(x)  # eg local_features = 3 in DM21
        self.sow("intermediates", "head_dense", x)
        x = sigmoid(x / sigmoid_scale_factor)
        self.sow("intermediates", "sigmoid", x)
        out = sigmoid_scale_factor * x  # sigmoid_scale_factor = 2.0 in DM21
        self.sow("intermediates", "sigmoid_product", out)

        return jnp.squeeze(out)  # Eliminating unnecessary dimensions

    def save_checkpoints(
        self,
        params: PyTree,
        tx: GradientTransformation,
        step: Optional[int],
        orbax_checkpointer: Checkpointer = PyTreeCheckpointer(),
        ckpt_dir: str = "ckpts",
    ):
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

        state = train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=state,
            step=step,
            overwrite=True,
            orbax_checkpointer=orbax_checkpointer,
            keep_every_n_steps=50,
        )

    def load_checkpoint(
        self,
        tx: GradientTransformation = None,
        ckpt_dir: str = "ckpts",
        step: Optional[int] = None,
        orbax_checkpointer: Checkpointer = PyTreeCheckpointer(),
    ) -> TrainState:
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
        state = TrainState(
            params=freeze(state_dict["params"]),
            tx=tx,
            step=step,
            opt_state=tx.init(freeze(state_dict["params"])),
            apply_fn=self.apply,
        )

        return state


######################## DM21 ########################


def dm21_coefficient_inputs(molecule: Molecule, clip_cte: Optional[float] = 1e-30, *_, **__):
    r"""
    Computes the electronic density and derivatives

    Parameters
    ----------
    molecule:
        class Molecule
    clip_cte: Optional[float]
        Needed to make sure it
        default 1e-30 (chosen carefully, take care if decrease)

    Returns
    -------
        Array: shape (n_grid, 7) where 7 is the number of features
    """

    rho = molecule.density()
    # We need to clip rho away from 0 to obtain good gradients.
    rho = jnp.maximum(abs(rho), clip_cte) * jnp.sign(rho)
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()

    grad_rho_norm = jnp.sum(grad_rho**2, axis=-1)
    grad_rho_norm_sumspin = jnp.sum(grad_rho.sum(axis=1, keepdims=True) ** 2, axis=-1)

    features = jnp.concatenate((rho, grad_rho_norm_sumspin, grad_rho_norm, tau), axis=1)

    return features


def dm21_densities(
    molecule: Molecule,
    functional_type: Optional[Union[str, Dict[str, int]]] = "LDA",
    clip_cte: float = 1e-30,
    *_,
    **__,
):
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

    beta = 1 / 1024.0

    if isinstance(functional_type, str):
        if functional_type == "LDA" or functional_type == "DM21":
            u_range, w_range = range(0, 1), range(0, 1)
        elif functional_type == "GGA":
            u_range, w_range = range(0, 2), range(0, 1)
        elif functional_type == "MGGA":
            u_range, w_range = range(0, 2), range(0, 2)
        else:
            raise ValueError(
                f"Functional type {functional_type} not recognized, must be one of LDA, GGA, MGGA."
            )

    # Molecule preprocessing data
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    # LDA preprocessing data
    log_rho = jnp.log2(jnp.clip(rho, a_min=clip_cte))

    # GGA preprocessing data
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min=clip_cte)) / 2
    log_x_sigma = log_grad_rho_norm - 4 / 3.0 * log_rho
    log_u_sigma = jnp.where(
        jnp.greater(log_rho, jnp.log2(clip_cte)),
        log_x_sigma - jnp.log2(1 + beta * (2**log_x_sigma)) + jnp.log2(beta),
        0,
    )

    # MGGA preprocessing data
    log_tau = jnp.log2(jnp.clip(tau, a_min=clip_cte))
    log_1t_sigma = -(
        5 / 3.0 * log_rho - log_tau + 2 / 3.0 * jnp.log2(6 * jnp.pi**2) + jnp.log2(3 / 5.0)
    )
    log_w_sigma = jnp.where(
        jnp.greater(log_rho, jnp.log2(clip_cte)),
        log_1t_sigma - jnp.log2(1 + beta * (2**log_1t_sigma)) + jnp.log2(beta),
        0,
    )

    # Compute the local features
    localfeatures = jnp.empty((log_rho.shape[0], 0))
    for i, j in itertools.product(u_range, w_range):
        mgga_term = (2 ** (4 / 3.0 * log_rho + i * log_u_sigma + j * log_w_sigma)).sum(
            axis=1, keepdims=True
        ) * jnp.where(
            jnp.logical_and(i == 0, j == 0), -2 * jnp.pi * (3 / (4 * jnp.pi)) ** (4 / 3), 1
        )  # to match DM21
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=1)

    return localfeatures

def dm21_combine_cinputs(
    cinputs: Float[Array, "grid cinputs_whf"] , 
    ehf: Float[Array, "omega spin grid"]
) -> Float[Array, "grid cinputs"]:
    r"""
    Default way to combine Hartree-Fock and the rest of the input features to the neural network.

    Parameters
    ----------
    densities: Float[Array, "grid cinputs_whf"]
        The rest of input features that constitute the input to the neural network in a
        functional of the form similar to DM21.
    ehf: Float[Array, "omega spin grid"]
        The Hartree-Fock features

    Returns
    ----------
    Float[Array, "grid cinputs_whf+omega*spin"]
    """

    # Remember that DM concatenates the hf density in the x features by spin...
    return jnp.concatenate([cinputs, ehf[:, 0].T, ehf[:, 1].T], axis=1)


def dm21_combine_densities(
    densities: Float[Array, "grid densities_whf"], 
    ehf: Float[Array, "omega spin grid"]
) -> Float[Array, "grid densities"]:
    r"""
    Default way to combine Hartree-Fock and the rest of the input default features.

    Parameters
    ----------
    densities : Float[Array, "grid densities_whf"]
        The rest of local energy densities that get dot-multiplied by the output of a neural network
        in a functional of the form similar to DM21.
    ehf : Float[Array, "omega spin grid"]
        The Hartree-Fock densities

    Returns
    ----------
    Float[Array, "grid densities_whf+omega*spin"]
    """

    # ... and in the y features by omega.
    return jnp.concatenate(
        [densities] + [ehf[i].sum(axis=0, keepdims=True).T for i in range(len(ehf))], axis=1
    )

@jaxtyped
@typechecked
def dm21_hfgrads_densities(
    functional: nn.Module,
    params: PyTree,
    molecule: Molecule,
    ehf: Float[Array, "omega spin grid"],
    coefficient_inputs: Float[Array, "grid cinputs"],
    densities_wout_hf: Float[Array, "grid densities_whf"],
    omegas: Float[Array, "omega"] = jnp.array([0.0, 0.4]),
) -> Float[Array, "spin orbitals orbitals"]:
    r"""
    Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock energy density.

    Parameters
    ----------
    functional: nn.Module
        The functional to calculate the Hartree-Fock matrix contribution for.
    params: PyTree
        The parameters of the functional.
    molecule: Molecule
        The molecule to calculate the Hartree-Fock matrix contribution for.
    ehf: Float[Array, "omega spin grid"]
        The Hartree-Fock energy density.
    coefficient_inputs: Float[Array, "grid cinputs"]
        The inputs to the neural network.
    densities_wout_hf: Float[Array, "grid densities_whf"]
        The rest of the local energy densities that get dot-multiplied by the output of a neural network
        in a functional of the form similar to DM21.
    omegas: Float[Array, "omega"]
        The omegas to calculate the Hartree-Fock matrix contribution for.

    Returns
    ----------
    Float[Array, "spin orbitals orbitals"]
    """
    vxc_hf = molecule.HF_density_grad_2_Fock(
        functional, params, omegas, ehf, coefficient_inputs, densities_wout_hf
    )
    return vxc_hf.sum(axis=0)  # Sum over omega

@jaxtyped
@typechecked
def dm21_hfgrads_cinputs(
    functional: nn.Module,
    params: PyTree,
    molecule: Molecule,
    ehf: Float[Array, "omega spin grid"],
    cinputs_wout_hf: Float[Array, "grid cinputs_whf"],
    densities: Float[Array, "grid densities"],
    omegas: Float[Array, "omega"] = jnp.array([0.0, 0.4]),
) -> Float[Array, "spin orbitals orbitals"]:
    r"""
    Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock coefficient input.

    Parameters
    ----------
    functional: nn.Module
        The functional to calculate the Hartree-Fock matrix contribution for.
    params: PyTree
        The parameters of the functional.
    molecule: Molecule
        The molecule to calculate the Hartree-Fock matrix contribution for.
    ehf: Float[Array, "omega spin grid"]
        The Hartree-Fock energy density.
    cinputs_wout_hf: Float[Array, "grid cinputs_whf"]
        The rest of the inputs to the neural network.
    densities: Float[Array, "grid densities"]
        The local energy densities that get dot-multiplied by the output of a neural network.
    omegas: Float[Array, "omega"]
        The omegas to calculate the Hartree-Fock matrix contribution for.

    Returns
    ----------
    Float[Array, "spin orbitals orbitals"]
    """
    vxc_hf = molecule.HF_coefficient_input_grad_2_Fock(
        functional, params, omegas, ehf, cinputs_wout_hf, densities
    )
    return vxc_hf.sum(axis=0)  # Sum over omega


@dataclass
class DM21(NeuralFunctional):
    r"""
    Creates the architecture of the DM21 functional.
    Contains a function to generate the weights, called `generate_DM21_weights`.
    See the initialization parameters of NeuralFunctional.
    """

    coefficients: Callable = lambda self, inputs: self.default_nn(inputs)
    energy_densities: Callable = dm21_densities
    nograd_densities: staticmethod = lambda molecule, *_, **__: molecule.HF_energy_density(
        jnp.array([0.0, 0.4])
    )
    densitygrads: staticmethod = lambda self, params, molecule, nograd_densities, cinputs, grad_densities, *_, **__: dm21_hfgrads_densities(
        self, params, molecule, nograd_densities, cinputs, grad_densities, jnp.array([0.0, 0.4])
    )
    combine_densities: staticmethod = dm21_combine_densities

    coefficient_inputs: staticmethod = dm21_coefficient_inputs
    nograd_coefficient_inputs: staticmethod = lambda molecule, *_, **__: molecule.HF_energy_density(
        jnp.array([0.0, 0.4])
    )
    coefficient_input_grads: staticmethod = lambda self, params, molecule, nograd_cinputs, grad_cinputs, densities, *_, **__: dm21_hfgrads_cinputs(
        self, params, molecule, nograd_cinputs, grad_cinputs, densities, jnp.array([0.0, 0.4])
    )
    combine_inputs: staticmethod = dm21_combine_cinputs

    is_xc: bool = True
    exchange_mask: Array = None

    activation: Callable = elu
    squash_offset: float = 1e-4
    layer_widths = [256, 256, 256, 256, 256, 256]
    local_features: int = 3
    sigmoid_scale_factor: float = 2.0

    def default_nn(instance, rhoinputs, *_, **__):
        x = canonicalize_inputs(rhoinputs)  # Making sure dimensions are correct

        # Initial layer: log -> dense -> tanh
        x = jnp.log(jnp.abs(x) + instance.squash_offset)  # squash_offset = 1e-4
        instance.sow("intermediates", "log", x)
        x = instance.dense(features=instance.layer_widths[0])(x)  # features = 256
        instance.sow("intermediates", "initial_dense", x)
        x = jnp.tanh(x)
        instance.sow("intermediates", "tanh", x)

        # 6 Residual blocks with 256-features dense layer and layer norm
        for features, i in zip(
            instance.layer_widths, range(len(instance.layer_widths))
        ):  # layer_widths = [256]*6
            res = x
            x = instance.dense(features=features)(x)
            instance.sow("intermediates", "residual_dense_" + str(i), x)
            x = x + res  # nn.Dense + Residual connection
            instance.sow("intermediates", "residual_residual_" + str(i), x)
            x = instance.layer_norm()(x)  # + res # nn.LayerNorm
            instance.sow("intermediates", "residual_layernorm_" + str(i), x)
            x = instance.activation(x)  # activation = jax.nn.gelu
            instance.sow("intermediates", "residual_elu_" + str(i), x)

        return instance.head(x, instance.local_features, instance.sigmoid_scale_factor)

    def generate_DM21_weights(
        self,
        folder: str = "models/DM21_model",
        num_layers_with_dm_parameters: int = 7,
        n_input_features: int = 11,
        rng: PRNGKeyArray = PRNGKey(0),
    ):
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
        params: Frozen
            The DM21 weights and biases.
        """

        import tensorflow as tf

        tf.compat.v1.enable_eager_execution()

        path = os.path.dirname(os.path.dirname(__file__))
        folder = os.path.join(path, folder)

        variables = tf.saved_model.load(folder).variables

        def tf_tensor_to_jax(tf_tensor: tf.Tensor) -> Array:
            return jnp.asarray(tf_tensor.numpy())

        def vars_to_params(variables: List[tf.Variable]) -> PyTree:
            import re

            params = {}
            for var in variables:
                if "ResidualBlock_" in var.name:
                    number = int(re.findall("ResidualBlock_[0-9]", var.name)[0][-1]) + 1
                elif "ResidualBlock/" in var.name:
                    number = 1
                elif "Squash" in var.name:
                    number = 0
                elif "Output" in var.name:
                    number = 7
                else:
                    raise ValueError("Unknown variable name.")

                if "/linear/" in var.name:
                    if "Dense_" + str(number) not in params.keys():
                        params["Dense_" + str(number)] = {}
                    if "/w:" in var.name:
                        params["Dense_" + str(number)]["kernel"] = tf_tensor_to_jax(var.value())
                    elif "/b:" in var.name:
                        params["Dense_" + str(number)]["bias"] = tf_tensor_to_jax(var.value())
                elif "/layer_norm/" in var.name:
                    if "LayerNorm_" + str(number - 1) not in params.keys():
                        params["LayerNorm_" + str(number - 1)] = {}
                    if "gamma:" in var.name:
                        params["LayerNorm_" + str(number - 1)]["scale"] = tf_tensor_to_jax(
                            var.value()
                        )
                    elif "beta:" in var.name:
                        params["LayerNorm_" + str(number - 1)]["bias"] = tf_tensor_to_jax(
                            var.value()
                        )
            return params

        example_features = normal(rng, shape=(2, n_input_features))
        # example_local_features = normal(rng, shape=(2, self.local_features))
        params = self.init(rng, example_features)

        dm_params = vars_to_params(variables)

        new_params = {}
        for key in params["params"].keys():
            check_same_params = []
            for k in params["params"][key].keys():
                if key in dm_params.keys():
                    check_same_params.append(
                        params["params"][key][k].shape != dm_params[key][k].shape
                    )
                else:
                    check_same_params.append(True)
            if int(key.split("_")[1]) > num_layers_with_dm_parameters or any(check_same_params):
                new_params[key] = params["params"][key]
                if (
                    "Dense" in key
                    and new_params[key]["kernel"].shape[0] == new_params[key]["kernel"].shape[1]
                ):  # DM21 suggests initializing the kernel matrices close to the identity matrix
                    new_params[key] = unfreeze(new_params[key])
                    new_params[key]["kernel"] = new_params[key]["kernel"] + jnp.identity(
                        new_params[key]["kernel"].shape[0]
                    )
                    new_params[key] = freeze(new_params[key])
            else:
                new_params[key] = dm_params[key]

        params = unfreeze(params)
        params["params"] = new_params
        params = freeze(params)
        return params


######################### Helper functions #########################


def canonicalize_inputs(x):
    r"""
    Ensures that the input to the neural network has the correct dimensions.
    """
    x = jnp.asarray(x)

    if x.ndim == 1:
        return jnp.expand_dims(x, axis=1)
    elif x.ndim == 0:
        raise ValueError("`features` has to be at least 1D array!")
    else:
        return x

################ Spin polarization correction functions ################


def exchange_polarization_correction(
    e_PF: Float[Array, "spin grid"], 
    rho: Float[Array, "spin grid"]
) -> Float[Array, "grid"]:
    r"""Spin polarization correction to an exchange functional using eq 2.71 from
    Carsten A. Ullrich, "Time-Dependent Density-Functional Theory".

    Parameters
    ----------
    e_PF:
        Float[Array, "spin grid"]
        The paramagnetic/ferromagnetic energy contributions on the grid, to be combined.

    rho:
        Float[Array, "spin grid"]
        The electronic density of each spin polarization at each grid point.

    Returns
    ----------
    e_tilde
        Float[Array, "grid"]
        The ready to be integrated electronic energy density.
    """
    zeta = (rho[:, 0] - rho[:, 1]) / rho.sum(axis=1)

    def fzeta(z):
        return ((1 - z) ** (4 / 3) + (1 + z) ** (4 / 3) - 2) / (2 * (2 ** (1 / 3) - 1))

    # Eq 2.71 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    return e_PF[:, 0] + (e_PF[:, 1] - e_PF[:, 0]) * fzeta(zeta)


def correlation_polarization_correction(
    e_tilde_PF: Float[Array, "spin grid"], 
    rho: Float[Array, "spin grid"], 
    clip_cte: float = 1e-30
) -> Float[Array, "grid"]:
    r"""Spin polarization correction to a correlation functional using eq 2.75 from
    Carsten A. Ullrich, "Time-Dependent Density-Functional Theory".

    Parameters
    ----------
    e_tilde_PF: Float[Array, "spin grid"]
        The paramagnetic/ferromagnetic energy contributions on the grid, to be combined.

    rho: Float[Array, "spin grid"]
        The electronic density of each spin polarization at each grid point.

    clip_cte:
        float, defaults to 1e-30
        Small constant to avoid numerical issues when dividing by rho.

    Returns
    ----------
    e_tilde: Float[Array, "grid"]
        The ready to be integrated electronic energy density.
    """

    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1), a_min=clip_cte))
    # assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0

    zeta = jnp.where(rho.sum(axis=1) > clip_cte, (rho[:, 0] - rho[:, 1]) / (rho.sum(axis=1)), 0.0)

    def fzeta(z):
        zm = 2 ** (4 * jnp.log2(1 - z) / 3)
        zp = 2 ** (4 * jnp.log2(1 + z) / 3)
        return (zm + zp - 2) / (2 * (2 ** (1 / 3) - 1))

    A_ = 0.016887
    alpha1 = 0.11125
    beta1 = 10.357
    beta2 = 3.6231
    beta3 = 0.88026
    beta4 = 0.49671

    ars = 2 ** (jnp.log2(alpha1) + log_rs)
    brs_1_2 = 2 ** (jnp.log2(beta1) + log_rs / 2)
    brs = 2 ** (jnp.log2(beta2) + log_rs)
    brs_3_2 = 2 ** (jnp.log2(beta3) + 3 * log_rs / 2)
    brs2 = 2 ** (jnp.log2(beta4) + 2 * log_rs)

    alphac = 2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))
    # assert not jnp.isnan(alphac).any() and not jnp.isinf(alphac).any()

    fz = fzeta(zeta) #jnp.round(fzeta(zeta), int(math.log10(clip_cte)))
    z4 = zeta**4 #jnp.round(2 ** (4 * jnp.log2(jnp.clip(zeta, a_min=clip_cte))), int(math.log10(clip_cte)))

    e_tilde = (
        e_tilde_PF[:, 0]
        + alphac * (fz / (grad(grad(fzeta))(0.0))) * (1 - z4)
        + (e_tilde_PF[:, 1] - e_tilde_PF[:, 0]) * fz * z4
    )
    # assert not jnp.isnan(e_tilde).any() and not jnp.isinf(e_tilde).any()

    return e_tilde


def densities(
    molecule: Molecule,
    functional_type: Optional[Union[str, Dict[str, int]]] = "LDA",
    clip_cte: float = 1e-30,
    *_,
    **__,
):
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

    beta = 1 / 1024.0

    if isinstance(functional_type, str):
        if functional_type == "LDA" or functional_type == "DM21":
            u_range, w_range = range(0, 1), range(0, 1)
        elif functional_type == "GGA":
            u_range, w_range = range(0, 2), range(0, 1)
        elif functional_type == "MGGA":
            u_range, w_range = range(0, 2), range(0, 2)
        else:
            raise ValueError(
                f"Functional type {functional_type} not recognized, must be one of LDA, GGA, MGGA."
            )

    # Molecule preprocessing data
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    # LDA preprocessing data
    log_rho = jnp.log2(jnp.clip(rho, a_min=clip_cte))

    # GGA preprocessing data
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min=clip_cte)) / 2
    log_x_sigma = log_grad_rho_norm - 4 / 3.0 * log_rho
    log_u_sigma = jnp.where(
        jnp.greater(log_rho, jnp.log2(clip_cte)),
        log_x_sigma - jnp.log2(1 + beta * (2**log_x_sigma)) + jnp.log2(beta),
        0,
    )

    # MGGA preprocessing data
    log_tau = jnp.log2(jnp.clip(tau, a_min=clip_cte))
    log_1t_sigma = log_tau - 5 / 3.0 * log_rho
    log_w_sigma = jnp.where(
        jnp.greater(log_rho, jnp.log2(clip_cte)),
        log_1t_sigma - jnp.log2(1 + beta * (2**log_1t_sigma)) + jnp.log2(beta),
        0,
    )

    # Compute the local features
    localfeatures = jnp.empty((log_rho.shape[0], 0))
    for i, j in itertools.product(u_range, w_range):
        mgga_term = 2 ** (4 / 3.0 * log_rho + i * log_u_sigma + j * log_w_sigma)

        # First we concatenate the exchange terms
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=1)

    ######### Correlation features ###############

    grad_rho_norm_sq_ss = jnp.sum((grad_rho.sum(axis=1)) ** 2, axis=-1)
    log_grad_rho_norm_ss = jnp.log2(jnp.clip(grad_rho_norm_sq_ss, a_min=clip_cte)) / 2
    log_rho_ss = jnp.log2(jnp.clip(rho.sum(axis=1), a_min=clip_cte))
    log_x_ss = log_grad_rho_norm_ss - 4 / 3.0 * log_rho_ss

    log_u_ss = jnp.where(
        jnp.greater(log_rho_ss, jnp.log2(clip_cte)),
        log_x_ss - jnp.log2(1 + beta * (2**log_x_ss)) + jnp.log2(beta),
        0,
    )

    log_u_ab = jnp.where(
        jnp.greater(log_rho_ss, jnp.log2(clip_cte)),
        log_x_ss - 1 - jnp.log2(1 + beta * (2 ** (log_x_ss - 1))) + jnp.log2(beta),
        0,
    )

    log_u_c = jnp.stack((log_u_ss, log_u_ab), axis=1)

    log_tau_ss = jnp.log2(jnp.clip(tau.sum(axis=1), a_min=clip_cte))
    log_1t_ss = log_tau_ss - 5 / 3.0 * log_rho_ss
    log_w_ss = jnp.where(
        jnp.greater(log_rho.sum(axis=1), jnp.log2(clip_cte)),
        log_1t_ss - jnp.log2(1 + beta * (2**log_1t_ss)) + jnp.log2(beta),
        0,
    )

    log_w_ab = jnp.where(
        jnp.greater(log_rho.sum(axis=1), jnp.log2(clip_cte)),
        log_1t_ss - 1 - jnp.log2(1 + beta * (2 ** (log_1t_ss - 1))) + jnp.log2(beta),
        0,
    )

    log_w_c = jnp.stack((log_w_ss, log_w_ab), axis=1)

    A_ = jnp.array([[0.031091, 0.015545]])
    alpha1 = jnp.array([[0.21370, 0.20548]])
    beta1 = jnp.array([[7.5957, 14.1189]])
    beta2 = jnp.array([[3.5876, 6.1977]])
    beta3 = jnp.array([[1.6382, 3.3662]])
    beta4 = jnp.array([[0.49294, 0.62517]])

    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1, keepdims=True), a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))

    e_PW92 = jnp.round(
        -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2)),
        int(math.log10(clip_cte)),
    )

    # Compute the local features
    for i, j in itertools.product(u_range, w_range):
        mgga_term = jnp.where(
            jnp.greater(e_PW92, clip_cte), 2 ** (jnp.log2(e_PW92) + i * log_u_c + j * log_w_c), 0
        )

        # First we concatenate the exchange terms
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=1)

    return localfeatures


############# Dispersion functional #############


@dataclass
class DispersionFunctional(nn.Module):
    r"""
    Dispersion functional
    """

    dispersion: staticmethod
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

        self.layer_norm = partial(nn.LayerNorm, param_dtype=self.param_dtype)

    @nn.compact
    def __call__(self, *inputs) -> Scalar:
        r"""Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Parameters
        ---------
        inputs: inputs to the function f

        Returns
        -------
        Union[Array, Scalar]
        """

        return self.dispersion(self, *inputs)

    def head(self, x: Array, local_features, sigmoid_scale_factor):
        r"""
        The head of the neural network, which is the final layer of the neural network.
        """
        # Final layer: dense -> sigmoid -> scale (x2)
        x = self.dense(features=local_features)(x)
        self.sow("intermediates", "head_dense", x)
        x = sigmoid(x / sigmoid_scale_factor)
        self.sow("intermediates", "sigmoid", x)
        out = sigmoid_scale_factor * x
        self.sow("intermediates", "sigmoid_product", out)

        return jnp.squeeze(out)  # Eliminating unnecessary dimensions

    def energy(self, params: PyTree, molecule: Molecule):
        r"""
        Calculates the energy of the functional.
        """
        R_AB, ai = calculate_distances(molecule.nuclear_pos, molecule.atom_index)

        result = 0
        for n in range(3, 6):
            x = jnp.concatenate((R_AB, ai, n * jnp.ones(R_AB.shape)), axis=-1)
            y = self.apply(params, x) / jnp.squeeze(R_AB) ** (2 * n)
            result = result + jnp.sum(y)
        return -result / 2


def calculate_distances(positions, atoms):
    r"""
    Calculates the distances between all atoms in the molecule.
    """
    pairwise_distances = jnp.linalg.norm(positions[:, None] - positions, axis=-1)
    atom_pairs = jnp.array(
        [(atoms[i], atoms[j]) for i in range(len(atoms)) for j in range(len(atoms))]
    )

    pairwise_distances = jnp.reshape(pairwise_distances, newshape=(-1))

    non_zero_mask = jnp.greater(pairwise_distances, 0.0)
    pairwise_distances = pairwise_distances[non_zero_mask]
    atom_pairs = atom_pairs[non_zero_mask]
    return pairwise_distances[:, None], atom_pairs
