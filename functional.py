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

class LocalFunctional(nn.Module):
    ''' A base class of local functionals.
    F[n(r)] = \int f(n(r)) dr^3

    Parameters
    ----------
    function: Callable
        Implements the function f above.
        Should take as input an object of size n_inputs,
        and output a scalar, as it will be vectorized.
    '''

    f: staticmethod  # Decorator to define f as a static method

    def setup(self):
        pass

    @nn.compact
    def __call__(self, inputs) -> Scalar:
        '''Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Paramters
        ---------
        rhoinputs: Array
            Shape: (..., n_grid)

        Returns
        -------
        Array
            Shape: (n_grid)
        '''

        return self.f(**inputs)
    
    def energy(self, gridweights, inputs):

        localfeatures = self.apply({"params": {}}, inputs)
        return self._integrate(localfeatures, gridweights)

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

        return jnp.einsum("r,r...->...", gridweights, features)


def external_f(instance, x):
    x = instance.dense(x)
    x = 0.5*jnp.tanh(x)
    return x

class Functional(nn.Module):
    ''' A base class of local functionals.
    F[n(r)] = \int f(n(r)) dr^3

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

        localfeatures = self.apply(params, *args)
        xc_energy = self._integrate(localfeatures, molecule.grid.weights)
        nonxc_energy = nonXC(molecule.rdm1, molecule.h1e, molecule.rep_tensor, molecule.nuclear_repulsion)
        return xc_energy + nonxc_energy

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

def nonXC(
    rdm1: Array, h1e: Array, rep_tensor: Array, nuclear_repulsion: Scalar, precision = Precision.HIGHEST
) -> Scalar:

    """A function that computes the non-XC part of a DFT functional.

    Parameters
    ----------
    rdm1 : Array
        The 1-Reduced Density Matrix.
        Equivalent to mf.make_rdm1() in pyscf.
        Expected shape: (n_spin, n_orb, n_orb)
    h1e : Array
        The 1-electron Hamiltonian.
        Equivalent to mf.get_hcore(mf.mol) in pyscf.
        Expected shape: (n_orb, n_orb)
    rep_tensor : Array
        The repulsion tensor.
        Equivalent to mf.mol.intor('int2e') in pyscf.
        Expected shape: (n_orb, n_orb, n_orb, n_orb)
    nuclear_repulsion : Scalar
        Equivalent to mf.mol.energy_nuc() in pyscf.
        The nuclear repulsion energy.
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
        The non-XC energy of the DFT functional.
    """

    v_coul = 2 * jnp.einsum("pqrt,srt->spq", rep_tensor, rdm1, precision=precision) 

    h1e_energy = jnp.einsum("sij,ji->", rdm1, h1e, precision=precision)
    coulomb2e_energy = jnp.einsum('sji,sij->', rdm1, v_coul, precision=precision)/2.

    return nuclear_repulsion + h1e_energy + coulomb2e_energy


def HF_exact_exchange(
    chi, rdm1, ao, precision = Precision.HIGHEST
) -> Array:
    
        """A function that computes the exact exchange energy of a DFT functional.

        Parameters
        ----------
        chi : Array
            Xc^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψc(r') ψd(r')
            Expected shape: (n_grid, n_omega, n_spin, n_orbitals)
        rdm1 : Array
            The 1-Reduced Density Matrix.
            Equivalent to mf.make_rdm1() in pyscf.
            Expected shape: (n_spin, n_orb, n_orb)
        ao : Array
            The atomic orbital basis.
            Equivalent to pyscf.dft.numint.eval_ao(mf.mol, grids.coords, deriv=0) in pyscf.
            Expected shape: (n_grid, n_orb)
        precision : Precision, optional
            The precision to use for the computation, by default Precision.HIGHEST
    
				Notes
				-------
				n_omega makes reference to different possible kernels, for example using
				the kernel f(|r-r'|) = erf(w |r-r'|)/|r-r'|.

        Returns
        -------
        Array
            The exact exchange energy of the DFT functional at each point of the grid.
        """

        _hf_energy = lambda _chi, _dm, _ao: - jnp.einsum("wsc,sac,a->ws", _chi, _dm, _ao, precision=precision)/2
        return vmap(_hf_energy, in_axes=(0, None, 0), out_axes=2)(chi, rdm1, ao)

def canonicalize_inputs(x):

    x = jnp.asarray(x)

    if x.ndim == 1:
        return jnp.expand_dims(x, axis=1)
    elif x.ndim == 0:
        raise ValueError("`features` has to be at least 1D array!")
    else:
        return x
    

@partial(value_and_grad, has_aux = True)
def defaultcost(params, functional, molecule, trueenergy, *functioninputs):
    ''' Computes the loss function, here MSE, between predicted and true energy'''

    predictedenergy = functional.energy(params, molecule, *functioninputs)
    cost_value = (predictedenergy - trueenergy) ** 2

    return cost_value, predictedenergy