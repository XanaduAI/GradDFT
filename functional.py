from jax import numpy as jnp
from flax import linen as nn
from jax.lax import Precision
from jax import vmap
from pyrsistent import v

from utils import Scalar, Array, Callable

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

    f: Callable

    def setup(self):
        pass

    @nn.compact
    def __call__(self, rhoinputs, *args) -> Scalar:
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

        return self.f(rhoinputs, *args)
    
    def energy(self, rhoinputs, gridweights, *args):

        localfeatures = self.apply(rhoinputs, *args)
        return self._integrate(localfeatures, gridweights)

    def _integrate(
        self, features: Array, gridweights: Array, precision: Precision.HIGHEST
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

        return jnp.einsum("r,...r,->...", gridweights, features, precision = precision)


######################### Helper functions #########################

def integrate_local_weights(
    gridweights: Array, features: Array
) -> Array:

    """A function that performs grid quadrature (integration) in a differentiable way (using jax.numpy).

    Parameters
    ----------
    gridweights : Array
        Quadrature weights.
        Expected shape: (n_grid,)
    features : Array
        Neural network outputs.
        Expected shape: (*batch, n_grid)
    Returns
    -------
    Array
        Integrals of shape (*batch,). If batch==(1,),
        then the output is squeezed to a scalar.
    """

    return jnp.einsum("r,...r,->...", gridweights, features)


def nonXC(
    rdm1, h1e, rep_tensor, nuclear_repulsion, precision = Precision.HIGHEST
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