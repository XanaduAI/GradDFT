from utils import Scalar, Array
from typing import Optional, Sequence, Union, List, Dict
from dataclasses import fields
from utils import Array, Scalar
from functools import partial, reduce

from jax import numpy as jnp
from jax.lax import Precision
from jax import vmap, grad
from flax import linen as nn
from flax import struct
import itertools
from pyscf.dft import Grids, numint  # type: ignore
import jax


@struct.dataclass
class Grid:

    coords: Array
    weights: Array

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"

    def __len__(self):
        return self.weights.shape[0]

    def to_dict(self) -> dict:
        return {"coords": self.coords, "weights": self.weights}

    def integrate(self, vals: Array, axis: int = 0) -> Array:

        """A function that performs grid quadrature (integration) in a differentiable way (using jax.numpy).

        This function is glorified tensor contraction but it sometimes helps
        with readability and expresing intent in the rest of the code.

        Parameters
        ----------
        vals : Array
            Local features/ function values to weigh.
            Expected shape: (..., n_lattice, ...)
        axis : int, optional
            Axis to contract. vals.shape[axis] == n_lattice
            has to hold.

        Returns
        -------
        Array
            Integrals of the same as `vals` but with `axis` contracted out.
            If vals.ndim==(1,), then the output is squeezed to a scalar.
        """

        return jnp.tensordot(self.weights, vals, axes=(0, axis))

@struct.dataclass
class Molecule:

    grid: Grid
    atom_index: Array
    nuclear_pos: Array
    ao: Array
    grad_ao: Array
    rdm1: Array
    nuclear_repulsion: Scalar
    h1e_energy: Scalar
    coulomb2e_energy: Scalar
    h1e: Array
    vj: Array
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Array
    mf_energy: Optional[Scalar] = None
    s1e: Optional[Array] = None # Not used during training
    omegas: Optional[Array] = None
    chi: Optional[Array] = None
    rep_tensor: Optional[Array] = None
    energy: Optional[Scalar] = None
    basis: Optional[str] = None
    name: Optional[str] = None
    spin: Optional[Scalar] = 0
    charge: Optional[Scalar] = 0
    unit_Angstrom: Optional[bool] = True
    grid_level: Optional[int] = 2
    scf_iteration: Optional[int] = 50
    fock: Optional[Array] = None

    def __repr__(self):
        return f"{self.__class__.__name__}(grid_size={self.grid_size})"

    @staticmethod
    def _rdm1_idx():
        # The index of the density matrix field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 6

    @staticmethod
    def _h1e_energy_idx():
        # The index of the h1e field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 8

    @staticmethod
    def _coulomb2e_energy_idx():
        # The index of the coulomb2e field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 9

    @staticmethod
    def _mo_coeff_idx():
        # The index of the mo_coeff field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 12

    @staticmethod
    def _mo_occ_idx():
        # The index of the mo_occ field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 13

    @staticmethod
    def _mo_energy_idx():
        # The index of the mo_energy field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 14

    @staticmethod
    def _mf_energy_idx():
        # The index of the mean field energy field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 15

    @staticmethod
    def _chi_idx():
        # The index of the chi field,
        # when you flatten the `Molecule` tree.
        # Needed for internal use.
        return 19

    @property
    def grid_size(self):
        return len(self.grid)

    def density(self, *args, **kwargs):
        return density(self.rdm1, self.ao, *args, **kwargs)

    def grad_density(self, *args, **kwargs):
        return grad_density(self.rdm1, self.ao, self.grad_ao, *args, **kwargs)

    def kinetic_density(self, *args, **kwargs):
        return kinetic_density(self.rdm1, self.grad_ao, *args, **kwargs)

    def to_dict(self) -> dict:
        grid_dict = self.grid.to_dict()
        rest = {field.name: getattr(self, field.name) for field in fields(self)[1:]}
        return dict(**grid_dict, **rest)




@partial(jax.jit, static_argnames="precision")
def density(dm: Array, ao: Array, precision: Precision = Precision.HIGHEST) -> Array:

    """Calculate electronic density from atomic orbitals.

    Parameters
    ----------
    dm : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density. Shape: (*batch, n_spin, n_grid_points)
    """

    return jnp.einsum("...ab,ra,rb->...r", dm, ao, ao, precision=precision)

@partial(jax.jit, static_argnames="precision")
def grad_rho_DM(dm: Array, ao: Array, precision: Precision = Precision.HIGHEST, chunk_size = None) -> Array:

    """Calculate the gradient of the density matrix.

    Parameters
    ----------
    dm : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.
    chunk_size : int, optional
        The chunk size to use for the vmap. By default None.

    Returns
    -------
    Array
        The gradient of the density matrix.
        Shape: (*batch, n_spin, n_grid_points, n_orbitals, n_orbitals)
    """

    def vmapped_integrand(dm_tensor: Array, ao_tensor: Array):

        return jnp.einsum("ab,a,b->", dm_tensor, ao_tensor, ao_tensor, precision=precision)

    return jnp.stack([vmap(grad(vmapped_integrand), in_axes = (None, 0))(dm[0], ao), vmap(grad(vmapped_integrand), in_axes = (None, 0))(dm[1], ao)], axis=0)

@partial(jax.jit, static_argnames="precision")
def grad_density(
    dm: Array, ao: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST
) -> Array:

    """Calculate the electronic density gradient from atomic orbitals.

    Parameters
    ----------
    dm : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    grad_ao : Array
        Gradients of atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals, 3)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density gradient. Shape: (*batch, n_spin, n_grid_points, 3)
    """

    return 2 * jnp.einsum("...ab,ra,rbj->...rj", dm, ao, grad_ao, precision=precision)

@partial(jax.jit, static_argnames="precision")
def partial_grad_density_DM(
    dm: Array, ao: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST
) -> Array:
    
        """Calculate the partial derivative of (the gradient of the density matrix with respect to r) with respect to dm.
    
        Parameters
        ----------
        dm : Array
            The density matrix.
            Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
        ao : Array
            Atomic orbitals.
            Expected shape: (n_grid_points, n_orbitals)
        grad_ao : Array
            Gradients of atomic orbitals.
            Expected shape: (n_grid_points, n_orbitals, 3)
        precision : jax.lax.Precision, optional
            Jax `Precision` enum member, indicating desired numerical precision.
            By default jax.lax.Precision.HIGHEST.
    
        Returns
        -------
        Array
            The gradient of the density matrix.
            Shape: (*batch, n_spin, n_grid_points, n_orbitals, n_orbitals)
        """

        def vmapped_integrand(dm_tensor: Array, ao_tensor: Array, grad_ao_tensor: Array):

            return 2 * jnp.linalg.norm(jnp.einsum("ab,a,bd->d", dm_tensor, ao_tensor, grad_ao_tensor, precision=precision))

        return jnp.stack([vmap(grad(vmapped_integrand), in_axes = (None, 0, 0), out_axes=0)(dm[0], ao, grad_ao), 
                        vmap(grad(vmapped_integrand), in_axes = (None, 0, 0), out_axes=0)(dm[1], ao, grad_ao)], axis=0)

@partial(jax.jit, static_argnames="precision")
def kinetic_density(dm: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST) -> Array:

    """Calculate kinetic energy density from atomic orbitals.

    Parameters
    ----------
    dm : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    grad_ao : Array
        Gradients of atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals, 3)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jaxx.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The kinetic energy density. Shape: (*batch, n_spin, n_grid_points)
    """

    return 0.5 * jnp.einsum("...ab,raj,rbj->...r", dm, grad_ao, grad_ao, precision=precision)









def default_features(molecule: Molecule, functional_type: Optional[Union[str, Dict[str, int]]] = 'LDA', clip_cte: float = 1e-27, *_, **__):
    """
    Generates all features except the HF energy features.
    """
    beta = 1/1024.

    rho = molecule.density()
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()

    grad_rho_norm = jnp.sum(grad_rho**2, axis=-1)
    grad_rho_norm_sumspin = jnp.sum(grad_rho.sum(axis=0, keepdims=True) ** 2, axis=-1)

    features = jnp.concatenate((rho, grad_rho_norm_sumspin, grad_rho_norm, tau), axis=0)

    log_rho = jnp.log2(jnp.clip(rho, a_min = clip_cte))
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm, a_min = clip_cte))
    log_x_sigma = log_grad_rho_norm/2 - 4/3.*log_rho
    log_u_sigma = jnp.where(jnp.greater(log_rho,jnp.log2(clip_cte)), log_x_sigma - jnp.log2(1 + beta*(2**log_x_sigma)) + jnp.log2(beta), 0)

    log_tau = jnp.log2(jnp.clip(tau, a_min = clip_cte))
    log_1t_sigma = -(5/3.*log_rho - log_tau + 2/3.*jnp.log2(6*jnp.pi**2) + jnp.log2(3/5.))
    log_w_sigma = jnp.where(jnp.greater(log_rho, jnp.log2(clip_cte)), log_1t_sigma - jnp.log2(1 + beta*(2**log_1t_sigma)) + jnp.log2(beta), 0)

    if type(functional_type) == str:
        if functional_type == 'LDA' or functional_type == 'DM21': u_power, w_power, uw_power = [0,0], [0,0], [0,0]
        elif functional_type == 'GGA': u_power, w_power, uw_power = [0,1], [0,0], [0,1]
        elif functional_type == 'MGGA': u_power, w_power, uw_power = [0,1], [0,1], [0,2]
        else: raise ValueError(f'Functional type {functional_type} not recognized, must be one of LDA, GGA, MGGA.')
    else: u_power, w_power, uw_power= functional_type['u'], functional_type['w'], functional_type['u+w']

    # Here we use the LDA form from DM21 to be able to replicate its behavior if desired.
    localfeatures = jnp.expand_dims((-2 * jnp.pi * (3 / (4 * jnp.pi)) ** (4 / 3) * 2**(4/3.*log_rho)).sum(axis=0), axis = 0)

    for i, j in itertools.product(range(u_power[0], u_power[1]+1), range(w_power[0], w_power[1]+1)):
        
        if i+j < uw_power[0] or i+j > uw_power[1] or (i == 0 and j == 0): continue

        mgga_term = jnp.expand_dims((2**(4/3.*log_rho + i * log_u_sigma + j * log_w_sigma)).sum(axis=0), axis = 0)

        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=0)

    return features.T, localfeatures.T 