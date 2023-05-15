from utils import Scalar, Array, Callable
from typing import Optional, Sequence, Union, List
from dataclasses import fields

from jax import numpy as jnp
from jax.lax import Precision
from jax import vmap, grad
from flax import linen as nn
from flax import struct


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

    def to_dict(self) -> dict:
        grid_dict = self.grid.to_dict()
        rest = {field.name: getattr(self, field.name) for field in fields(self)[1:]}
        return dict(**grid_dict, **rest)

