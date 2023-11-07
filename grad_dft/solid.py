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

import jax.numpy as jnp
from jax.lax import Precision
from typing import List, Optional

from typeguard import typechecked
from grad_dft.utils import vmap_chunked
from functools import partial
from jax import jit

from flax import struct
from jaxtyping import Array, PyTree, Scalar, Float, Int, Complex, jaxtyped


@struct.dataclass
class Grid:
    r""" Base class for the grid coordinates and integration grids."""
    coords: Array
    weights: Array

    def __len__(self):
        return self.weights.shape[0]

    def to_dict(self) -> dict:
        return {"coords": self.coords, "weights": self.weights}

    def integrate(self, vals: Array, axis: int = 0) -> Array:
        r"""
        A function that performs grid quadrature (integration) in a differentiable way (using jax.numpy).

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
class KPointInfo:
    r"""Contains the neccesary information about BZ sampling needed for total energy calculations.
    Most simply, we need the array of k-points in absolute and fractional forms with equal weights.
    To properly take advantage of space-group and time-reversal symmetry, informations about mappings
    between the BZ -> IBZ and vice versa is needed as well as weights which are not neccesarily equal.
        
    n_kpts_or_n_ikpts in weights could be the total number of points in the full BZ or the number of
    points in the IBZ, context dependent. I.e, if the next variables are set to None,
    the first case applies. If they are not None, the second does.
    """
    
    kpts_abs: Float[Array, "n_kpts 3"]
    kpts_scaled: Float[Array, "n_kpts 3"] 
    weights: Float[Array, "n_kpts_or_n_ir_kpts"]
    # Coming Soon: take advantage of Space Group symmetry for efficient simulation
    # bz2ibz_map: Optional[Float[Array, "n_kpts"]]
    # ibz2bz_map: Optional[Float[Array, "n_kpts_ir"]]
    # kpts_ir_abs: Optional[Float[Array, "n_kpts_ir 3"]]
    # kpts_ir_scaled: Optional[Float[Array, "n_kpts_ir 3"]]

@struct.dataclass
class Solid:
    r"""Base class for storing data pertaining to DFT calculations with solids.
    This shares many simlarities ~/grad_dft/molecule.py's `Molecule` class, but many arrays
    must have an extra dimension to house the number of k-points.
    
    Typically, for those arrays which need a k-point index, if a spin index is required,
    dimension 1 will be dimension n_kpt. If spin is not required, dimension 0 will be 
    n_kpt.
    """

    grid: Grid
    atom_index: Int[Array, "n_atom"]
    lattice_vectors: Float[Array, "3 3"] 
    nuclear_pos: Float[Array, "n_atom 3"]
    ao: Float[Array, "n_flat_grid n_orbitals"]
    grad_ao: Float[Array, "n_flat_grid n_orbitals 3"]
    grad_n_ao: PyTree
    rdm1: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    nuclear_repulsion: Scalar
    h1e: Complex[Array, "n_kpt n_orbitals n_orbitals"]
    vj: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    mo_coeff: Float[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    mo_occ: Float[Array, "n_spin n_kpt n_orbitals"]
    mo_energy: Float[Array, "n_spin n_kpt n_orbitals"]
    kpt_info: KPointInfo
    mf_energy: Optional[Scalar] = None
    s1e: Optional[Complex[Array, "n_kpt n_orbitals n_orbitals"]] = None
    omegas: Optional[Float[Array, "omega"]] = None
    chi: Optional[Float[Array, "grid omega spin orbitals"]] = None # Come back to this to figure out correct dims for k-points
    rep_tensor: Optional[Complex[Array, "n_k4pt n_orbitals n_orbitals n_orbitals n_orbitals"]] = None
    energy: Optional[Scalar] = None
    basis: Optional[Int[Array, '...']] = None # The name is saved as a list of integers, JAX does not accept str
    name: Optional[Int[Array, '...']] = None # The name is saved as a list of integers, JAX does not accept str
    spin: Optional[Scalar] = 0
    charge: Optional[Scalar] = 0
    unit_Angstrom: Optional[bool] = True
    grid_level: Optional[Scalar] = 2
    scf_iteration: Optional[Scalar] = 50
    fock: Optional[Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]] = None
    
@jaxtyped
@typechecked
@partial(jit, static_argnames=["precision"])
def one_body_energy(
    rdm1: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    h1e: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision=Precision.HIGHEST,
) -> Scalar:
    r"""A function that computes the one-body energy of a DFT functional.

    Parameters
    ----------
    rdm1 : Float[Array, "n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix for each k-point.
    h1e : Float[Array, "n_kpt orbitals orbitals"]
        The 1-electron Hamiltonian for each k-point.
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.

    Returns
    -------
    Scalar
    """
    h1e_energy = jnp.einsum("k,kij,kij->", weights, rdm1, h1e, precision=precision)
    return h1e_energy.real



def coulomb_energy(
    rdm1: Float[Array, "n_kpt n_orbitals n_orbitals"],
    rep_tensor: Float[Array, "n_kpt_quartets n_orbitals n_orbitals n_orbitals n_orbitals"],
    weights_k4: Float[Array, "n_kpt_quartets"],
    k4_idxs: Int[Array, "n_k4pts 4"],
    bz2ibz_map: Float[Array, "n_kpts"],
    precision=Precision.HIGHEST,
) -> Scalar:
    """
    Compute the Coulomb energy considering crystal momentum conserving k-point quartets.

    Parameters
    ----------
    rdm1 : Float[Array, "n_ir_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix at each irreducible k-point.
    rep_tensor : Float[Array, "n_ir_kpt_quartets n_orbitals n_orbitals n_orbitals n_orbitals"]
        The repulsion tensor indexed by k-point quartets and orbitals.
    weights_k4 : Float[Array, "n_ir_kpt_quartets"]
        The weights associated with each k-point quartet.
    k4_idxs : Int[Array, "n_ir_kpt_quartets 4"].
        Each element in the first dimension gives the indices in the full 1BZ for 
        four crystal momentum conserving k-points. I.e, the k-point quartet indices.
    bz2ibz_map : Float[Array, "n_kpts"].
        Given an index in the 1BZ, return the corresponding index in the irreducible 1BZ.
    precision : Precision, optional
        The precision to use for the computation.

    Returns
    -------
    Scalar
        The Coulomb energy as:
        .. math::
            E_C = \frac{1}{2} \sum_{\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4} \delta_{\mathbf{k}_1 - \mathbf{k}_2 + \mathbf{k}_3 - \mathbf{k}_4, \mathbf{G}} w(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4) \sum_{pqrs} D_{pq}(\mathbf{k}_1) (pq|rs)_{\mathbf{k}_1\mathbf{k}_2\mathbf{k}_3\mathbf{k}_4} D_{rs}(\mathbf{k}_3)
    """

    # Initialize Coulomb energy to zero
    coulomb_energy = 0.0

    # Loop over all k-point quartets
    for i, k_quartet_idxs in enumerate(k4_idxs):
        # Extract the ERIs for this k-point quartet
        eri_kpt_quartet = rep_tensor[i]
        
        # Determine the indices for the k-points involved in this quartet
        k1_idx = k_quartet_idxs[0]
        k2_idx = k_quartet_idxs[1]
        k3_idx = k_quartet_idxs[2]
        k4_idx = k_quartet_idxs[3]
        
        # k1_idx_ibz = bz2ibz_map[k1_idx]
        # k3_idx_ibz = bz2ibz_map[k3_idx]
        
        
        # Compute the contribution to the Coulomb energy from this k-point quartet
        # energy_contribution = jnp.einsum(
        #     "qp,pqrs,sr->", rdm1[k1_idx], eri_kpt_quartet, rdm1[k3_idx], precision=precision
        # )
        energy_contribution = jnp.trace(
            rdm1[k1_idx] @ eri_kpt_quartet @ rdm1[k3_idx]
        )
        # Accumulate the weighted energy contribution
        coulomb_energy += weights_k4[i] * energy_contribution
    
    # Account for double-counting in the ERI
    coulomb_energy /= 2.0

    return coulomb_energy


