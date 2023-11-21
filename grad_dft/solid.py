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
import jax

from typeguard import typechecked
from grad_dft.utils import vmap_chunked
from functools import partial
from jax import jit, vmap
from jax.lax import fori_loop, cond

from dataclasses import fields

from flax import struct
from flax import linen as nn
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
    
    def to_dict(self) -> dict:
        info = {
            "kpts_abs": self.kpts_abs,
            "kpts_scaled": self.kpts_scaled,
            "kpt_weights": self.weights
        }
        return info

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
    kpt_info: KPointInfo
    atom_index: Int[Array, "n_atom"]
    lattice_vectors: Float[Array, "3 3"] 
    nuclear_pos: Float[Array, "n_atom 3"]
    ao: Complex[Array, "n_kpt n_flat_grid n_orbitals"] # ao = Crystal Atomic Orbitals in PBC case
    grad_ao: Complex[Array, "nkpt n_flat_grid n_orbitals 3"]
    grad_n_ao: PyTree
    rdm1: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    nuclear_repulsion: Scalar
    h1e: Complex[Array, "n_kpt n_orbitals n_orbitals"]
    vj: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    mo_coeff: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    mo_occ: Float[Array, "n_spin n_kpt n_orbitals"]
    mo_energy: Float[Array, "n_spin n_kpt n_orbitals"]
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
    
    def density(self, *args, **kwargs) -> Array:
        r"""Compute the electronic density at each grid point.
        
        Returns
        -------
        Float[Array, "grid spin"]
        """
        return density(self.rdm1, self.ao, self.kpt_info.weights, *args, **kwargs)
    
    def nonXC(self, *args, **kwargs) -> Scalar:
        r"""Compute all terms in the KS total energy with the exception of the XC component
        
        Returns
        -------
        Scalar
        """
        return non_xc(self.rdm1.sum(axis=0), self.h1e, self.rep_tensor, self.nuclear_repulsion, self.kpt_info.weights, *args, **kwargs)
    
    def make_rdm1(self, *args, **kwargs) -> Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]:
        r"""Compute the 1-body reduced density matrix for each k-point.
        
        Returns
        -------
        Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
        """
        return make_rdm1(self.mo_coeff, self.mo_occ, *args, **kwargs)
    
    def get_occ(self) -> Array:
        r"""Compute the occupations of the molecular orbitals for each spin and k-point.
        
        Returns
        -------
        Float[Array, "n_spin n_kpt n_orbitals"]
        """
        # each k-channel has same total number of electrons, so just use index 0 in nelec calculation
        # when indexing self.mo_occ
        nelecs = jnp.array([self.mo_occ[i, 0].sum() for i in range(2)], dtype=jnp.int64)
        return get_occ(self.mo_energy, nelecs)
    
    def grad_density(self, *args, **kwargs) -> Array:
        r"""Compute the gradient of the electronic density at each grid point.
        
        Returns
        -------
        Float[Array, "n_flat_grid n_spin 3"]
        """
        return grad_density(self.rdm1, self.ao, self.grad_ao, self.kpt_info.weights, *args, **kwargs)

    def lapl_density(self, *args, **kwargs) -> Array:
        r"""Compute the laplacian of the electronic density at each grid point.
        
        Returns
        -------
        Float[Array, "n_flat_grid n_spin"]
        """
        return lapl_density(self.rdm1, self.ao, self.grad_ao, self.grad_n_ao[2], self.kpt_info.weights, *args, **kwargs)

    def kinetic_density(self, *args, **kwargs) -> Array:
        r"""Compute the kinetic energy density at each grid point.
        
        Returns
        -------
        Float[Array, "n_flat_grid n_spin"]
        """
        return kinetic_density(self.rdm1, self.grad_ao, self.kpt_info.weights, *args, **kwargs)
    
    def to_dict(self) -> dict:
        r"""Return a dictionary with the attributes of the solid.
        
        Returns
        -------
        Dict
        """
        grid_dict = self.grid.to_dict()
        kpt_dict = self.kpt_info.to_dict()
        rest = {field.name: getattr(self, field.name) for field in fields(self)[2:]}
        return dict(**grid_dict, **kpt_dict, **rest)
    
    def get_coulomb_potential(self, *args, **kwargs) -> Complex[Array, "n_kpts_or_n_ir_kpts n_orbitals n_orbitals"]:
        r"""
        Computes the Coulomb potential matrix for all k-points.

        Returns
        -------
        Complex[Array, "n_kpts_or_n_ir_kpts n_orbitals n_orbitals"]
        """
        return coulomb_potential(self.rdm1.sum(axis=0), self.rep_tensor, self.kpt_info.weights, *args, **kwargs)
    
    def select_HF_omegas(self, omegas: Float[Array, "omega"]) -> Array:
        raise NotImplementedError("Hartree-Fock methods (for computation of Hybrid functionals) will come in a later release.")

    def HF_energy_density(self, omegas: Float[Array, "omega"], *args, **kwargs) -> Array:
        raise NotImplementedError("Hartree-Fock methods (for computation of Hybrid functionals) will come in a later release.")

    def HF_density_grad_2_Fock(
        self,
        functional: nn.Module,
        params: PyTree,
        omegas: Float[Array, "omega"],
        ehf: Float[Array, "omega spin grid"],
        coefficient_inputs: Float[Array, "grid cinputs"],
        densities_wout_hf: Float[Array, "grid densities_w"],
        **kwargs,
    ) -> Float[Array, "omega spin orbitals orbitals"]:
        raise NotImplementedError("Hartree-Fock methods (for computation of Hybrid functionals) will come in a later release.")

    def HF_coefficient_input_grad_2_Fock(
        self,
        functional: nn.Module,
        params: PyTree,
        omegas: Float[Array, "omega"],
        ehf: Float[Array, "omega spin grid"],
        cinputs_wout_hf: Float[Array, "grid cinputs_w"],
        densities: Float[Array, "grid densities"],
        **kwargs,
    ) -> Float[Array, "omega spin orbitals orbitals"]:
        raise NotImplementedError("Hartree-Fock methods (for computation of Hybrid functionals) will come in a later release.")
    
    def get_mo_grads(self, *args, **kwargs):
        r"""Compute the gradient of the electronic energy with respect 
        to the molecular orbital coefficients.

        Returns:
        -------
        Float[Array, "orbitals orbitals"]
        """
        return orbital_grad(self.mo_coeff, self.mo_occ, self.fock, *args, **kwargs)
    

@jaxtyped
@typechecked
@partial(jit, static_argnames=["precision"])
def one_body_energy(
    rdm1: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    h1e: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision=Precision.HIGHEST,
) -> Scalar:
    r"""Compute the one-body (kinetic + external) component of the KS total energy.

    Parameters
    ----------
    rdm1 : Complex[Array, "n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix for each k-point. Spin has been summed over before input.
    h1e : Complex[Array, "n_kpt orbitals orbitals"]
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
    h1e_energy = jnp.einsum("k,kij,kji->", weights, rdm1, h1e, precision=precision)
    return h1e_energy.real

@jaxtyped
@typechecked
@partial(jit, static_argnames=["precision"])
def coulomb_potential(
    rdm1: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    rep_tensor: Complex[Array, "n_kpt n_kpt n_orbitals n_orbitals n_orbitals n_orbitals"],
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision=Precision.HIGHEST
) -> Complex[Array, "n_kpts_or_n_ir_kpts n_orbitals n_orbitals"]:
    r"""
    Computes the Coulomb potential matrix for all k-points.

    Parameters
    ----------
    rdm1 : Complex[Array, "n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix. Spin has been summed over before input.
    rep_tensor : Complex[Array, "n_kpt n_kpt n_orbitals n_orbitals n_orbitals n_orbitals"]
        The repulsion tensor computed on a grid of nkpt x nkpt
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Complex[Array, "n_kpts_or_n_ir_kpts n_orbitals n_orbitals"]
    """
    
    # k and q are k-point indices while i, j, l and m are orbital indices
    v_k = jnp.einsum("k,kqijlm,qml->kij", weights, rep_tensor, rdm1, precision=precision)
    return v_k
    

@jaxtyped
@typechecked
@partial(jit, static_argnames=["precision"])
def coulomb_energy(
    rdm1: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    rep_tensor: Complex[Array, "n_kpt n_kpt n_orbitals n_orbitals n_orbitals n_orbitals"],
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision=Precision.HIGHEST
) -> Scalar:
    """
    Compute the Coulomb energy
    
    Parameters
    ----------
    rdm1 : Complex[Array, "n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix. Spin has been summed over before input.
    rep_tensor : Complex[Array, "n_kpt n_kpt n_orbitals n_orbitals n_orbitals n_orbitals"]
        The repulsion tensor computed on a grid of nkpt x nkpt
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
    """
    v_k = coulomb_potential(rdm1, rep_tensor, weights, precision)
    coulomb_energy = jnp.einsum("k,kij,kji->", weights, rdm1, v_k)/2.0
    return coulomb_energy.real

@jaxtyped
@typechecked
@partial(jit, static_argnames=["precision"])
def non_xc(
    rdm1: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    h1e: Complex[Array, "n_kpt n_orbitals n_orbitals"],
    rep_tensor: Complex[Array, "n_kpt n_kpt n_orbitals n_orbitals n_orbitals n_orbitals"],
    nuclear_repulsion: Scalar,
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision=Precision.HIGHEST,
) -> Scalar:
    r"""Compute all terms in the KS total energy with the exception of the XC component

    Parameters
    ----------
    rdm1 : Complex[Array, "n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix. Spin has been summed over before input.
    h1e : Complex[Array, "n_kpt orbitals orbitals"]
        The 1-electron Hamiltonian for each k-point.
        Equivalent to mf.get_hcore(mf.mol) in pyscf.
    rep_tensor : Complex[Array, "n_kpt n_kpt n_orbitals n_orbitals n_orbitals n_orbitals"]
        The repulsion tensor computed on a grid of nkpt x nkpt
    nuclear_repulsion : Scalar
        The nuclear repulsion energy.
        Equivalent to mf.mol.energy_nuc() in pyscf.
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
    """
    kinetic_and_external = one_body_energy(rdm1, h1e, weights, precision)
    # jax.debug.print("h1e_energy is {x}", x=h1e_energy)
    coulomb = coulomb_energy(rdm1, rep_tensor, weights, precision)
    # jax.debug.print("coulomb2e_energy is {x}", x=coulomb2e_energy)
    # jax.debug.print("nuclear_repulsion is {x}", x=nuclear_repulsion)

    return nuclear_repulsion + kinetic_and_external + coulomb


@jaxtyped
@typechecked
@partial(jit, static_argnames=["precision"])
def make_rdm1(
    mo_coeff: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"],
    mo_occ: Float[Array, "n_spin n_kpt n_orbitals"],
    precision=Precision.HIGHEST
) -> Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]:
    r"""
    One-body reduced density matrix for each k-point in AO representation

    Parameters:
    ----------
        mo_coeff : Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
            Spin-orbital coefficients for each k-point.
        mo_occ : Float[Array, "n_spin n_kpt n_orbitals"]
            Spin-orbital occupancies for each k-point.

    Returns:
    -------
        Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
    """

    return jnp.einsum("skij,skj,sklj -> skil", mo_coeff, mo_occ, mo_coeff.conj(), precision=precision)


@jaxtyped
@typechecked
def get_occ(
    mo_energies: Float[Array, "n_spin n_kpt n_orbitals"],
    nelecs: Int[Array, "spin"],
) -> Float[Array, "n_spin n_kpt n_orbitals"]:
    r"""Compute the occupations of the molecular orbitals for each
    spin and k-point.

    Parameters
    ----------
    mo_energies : Float[Array, "n_spin n_kpt n_orbitals"]
        The molecular orbital energies.
    nelecs : Int[Array, "n_spin"]
        The number of electrons in each spin channel.

    Returns
    -------
    Int[Array, "n_spin n_kpt n_orbitals"]
    """
    nkpt = mo_energies.shape[1]
    nmo = mo_energies.shape[2]
    def get_occ_spin_k_pair(mo_energy_spin_k, nelec_spin, nmo):
        sorted_indices = jnp.argsort(mo_energy_spin_k)

        mo_occ = jnp.zeros_like(mo_energy_spin_k)

        def assign_values(i, mo_occ):
            value = cond(jnp.less(i, nelec_spin), lambda _: 1, lambda _: 0, operand=None)
            idx = sorted_indices[i]
            mo_occ = mo_occ.at[idx].set(value)
            return mo_occ

        mo_occ = fori_loop(0, nmo, assign_values, mo_occ)

        return mo_occ
    

    mo_occ = jnp.stack(
        jnp.asarray([[get_occ_spin_k_pair(mo_energies[s, k], jnp.int64(nelecs[s]), nmo) for k in range(nkpt)] for s in range(2)]), axis=0
    )

    return mo_occ


"""
Note: while the below functions related to the density and it's gradients take a k-point weights parameter,
modification is needed before they support unequal weights as they would appear in a symmetry adapted code. I.e, 
the whole 1BZ need to be considered which would involve use of rotation matrices to map 1RDM's in the IBZ to the full
1BZ. 
"""

@jaxtyped
@typechecked
@partial(jit, static_argnames="precision")
def density(rdm1: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"], 
            ao: Complex[Array, "n_kpt n_flat_grid n_orbitals"], 
            weights: Float[Array, "n_kpts_or_n_ir_kpts"],
            precision: Precision = Precision.HIGHEST
) -> Float[Array, "n_flat_grid n_spin"]:
    r""" Calculates electronic density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix.
    ao : Complex[Array, "n_kpt n_flat_grid n_orbitals"]
        Crystal atomic orbitals.
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Float[Array, "n_flat_grid n_spin"]
    """
    den = jnp.einsum("k,skab,kra,krb->rs", weights, rdm1, ao, ao, precision=precision).real
    # den = jnp.einsum("...kab,ra,rb->r...", rdm1, ao, ao, precision=precision)
    print(jnp.sum(den.imag))
    jax.debug.print("imag remainder is {x}", x=jnp.sum(den.imag))
    return den
@jaxtyped
@typechecked
@partial(jit, static_argnames="precision")
def grad_density(
    rdm1: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"], 
    ao: Complex[Array, "n_kpt n_flat_grid n_orbitals"], 
    grad_ao: Complex[Array, "n_kpt n_flat_grid n_orbitals 3"], 
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision: Precision = Precision.HIGHEST
) -> Float[Array, "n_flat_grid n_spin 3"]:
    r"""Compute the electronic density gradient using crystal atomic orbitals.

    Parameters
    ----------
    rdm1 : Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix.
    ao : Complex[Array, "n_kpt n_flat_grid n_orbitals"]
        Crystal atomic orbitals.
    grad_ao : Complex[Array, "n_kpt n_flat_grid n_orbitals 3"]
        Gradients of crystal atomic orbitals.
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density gradient: Float[Array, "n_flat_grid n_spin 3"]
    """

    return 2 * jnp.einsum("k,...kab,kra,krbj->r...j", weights, rdm1, ao, grad_ao, precision=precision).real

@jaxtyped
@typechecked
@partial(jit, static_argnames="precision")
def lapl_density(
    rdm1: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"], 
    ao: Complex[Array, "n_kpt n_flat_grid n_orbitals"], 
    grad_ao: Complex[Array, "n_kpt n_flat_grid n_orbitals 3"], 
    grad_2_ao: Complex[Array, "n_kpt n_flat_grid n_orbitals 3"],
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision: Precision = Precision.HIGHEST,
) -> Float[Array, "n_flat_grid n_spin"]:
    r"""Compute the laplacian of the electronic density.

    Parameters
    ----------
    rdm1 : Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix.
    ao : Complex[Array, "b_flat_grid n_orbitals"]
        Crystal atomic orbitals.
    grad_ao : Complex[Array, "n_flat_grid n_orbitals 3"]
        Gradients of crystal atomic orbitals.
    grad_2_ao : Complex[Array, "n_flat_grid n_orbitals 3"]
        Vector of second derivatives of crystal atomic orbitals.
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Float[Array, "n_flat_grid n_spin"]
    """
    return (2 * jnp.einsum(
        "k,...kab,kraj,krbj->r...", weights, rdm1, grad_ao, grad_ao, precision=precision
    ) + 2 * jnp.einsum("k,...kab,kra,krbi->r...", weights, rdm1, ao, grad_2_ao, precision=precision)).real

@jaxtyped
@typechecked
@partial(jit, static_argnames="precision")
def kinetic_density(
    rdm1 : Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"],
    grad_ao: Complex[Array, "n_kpt n_flat_grid n_orbitals 3"],
    weights: Float[Array, "n_kpts_or_n_ir_kpts"],
    precision: Precision = Precision.HIGHEST
) -> Float[Array, "n_flat_grid n_spin"]:
    r""" Compute the kinetic energy density using crystal atomic orbitals.

    Parameters
    ----------
    rdm1 : Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
        The 1-body reduced density matrix.
    grad_ao : Complex[Array, "n_kpt n_flat_grid n_orbitals 3"]
        Gradients of crystal atomic orbitals.
    weights : Float[Array, "n_kpts_or_n_ir_kpts"]
        The weights for each k-point which together sum to 1. If we are working
        in the full 1BZ, weights are equal. If we are working in the
        irreducible 1BZ, weights may not be equal if symmetry can be 
        exploited.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jaxx.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The kinetic energy density: Float[Array, "n_flat_grid n_spin"]
    """

    return 0.5 * jnp.einsum("k,...kab,kraj,krbj->r...", weights, rdm1, grad_ao, grad_ao, precision=precision).real

@jaxtyped
@typechecked
@partial(jit, static_argnames="precision")
def orbital_grad(
        mo_coeff: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"],
        mo_occ: Float[Array, "n_spin n_kpt n_orbitals"],
        fock: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"],
        precision: Precision = Precision.HIGHEST
    ) -> Float[Array, "n_kpt n_orbitals n_orbitals"]:
    r"""Compute the gradient of the electronic energy with respect 
    to the molecular orbital coefficients.

    Parameters:
    ----------
        mo_coeff: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
            Orbital coefficients
        mo_occ: Float[Array, "n_spin n_kpt n_orbitals"]
            Orbital occupancy
        fock: Complex[Array, "n_spin n_kpt n_orbitals n_orbitals"]
            Fock matrix in AO representation
        precision: jax.lax.Precision, optional

    Returns:
    -------
    Float[Array, "n_kpt n_orbitals n_orbitals"]


    Notes:
    -----
    # Performs same task as pyscf/scf/hf.py but we have k-point sampling:
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(jnp.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()
    """

    C_occ = vmap(jnp.where, in_axes=(None, 2, None), out_axes=2)(mo_occ > 0, mo_coeff, 0)
    C_vir = vmap(jnp.where, in_axes=(None, 2, None), out_axes=2)(mo_occ == 0, mo_coeff, 0)

    return jnp.einsum("skab,skac,skcd->kbd", C_vir.conj(), fock, C_occ, precision = precision).real



