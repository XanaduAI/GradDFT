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

from typing import List, Optional, Union, Sequence, Tuple, NamedTuple
from dataclasses import fields

from typeguard import typechecked
from grad_dft.utils import vmap_chunked
from functools import partial

from jax import numpy as jnp
from jax import scipy as jsp
from jax.lax import Precision
from jax import vmap, grad
from jax.lax import fori_loop, cond
from flax import struct
from flax import linen as nn
import jax 

from jaxtyping import Array, PyTree, Scalar, Float, Int, jaxtyped


@struct.dataclass
class Grid:
    r""" Base class for the grid coordinates and integration grids."""
    coords: Array
    weights: Array

    # def __repr__(self):
    #    return f"{self.__class__.__name__}(size={len(self)})"

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
class Molecule:
    r""" Base class for storing molecule properties and methods."""

    grid: Grid
    atom_index: Int[Array, "atom"]
    nuclear_pos: Float[Array, "atom 3"]
    ao: Float[Array, "grid orbitals"]
    grad_ao: Float[Array, "grid orbitals 3"]
    grad_n_ao: PyTree
    rdm1: Float[Array, "spin orbitals orbitals"]
    nuclear_repulsion: Scalar
    h1e: Float[Array, "orbitals orbitals"]
    vj: Float[Array, "spin orbitals orbitals"]
    mo_coeff: Float[Array, "spin orbitals orbitals"]
    mo_occ: Float[Array, "spin orbitals"]
    mo_energy: Float[Array, "spin orbitals"]
    mf_energy: Optional[Scalar] = None
    s1e: Optional[Float[Array, "orbitals orbitals"]] = None  # Not used during training
    omegas: Optional[Float[Array, "omega"]] = None
    chi: Optional[Float[Array, "grid omega spin orbitals"]] = None
    rep_tensor: Optional[Float[Array, "orbitals orbitals orbitals orbitals"]] = None
    energy: Optional[Scalar] = None
    basis: Optional[Int[Array, '...']] = None # The name is saved as a list of integers, JAX does not accept str
    name: Optional[Int[Array, '...']] = None # The name is saved as a list of integers, JAX does not accept str
    spin: Optional[Scalar] = 0
    charge: Optional[Scalar] = 0
    unit_Angstrom: Optional[bool] = True
    grid_level: Optional[Scalar] = 2
    scf_iteration: Optional[Scalar] = 50
    fock: Optional[Array] = None

    @property
    def grid_size(self):
        return len(self.grid)

    def density(self, *args, **kwargs) -> Array:
        r""" Computes the electronic density of a molecule at each grid point.
        
        Returns
        -------
        Float[Array, "grid spin"]
        """
        return density(self.rdm1, self.ao, *args, **kwargs)

    def grad_density(self, *args, **kwargs) -> Array:
        r""" Computes the gradient of the electronic density of a molecule at each grid point.
        
        Returns
        -------
        Float[Array, "grid spin 3"]
        """
        return grad_density(self.rdm1, self.ao, self.grad_ao, *args, **kwargs)

    def lapl_density(self, *args, **kwargs) -> Array:
        r""" Computes the laplacian of the electronic density of a molecule at each grid point.
        
        Returns
        -------
        Float[Array, "grid spin"]
        """
        return lapl_density(self.rdm1, self.ao, self.grad_ao, self.grad_n_ao[2], *args, **kwargs)

    def kinetic_density(self, *args, **kwargs) -> Array:
        r""" Computes the kinetic energy density of a molecule at each grid point.
        
        Returns
        -------
        Float[Array, "grid spin"]
        """
        return kinetic_density(self.rdm1, self.grad_ao, *args, **kwargs)

    def select_HF_omegas(self, omegas: Float[Array, "omega"]) -> Array:
        r""" Selects the chi tensor according to the omegas passed.

        Parameters
        ----------
        omegas: Float[Array, "omega"]
            The parameter omega with which the kernel
            .. math::
                f(|r-r'|) = \erf(\omega|r-r'|)/|r-r'|

            has been computed.

        Returns
        ----------
        chi: Float[Array, "grid omega spin orbitals"]

        .. math::
            Xc^{ω,σ} = Γbd^σ ψb(r) ∫ dr' f_{ω}(|r-r'|) ψc(r') ψd(r') = Γbd^σ ψb(r) v_{cd}(r)

        """

        if self.chi is None:
            raise ValueError("Precomputed chi tensor has not been loaded.")
        for o in omegas:
            if o not in self.omegas:
                raise ValueError(
                    f"The molecule.chi tensor does not contain omega value {o}, only {self.omegas}"
                )
        indices = [list(self.omegas).index(o) for o in omegas]
        chi = jnp.stack([self.chi[:, i] for i in indices], axis=1)
        return chi

    def HF_energy_density(self, omegas: Float[Array, "omega"], *args, **kwargs) -> Array:
        r""" Computes the Hartree-Fock energy density of a molecule at each grid point,
        for a given set of omegas in the range-separated Coulomb kernel.

        Parameters
        ----------
        omegas: Float[Array, "omega"]
        The parameter omega with which the kernel
            .. math::
                f(|r-r'|) = \erf(\omega|r-r'|)/|r-r'|

            has been computed.


        Returns
        -------
        Float[Array, "grid omega spin"]
        """
        chi = self.select_HF_omegas(omegas)
        return HF_energy_density(self.rdm1, self.ao, chi, *args, **kwargs)

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
        r""" Computes the Hartree-Fock matrix contribution due to the partial derivative
        with respect to the Hartree Fock energy energy density.

        Parameters
        ----------
        functional : nn.Module
            The functional.
        params : PyTree
            The parameters of the neural network.
        omegas : Float[Array, "omega"]
            The parameter omega with which the kernel.
        ehf : Float[Array, "grid omega spin"]
            The Hartree-Fock energy density.
        cinputs : Float[Array, "grid cinputs"]
            Contains a list of the features to input to fxc, potentially including the HF density.
        densities_wout_hf: Float[Array, "grid densities_w"]
            Contains a list of the features to input to fxc, excluding the HF density.

        Returns
        -------
        Float[Array, "omega spin orbitals orbitals"]
        """

        chi = self.select_HF_omegas(omegas)
        return HF_density_grad_2_Fock(
            self.grid,
            functional,
            params,
            chi,
            self.ao,
            ehf,
            coefficient_inputs,
            densities_wout_hf,
            **kwargs
        )

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
        r""" Computes the Hartree-Fock matrix contribution due to the partial derivative
        with respect to the Hartree Fock energy energy density.

        Parameters
        ----------
        functional : nn.Module
            The functional.
        params : PyTree
            The parameters of the neural network.
        omegas : Float[Array, "omega"]
            The parameter omega with which the kernel.
        ehf : Float[Array, "grid omega spin"]
            The Hartree-Fock energy density.
        cinputs_wout_hf : Float[Array, "grid cinputs_w"]
            Contains a list of the features to input to fxc, excluding the HF density.
        densities: Float[Array, "grid densities"]
            Contains a list of the features to input to fxc, potentially including the HF density.

        Returns
        -------
        Float[Array, "omega spin orbitals orbitals"]
        """

        chi = self.select_HF_omegas(omegas)
        return HF_coefficient_input_grad_2_Fock(
            self.grid, 
            functional, 
            params, 
            chi, 
            self.ao, 
            ehf, 
            cinputs_wout_hf, 
            densities, 
            **kwargs
        )

    def nonXC(self, *args, **kwargs) -> Scalar:
        r""" Computes the non-XC energy of a DFT functional."""
        return nonXC(self.rdm1.sum(axis = 0), self.h1e, self.rep_tensor, self.nuclear_repulsion, *args, **kwargs)

    def make_rdm1(self) -> Array:
        r""" Computes the 1-Reduced Density Matrix for the molecule.
        
        Returns
        -------
        Float[Array, "spin orbitals orbitals"]
        """
        return make_rdm1(self.mo_coeff, self.mo_occ)

    def get_occ(self) -> Array:
        r""" Computes the orbital occupancy for the molecule.
        
        Returns
        -------
        Float[Array, "spin orbitals"]
        """
        nelecs = jnp.array([self.mo_occ[i].sum() for i in range(2)], dtype=jnp.int64)
        naos = self.mo_occ.shape[1]
        return get_occ(self.mo_energy, nelecs, naos)

    def to_dict(self) -> dict:
        r""" Returns a dictionary with the attributes of the molecule."""
        grid_dict = self.grid.to_dict()
        rest = {field.name: getattr(self, field.name) for field in fields(self)[1:]}
        return dict(**grid_dict, **rest)


#######################################################################

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames="precision")
def orbital_grad(
        mo_coeff: Float[Array, "spin orbitals orbitals"],
        mo_occ: Float[Array, "spin orbitals"],
        F: Float[Array, "spin orbitals orbitals"],
        precision: Precision = Precision.HIGHEST
    ) -> Float[Array, "orbitals orbitals"]:
    r""" Computes the restricted Hartree Fock orbital gradients

    Parameters:
    ----------
        mo_coeff: Float[Array, "spin orbitals orbitals"]
            Orbital coefficients
        mo_occ: Float[Array, "spin orbitals"]
            Orbital occupancy
        F: Float[Array, "spin orbitals orbitals"]
            Fock matrix in AO representation
        precision: jax.lax.Precision, optional

    Returns:
    -------
    Float[Array, "orbitals orbitals"]


    Notes:
    -----
    # Similar to pyscf/scf/hf.py:
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(jnp.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()
    """

    C_occ = jax.vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(mo_occ > 0, mo_coeff, 0)
    C_vir = jax.vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(mo_occ == 0, mo_coeff, 0)

    return jnp.einsum("sab,sac,scd->bd", C_vir.conj(), F, C_occ, precision = precision)


##########################################################
@jaxtyped
@typechecked
@partial(jax.jit, static_argnames="precision")
def density(rdm1: Float[Array, "spin orbitals orbitals"], 
            ao: Float[Array, "grid orbitals"], 
            precision: Precision = Precision.HIGHEST
) -> Float[Array, "grid spin"]:
    r""" Calculates electronic density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The density matrix.
    ao : Float[Array, "grid orbitals"]
        Atomic orbitals.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Float[Array, "grid spin"]
    """

    return jnp.einsum("...ab,ra,rb->r...", rdm1, ao, ao, precision=precision)

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames="precision")
def grad_density(
    rdm1: Float[Array, "spin orbitals orbitals"], 
    ao: Float[Array, "grid orbitals"], 
    grad_ao: Float[Array, "grid orbitals 3"], 
    precision: Precision = Precision.HIGHEST
) -> Float[Array, "grid spin 3"]:
    r""" Calculate the electronic density gradient from atomic orbitals.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The density matrix.
    ao : Float[Array, "grid orbitals"]
        Atomic orbitals.
    grad_ao : Float[Array, "grid orbitals 3"]
        Gradients of atomic orbitals.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density gradient. Shape: (n_grid_points, n_spin, 3)
    """

    return 2 * jnp.einsum("...ab,ra,rbj->r...j", rdm1, ao, grad_ao, precision=precision)

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames="precision")
def lapl_density(
    rdm1: Float[Array, "spin orbitals orbitals"], 
    ao: Float[Array, "grid orbitals"], 
    grad_ao: Float[Array, "grid orbitals 3"], 
    grad_2_ao: Float[Array, "grid orbitals 3"],
    precision: Precision = Precision.HIGHEST,
) -> Float[Array, "grid spin"]:
    r""" Calculates the laplacian of the electronic density.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The density matrix.
    ao : Float[Array, "grid orbitals"]
        Atomic orbitals.
    grad_ao : Float[Array, "grid orbitals 3"]
        Gradients of atomic orbitals.
    grad_2_ao : Float[Array, "grid orbitals 3"]
        Vector of second derivatives of atomic orbitals.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Float[Array, "grid spin"]
    """
    return 2 * jnp.einsum(
        "...ab,raj,rbj->r...", rdm1, grad_ao, grad_ao, precision=precision
    ) + 2 * jnp.einsum("...ab,ra,rbi->r...", rdm1, ao, grad_2_ao, precision=precision)

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames="precision")
def kinetic_density(
    rdm1: Float[Array, "spin orbitals orbitals"], 
    grad_ao: Float[Array, "grid orbitals 3"],
    precision: Precision = Precision.HIGHEST
) -> Float[Array, "grid spin"]:
    r""" Calculate kinetic energy density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The density matrix.
    grad_ao : Float[Array, "grid orbitals 3"]
        Gradients of atomic orbitals.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jaxx.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The kinetic energy density. Shape: (n_spin, n_grid_points)
    """

    return 0.5 * jnp.einsum("...ab,raj,rbj->r...", rdm1, grad_ao, grad_ao, precision=precision)

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames=["precision"])
def HF_energy_density(
    rdm1: Float[Array, "spin orbitals orbitals"],
    ao: Float[Array, "grid orbitals"],
    chi: Float[Array, "grid omega spin orbitals"],
    precision: Precision = Precision.HIGHEST
) -> Float[Array, "omega spin grid"]:
    r""" Calculate the Hartree-Fock energy density.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The density matrix.
    ao : Float[Array, "grid orbitals"]
        Atomic orbitals.
    chi : Float[Array, "grid omega spin orbitals"]
        Precomputed chi density.

        .. math::
            Xc^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψc(r') ψd(r') = Γbd^σ ψb(r) v_{cd}(r)

        where :math:`v_{ab}(r) = ∫ dr' f(|r-r'|) ψa(r') ψd(r')`.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.

    Returns
    -------
    Float[Array, "omega spin grid"]
    """

    # return - jnp.einsum("rwsc,sac,ra->wsr", chi, rdm1, ao, precision=precision)/2
    _hf_energy = (
        lambda _chi, _rdm1, _ao: - jnp.einsum("wsc,sac,a->ws", _chi, _rdm1, _ao, precision=precision)
        / 2
    )
    return vmap(_hf_energy, in_axes=(0, None, 0), out_axes=2)(chi, rdm1, ao)

@jaxtyped
@typechecked
def HF_density_grad_2_Fock(
    grid: Grid,
    functional: nn.Module,
    params: PyTree,
    chi: Float[Array, "grid omega spin orbitals"],
    ao: Float[Array, "grid orbitals"],
    ehf: Float[Array, "omega spin grid"],
    coefficient_inputs: Optional[Float[Array, "grid cinputs"]],
    densities_wout_hf: Float[Array, "grid densities_w"],
    chunk_size: Optional[Int] = None,
    precision: Precision = Precision.HIGHEST
) -> Float[Array, "omega spin orbitals orbitals"]:
    r""" Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock energy density.

    .. math::
        F = ∂f(x)/∂e_{HF}^{ωσ} * ∂e_{HF}^{ωσ}/∂Γab^σ = ∂f(x)/∂e_{HF}^{ωσ} * ψa(r) Xb^σ.

    Parameters
    ----------
    grid: Grid
    functional : Callable
        Functional object.
    params: PyTree
        The parameters of the neural network.
    chi : Float[Array, "grid omega spin orbitals"]
        .. math::
            Xa^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψa(r') ψd(r')

        Expected shape: (n_grid_points, n_omegas, n_spin, n_orbitals)
    ao : Float[Array, "grid orbitals"]
        The orbitals.
    ehf : Float[Array, "grid omega spin"]
        The Hartree-Fock energy density.
    coefficient_inputs: Float[Array, "grid cinputs"]
        The inputs to the coefficients function in the functional.
    densities_wout_hf: Float[Array, "grid densities_w"]
        The energy densities excluding the Hartree-Fock energy density.
    chunk_size : int, optional
        The batch size for the number of atomic orbitals the integral
        evaluation is looped over. For a grid of N points, the solution
        formally requires the construction of a N x N matrix in an intermediate
        step. If `chunk_size` is given, the calculation is broken down into
        smaller subproblems requiring construction of only chunk_size x N matrices.
        Practically, higher `chunk_size`s mean faster calculations with larger
        memory requirements and vice-versa.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Float[Array, "omega spin orbitals orbitals"]
    """

    def partial_fxc(params, grid, ehf, coefficient_inputs, densities_wout_hf):
        densities = functional.combine_densities(densities_wout_hf, ehf)
        return functional.xc_energy(params, grid, coefficient_inputs, densities)

    gr = grad(partial_fxc, argnums=2)(params, grid, ehf, coefficient_inputs, densities_wout_hf)

    @partial(vmap_chunked, in_axes=(0, None, None), chunk_size=chunk_size)
    def chunked_jvp(chi_tensor, gr_tensor, ao_tensor):
        return (
            -jnp.einsum("rws,wsr,ra->wsa", chi_tensor, gr_tensor, ao_tensor, precision=precision)
            / 2.0
        )

    return (jax.jit(chunked_jvp)(chi.transpose(3, 0, 1, 2), gr, ao)).transpose(1, 2, 3, 0)

@jaxtyped
@typechecked
def HF_coefficient_input_grad_2_Fock(
    grid: Grid,
    functional: nn.Module,
    params: PyTree,
    chi: Float[Array, "grid omega spin orbitals"],
    ao: Float[Array, "grid orbitals"],
    ehf: Float[Array, "omega spin grid"],
    cinputs_wout_hf: Optional[Float[Array, "grid cinputs_w"]],
    densities: Float[Array, "grid densities"],
    chunk_size: Optional[int] = None,
    precision: Precision = Precision.HIGHEST,
) -> Float[Array, "omega spin orbitals orbitals"]:
    r""" Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock coefficient inputs.

    .. math::
        F = ∂f(x)/∂e_{HF}^{ωσ} * ∂e_{HF}^{ωσ}/∂Γab^σ = ∂f(x)/∂e_{HF}^{ωσ} * ψa(r) Xb^σ.

    Parameters
    ----------
    grid: Grid
    functional : Callable
        Functional object.
    params: PyTree
        The parameters of the neural network.
    chi : Float[Array, "grid omega spin orbitals"]
        .. math::
            Xa^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψa(r') ψd(r')

        Expected shape: (n_grid_points, n_omegas, n_spin, n_orbitals)
    ao : Float[Array, "grid orbitals"]
        The orbitals.
    ehf : Float[Array, "grid omega spin"]
        The Hartree-Fock energy density.
    cinputs_wout_hf: Float[Array, "grid cinputs"]
        The inputs to the coefficients function in the functional, excluding the Hartree-Fock contributions.
    densities: Float[Array, "grid densities_w"]
        The energy densities.
    chunk_size : int, optional
        The batch size for the number of atomic orbitals the integral
        evaluation is looped over. For a grid of N points, the solution
        formally requires the construction of a N x N matrix in an intermediate
        step. If `chunk_size` is given, the calculation is broken down into
        smaller subproblems requiring construction of only chunk_size x N matrices.
        Practically, higher `chunk_size`s mean faster calculations with larger
        memory requirements and vice-versa.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Float[Array, "omega spin orbitals orbitals"]
    """

    def partial_fxc(params, grid, ehf, coefficient_inputs_wout_hf, densities):
        coefficient_inputs = functional.combine_inputs(coefficient_inputs_wout_hf, ehf)
        return functional.xc_energy(params, grid, coefficient_inputs, densities)

    gr = grad(partial_fxc, argnums=2)(params, grid, ehf, cinputs_wout_hf, densities)

    @partial(vmap_chunked, in_axes=(0, None, None), chunk_size=chunk_size)
    def chunked_jvp(chi_tensor, gr_tensor, ao_tensor):
        return (
            -jnp.einsum("rws,wsr,ra->wsa", chi_tensor, gr_tensor, ao_tensor, precision=precision)
            / 2.0
        )

    return (jax.jit(chunked_jvp)(chi.transpose(3, 0, 1, 2), gr, ao)).transpose(1, 2, 3, 0)

def abs_clip(arr, threshold):
    r"""If the absolute value of an array is below a threshold, set it to zero."""
    return jnp.where(jnp.abs(arr) > threshold, arr, 0)


######################################################################

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames=["precision"])
def nonXC(
    rdm1: Float[Array, "orbitals orbitals"],
    h1e: Float[Array, "orbitals orbitals"],
    rep_tensor: Float[Array, "orbitals orbitals orbitals orbitals"],
    nuclear_repulsion: Scalar,
    precision=Precision.HIGHEST,
) -> Scalar:
    r""" A function that computes the non-XC part of a DFT functional.

    Parameters
    ----------
    rdm1 : Float[Array, "orbitals orbitals"]
        The 1-Reduced Density Matrix.
        Equivalent to mf.make_rdm1() in pyscf.
    h1e : Float[Array, "orbitals orbitals"]
        The 1-electron Hamiltonian.
        Equivalent to mf.get_hcore(mf.mol) in pyscf.
    rep_tensor : Float[Array, "orbitals orbitals orbitals orbitals"]
        The repulsion tensor.
        Equivalent to mf.mol.intor('int2e') in pyscf.
    nuclear_repulsion : Scalar
        The nuclear repulsion energy.
        Equivalent to mf.mol.energy_nuc() in pyscf.
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
    """
    h1e_energy = one_body_energy(rdm1, h1e, precision)
    jax.debug.print("h1e_energy is {x}", x=h1e_energy)
    coulomb2e_energy = coulomb_energy(rdm1, rep_tensor, precision)
    jax.debug.print("coulomb2e_energy is {x}", x=coulomb2e_energy)
    jax.debug.print("nuclear_repulsion is {x}", x=nuclear_repulsion)

    return nuclear_repulsion + h1e_energy + coulomb2e_energy

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames=["precision"])
def one_body_energy(
    rdm1: Float[Array, "orbitals orbitals"],
    h1e: Float[Array, "orbitals orbitals"],
    precision=Precision.HIGHEST,
) -> Scalar:
    r"""A function that computes the one-body energy of a DFT functional.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The 1-Reduced Density Matrix.
    h1e : Float[Array, "orbitals orbitals"]
        The 1-electron Hamiltonian.

    Returns
    -------
    Scalar
    """
    h1e_energy = jnp.einsum("ij,ij->", rdm1, h1e, precision=precision)
    return h1e_energy


@jaxtyped
@typechecked
@partial(jax.jit, static_argnames=["precision"])
def coulomb_energy(
    rdm1: Float[Array, "orbitals orbitals"],
    rep_tensor: Float[Array, "orbitals orbitals orbitals orbitals"],
    precision=Precision.HIGHEST,
) -> Scalar:
    r"""A function that computes the Coulomb two-body energy of a DFT functional.
    
    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The 1-Reduced Density Matrix.
    rep_tensor : Float[Array, "orbitals orbitals orbitals orbitals"]
        The repulsion tensor.

    Returns
    -------
    Scalar
    """
    v_coul = coulomb_potential(rdm1, rep_tensor, precision)
    coulomb2e_energy = jnp.einsum("pq,pq->", rdm1, v_coul, precision=precision) / 2.0
    return coulomb2e_energy

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames=["precision"])
def coulomb_potential(
    rdm1: Float[Array, "orbitals orbitals"],
    rep_tensor: Float[Array, "orbitals orbitals orbitals orbitals"],
    precision=Precision.HIGHEST,
) -> Float[Array, "orbitals orbitals"]:
    r"""
    A function that computes the Coulomb potential matrix.

    Parameters
    ----------
    rdm1 : Float[Array, "spin orbitals orbitals"]
        The 1-Reduced Density Matrix.
        Equivalent to mf.make_rdm1() in pyscf.
    rep_tensor : Float[Array, "orbitals orbitals orbitals orbitals"]
        The repulsion tensor.
        Equivalent to mf.mol.intor('int2e') in pyscf.
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Float[Array, "spin orbitals orbitals"]
    """
    return jnp.einsum("pqrt,rt->pq", rep_tensor, rdm1, precision=precision)

@jaxtyped
@typechecked
@partial(jax.jit, static_argnames=["precision"])
def make_rdm1(
    mo_coeff: Float[Array, "spin orbitals orbitals"],
    mo_occ: Float[Array, "spin orbitals"],
    precision=Precision.HIGHEST
) -> Float[Array, "spin orbitals orbitals"]:
    r"""
    One-particle density matrix in AO representation

    Parameters:
    ----------
        mo_coeff : Float[Array, "spin orbitals orbitals"]
            Spin-orbital coefficients.
        mo_occ : Float[Array, "spin orbitals"]
            Spin-orbital occupancies.

    Returns:
    -------
        Float[Array, "spin orbitals orbitals"]

    Notes:
    -----
    # Pyscf code
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = jnp.dot(mo_a*mo_occ[0], mo_a.conj().T)
    dm_b = jnp.dot(mo_b*mo_occ[1], mo_b.conj().T)

    return jnp.array((dm_a, dm_b))
    """

    return jnp.einsum("sij,sj,skj -> sik", mo_coeff, mo_occ, mo_coeff.conj(), precision=precision)

@jaxtyped
@typechecked
@jax.jit
def get_occ(
    mo_energies: Float[Array, "spin orbitals"],
    nelecs: Int[Array, "spin"],
    naos: int,
) -> Float[Array, "spin orbitals"]:
    r""" Computes the occupation of the molecular orbitals.

    Parameters
    ----------
    mo_energies : Float[Array, "spin orbitals"]
        The molecular orbital energies.
    nelecs : Int[Array, "spin"]
        The number of electrons in each spin channel.
    naos : Int

    Returns
    -------
    Int[Array, "spin orbitals"]
    """
    def get_occ_spin(mo_energy, nelec_spin, naos):
        sorted_indices = jnp.argsort(mo_energy)

        mo_occ = jnp.zeros_like(mo_energy)

        def assign_values(i, mo_occ):
            value = cond(jnp.less(i, nelec_spin), lambda _: 1, lambda _: 0, operand=None)
            idx = sorted_indices[i]
            mo_occ = mo_occ.at[idx].set(value)
            return mo_occ

        mo_occ = fori_loop(0, naos, assign_values, mo_occ)

        return mo_occ

    mo_occ = jnp.stack(
        [get_occ_spin(mo_energies[s], jnp.int64(nelecs[s]), naos) for s in range(2)], axis=0
    )

    return mo_occ


######################################################################


class Reaction(NamedTuple):
    r""" Base class for storing reactions as set of reactants and ."""
    reactants: Sequence[Molecule]
    products: Sequence[Molecule]
    reactant_numbers: Sequence[int]
    product_numbers: Sequence[int]
    energy: float
    name: Optional[List[int]] = None


def make_reaction(
    reactants: Union[Molecule, Sequence[Molecule]],
    products: Union[Molecule, Sequence[Molecule]],
    reactant_numbers: Optional[Sequence[int]] = None,
    product_numbers: Optional[Sequence[int]] = None,
    energy: Optional[float] = None,
    name: Optional[str] = None,
) -> Reaction:
    r"""
    Parse inputs and make a `Reaction` object.

    Parameters
    ----------
    reactants : Union[Molecule, Sequence[Molecule]]
        A sequence of `Molecule` objects representing reactants.
    products : Union[Molecule, Sequence[Molecule]]
        A sequence of `Molecule` objects representing products.
    reactant_numbers : Sequence[int], optional
        A sequence of reactant multiplicities for the given reaction. Defaults to 1 for each reactant.
    product_numbers : Sequence[int], optional
        A sequence of product multiplicities for the given reaction. Defaults to 1 for each product.

    Returns
    -------
    Reaction
        A `Reaction` NamedTuple.
    """

    reactants, reactant_numbers = _canonicalize_molecules(reactants, reactant_numbers)
    products, product_numbers = _canonicalize_molecules(products, product_numbers)

    return Reaction(reactants, products, reactant_numbers, product_numbers, energy, name)


def _canonicalize_molecules(
    molecules: Union[Molecule, Sequence[Molecule]], numbers: Optional[Sequence[int]]
) -> Tuple[Sequence[Molecule], Sequence[int]]:
    r"""
    Makes sure that the molecules are a sequence and that the numbers are a sequence of the same length.
    """
    if isinstance(molecules, Molecule):
        molecules = (molecules,)

    if numbers is None:
        numbers = (1,) * len(molecules)
    elif not isinstance(numbers, Sequence):
        numbers = (numbers,)

    return molecules, numbers
