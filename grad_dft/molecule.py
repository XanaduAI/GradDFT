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

from typing import Optional, Union, Sequence, Tuple, NamedTuple
from dataclasses import fields
from grad_dft.utils import Array, Scalar, PyTree, vmap_chunked
from functools import partial
from grad_dft.external.eigh_impl import eigh2d

from jax import numpy as jnp
from jax.lax import Precision
from jax import vmap, grad
from jax.lax import fori_loop, cond
from flax import struct
from flax import linen as nn
import jax


@struct.dataclass
class Grid:
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
    r"""
    Base class for storing molecule properties
    and methods
    """

    grid: Grid
    atom_index: Array
    nuclear_pos: Array
    ao: Array
    grad_ao: Array
    grad_n_ao: PyTree
    rdm1: Array
    nuclear_repulsion: Scalar
    h1e: Array
    vj: Array
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Array
    mf_energy: Optional[Scalar] = None
    s1e: Optional[Array] = None  # Not used during training
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

    # def __repr__(self):
    #    return f"{self.__class__.__name__}(grid_size={self.grid_size})"

    @property
    def grid_size(self):
        return len(self.grid)

    def density(self, *args, **kwargs):
        return density(self.rdm1, self.ao, *args, **kwargs)

    def grad_density(self, *args, **kwargs):
        return grad_density(self.rdm1, self.ao, self.grad_ao, *args, **kwargs)

    def lapl_density(self, *args, **kwargs):
        return lapl_density(self.rdm1, self.ao, self.grad_ao, self.grad_n_ao[2], *args, **kwargs)

    def kinetic_density(self, *args, **kwargs):
        return kinetic_density(self.rdm1, self.grad_ao, *args, **kwargs)

    def select_HF_omegas(self, omegas):
        r"""
        Selects the chi tensor according to the omegas passed.
        self.chi is ordered according to self.omegas in dimension = 1.

        Parameters
        ----------
        omegas: List[float]
            The parameter omega with which the kernel
            .. math::
                f(|r-r'|) = \erf(\omega|r-r'|)/|r-r'|

            has been computed.

        Returns
        ----------
        chi: Array

        .. math::
            Xc^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψc(r') ψd(r') = Γbd^σ ψb(r) v_{cd}(r)

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

    def HF_energy_density(self, omegas, *args, **kwargs):
        chi = self.select_HF_omegas(omegas)
        return HF_energy_density(self.rdm1, self.ao, chi, *args, **kwargs)

    def HF_density_grad_2_Fock(
        self,
        functional: nn.Module,
        params: PyTree,
        omegas: Array,
        ehf: Array,
        coefficient_inputs: Array,
        densities_wout_hf: Array,
        **kwargs,
    ):
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
            **kwargs,
        )

    def HF_coefficient_input_grad_2_Fock(
        self,
        functional: nn.Module,
        params: PyTree,
        omegas: Array,
        ehf: Array,
        cinputs_wout_hf: Array,
        densities: Array,
        **kwargs,
    ):
        chi = self.select_HF_omegas(omegas)
        return HF_coefficient_input_grad_2_Fock(
            self.grid, functional, params, chi, self.ao, ehf, cinputs_wout_hf, densities, **kwargs
        )

    def nonXC(self, *args, **kwargs):
        return nonXC(self.rdm1, self.h1e, self.rep_tensor, self.nuclear_repulsion, *args, **kwargs)

    def make_rdm1(self):
        return make_rdm1(self.mo_coeff, self.mo_occ)

    def get_occ(self):
        nelecs = [self.mo_occ[i].sum() for i in range(2)]
        naos = self.mo_occ.shape[1]
        return get_occ(self.mo_energy, nelecs, naos)

    def to_dict(self) -> dict:
        grid_dict = self.grid.to_dict()
        rest = {field.name: getattr(self, field.name) for field in fields(self)[1:]}
        return dict(**grid_dict, **rest)


#######################################################################


def orbital_grad(mo_coeff, mo_occ, F):
    r"""
    RHF orbital gradients

    Args:
        mo_coeff: 2D ndarray
            Orbital coefficients
        mo_occ: 1D ndarray
            Orbital occupancy
        F: 2D ndarray
            Fock matrix in AO representation

    Returns:
        Gradients in MO representation.  It's a num_occ*num_vir vector.

    # Similar to pyscf/scf/hf.py:
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(jnp.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()
    """

    C_occ = jax.vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(mo_occ > 0, mo_coeff, 0)
    C_vir = jax.vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(mo_occ == 0, mo_coeff, 0)

    return jnp.einsum("sab,sac,scd->bd", C_vir.conj(), F, C_occ)


##########################################################


@partial(jax.jit, static_argnames="precision")
def density(rdm1: Array, ao: Array, precision: Precision = Precision.HIGHEST) -> Array:
    r"""
    Calculate electronic density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density. Shape: (n_spin, n_grid_points)
    """

    return jnp.einsum("...ab,ra,rb->r...", rdm1, ao, ao, precision=precision)


@partial(jax.jit, static_argnames="precision")
def grad_density(
    rdm1: Array, ao: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST
) -> Array:
    r"""
    Calculate the electronic density gradient from atomic orbitals.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (n_spin, n_orbitals, n_orbitals)
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
        The density gradient. Shape: (n_spin, n_grid_points, 3)
    """

    return 2 * jnp.einsum("...ab,ra,rbj->r...j", rdm1, ao, grad_ao, precision=precision)


@partial(jax.jit, static_argnames="precision")
def lapl_density(
    rdm1: Array,
    ao: Array,
    grad_ao: Array,
    grad_2_ao: PyTree,
    precision: Precision = Precision.HIGHEST,
):
    return 2 * jnp.einsum(
        "...ab,raj,rbj->r...", rdm1, grad_ao, grad_ao, precision=precision
    ) + 2 * jnp.einsum("...ab,ra,rbi->r...", rdm1, ao, grad_2_ao, precision=precision)


@partial(jax.jit, static_argnames="precision")
def kinetic_density(rdm1: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST) -> Array:
    r"""
    Calculate kinetic energy density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (n_spin, n_orbitals, n_orbitals)
    grad_ao : Array
        Gradients of atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals, 3)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jaxx.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The kinetic energy density. Shape: (n_spin, n_grid_points)
    """

    return 0.5 * jnp.einsum("...ab,raj,rbj->r...", rdm1, grad_ao, grad_ao, precision=precision)


@partial(jax.jit, static_argnames=["precision"])
def HF_energy_density(rdm1: Array, ao: Array, chi: Array, precision: Precision = Precision.HIGHEST):
    r"""
    Calculate the Hartree-Fock energy density.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    chi : Array
        Precomputed chi density.

        .. math::
            Xc^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψc(r') ψd(r') = Γbd^σ ψb(r) v_{cd}(r)

        where :math:`v_{ab}(r) = ∫ dr' f(|r-r'|) ψa(r') ψd(r')`.
        Expected shape: (n_grid_points, n_omega, n_spin, n_orbitals)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.

    Returns
    -------
    Array
        The Hartree-Fock energy density. Shape: (n_omega, n_spin, n_grid_points).
    """

    # return - jnp.einsum("rwsc,sac,ra->wsr", chi, rdm1, ao, precision=precision)/2
    _hf_energy = (
        lambda _chi, _rdm1, _ao: -jnp.einsum("wsc,sac,a->ws", _chi, _rdm1, _ao, precision=precision)
        / 2
    )
    return vmap(_hf_energy, in_axes=(0, None, 0), out_axes=2)(chi, rdm1, ao)


def HF_density_grad_2_Fock(  # todo: change the documentation
    grid: Grid,
    functional: nn.Module,
    params: PyTree,
    chi: Array,
    ao: Array,
    ehf: Array,
    coefficient_inputs: Array,
    densities_wout_hf: Array,
    chunk_size: Optional[int] = None,
    precision: Precision = Precision.HIGHEST,
    fxc_kwargs: dict = {},
) -> Array:
    r"""
    Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock energy energy density.

    .. math::
        F = ∂f(x)/∂e_{HF}^{ωσ} * ∂e_{HF}^{ωσ}/∂Γab^σ = ∂f(x)/∂e_{HF}^{ωσ} * ψa(r) Xb^σ.

    Parameters
    ----------
    fxc : Callable
        FeedForwardFunctional object.
    feat_wout_hf : Array
        Contains a list of the features to input to fxc, except the HF density.
        Expected shape: (n_features, n_grid_points)
    loc_feat_wout_hf : Array
        Contains a list of the features to dot multiply with the output of the functional, except the HF density.
        Expected shape: (n_features, n_grid_points)
    ehf : Array
        The Hartree-Fock energy density.
        Expected shape: (n_omega, n_spin, n_grid_points).
    chi : Array
        .. math::
            Xa^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψa(r') ψd(r')

        Expected shape: (n_grid_points, n_omegas, n_spin, n_orbitals)
    ao : Array
        The orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    grid_weights : Array
        The weights of the grid points.
        Expected shape: (n_grid_points)
    params : PyTree
        The parameters of the neural network.
    combine_features_hf: Callable
        A function that takes the features and ehf, and outputs the updated features
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

    Notes
    -----
    There are other contributions to the Fock matrix from the other features,
    computed separately by automatic differentiation and added later on.
    Also, the chunking is done over the atomic orbitals, not the grid points,
    because the resulting Fock matrix calculation sums over the grid points.

    Returns
    -------
    Array
        The Hartree-Fock matrix contribution due to the partial derivative
        with respect to the Hartree Fock energy density.
        Shape: (n_omegas, n_spin, n_orbitals, n_orbitals).
    """

    def partial_fxc(params, grid, ehf, coefficient_inputs, densities_wout_hf):
        densities = functional.combine_densities(densities_wout_hf, ehf)
        return functional.apply_and_integrate(params, grid, coefficient_inputs, densities)

    gr = grad(partial_fxc, argnums=2)(params, grid, ehf, coefficient_inputs, densities_wout_hf)

    @partial(vmap_chunked, in_axes=(0, None, None), chunk_size=chunk_size)
    def chunked_jvp(chi_tensor, gr_tensor, ao_tensor):
        return (
            -jnp.einsum("rws,wsr,ra->wsa", chi_tensor, gr_tensor, ao_tensor, precision=precision)
            / 2.0
        )

    return (jax.jit(chunked_jvp)(chi.transpose(3, 0, 1, 2), gr, ao)).transpose(1, 2, 3, 0)


def HF_coefficient_input_grad_2_Fock(
    grid: Grid,
    functional: nn.Module,
    params: PyTree,
    chi: Array,
    ao: Array,
    ehf: Array,
    cinputs_wout_hf: Array,
    densities: Array,
    chunk_size: Optional[int] = None,
    precision: Precision = Precision.HIGHEST,
    fxc_kwargs: dict = {},
) -> Array:
    r"""
    Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock energy energy density.

    .. math::
        F = ∂f(x)/∂e_{HF}^{ωσ} * ∂e_{HF}^{ωσ}/∂Γab^σ = ∂f(x)/∂e_{HF}^{ωσ} * ψa(r) Xb^σ.

    Parameters
    ----------
    fxc : Callable
        FeedForwardFunctional object.
    feat_wout_hf : Array
        Contains a list of the features to input to fxc, except the HF density.
        Expected shape: (n_features, n_grid_points)
    loc_feat_wout_hf : Array
        Contains a list of the features to dot multiply with the output of the functional, except the HF density.
        Expected shape: (n_features, n_grid_points)
    ehf : Array
        The Hartree-Fock energy density.
        Expected shape: (n_omega, n_spin, n_grid_points).
    chi : Array
        .. math::
            Xa^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψa(r') ψd(r')

        Expected shape: (n_grid_points, n_omegas, n_spin, n_orbitals)
    ao : Array
        The orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    grid_weights : Array
        The weights of the grid points.
        Expected shape: (n_grid_points)
    params : PyTree
        The parameters of the neural network.
    combine_features_hf: Callable
        A function that takes the features and ehf, and outputs the updated features
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

    Notes
    -----
    There are other contributions to the Fock matrix from the other features,
    computed separately by automatic differentiation and added later on.
    Also, the chunking is done over the atomic orbitals, not the grid points,
    because the resulting Fock matrix calculation sums over the grid points.

    Returns
    -------
    Array
        The Hartree-Fock matrix contribution due to the partial derivative
        with respect to the Hartree Fock energy density.
        Shape: (n_omegas, n_spin, n_orbitals, n_orbitals).
    """

    def partial_fxc(params, grid, ehf, coefficient_inputs_wout_hf, densities):
        coefficient_inputs = functional.combine_inputs(coefficient_inputs_wout_hf, ehf)
        return functional.apply_and_integrate(params, grid, coefficient_inputs, densities)

    gr = grad(partial_fxc, argnums=2)(params, grid, ehf, cinputs_wout_hf, densities)

    @partial(vmap_chunked, in_axes=(0, None, None), chunk_size=chunk_size)
    def chunked_jvp(chi_tensor, gr_tensor, ao_tensor):
        return (
            -jnp.einsum("rws,wsr,ra->wsa", chi_tensor, gr_tensor, ao_tensor, precision=precision)
            / 2.0
        )

    return (jax.jit(chunked_jvp)(chi.transpose(3, 0, 1, 2), gr, ao)).transpose(1, 2, 3, 0)


def eig(h, x):
    e0, c0 = eigh2d(h[0], x)
    e1, c1 = eigh2d(h[1], x)
    return jnp.stack((e0, e1), axis=0), jnp.stack((c0, c1), axis=0)


######################################################################


@partial(jax.jit, static_argnames=["precision"])
def nonXC(
    rdm1: Array,
    h1e: Array,
    rep_tensor: Array,
    nuclear_repulsion: Scalar,
    precision=Precision.HIGHEST,
) -> Scalar:
    r"""
    A function that computes the non-XC part of a DFT functional.

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
    rdm1 = symmetrize_rdm1(rdm1)
    h1e_energy = one_body_energy(rdm1, h1e, precision)
    coulomb2e_energy = two_body_energy(rdm1, rep_tensor, precision)

    return nuclear_repulsion + h1e_energy + coulomb2e_energy


@partial(jax.jit)
def symmetrize_rdm1(rdm1):
    dm = rdm1.sum(axis=0)
    rdm1 = jnp.stack([dm, dm], axis=0) / 2.0
    return rdm1


@partial(jax.jit, static_argnames=["precision"])
def two_body_energy(rdm1, rep_tensor, precision=Precision.HIGHEST):
    v_coul = 2 * jnp.einsum(
        "pqrt,srt->spq", rep_tensor, rdm1, precision=precision
    )  # The 2 is to compensate for the /2 in the dm definition
    coulomb2e_energy = jnp.einsum("sji,sij->", rdm1, v_coul, precision=precision) / 2.0
    return coulomb2e_energy


@partial(jax.jit, static_argnames=["precision"])
def one_body_energy(rdm1, h1e, precision=Precision.HIGHEST):
    h1e_energy = jnp.einsum("sij,ji->", rdm1, h1e, precision=precision)
    return h1e_energy


def coulomb_potential(rdm1, rep_tensor, precision=Precision.HIGHEST):
    r"""
    A function that computes the non-XC part of a DFT functional.

    Parameters
    ----------
    rdm1 : Array
        The 1-Reduced Density Matrix.
        Equivalent to mf.make_rdm1() in pyscf.
        Expected shape: (n_spin, n_orb, n_orb)
    rep_tensor : Array
        The repulsion tensor.
        Equivalent to mf.mol.intor('int2e') in pyscf.
        Expected shape: (n_orb, n_orb, n_orb, n_orb)
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
        Coulomb potential matrix.
    """
    return 2 * jnp.einsum("pqrt,srt->spq", rep_tensor, rdm1, precision=precision)


@partial(jax.jit, static_argnames=["precision"])
def HF_exact_exchange(chi, rdm1, ao, precision=Precision.HIGHEST) -> Array:
    r"""
    A function that computes the exact exchange energy of a DFT functional.

    Parameters
    ----------
    chi : Array
        .. math::
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
            the kernel :math:`f(|r-r'|) = erf(w |r-r'|)/|r-r'|`.

    Returns
    -------
    Array
        The exact exchange energy of the DFT functional at each point of the grid.
    """

    _hf_energy = (
        lambda _chi, _dm, _ao: -jnp.einsum("wsc,sac,a->ws", _chi, _dm, _ao, precision=precision) / 2
    )
    return vmap(_hf_energy, in_axes=(0, None, 0), out_axes=2)(chi, rdm1, ao)


def make_rdm1(mo_coeff, mo_occ):
    r"""
    One-particle density matrix in AO representation

    Args:
        mo_coeff : tuple of 2D ndarrays
            Orbital coefficients for alpha and beta spins. Each column is one orbital.
        mo_occ : tuple of 1D ndarrays
            Occupancies for alpha and beta spins.
    Returns:
        A list of 2D ndarrays for alpha and beta spins

    # Pyscf code
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = jnp.dot(mo_a*mo_occ[0], mo_a.conj().T)
    dm_b = jnp.dot(mo_b*mo_occ[1], mo_b.conj().T)

    return jnp.array((dm_a, dm_b))
    """

    return jnp.einsum("sij,sj,skj -> sik", mo_coeff, mo_occ, mo_coeff.conj())


def get_occ(mo_energies, nelecs, naos):
    def get_occ_spin(mo_energy, nelec_spin, naos):
        sorted_indices = jnp.argsort(mo_energy)

        mo_occ = jnp.zeros_like(mo_energy)

        def assign_values(i, mo_occ):
            value = cond(jnp.less(i, nelec_spin), lambda _: 1.0, lambda _: 0.0, operand=None)
            idx = sorted_indices[i]
            mo_occ = mo_occ.at[idx].set(value)
            return mo_occ

        mo_occ = fori_loop(0, naos, assign_values, mo_occ)

        return mo_occ

    mo_occ = jnp.stack(
        [get_occ_spin(mo_energies[s], jnp.int32(nelecs[s]), naos) for s in range(2)], axis=0
    )

    return mo_occ


######################################################################


class Reaction(NamedTuple):
    reactants: Sequence[Molecule]
    products: Sequence[Molecule]
    reactant_numbers: Sequence[int]
    product_numbers: Sequence[int]
    energy: float
    name: Optional[str] = None


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
    if isinstance(molecules, Molecule):
        molecules = (molecules,)

    if numbers is None:
        numbers = (1,) * len(molecules)
    elif not isinstance(numbers, Sequence):
        numbers = (numbers,)

    return molecules, numbers
