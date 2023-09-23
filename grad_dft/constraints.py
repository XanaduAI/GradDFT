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

from flax import struct
from jax.lax import cond
import jax.numpy as jnp
from jax.lax import Precision
from jax.nn import relu

from grad_dft import (
    Functional,
    Molecule,
    density,
    grad_density,
    abs_clip,
    energy_predictor
)
from grad_dft.interface.pyscf import (
    generate_chi_tensor,
)

from grad_dft.popular_functionals import (
    LSDA,
)
from jaxtyping import Array, PyTree, Scalar, Float

r"""
In this document we implement some of the constraints listed in the review paper
https://doi.org/10.1146/annurev-physchem-062422-013259 that the exact functional should satisfy,
as quadratic loss functions.
"""

###################### Constraints ############################


def x1_c1(
    functional: Functional, params: PyTree, molecule: Molecule, precision=Precision.HIGHEST
) -> (Float, Float):
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        \epsilon_x[n] < 0 (x1)
        \epsilon_c[n] < 0 (c1)

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    precision: Precision

    Returns
    -------
    Float, Float
    """

    # Compute the input features
    densities = functional.compute_densities(molecule)
    cinputs = functional.compute_coefficient_inputs(molecule)
    coefficients = functional.apply(params,cinputs)

    ex = jnp.einsum('rf,rf,f->r', coefficients, densities,functional.exchange_mask, precision=precision)
    Ex = functional._integrate(relu(ex), molecule.grid.weights)

    ec = jnp.einsum('rf,rf,f->r', coefficients, densities,1-functional.exchange_mask, precision=precision)
    Ec = functional._integrate(relu(ec), molecule.grid.weights)

    # return jnp.less_equal(ex, 0.), jnp.less_equal(ec, 0.)
    return Ex**2, Ec**2


def c2(
    functional: Functional, params: PyTree, molecule: Molecule, precision=Precision.HIGHEST
) -> Float:
    r"""
    Instantiates a loss function checking the following constraint:
    Assuming there is a single electron in the molecule,
    .. math::
        \epsilon_c[n] = 0.

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule (single electron molecule)
    precision: Precision

    Returns
    -------
    Float

    Notes
    -----
    Assumes molecule has a single electron.
    """

    # Compute the input features
    densities = functional.compute_densities(molecule)
    cinputs = functional.compute_coefficient_inputs(molecule)
    coefficients = functional.apply(params,cinputs)

    # Mask the exchange features
    ec = jnp.einsum('rf,rf,f->r', coefficients, densities,1-functional.exchange_mask, precision=precision)

    # Compute the exchange-correlation energy at each point
    Ec = functional._integrate(ec, molecule.grid.weights)

    # return jnp.isclose(Ec, 0.)

    return Ec**2


def x2(
    functional: Functional, params: PyTree, molecule: Molecule, precision=Precision.HIGHEST
) -> Float:
    r"""
    Instantiates a loss function checking the following constraint:
    .. math::
        E_x[n]  = \frac{1}{2}(E_x[2n_\uparrow] + E_x[2n_\downarrow])

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    precision: Precision

    Returns
    -------
    Float
    """

    rdm1 = molecule.rdm1

    rdm1u = rdm1[0]
    rdm1u = jnp.stack([rdm1u, rdm1u], axis=0)
    rdm1d = rdm1[1]
    rdm1d = jnp.stack([rdm1d, rdm1d], axis=0)

    # Compute the input coefficient inputs and densities
    densities = functional.compute_densities(molecule)
    cinputs = functional.compute_coefficient_inputs(molecule)
    # Apply the functional
    coefficients = functional.apply(params,cinputs)
    # Mask the exchange features
    ex = jnp.einsum('rf,rf,f->r', coefficients, densities,functional.exchange_mask, precision=precision)
    Exc = functional._integrate(ex, molecule.grid.weights)

    # Replace rdm1 with rdm1u
    molecule = molecule.replace(rdm1 = rdm1u)
    # Compute the input coefficient inputs and densities
    densities = functional.compute_densities(molecule)
    cinputs = functional.compute_coefficient_inputs(molecule)
    # Apply the functional
    coefficients = functional.apply(params,cinputs)
    # Mask the exchange features
    ex = jnp.einsum('rf,rf,f->r', coefficients, densities,functional.exchange_mask, precision=precision)
    Excu = functional._integrate(ex, molecule.grid.weights)

    # Replace rdm1 with rdm1d
    molecule = molecule.replace(rdm1 = rdm1d)
    # Compute the input coefficient inputs and densities
    densities = functional.compute_densities(molecule)
    cinputs = functional.compute_coefficient_inputs(molecule)
    # Apply the functional
    coefficients = functional.apply(params,cinputs)
    # Mask the exchange features
    ex = jnp.einsum('rf,rf,f->r', coefficients, densities,functional.exchange_mask, precision=precision)
    Excd = functional._integrate(ex, molecule.grid.weights)

    # Reinserting original rdm1 into molecule
    molecule = molecule.replace(rdm1=rdm1)

    # return jnp.isclose(Exc, (Excu + Excd)/2.)
    return (Exc - (Excu + Excd) / 2.0) ** 2


def x3_c3_c4(
    functional: Functional,
    params: PyTree,
    molecule: Molecule,
    gamma: Scalar = 2,
    precision=Precision.HIGHEST,
) -> (Float, Float, Float):
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        E_x[n_\gamma]  = \gamma E_x[n], (x3) \\
        E_c[n_\gamma]  > \gamma E_c[n], (c3)  & \gamma > 1\\
        E_c[n_\gamma]  < \gamma E_c[n], (c4)  & \gamma < 1\\

    where :math: `\gamma` is a constant, and
    .. math::
        n_\gamma(r) = \gamma^3 n(\gamma r)

    Instead of working with :math: `r` it is better to work with the scaled coordinates :math: `u = \gamma r`.
    Then, we have
    .. math::
        dr = du/\gamma^3\\
        \psi(u) = \gamma^{3/2} \psi(r)\\
        \nabla \psi(u) = \gamma^{3/2+1} \nabla \psi(r)\\
        \nabla^k \psi(u) = \gamma^{3/2+k} \nabla^k \psi(r)\\
        \rho(u) = \gamma^3 \rho(r)\\
        \frac{\nabla \rho(u)}{\rho(u)^{4/3}} = \frac{\gamma^{5/2+3/2} \nabla \rho(r)}{\gamma^4 \rho(r)^{4/3}} = 1
        \frac{\nabla^k \rho(u)}{\rho(u)^{1+k/3}} = \frac{\gamma^{3/2+3/2 + k} \nabla^k \rho(r)}{\gamma^{3+k} \rho(r)^{1+k/3}} = 1
        \chi(u) = \gamma^{5/2} \chi(r)\\
        \tau(u) = \gamma^5 \tau(r)

    Parameters
    ----------
    functional : Functional
        The functional to be tested.
    params : PyTree
        The parameters of the functional.
    molecule : Molecule
        The molecule to be tested.
    gamma : Scalar
        The scaling factor. Default is 2.

    Returns
    -------
    Float, Float, Float
    """

    densities = functional.compute_densities(molecule)
    cinputs = functional.compute_coefficient_inputs(molecule)
    # Apply the functional
    coefficients = functional.apply(params,cinputs)
    # Mask the exchange features
    ex = jnp.einsum('rf,rf,f->r', coefficients, densities, functional.exchange_mask, precision=precision)
    Ex = functional._integrate(ex, molecule.grid.weights)

    # Mask the correlation features
    ec = jnp.einsum('rf,rf,f->r', coefficients, densities, 1-functional.exchange_mask, precision=precision)
    Ec = functional._integrate(ec, molecule.grid.weights)

    # Replace the molecule properties with the scaled ones
    ao1 = molecule.ao
    grad_ao1 = molecule.grad_ao
    grad_n_ao1 = molecule.grad_n_ao
    chi1 = molecule.chi
    grid_coords1 = molecule.grid.coords
    grid_weights1 = molecule.grid.weights

    exchange_constraints = jnp.array((False, False), dtype=bool)
    correlation_constraints = jnp.array((False, False), dtype=bool)

    exchange_constraint_losses = jnp.array((0.0, 0.0))
    correlation_constraint_losses = jnp.array((0.0, 0.0))

    # Computing the scaled properties
    for i, g in enumerate((gamma, 1 / gamma)):
        ao = ao1 * g ** (3 / 2)
        grad_ao = grad_ao1 * g ** (3 / 2 + 1)
        grad_n_ao = {}
        for k in grad_n_ao1.keys():
            grad_n_ao[k] = grad_n_ao1[k] * g ** (3 / 2 + k)
        chi = chi1 * g ** (5 / 2)
        molecule = molecule.replace(ao=ao, grad_ao=grad_ao, grad_n_ao=grad_n_ao, chi=chi)

        grid = molecule.grid
        grid = grid.replace(coords=grid_coords1 * g)
        grid = grid.replace(weights=grid_weights1 / g**3)
        molecule = molecule.replace(grid=grid)

        # Compute the input features
        densities = functional.compute_densities(molecule)
        cinputs = functional.compute_coefficient_inputs(molecule)
        # Apply the functional
        coefficients = functional.apply(params,cinputs)
        # Mask the exchange features
        exg = jnp.einsum('rf,rf,f->r', coefficients, densities, functional.exchange_mask, precision=precision)
        Exg = functional._integrate(exg, molecule.grid.weights)

        # Mask the correlation features
        ecg = jnp.einsum('rf,rf,f->r', coefficients, densities, 1-functional.exchange_mask, precision=precision)
        Ecg = functional._integrate(ecg, molecule.grid.weights)

        # Checking the x3 constraint
        exchange_constraint_losses = exchange_constraint_losses.at[i].set((g * Ex - Exg) ** 2)

        # using jax, check whether gamma*Ec > Ecg or gamma*Ec < Ecg depending on i:
        c3c4 = cond(
            jnp.greater(g, 1),
            lambda Ecg, Ec, g: jnp.greater(Ecg, g * Ec),
            lambda Ecg, Ec, g: jnp.less(Ecg, g * Ec),
            Ecg,
            Ec,
            g,
        )
        correlation_constraints = correlation_constraints.at[i].set(c3c4)

        c3c4_loss = cond(
            jnp.greater(g, 1),
            lambda Ecg, Ec, g: relu(g * Ec - Ecg) ** 2,
            lambda Ecg, Ec, g: relu(Ecg - g * Ec) ** 2,
            Ecg,
            Ec,
            g,
        )
        correlation_constraint_losses = correlation_constraint_losses.at[i].set(c3c4_loss)

    # Reinserting original molecule properties
    molecule = molecule.replace(ao=ao1, grad_ao=grad_ao1, grad_n_ao=grad_n_ao1, chi=chi1)
    grid = molecule.grid
    grid = grid.replace(coords=grid_coords1)
    grid = grid.replace(weights=grid_weights1)
    molecule = molecule.replace(grid=grid)
    # return exchange_constraints, correlation_constraints

    return exchange_constraint_losses, correlation_constraint_losses


def x4(
    functional: Functional,
    params: PyTree,
    molecule: Molecule,
    s2_mask: Array,
    q2_mask: Array,
    qs2_mask: Array,
    s4_mask: Array,
    precision=Precision.HIGHEST,
    lower_bound=1e-15,
    upper_bound=1e-5,
    clip_cte = 1e-30,
) -> Float:
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        E_x[n] \approx \frac{-3}{2}(\frac{3}{4\pi})^{1/3}\int dr [1 + \frac{10}{81}s^2 + \frac{146}{2025}q^2 - \frac{146}{2025}\frac{2}{5}qs^2 + 0 s^4 + O(|\nabla^6 \rho|)] \rho(r)^{4/3}\\
        
    where
    .. math::
        s = |\nabla \rho| / (2k_f \rho^{4/3})
        q = \nabla^2 \rho / (2k_f^2 \rho^{5/3})
    
    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    s2_mask : Array
    q2_mask : Array
    qs2_mask : Array
    s4_mask : Array
    precision: Precision
    lower_bound: Float
    upper_bound: Float
    clip_cte: Float

    Returns
    -------
    Float
    """

    density = molecule.density()
    grad_density = molecule.grad_density().sum(axis=-1)
    lapl_density = molecule.lapl_density().sum(axis=-1)

    s = jnp.where(jnp.greater_equal(density, clip_cte), grad_density / density ** (4 / 3), 0)
    q = jnp.where(jnp.greater_equal(density, clip_cte), lapl_density / density ** (5 / 3), 0)

    s2_cond = jnp.logical_and(
        jnp.logical_and(
            jnp.less_equal(jnp.abs(grad_density), upper_bound),
            jnp.greater_equal(jnp.abs(grad_density), lower_bound),
        ),
        jnp.greater_equal(jnp.abs(density), lower_bound),
    )

    q2_cond = jnp.logical_and(
        jnp.logical_and(
            jnp.less_equal(jnp.abs(lapl_density), upper_bound),
            jnp.greater_equal(jnp.abs(lapl_density), lower_bound),
        ),
        jnp.greater_equal(jnp.abs(density), lower_bound),
    )

    qs2_cond = jnp.logical_and(s2_cond, q2_cond)

    lda_functional = LSDA
    lda_densities = lda_functional.compute_densities(molecule)
    lda_cinputs = lda_functional.compute_coefficient_inputs(molecule)
    lda_coefficients = lda_functional.apply(params, lda_cinputs)
    lda_e = jnp.einsum("rf,rf->r", lda_coefficients, lda_densities)
    lda_e = abs_clip(lda_e, clip_cte)

    cinputs = functional.compute_coefficient_inputs(molecule)
    densities = functional.compute_densities(molecule)

    if s2_mask:
        cinputsxs2 = jnp.einsum("rf,f->rf", cinputs, s2_mask, precision=precision)
        coefficients = functional.apply(params, cinputsxs2)
        exs2 = jnp.einsum('rf,rf,f->r', coefficients, densities, functional.exchange_mask, precision=precision)
        s2loss = jnp.where(s2_cond, (exs2 / (lda_e * s**2) - 10 / 81) ** 2, 0)
    else:
        s2loss = 0

    if q2_mask:
        cinputsxq2 = jnp.einsum("rf,f->rf", cinputs, q2_mask, precision=precision)
        exq2 = functional.apply(params, cinputsxq2)
        q2loss = jnp.where(q2_cond, (exq2 / (lda_e * q**2) - 146 / 2025) ** 2, 0)
    else:
        q2loss = 0

    if qs2_mask:
        cinputsxqs2 = jnp.einsum("rf,f->rf", cinputs, qs2_mask, precision=precision)
        exqs2 = functional.apply(params, cinputsxqs2)
        qs2loss = jnp.where(qs2_cond, (exqs2 / (lda_e * s**2 * q) + 146 / 2025 * 5 / 2) ** 2, 0)
    else:
        qs2loss = 0

    if s4_mask:
        cinputsxs4 = jnp.einsum("rf,f->rf", cinputs, s4_mask, precision=precision)
        exs4 = functional.apply(params, cinputsxs4)
        s4loss = jnp.where(s2_cond, (exs4 / (lda_e * s**4)) ** 2, 0)
    else:
        s4loss = 0

    # return s2, q2, qs2, s4
    return (
        functional._integrate(s2loss, molecule.grid.weights),
        functional._integrate(q2loss, molecule.grid.weights),
        functional._integrate(qs2loss, molecule.grid.weights),
        functional._integrate(s4loss, molecule.grid.weights),
    )


def x5(
    functional: Functional,
    params: PyTree,
    molecule: Molecule,
    precision=Precision.HIGHEST,
    multiplier1=1e5,
    multiplier2=1e7,
    clip_cte = 1e-30,
) -> Float:
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        lim_{s\rightarrow \infty}F_x(s, ...) \propto s^{-1/2}

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    precision: Precision
    multiplier1: Float
    multiplier2: Float
    clip_cte: Float

    Returns
    -------
    Float
    """

    # Compute the lda energy
    lda_functional = LSDA
    lda_densities = lda_functional.compute_densities(molecule)
    lda_cinputs = lda_functional.compute_coefficient_inputs(molecule)
    lda_coefficients = lda_functional.apply(params, lda_cinputs)
    lda_e = jnp.einsum("rf,rf->r", lda_coefficients, lda_densities)
    lda_e = jnp.expand_dims(abs_clip(lda_e, clip_cte), axis=-1)

    @struct.dataclass
    class modMolecule(Molecule):
        s_multiplier: Scalar = 1.0

        def grad_density(self, *args, **kwargs) -> Array:
            r"""Scaled kinetic energy"""
            return (
                grad_density(self.rdm1, self.ao, self.grad_ao, *args, **kwargs) * self.s_multiplier
            )

    modmolecule = modMolecule(
        molecule.grid,
        molecule.atom_index,
        molecule.nuclear_pos,
        molecule.ao,
        molecule.grad_ao,
        molecule.grad_n_ao,
        molecule.rdm1,
        molecule.nuclear_repulsion,
        molecule.h1e,
        molecule.vj,
        molecule.mo_coeff,
        molecule.mo_occ,
        molecule.mo_energy,
        molecule.mf_energy,
        molecule.s1e,
        molecule.omegas,
        molecule.chi,
        molecule.rep_tensor,
        molecule.energy,
        molecule.basis,
        molecule.name,
        molecule.spin,
        molecule.charge,
        molecule.unit_Angstrom,
        molecule.grid_level,
        molecule.scf_iteration,
        molecule.fock,
    )

    # Multiplying s by multiplier 1
    modmolecule = modmolecule.replace(s_multiplier=multiplier1)
    density1 = modmolecule.density()
    grad_density1 = modmolecule.grad_density().sum(axis=-1)

    s1 = jnp.where(jnp.greater_equal(density1, 1e-30), grad_density1 / density1 ** (4 / 3), 0)
    a = jnp.isnan(s1).any()

    cinputs1 = functional.compute_coefficient_inputs(modmolecule)
    densities1 = functional.compute_densities(modmolecule)
    coefficients1 = functional.apply(params, cinputs1)
    ex1 = jnp.einsum('rf,rf,f->r', coefficients1, densities1, functional.exchange_mask, precision=precision)
    ex1 = jnp.expand_dims(ex1, axis=-1)

    fx1 = jnp.where(
        jnp.greater_equal(jnp.abs(lda_e * jnp.sqrt(s1)), 1e-30), ex1 / (lda_e * jnp.sqrt(s1)), 0
    )
    a = jnp.isnan(fx1).any()

    # Multiplying s by multiplier 2
    modmolecule = modmolecule.replace(s_multiplier=multiplier2)
    density2 = modmolecule.density()
    grad_density2 = modmolecule.grad_density().sum(axis=-1)

    s2 = jnp.where(jnp.greater_equal(density2, 1e-30), grad_density2 / density2 ** (4 / 3), 0)
    a = jnp.isnan(s2).any()

    cinputs2 = functional.compute_coefficient_inputs(modmolecule)
    densities2 = functional.compute_densities(modmolecule)
    coefficients2 = functional.apply(params, cinputs2)
    ex2 = jnp.einsum('rf,rf,f->r', coefficients2, densities2, functional.exchange_mask, precision=precision)
    ex2 = jnp.expand_dims(ex2, axis=-1)

    fx2 = jnp.where(
        jnp.greater_equal(jnp.abs(lda_e * jnp.sqrt(s2)), 1e-30), ex2 / (lda_e * jnp.sqrt(s2)), 0
    )
    a = jnp.isnan(fx2).any()

    # Dividing s by multiplier 1
    modmolecule = modmolecule.replace(s_multiplier=1 / multiplier1)
    density_1 = modmolecule.density()
    grad_density_1 = modmolecule.grad_density().sum(axis=-1)

    s_1 = jnp.where(jnp.greater_equal(density_1, 1e-30), grad_density_1 / density_1 ** (4 / 3), 0)
    a = jnp.isnan(s_1).any()

    cinputs_1 = functional.compute_coefficient_inputs(modmolecule)
    densities_1 = functional.compute_densities(modmolecule)
    coefficients_1 = functional.apply(params, cinputs_1)
    ex_1 = jnp.einsum('rf,rf,f->r', coefficients_1, densities_1, functional.exchange_mask, precision=precision)
    ex_1 = jnp.expand_dims(ex_1, axis=-1)

    fx_1 = jnp.where(
        jnp.greater_equal(jnp.abs(lda_e * jnp.sqrt(s_1)), 1e-30),
        ex_1 / (lda_e * jnp.sqrt(s_1)),
        0,
    )
    a = jnp.isnan(fx_1).any()

    # Dividing s by multiplier 2
    modmolecule = modmolecule.replace(s_multiplier=1 / multiplier2)
    density_2 = modmolecule.density()
    grad_density_2 = modmolecule.grad_density().sum(axis=-1)

    s_2 = jnp.where(jnp.greater_equal(density_2, 1e-30), grad_density_2 / density_2 ** (4 / 3), 0)
    a = jnp.isnan(s2).any()

    cinputs_2 = functional.compute_coefficient_inputs(modmolecule)
    densities_2 = functional.compute_densities(modmolecule)
    coefficients_2 = functional.apply(params, cinputs_2)
    ex_2 = jnp.einsum('rf,rf,f->r', coefficients_2, densities_2, functional.exchange_mask, precision=precision)
    ex_2 = jnp.expand_dims(ex_2, axis=-1)

    fx_2 = jnp.where(
        jnp.greater_equal(jnp.abs(lda_e * jnp.sqrt(s_2)), 1e-30),
        ex_2 / (lda_e * jnp.sqrt(s_2)),
        0,
    )
    a = jnp.isnan(fx_2).any()

    # Checking the condition
    # return jnp.isclose(fx1, fx2, rtol = 1e-4, atol = 1).all(), jnp.isclose(fx_1, fx_2, rtol = 1e-4, atol = 1).all()
    f = ((fx1 - fx2) ** 2).sum(axis=1)
    f_ = ((fx_1 - fx_2) ** 2).sum(axis=1)

    e = functional._integrate(f, molecule.grid.weights)
    e_ = functional._integrate(f_, molecule.grid.weights)

    return e, e_


def x6(
    functional: Functional, params: PyTree, molecule: Molecule, precision=Precision.HIGHEST, clip_cte = 1e-30
) -> Float:
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        E_x[\rho/2, \rho/2] \geq E_{xc}[\rho/2, \rho/2] \geq 1.804 E_x^{LSDA}[\rho]

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    precision: Precision
    clip_cte: Float

    Returns
    -------
    Float
    """

    # Compute the lda energy
    lda_functional = LSDA
    lda_densities = lda_functional.compute_densities(molecule)
    lda_cinputs = lda_functional.compute_coefficient_inputs(molecule)
    lda_coefficients = lda_functional.apply(params, lda_cinputs)
    lda_e = jnp.einsum("rf,rf->r", lda_coefficients, lda_densities)
    lda_e = abs_clip(lda_e, clip_cte)

    # Symmetrize the reduced density matrix
    rdm1 = molecule.rdm1
    molecule = molecule.replace(rdm1=(rdm1 + rdm1[::-1]) / 2)

    # Compute the exchange energy
    cinputs = functional.compute_coefficient_inputs(molecule)
    densities = functional.compute_densities(molecule)
    coefficients = functional.apply(params, cinputs)
    ex = jnp.einsum('rf,rf,f->r', coefficients, densities, functional.exchange_mask, precision=precision)
    exc = jnp.einsum('rf,rf->r', coefficients, densities, precision=precision)

    # Put rdm1 back in place
    molecule = molecule.replace(rdm1=rdm1)

    # return jnp.greater_equal(ex, exc).all(), jnp.greater_equal(exc, 1.804*lsda_e).all()
    return functional._integrate(
        (relu(exc - ex)) ** 2, molecule.grid.weights
    ), functional._integrate(
        (relu(1.804 * lda_e - exc)) ** 2, molecule.grid.weights
    )


def x7(
    functional: Functional, params: PyTree, molecule2e: Molecule, precision=Precision.HIGHEST, clip_cte = 1e-30
) -> Float:
    r"""
    Instantiates a loss function checking the following constraints:
    For a two electron system:
    .. math::
        F_x[s, \alpha = 0] \geq 1.174

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule with two electrons
    precision: Precision
    clip_cte: Float

    Returns
    -------
    Float
    """

    @struct.dataclass
    class modMolecule(Molecule):
        def kinetic_density(self: Molecule, *args, **kwargs) -> Array:
            r"""Weizsacker kinetic energy"""
            drho = self.grad_density(*args, **kwargs)
            rho = self.density(*args, **kwargs)
            return jnp.where(jnp.greater_equal(rho, 1e-30), drho**2 / (8 * rho), 0)

    modmolecule = modMolecule(
        molecule2e.grid,
        molecule2e.atom_index,
        molecule2e.nuclear_pos,
        molecule2e.ao,
        molecule2e.grad_ao,
        molecule2e.grad_n_ao,
        molecule2e.rdm1,
        molecule2e.nuclear_repulsion,
        molecule2e.h1e,
        molecule2e.vj,
        molecule2e.mo_coeff,
        molecule2e.mo_occ,
        molecule2e.mo_energy,
        molecule2e.mf_energy,
        molecule2e.s1e,
        molecule2e.omegas,
        molecule2e.chi,
        molecule2e.rep_tensor,
        molecule2e.energy,
        molecule2e.basis,
        molecule2e.name,
        molecule2e.spin,
        molecule2e.charge,
        molecule2e.unit_Angstrom,
        molecule2e.grid_level,
        molecule2e.scf_iteration,
        molecule2e.fock,
    )

    cinputs = functional.compute_coefficient_inputs(modmolecule)
    densities = functional.compute_densities(modmolecule)
    coefficients = functional.apply(params, cinputs)
    functional_e = jnp.einsum('rf,rf,f->r', coefficients, densities, functional.exchange_mask, precision=precision)

    lda_functional = LSDA
    lda_densities = lda_functional.compute_densities(modmolecule)
    lda_cinputs = lda_functional.compute_coefficient_inputs(modmolecule)
    lda_coefficients = lda_functional.apply(params, lda_cinputs)
    lda_e = jnp.einsum("rf,rf->r", lda_coefficients, lda_densities)
    lda_e = abs_clip(lda_e, clip_cte)

    # return jnp.where(jnp.greater_equal(lsda_e, 1e-30), jnp.less_equal(functional_e / lsda_e, 1.174), True).all()
    return functional._integrate(
        jnp.where(jnp.greater_equal(lda_e, 1e-30), (relu(functional_e / lda_e - 1.174) ** 2), 0),
        molecule2e.grid.weights,
    )


def c6(
    functional: Functional, params: PyTree, molecule: Molecule, multiplier: Scalar = 1e-7, precision=Precision.HIGHEST
) -> Float:
    r"""
    Instantiates a loss function checking the following constraints:
    For a two electron system:
    .. math::
        E_c[s \rightarrow 0] = 0

        Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    multiplier: Float; multiplicative scaling factor for the electronic density
    precision: Precision

    Returns
    -------
    Float
    """

    @struct.dataclass
    class modMolecule(Molecule):
        rho_scale: Scalar = multiplier

        def density(self, *args, **kwargs) -> Array:
            r"""Scaled kinetic energy"""
            return (
                density(self.rdm1, self.ao, *args, **kwargs)
                * self.rho_scale
            )

    modmolecule = modMolecule(
        molecule.grid,
        molecule.atom_index,
        molecule.nuclear_pos,
        molecule.ao,
        molecule.grad_ao,
        molecule.grad_n_ao,
        molecule.rdm1,
        molecule.nuclear_repulsion,
        molecule.h1e,
        molecule.vj,
        molecule.mo_coeff,
        molecule.mo_occ,
        molecule.mo_energy,
        molecule.mf_energy,
        molecule.s1e,
        molecule.omegas,
        molecule.chi,
        molecule.rep_tensor,
        molecule.energy,
        molecule.basis,
        molecule.name,
        molecule.spin,
        molecule.charge,
        molecule.unit_Angstrom,
        molecule.grid_level,
        molecule.scf_iteration,
        molecule.fock,
    )

    cinputs = functional.compute_coefficient_inputs(modmolecule)
    densities = functional.compute_densities(modmolecule)
    coefficients = functional.apply(params, cinputs)
    ec = jnp.einsum('rf,rf,f->r', coefficients, densities, 1-functional.exchange_mask, precision=precision)
    Ec = functional._integrate(ec, modmolecule.grid.weights)

    # return jnp.isclose(Ec, 0)
    return Ec**2


def xc2(
    functional: Functional, params: PyTree, molecule: Molecule, gamma: Scalar = 1e-7, precision=Precision.HIGHEST
) -> Float:
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        \lim_{\gamma \rightarrow 0} E_{xc}[n_\gamma^\uparrow, n_\gamma^\downarrow] = E_{xc}[n_\gamma]

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule : Molecule
    gamma: Float; multiplicative scaling factor for the atomic orbitals and their gradients
    precision: Precision

    Returns
    -------
    Float
    """

    ao1 = molecule.ao
    grad_ao1 = molecule.grad_ao
    grad_n_ao1 = molecule.grad_n_ao
    chi1 = molecule.chi
    grid_coords1 = molecule.grid.coords
    grid_weights1 = molecule.grid.weights
    rdm11 = molecule.rdm1

    # Computing the scaled properties

    ao = ao1 * gamma ** (3 / 2)
    grad_ao = grad_ao1 * gamma ** (3 / 2 + 1)
    grad_n_ao = {}
    for k in grad_n_ao1.keys():
        grad_n_ao[k] = grad_n_ao1[k] * gamma ** (3 / 2 + k)
    chi = chi1 * gamma ** (5 / 2)
    molecule = molecule.replace(ao=ao, grad_ao=grad_ao, grad_n_ao=grad_n_ao, chi=chi)

    grid = molecule.grid
    grid = grid.replace(coords=grid_coords1 * gamma)
    grid = grid.replace(weights=grid_weights1 / gamma**3)
    molecule = molecule.replace(grid=grid)

    cinputs = functional.compute_coefficient_inputs(molecule)
    densities = functional.compute_densities(molecule)
    coefficients = functional.apply(params, cinputs)
    ec = jnp.einsum('rf,rf,f->r', coefficients, densities, 1-functional.exchange_mask, precision=precision)
    Ec = functional._integrate(ec, molecule.grid.weights)

    molecule = molecule = molecule.replace(rdm1=(rdm11 + rdm11[::-1]) / 2)
    cinputs = functional.compute_coefficient_inputs(molecule)
    densities = functional.compute_densities(molecule)
    coefficients = functional.apply(params, cinputs)
    ec = jnp.einsum('rf,rf,f->r', coefficients, densities, 1-functional.exchange_mask, precision=precision)
    Ec_sym = functional._integrate(ec, molecule.grid.weights)

    # Reinserting original molecule properties
    molecule = molecule.replace(
        ao=ao1, grad_ao=grad_ao1, grad_n_ao=grad_n_ao1, chi=chi1, rdm1=rdm11
    )
    grid = molecule.grid
    grid = grid.replace(coords=grid_coords1)
    grid = grid.replace(weights=grid_weights1)
    molecule = molecule.replace(grid=grid)

    # return jnp.isclose(Ec, Ec_sym)
    return (Ec - Ec_sym) ** 2


def xc4(
    functional: Functional, params: PyTree, molecule2e: Molecule, precision=Precision.HIGHEST
):
    r"""
    Instantiates a loss function checking the following constraints:
    For a two electron system:
    .. math::
        E_{xc}[n2] \geq 1.671 E_{xc}^{LDA}[n2]
    
    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule2e : Molecule with two electrons
    precision: Precision

    Returns
    -------
    Float
    """
    cinputs = functional.compute_coefficient_inputs(molecule2e)
    densities = functional.compute_densities(molecule2e)
    Exc = functional.xc_energy(params, molecule2e.grid, cinputs, densities)

    lda_functional = LSDA
    lda_cinputs = lda_functional.compute_coefficient_inputs(molecule2e)
    lda_densities = lda_functional.compute_densities(molecule2e)
    Exc_lda = lda_functional.xc_energy(params, molecule2e.grid, lda_cinputs, lda_densities)

    # return jnp.greater_equal(Exc, 1.671*Ex_lda)
    return (relu(1.671 * Exc_lda - Exc)) ** 2


def xc1(
    functional: Functional,
    params: PyTree,
    molecule1: Molecule,
    molecule2: Molecule,
    gamma: Scalar = 0.5,
    mol=None,
):
    r"""
    Instantiates a loss function checking the following constraints:
    .. math::
        E_U[\gamma \rho_1 + (1-\gamma) \rho_2]  = \gamma E_U[\rho_1] + (1-\gamma) E_U[\rho_2]

    Parameters
    ----------
    functional : Functional
    params: PyTree
    molecule1 : Molecule
    molecule2 : Molecule
    gamma: Float, mixing factor
    precision: Precision

    Returns
    -------
    Float

    Notes
    -----
    Assumes gamma is between 0 and 1.
    Assumes molecule1 and molecule2 have the same nuclear positions and grid, and differ in that
    molecule2 has one (or 0) more electrons than molecule1.

    Also named as the fractional charge and spin constraint.
    """

    predict = energy_predictor(functional)
    E1, _ = predict(params, molecule1)
    E2, _ = predict(params, molecule2)

    grid = molecule1.grid
    atom_index = molecule1.atom_index
    if not jnp.isclose(molecule1.atom_index, molecule2.atom_index).all():
        raise ValueError("The two molecules must have the same atom_index")
    nuclear_pos = molecule1.nuclear_pos
    if not jnp.isclose(molecule1.nuclear_pos, molecule2.nuclear_pos).all():
        raise ValueError("The two molecules must have the same nuclear_pos")
    ao = molecule1.ao
    if not jnp.isclose(molecule1.ao, molecule2.ao).all():
        raise ValueError("The two molecules must have the same ao")
    grad_ao = molecule1.grad_ao
    if not jnp.isclose(molecule1.grad_ao, molecule2.grad_ao).all():
        raise ValueError("The two molecules must have the same grad_ao")
    grad_n_ao = molecule1.grad_n_ao
    if (
        not all(
            [
                jnp.isclose(molecule1.grad_n_ao[k], molecule2.grad_n_ao[k]).all()
                for k in molecule1.grad_n_ao.keys()
            ]
        )
        and jnp.isclose(
            jnp.array(molecule1.grad_n_ao.keys()), jnp.array(molecule2.grad_n_ao.keys())
        ).all()
    ):
        raise ValueError("The two molecules must have the same grad_n_ao")
    nuclear_repulsion = molecule1.nuclear_repulsion
    if not jnp.isclose(molecule1.nuclear_repulsion, molecule2.nuclear_repulsion):
        raise ValueError("The two molecules must have the same nuclear_repulsion")
    h1e = molecule1.h1e
    if not jnp.isclose(molecule1.h1e, molecule2.h1e).all():
        raise ValueError("The two molecules must have the same h1e")
    vj = jnp.empty((0))
    mo_coeff = jnp.empty((0))
    mo_occ = jnp.empty((0))
    mo_energy = jnp.empty((0))
    mf_energy = jnp.empty((0))
    s1e = molecule1.s1e
    if not jnp.isclose(molecule1.s1e, molecule2.s1e).all():
        raise ValueError("The two molecules must have the same s1e")
    omegas = molecule1.omegas
    if not jnp.isclose(jnp.array(molecule1.omegas), jnp.array(molecule2.omegas)).all():
        raise ValueError("The two molecules must have the same omegas")

    rep_tensor = molecule1.rep_tensor
    if not jnp.isclose(molecule1.rep_tensor, molecule2.rep_tensor).all():
        raise ValueError("The two molecules must have the same rep_tensor")
    energy = jnp.empty((0))
    basis = molecule1.basis
    if not jnp.isclose(molecule1.basis, molecule2.basis).all():
        raise ValueError("The two molecules must have the same basis")
    name = jnp.empty((0))
    unit_Angstrom = molecule1.unit_Angstrom
    if not jnp.equal(molecule1.unit_Angstrom, molecule2.unit_Angstrom):
        raise ValueError("The two molecules must have the same unit_Angstrom")
    grid_level = molecule1.grid_level
    if not jnp.isclose(molecule1.grid_level, molecule2.grid_level).all():
        raise ValueError("The two molecules must have the same grid_level")
    scf_iteration = jnp.empty((0))
    fock = jnp.empty((0))

    rdm1 = gamma * molecule1.rdm1 + (1 - gamma) * molecule2.rdm1
    spin = gamma * molecule1.spin + (1 - gamma) * molecule2.spin
    charge = gamma * molecule1.charge + (1 - gamma) * molecule2.charge
    chi = generate_chi_tensor(rdm1, ao, grid.coords, mol, omegas)

    molecule = Molecule(
        grid,
        atom_index,
        nuclear_pos,
        ao,
        grad_ao,
        grad_n_ao,
        rdm1,
        nuclear_repulsion,
        h1e,
        vj,
        mo_coeff,
        mo_occ,
        mo_energy,
        mf_energy,
        s1e,
        omegas,
        chi,
        rep_tensor,
        energy,
        basis,
        name,
        spin,
        charge,
        unit_Angstrom,
        grid_level,
        scf_iteration,
        fock,
    )

    E, _ = predict(params, molecule)

    # return jnp.isclose(E, gamma*E1 + (1-gamma)*E2)
    return (gamma * E1 + (1 - gamma) * E2 - E) ** 2
