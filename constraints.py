from jax.lax import cond
import jax.numpy as jnp
from jax.lax import Precision
from functional import Functional
from molecule import Molecule, kinetic_density

from train import compute_features
from utils.types import PyTree, Scalar

r"""
In this document we implement some of the constraints listed in the review paper
https://doi.org/10.1146/annurev-physchem-062422-013259 that the exact functional should satisfy.
"""

###################### Constraints ############################

def constraint_x1(functional: Functional, params: PyTree, molecule: Molecule, precision = Precision.HIGHEST):
    r"""
    ::math::
        \epsilon_x[n] < 0
    """

    # Compute the input features
    features = compute_features(functional, molecule)[0]

    # Mask the correlation features
    features = jnp.einsum('rf,f->rf', features, functional.exchange_mask, precision=precision)

    # Compute the exchange-correlation energy at each point
    ex = functional.apply(params, features)

    return jnp.less_equal(ex, 0.)

def constraint_x2(functional: Functional, params: PyTree, molecule: Molecule, precision = Precision.HIGHEST):
    r"""
    ::math::
        E_x[n]  = \frac{1}{2}(E_x[2n_\uparrow] + E_x[2n_\downarrow])
    """

    rdm1 = molecule.rdm1

    rdm1u = rdm1[0]
    rdm1u = jnp.stack([rdm1u, rdm1u], axis=0)
    rdm1d = rdm1[1]
    rdm1d = jnp.stack([rdm1d, rdm1d], axis=0)

    # Compute the input features
    features = compute_features(functional, molecule)[0]
    # Mask the correlation features
    features = jnp.einsum('rf,f->rf', features, functional.exchange_mask, precision=precision)
    # Compute the energy
    Exc = functional.apply_and_integrate(params, molecule, features)

    # Replace rdm1 with rdm1u
    molecule = molecule.replace(rdm1 = rdm1u)
    # Compute the input features
    featuresu = compute_features(functional, molecule)[0]
    # Mask the correlation features
    featuresu = jnp.einsum('rf,f->rf', featuresu, functional.exchange_mask, precision=precision)
    # Compute the energy
    Excu = functional.apply_and_integrate(params, molecule, featuresu)

    # Replace rdm1 with rdm1d
    molecule = molecule.replace(rdm1 = rdm1d)
    # Compute the input features
    featuresd = compute_features(functional, molecule)[0]
    # Mask the correlation features
    featuresd = jnp.einsum('rf,f->rf', featuresd, functional.exchange_mask, precision=precision)
    # Compute the energy
    Excd = functional.apply_and_integrate(params, molecule, featuresd)

    # Reinserting original rdm1 into molecule
    molecule = molecule.replace(rdm1 = rdm1)

    return jnp.isclose(Exc, (Excu + Excd)/2.)

def constraints_x3_c3_c4(functional: Functional, params: PyTree, molecule: Molecule, gamma: Scalar = 2, precision = Precision.HIGHEST):
    r"""
    ::math::
        E_x[n_\gamma]  = \gamma E_x[n], (x3) \\
        E_c[n_\gamma]  > \gamma E_c[n], (c3)  & \gamma > 1\\
        E_c[n_\gamma]  < \gamma E_c[n], (c4)  & \gamma < 1\\

    where :math: `\gamma` is a constant, and
    ::math::
        n_\gamma(r) = \gamma^3 n(\gamma r)

    Instead of working with :math: `r` it is better to work with the scaled coordinates :math: `u = \gamma r`.
    Then, we have
    ::math::
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
    exchange_constraint : Array[bool]
        The x3 constraint results for gamma and 1/gamma
    correlation_constraints : Array[bool]
        The c3 and c4 constraint results for gamma and 1/gamma respectively
    """

    # Compute the input features
    features = compute_features(functional, molecule)[0]
    # Mask the correlation features
    features_x = jnp.einsum('rf,f->rf', features, functional.exchange_mask, precision=precision)
    # Compute the energy
    Ex = functional.apply_and_integrate(params, molecule, features_x)

    # Mask the exchange features
    features_c = jnp.einsum('rf,f->rf', features, 1-functional.exchange_mask, precision=precision)
    # Compute the energy
    Ec = functional.apply_and_integrate(params, molecule, features_c)

    # Replace the molecule properties with the scaled ones
    ao1 = molecule.ao
    grad_ao1 = molecule.grad_ao
    grad_n_ao1 = molecule.grad_n_ao
    chi1 = molecule.chi
    grid_coords1 = molecule.grid.coords
    grid_weights1 = molecule.grid.weights

    exchange_constraints =jnp.array((False, False), dtype=bool)
    correlation_constraints =jnp.array((False, False), dtype=bool)

    # Computing the scaled properties
    for i, g in enumerate((gamma, 1/gamma)):
        ao = ao1 * g**(3/2)
        grad_ao = grad_ao1 * g**(3/2+1)
        grad_n_ao ={}
        for k in grad_n_ao1.keys():
            grad_n_ao[k] = grad_n_ao1[k] * g**(3/2+k)
        chi = chi1 * g**(5/2)
        molecule = molecule.replace(ao = ao, grad_ao = grad_ao, grad_n_ao = grad_n_ao, chi = chi)

        grid = molecule.grid
        grid = grid.replace(coords = grid_coords1 * g)
        grid = grid.replace(weights = grid_weights1 / g**3 )
        molecule = molecule.replace(grid = grid)

        # Compute the input features
        features_gamma = compute_features(functional, molecule)[0]

        # Mask the correlation features
        features_gamma_x = jnp.einsum('rf,f->rf', features_gamma, functional.exchange_mask, precision=precision)
        # Compute the energy
        Exg = functional.apply_and_integrate(params, molecule, features_gamma_x)
        # Checking the x3 constraint
        exchange_constraints = exchange_constraints.at[i].set(jnp.isclose(g*Ex, Exg))

        # Mask the correlation features
        features_gamma_c = jnp.einsum('rf,f->rf', features_gamma, 1-functional.exchange_mask, precision=precision)
        # Compute the energy
        Ecg = functional.apply_and_integrate(params, molecule, features_gamma_c)

        # using jax, check whether gamma*Ec > Ecg or gamma*Ec < Ecg depending on i:
        c3c4 = cond(jnp.greater(g, 1), 
                lambda Ecg, Ec, g: jnp.greater(Ecg, g*Ec), 
                lambda Ecg, Ec, g: jnp.less(Ecg, g*Ec),
                Ecg, Ec, g)
        correlation_constraints = correlation_constraints.at[i].set(c3c4)

    # Reinserting original molecule properties
    molecule = molecule.replace(ao = ao1, grad_ao = grad_ao1, grad_n_ao = grad_n_ao1, chi = chi1)
    grid = molecule.grid
    grid = grid.replace(coords = grid_coords1)
    grid = grid.replace(weights = grid_weights1)
    molecule = molecule.replace(grid = grid)
    return exchange_constraints, correlation_constraints

