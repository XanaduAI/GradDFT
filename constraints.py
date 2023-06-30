import jax.numpy as jnp
from jax.lax import Precision
from functional import Functional
from molecule import Molecule, kinetic_density

from train import compute_features
from utils.types import PyTree, Scalar

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

def constraint_x3(functional: Functional, params: PyTree, molecule: Molecule, gamma: Scalar = 2, precision = Precision.HIGHEST):
    r"""
    ::math::
        E_x[n_\gamma]  = \gamma E_x[n],

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
    """

    # Compute the input features
    features = compute_features(functional, molecule)[0]
    # Mask the correlation features
    features = jnp.einsum('rf,f->rf', features, functional.exchange_mask, precision=precision)
    # Compute the energy
    Exc = functional.apply_and_integrate(params, molecule, features)

    # Replace the molecule properties with the scaled ones
    ao0 = molecule.ao
    grad_ao0 = molecule.grad_ao
    grad_n_ao0 = molecule.grad_n_ao
    grid_coords0 = molecule.grid.coords
    grid_weights0 = molecule.grid.weights
    chi0 = molecule.chi

    # Computing the scaled properties
    ao = molecule.ao*gamma**(3/2)
    grad_ao = molecule.grad_ao * gamma**(3/2+1)
    grad_n_ao ={}
    for k in molecule.grad_n_ao.keys():
        grad_n_ao[k] = molecule.grad_n_ao[k] * gamma**(3/2+k)
    chi = molecule.chi * gamma**(5/2)
    molecule = molecule.replace(ao = ao, grad_ao = grad_ao, grad_n_ao = grad_n_ao, chi = chi)

    grid = molecule.grid
    grid = grid.replace(coords = grid.coords*gamma)
    grid = grid.replace(weights = grid.weights/gamma**3)
    molecule = molecule.replace(grid = grid)

    # Compute the input features
    featuresg = compute_features(functional, molecule)[0]
    # Mask the correlation features
    featuresg = jnp.einsum('rf,f->rf', featuresg, functional.exchange_mask, precision=precision)
    # Compute the energy
    Excg = functional.apply_and_integrate(params, molecule, featuresg)

    # Reinserting original molecule properties
    molecule = molecule.replace(ao = ao0, grad_ao = grad_ao0, grad_n_ao = grad_n_ao0, chi = chi0)
    grid = molecule.grid
    grid = grid.replace(coords = grid_coords0)
    grid = grid.replace(weights = grid_weights0)
    molecule = molecule.replace(grid = grid)

    return jnp.isclose(gamma*Exc, Excg)







