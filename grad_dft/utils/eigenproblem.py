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

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jaxtyping import Array, Float, Scalar
from typing import Tuple
#from .types import Array, Scalar

# Probably don't alter these unless you know what you're doing
DEGEN_TOL = 1e-6 #
BROADENING = 1e-10

@custom_vjp
def safe_eigh(A: Array) -> tuple[Array, Array]:
    r"""Get the eigenvalues and eigenvectors for an input real symmetric matrix.
        A safe reverse mode gradient is implemented in safe_eigh_rev below.

    Args:
        A (Array): a 2D Jax array representing a real symmetric matrix.

    Returns:
        tuple[Array, Array]: the eigenvalues and eigenvectors of the input real symmetric matrix.
    """
    evecs, evals = jnp.linalg.eigh(A)
    return evecs, evals


def safe_eigh_fwd(A: Array) -> tuple[tuple[Array, Array], tuple[tuple[Array, Array], Array]]:
    r"""Forward mode operation of safe_eigh. Saves evecs and evals for the reverse pass.

    Args:
        A (Array): a 2D Jax array representing a real symmetric matrix.

    Returns:
        tuple[tuple[Array, Array], tuple[tuple[Array, Array], Array]]: eigenvectors, eigenvalues and the input real symmetric matrix A.
    """
    evecs, evals = safe_eigh(A)
    return (evecs, evals), ((evecs, evals), A)


def safe_eigh_rev(res: tuple[tuple[Array, Array], Array], g: Array) -> tuple[Array]:
    r"""Use the Lorentzian broading approach suggested in https://doi.org/10.1038/s42005-021-00568-6
        to calculate stable backward mode gradients for degenerate eigenvectors. We only apply this
        technique if eigenvalues are detected to be degenerate according to the constant DEGEN_TOL
        in this module. When degeneracies are detected, the are broadened according to the constant
        BROADENING also defined in this module.

    Args:
        res (tuple[tuple[Array, Array]): eigenvectors, eigenvales and the input real symmetric matrix A saved from the forward pass
        g (Array): the gradients d[eigenvalues]/dA and d[eigenvectors]/dA

    Returns:
        tuple[Array]: the matrix of reverse mode gradients.

    """
    (evals, evecs), A = res
    grad_evals, grad_evecs = g
    grad_evals_diag = jnp.diag(grad_evals)
    evecs_trans = evecs.T

    # Generate eigenvalue difference matrix
    eval_diff = evals.reshape((1, -1)) - evals.reshape((-1, 1))
    # Find elements where degen_tol condition was or wasn't was met
    mask_degen = (jnp.abs(eval_diff) < DEGEN_TOL).astype(jnp.int64)
    mask_non_degen = (jnp.abs(eval_diff) >= DEGEN_TOL).astype(jnp.int64)

    # Regular gap for non_degen terms => 1/(e_j - e_i)
    # Will get +infs turning to large numbers here if degeneracies are present.
    # This doesn't matter as they multiply by 0 in the forthcoming mask when calculating
    # the F-matrix
    regular_gap = jnp.nan_to_num(jnp.divide(1, eval_diff))

    # Lorentzian broadened gap for degen terms => (e_j - e_i)/((e_j - e_i)^2 + eps)
    broadened_gap = eval_diff / (eval_diff * eval_diff + BROADENING)

    # Calculate full F matrix. large numbers generated by NaNs from regular_gap are deleted here
    F = 0.5 * (jnp.multiply(mask_non_degen, regular_gap) + jnp.multiply(mask_degen, broadened_gap))

    # Set diagonals to 0
    F = F.at[jnp.diag_indices_from(F)].set(0)

    # Calculate the gradient
    grad = (
        jnp.linalg.inv(evecs_trans)
        @ (0.5 * grad_evals_diag + jnp.multiply(F, evecs_trans @ grad_evecs))
        @ evecs_trans
    )
    # Symmetrize
    grad_sym = grad + grad.T
    return (grad_sym,)


safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_rev)


def safe_general_eigh(A: Array, B: Array) -> tuple[Array, Array]:
    r"""Solve the general eigenproblem for the eigenvalues and eigenvectors. I.e,
        . math::
            AC = ECB
        for matrix of eigenvectors C and diagonal matrix of eigenvalues E. This function requires all input
        matrices to real and symmetric and the matrix B to be invertible.

    Args:
        A (Array): a real symmetric matrix
        B (Array): another real symmetric matrix

    Returns:
        tuple[Array, Array]: the eigenvalues and matrix of eigenvectors
    """
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = safe_eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original


def safe_fock_solver(
    fock: Float[Array, 'spin orbitals orbitals'], 
    overlap: Float[Array, 'orbitals orbitals']
) -> (Float[Array, 'spin orbitals'], Float[Array, 'spin orbitals orbitals']):
    """Get the eigenenergies and molecular orbital coefficients for the
        up and down fock spin matrices.
    Args:
        fock (tuple[Array, Array]): the up and down fock spin matrices
        overlap (Array): the overlap matrix

    Returns:
        tuple[Array, Array]: the eigenenergies and matrix of molecular orbital coefficients.
    """
    mo_energies_up, mo_coeffs_up = safe_general_eigh(fock[0], overlap)
    mo_energies_dn, mo_coeffs_dn = safe_general_eigh(fock[1], overlap)
    return jnp.stack((mo_energies_up, mo_energies_dn), axis=0), jnp.stack(
        (mo_coeffs_up, mo_coeffs_dn), axis=0
    )
