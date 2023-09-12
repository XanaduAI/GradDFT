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

"""The goal of this module is to test that the code in ~/utils/eigenproblem.py produces:

(1) The same gradients as jnp.linalg.eigh for input real symmetric matrices with non-degenerate eigenvalues.

(2) Gradients free of NaN's when the problem is degenerate.

Subsequently, we only aim to test the implementation safe_eigh
"""

from jax import config
from jax.random import PRNGKey, normal, randint
import jax.numpy as jnp
from jax import jacrev

import numpy as np

from grad_dft.utils.eigenproblem import safe_eigh
from jaxtyping import Array, Scalar

import pytest

config.update("jax_enable_x64", True)

ABS_THRESH = 1e-10
SEEDS = [1984, 1993, 1945, 2001, 10, 29, 101, 1992]
RANDOM_KEYS = [PRNGKey(seed) for seed in SEEDS]
MATRIX_SIZES = jnp.arange(2, 10)
GRAD_REV_FN_JNP = jacrev(jnp.linalg.eigh)
GRAD_REV_FN_SAFE_EIGH = jacrev(safe_eigh)


def rand_sym_mat(matrix_size: Scalar, rand_key: PRNGKey) -> Array:
    """Generate a real symmetric matrix

    Args:
        matrix_size (Scalar): the square dimensions of the real symmetric matrix to be generated.
        rand_key (PRNGKey): the jax-type random key for seeding RNG.

    Returns:
        Array: a random real symmetric matrix
    """
    random_matrix = normal(rand_key, (matrix_size, matrix_size))
    return 0.5 * (random_matrix + random_matrix.T)


def generate_symmetric_matrix_with_degenerate_eigenvalue(matrix_size: Scalar, rand_key: PRNGKey):
    """Generate a real symmetric matrix guaranteed to have one denegerate eigenvalue

    Args:
        matrix_size (Scalar): the square dimensions of the real symmetric matrix to be generated.
        rand_key (PRNGKey): the jax-type random key for seeding RNG.

    Returns:
        Array: a random real symmetric matrix guaranteed to have one degenerate eigenvalue
    """

    sym_mat = rand_sym_mat(matrix_size, rand_key)

    # Add a degenerate eigenvalue by duplicating one eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(sym_mat)
    index_to_duplicate = randint(rand_key, (1,), 0, matrix_size)
    eigenvalues[index_to_duplicate] = eigenvalues[index_to_duplicate - 1]

    # Reconstruct the matrix with the modified eigenvalues
    A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

    return A


def test_non_degen_rev_mode_jacobians() -> None:
    r"""Check that the reverse mode jacobians match between the jnp.linalg.eigh implementation and our
        custom safe_eigh implementation when no degeneracies are present.
    Args:
        None
    Returns:
        None
    """
    for mat_size in MATRIX_SIZES:
        for i, key in enumerate(RANDOM_KEYS):
            sym_mat = rand_sym_mat(mat_size, key)
            jac_jnp = GRAD_REV_FN_JNP(sym_mat)
            jac_safe_eigh = GRAD_REV_FN_SAFE_EIGH(sym_mat)
            assert jac_jnp[0] == pytest.approx(
                jac_safe_eigh[0], abs=1e-10
            ), f"Reverse mode jacobian difference comparing jnp.linalg.eigh and safe_eigh for seed {SEEDS[i]} and matrix_size {mat_size} exceeds threshold: {ABS_THRESH}"
            assert jac_jnp[1] == pytest.approx(
                jac_safe_eigh[1], abs=1e-10
            ), f"Reverse mode jacobian difference comparing jnp.linalg.eigh and safe_eigh for seed {SEEDS[i]} and matrix_size {mat_size} exceeds threshold: {ABS_THRESH}"


def test_degen_rev_mode_jacobians_for_nans() -> None:
    r"""Check that the reverse mode jacobian contains no NaNs when passed a symmetric real matrix with degenerate eigenvalues

    Args:
        None
    Returns:
        None
    """
    for mat_size in MATRIX_SIZES:
        for i, key in enumerate(RANDOM_KEYS):
            degen_sym_mat = generate_symmetric_matrix_with_degenerate_eigenvalue(mat_size, key)
            jac_safe_eigh = GRAD_REV_FN_SAFE_EIGH(degen_sym_mat)
            assert not jnp.isnan(
                jac_safe_eigh[0]
            ).any(), f"Reverse mode jacobian element 0 for safe_eigh for seed {SEEDS[i]} and matrix_size {mat_size} contained atleast one NaN when passed a matrix with degenerate eigenvalues/eigenvectors"
            assert not jnp.isnan(
                jac_safe_eigh[0]
            ).any(), f"Reverse mode jacobian element 1 for safe_eigh for seed {SEEDS[i]} and matrix_size {mat_size} contained atleast one NaN when passed a matrix with degenerate eigenvalues/eigenvectors"
