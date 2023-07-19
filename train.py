from typing import Callable, Optional, Tuple
from functools import partial

from jax import numpy as jnp, vmap
from jax import value_and_grad
from jax.profiler import annotate_function
from jax.lax import stop_gradient, cond, fori_loop
from flax import struct
from optax import OptState, GradientTransformation, apply_updates

from utils import Scalar, Array, PyTree
from functional import DispersionFunctional, Functional
from molecule import Molecule, coulomb_potential, symmetrize_rdm1, eig, orbital_grad

def compute_features(functional, molecule, *args, **kwargs):
    r"""
    Computes the features for the functional
    """

    if functional.nograd_features and functional.features:
        functional_inputs = functional.features(molecule, *args, **kwargs)
        nograd_functional_inputs = stop_gradient(functional.nograd_features(molecule, *args, **kwargs))
        functional_inputs = functional.combine(functional_inputs, nograd_functional_inputs)

    elif functional.features:
        functional_inputs = functional.features(molecule, *args, **kwargs)

    elif functional.nograd_features:
        functional_inputs = stop_gradient(functional.nograd_features(molecule, *args, **kwargs))
    return functional_inputs

def molecule_predictor(
    functional: Functional,
    nlc_functional: DispersionFunctional = None,
    **kwargs,
) -> Callable:
    
    r"""Generate a function that predicts the energy
    energy of a `Molecule` and a corresponding Fock matrix

    Parameters
    ----------
    functional : Functional
        A callable or a `flax.linen.Module` that predicts the
        exchange-correlation energy given some parameters.
        A callable must have the following signature:

        fxc.energy(params: Array, molecule: Molecule, **functional_kwargs) -> Scalar

        where `params` is any parameter pytree, and `molecule`
        is a Molecule class instance.

    Returns
    -------
    Callable
        A wrapped verison of `fxc` that calculates input/output features and returns
        the predicted energy with the corresponding Fock matrix.
        Signature:

        (params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Array]

    Notes
    -----
    In a nutshell, this takes any Jax-transformable functional and does two things:
        1.) Wraps it in a way to return the Fock matrix as well as
        the energy.
        2.) Explicitly takes a function to generate/load precomputed
        features to feed into a parameterized functional for flexible
        feature generation.

    Examples
    --------
    Given a `Molecule`:
    >>> from qdft import FeedForwardFunctional
    >>> Fxc = FeedForwardFunctional(layer_widths=[128, 128])
    >>> params = Fxc.init(jax.random.PRNGKey(42), jnp.zeros(shape=(32, 11)))
    >>> predictor = make_molecule_predictor(Fxc, chunk_size=1000)
    `chunk` size is forwarded to the default feature function as a keyword parameter.
    >>> e, fock = predictor(params, molecule) # `Might take a while for the default_features`
    >>> fock.shape == molecule.density_matrix.shape
    True
    """

    @partial(value_and_grad, argnums=1)
    def energy_and_grads(
        params: PyTree, rdm1: Array, molecule: Molecule, *args, **functional_kwargs
    ) -> Scalar:
        r"""
        Computes the energy and gradients with respect to the density matrix

        Parameters
        ----------
        params: Pytree
            Functional parameters
        rdm1: Array
            The reduced density matrix.
            Expected shape: (n_grid_points, n_orbitals, n_orbitals)
        molecule: Molecule
            the molecule

        Returns
        -----------
        Scalar
            The energy of the molecule when the state of the system is given by rdm1.
        """

        molecule = molecule.replace(rdm1 = rdm1)

        functional_inputs = compute_features(functional, molecule, *args, **kwargs)

        e = functional.energy(params, molecule, *functional_inputs, **functional_kwargs)
        if nlc_functional:
            e = e + nlc_functional.energy({'params': params['dispersion']}, molecule, **functional_kwargs)
        return e

    @partial(annotate_function, name="predict")
    def predict(params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Array]:

        r"""A DFT functional wrapper, returning the predicted exchange-correlation
        energy as well as the corresponding Fock matrix. This function does **not** require
        that the provided `feature_fn` returns derivatives (Jacobian matrix) of provided
        input features.

        Parameters
        ----------
        params : PyTree
            The functional parameters.
        molecule : Molecule
            The `Molecule` object to predict properties of.
        *args

        Returns
        -------
        Tuple[Scalar, Array]
            A tuple of the predicted exchange-correlation energy and the corresponding
            Fock matrix of the same shape as `molecule.density_matrix`:
            (*batch_size, n_spin, n_orbitals, n_orbitals).
        """

        energy, fock = energy_and_grads(params, molecule.rdm1, molecule, *args)
        fock = 1/2*(fock + fock.transpose(0,2,1))

        # HF Potential
        if functional.features:
            functional_inputs = functional.features(molecule, *args, **kwargs)
        else: functional_inputs = None

        if functional.featuregrads:
            nograd_functional_inputs = stop_gradient(functional.nograd_features(molecule, *args, **kwargs))
            vxc_expl = functional.featuregrads(functional, params, molecule, functional_inputs, nograd_functional_inputs)
            fock += vxc_expl+vxc_expl.transpose(0,2,1) # Sum over omega

        if functional.is_xc:
            rdm1 = symmetrize_rdm1(molecule.rdm1)
            fock += coulomb_potential(rdm1, molecule.rep_tensor)
            fock = cond(jnp.isclose(molecule.spin, 0), # Condition
                            lambda x: x, # Truefn branch
                            lambda x: jnp.stack([x.sum(axis = 0)/2., x.sum(axis = 0)/2.], axis=0), # Falsefn branch
                            fock) # Argument
            fock = fock + jnp.stack([molecule.h1e, molecule.h1e], axis=0)

        return energy, fock

    return predict


##################### Regularization #####################

def fock_grad_regularization(molecule: Molecule, F: Array) -> Scalar:
    """Calculates the Fock alternative regularization term for a `Molecule` given a Fock matrix.
    
    Parameters
    ----------
    molecule : Molecule
        A `Molecule` object.
    F : Array
        The Fock matrix array. Has to be of the same shape as `molecule.density_matrix`

    Returns
    -------
    Scalar
        The Fock gradient regularization term alternative.
    """
    return jnp.sqrt(jnp.einsum('sij->', (F - molecule.fock)**2 )) / jnp.sqrt(jnp.einsum('sij->', molecule.fock**2))

def dm21_grad_regularization(molecule: Molecule, F: Array) -> Scalar:

    """Calculates the default gradient regularization term for a `Molecule` given a Fock matrix.

    Parameters
    ----------
    molecule : Molecule
        A `Molecule` object.
    F : Array
        The Fock matrix array. Has to be of the same shape as `molecule.density_matrix`

    Returns
    -------
    Scalar
        The gradient regularization term of the DM21 variety.
    """

    n = molecule.mo_occ
    e = molecule.mo_energy
    C = molecule.mo_coeff

    #factors = jnp.einsum("sba,sac,scd->sbd", C.transpose(0,2,1), F, C) ** 2
    #factors = jnp.einsum("sab,sac,scd->sbd", C, F, C) ** 2
    #factors = jnp.einsum("sac,sab,scd->sbd", F, C, C) ** 2
    factors = jnp.einsum("sac,sab,scd->sbd", F, C, C) ** 2 # F is symmetric

    numerator = n[:, :, None] - n[:, None, :]
    denominator = e[:, :, None] - e[:, None, :]

    mask = jnp.logical_and(jnp.abs(factors) > 0, jnp.abs(numerator) > 0)

    safe_denominator = jnp.where(mask, denominator, 1.0)

    second_mask = jnp.abs(safe_denominator) > 0
    safe_denominator = jnp.where(second_mask, safe_denominator, 1.e-20)

    prefactors = numerator / safe_denominator

    dE = jnp.clip(0.5 * jnp.sum(prefactors * factors), a_min = -10, a_max = 10)

    return dE**2

def orbital_grad_regularization(molecule: Molecule, F: Array) -> Scalar:
    """Deprecated"""

    #  Calculate the gradient regularization term
    new_grad = get_grad(molecule.mo_coeff, molecule.mo_occ, F)

    dE = jnp.linalg.norm(new_grad - molecule.training_gorb_grad, ord = 'fro')

    return dE**2

def get_grad(mo_coeff, mo_occ, F):
    '''RHF orbital gradients

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
    '''

    C_occ = vmap(jnp.where, in_axes = (None, 1, None), out_axes=1)(mo_occ > 0, mo_coeff, 0)
    C_vir = vmap(jnp.where, in_axes = (None, 1, None), out_axes=1)(mo_occ == 0, mo_coeff, 0)

    return jnp.einsum("sab,sac,scd->bd", C_vir.conj(), F, C_occ)




##################### SCF training loop #####################


def make_scf_training_loop(functional: Functional, max_cycles: int = 25,
                            **kwargs) -> Callable:
    
    r"""
    Creates an scf_iterator object that can be called to implement a self-consistent loop,
    intented to be jax.jit compatible (fully self-differentiable).
    If you are looking for a more flexible but not differentiable scf loop, see evaluate.py make_scf_loop.

    Main parameters
    ---------------
    functional: Functional
    max_cycles: int, default to 25

    Returns
    ---------
    float
    """

    predict_molecule = molecule_predictor(functional, chunk_size = None, **kwargs)

    def scf_training_iterator(
        params: PyTree, molecule: Molecule, *args
    ) -> Tuple[Scalar, Scalar]:
        
        r"""
        Implements a scf loop intented for use in a jax.jit compiled function (training loop).
        If you are looking for a more flexible but not differentiable scf loop, see evaluate.py make_scf_loop.
        It asks for a Molecule and a functional implicitly defined predict_molecule with
        parameters params

        Parameters
        ----------
        params: PyTree
        molecule: Molecule
        *args: Arguments to be passed to predict_molecule function
        """

        if molecule.omegas:
            raise NotImplementedError("SCF training loop not implemented for (range-separated) exact-exchange functionals. \
                                    Doing so would require a differentiable way of recomputing the chi tensor.")

        old_e = jnp.inf
        norm_gorb = jnp.inf

        predicted_e, fock = predict_molecule(params, molecule, *args)

        # Initialize DIIS
        A = jnp.identity(molecule.s1e.shape[0])
        diis = TrainingDiis(overlap_matrix=molecule.s1e, A = A, max_diis = 10)
        diis_data = (jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])), 
                    jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])), 
                    jnp.zeros(diis.max_diis), 
                    jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])))

        state = (molecule, fock, predicted_e, old_e, norm_gorb, diis_data)

        def loop_body(cycle, state):

            old_state = state
            molecule, fock, predicted_e, old_e, norm_gorb, diis_data = old_state
            old_e = predicted_e

            # DIIS iteration
            new_data = (molecule.rdm1, fock, predicted_e)
            fock, diis_data = diis.run(new_data, diis_data, cycle)

            # Diagonalize Fock matrix
            mo_energy, mo_coeff = eig(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff = mo_coeff)
            molecule = molecule.replace(mo_energy = mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()

            # Update the density matrix
            rdm1 = molecule.make_rdm1()
            molecule = molecule.replace(rdm1 = rdm1)

            # Compute the new energy and Fock matrix
            predicted_e, fock = predict_molecule(params, molecule, *args)

            # Compute the norm of the gradient
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

            state = (molecule, fock, predicted_e, old_e, norm_gorb, diis_data)

            return state

        # Compute the scf loop
        final_state = fori_loop(0, max_cycles, body_fun = loop_body, init_val = state)
        molecule, fock, predicted_e, old_e, norm_gorb, diis_data = final_state

        # Perform a final diagonalization without diis (reinitializing)
        diis_data = (jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])), 
                    jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])), 
                    jnp.zeros(diis.max_diis), 
                    jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])))
        state = (molecule, fock, predicted_e, old_e, norm_gorb, diis_data)
        state = loop_body(0, state)
        molecule, fock, predicted_e, _, _, _ = final_state

        return predicted_e, fock, molecule.rdm1

    return scf_training_iterator


@struct.dataclass
class TrainingDiis:

    r"""DIIS extrapolation, intended for training of the resulting energy of a scf loop.
    If you are looking for a more flexible, not differentiable DIIS, see evaluate.py DIIS class
    The implemented CDIIS computes the Fock matrix as a linear combination of the previous Fock matrices, with
    ::math::
        F_{DIIS} = \sum_i x_i F_i,

    where the coefficients are determined by minimizing the error vector
    ::math::
        e_i = A^T (F_i D_i S - S D_i F_i) A,

    with F_i the Fock matrix at iteration i, D_i the density matrix at iteration i,
    and S the overlap matrix. The error vector is then used to compute the
    coefficients as
    ::math::
        B = \begin{pmatrix}
            <e_1|e_1> & <e_1|e_2> & \cdots & <e_1|e_n> & -1 \\
            <e_2|e_1> & <e_2|e_2> & \cdots & <e_2|e_n> & -1 \\
            \vdots & \vdots & \ddots & \vdots & \vdots \\
            <e_n|e_1> & <e_n|e_2> & \cdots & <e_n|e_n> & -1 \\
            -1 & -1 & \cdots & -1 & 0
        \end{pmatrix},

    ::math::
        x = \begin{pmatrix}
            x_1 \\
            x_2 \\
            \vdots \\
            x_n \\
            0
        \end{pmatrix}
    
    and
    ::math::
        C= \begin{pmatrix}
            0 \\
            0 \\
            \vdots \\
            0 \\
            1
        \end{pmatrix}

    where n is the number of stored Fock matrices. The coefficients are then
    computed as
    ::math::
        x = B^{-1} C.

    Diis attributes:
        overlap_matrix (jnp.array): Overlap matrix, molecule.s1e. Shape: (n_orbitals, n_orbitals).
        A (jnp.array): Transformation matrix for CDIIS, molecule.A. Shape: (n_orbitals, n_orbitals).
        max_diis (int): Maximum number of DIIS vectors to store. Defaults to 8.

    Other objects used during the calculation:
        density_vector (jnp.array): Density matrix vectorized.
            Shape: (n_iterations, spin, n_orbitals, n_orbitals).
        fock_vector (jnp.array): Fock matrix vectorized.
            Shape: (n_iterations, spin, n_orbitals, n_orbitals).
        energy_vector (jnp.array): Fock energy vector.
            Shape: (n_iterations).
        error_vector (jnp.array): Error vector.
            Shape: (n_iterations, spin, n_orbitals, n_orbitals).
    """

    overlap_matrix: Array
    A: Array
    max_diis: Optional[int] = 8

    def update(self, new_data, diis_data, cycle):

        density_matrix, fock_matrix, energy = new_data
        density_vector, fock_vector, energy_vector, error_vector = diis_data

        fds =  jnp.einsum('ij,sjk,skl,lm,mn->sin', self.A, fock_matrix, density_matrix, self.overlap_matrix, self.A.T) 
        error_matrix = fds - fds.transpose(0,2,1).conj()

        error_vector = cond(jnp.greater(cycle, self.max_diis),
                            lambda error_vector, error_matrix: jnp.concatenate((error_vector, jnp.expand_dims(error_matrix, axis = 0)), axis = 0)[1:],
                            lambda error_vector, error_matrix: error_vector.at[cycle].set(error_matrix),
                            error_vector, error_matrix)
        density_vector = cond(jnp.greater(cycle, self.max_diis),
                            lambda density_vector, density_matrix: jnp.concatenate((density_vector, jnp.expand_dims(density_matrix, axis = 0)), axis = 0)[1:],
                            lambda density_vector, density_matrix: density_vector.at[cycle].set(density_matrix),
                            density_vector, density_matrix)
        fock_vector = cond(jnp.greater(cycle, self.max_diis),
                            lambda fock_vector, fock_matrix: jnp.concatenate((fock_vector, jnp.expand_dims(fock_matrix, axis = 0)), axis = 0)[1:],
                            lambda fock_vector, fock_matrix: fock_vector.at[cycle].set(fock_matrix),
                            fock_vector, fock_matrix)
        energy_vector = cond(jnp.greater(cycle, self.max_diis),
                            lambda energy_vector, energy: jnp.concatenate((energy_vector, jnp.expand_dims(energy, axis = 0)), axis = 0)[1:],
                            lambda energy_vector, energy: energy_vector.at[cycle].set(energy),
                            energy_vector, energy)

        return density_vector, fock_vector, energy_vector, error_vector

    def run(self, new_data, diis_data, cycle = 0):

        diis_data = self.update(new_data, diis_data, cycle)
        density_vector, fock_vector, energy_vector, error_vector = diis_data

        x = self.cdiis_minimize(error_vector, cycle)
        F = jnp.einsum('si,isjk->sjk', x, fock_vector)
        return jnp.einsum('ji,sjk,kl->sil', self.A, F, self.A), diis_data

    def cdiis_minimize(self, error_vector, cycle):

        # Find the coefficients x that solve B @ x = C with B and C defined below
        B = jnp.zeros((2, len(error_vector) + 1, len(error_vector) + 1))
        B = B.at[:, 1:, 1:].set(jnp.einsum('iskl,jskl->sij', error_vector, error_vector))

        def assign_values(i, B):
            value = cond(jnp.less_equal(i,cycle), lambda _: 1.0, lambda _: 0.0, operand=None)
            B = B.at[:, 0, i+1].set(value) # Make 0 if i > cycle, else 1
            B = B.at[:, i+1, 0].set(value) # Make 0 if i > cycle, else 1
            return B

        def assign_values_diag(i, B):
            value = cond(jnp.less_equal(i,cycle), 
                        lambda error_vector: jnp.einsum('iskl,jskl->sij', error_vector, error_vector)[:, i, i], 
                        lambda _: jnp.array([1.0, 1.0]), 
                        error_vector)
            B = B.at[:, i+1, i+1].set(value)
            return B

        B = fori_loop(0, error_vector.shape[0]+2, assign_values, B)
        B = fori_loop(0, error_vector.shape[0]+2, assign_values_diag, B)
    
        C = jnp.zeros((2, len(error_vector) + 1))
        C = C.at[:, 0].set(1)

        x0 = jnp.linalg.inv(B[0]) @ C[0]
        x1 = jnp.linalg.inv(B[1]) @ C[1]
        x = jnp.stack([x0, x1], axis=0)

        return x[:,1:]
