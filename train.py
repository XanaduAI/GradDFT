from typing import Callable, Optional, Sequence, Tuple
from functools import partial

from jax import numpy as jnp
from jax import value_and_grad
from jax.profiler import annotate_function
from jax.lax import Precision, stop_gradient

from utils import Scalar, Array, PyTree
from functional import Functional
from molecule import Molecule, coulomb_potential, symmetrize_rdm1

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

        return functional.energy(params, molecule, *functional_inputs, **functional_kwargs)

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
            if int(molecule.spin) == 0: fock = fock.sum(axis = 0)/2.
            fock = fock + jnp.stack([molecule.h1e, molecule.h1e], axis=0)

        return energy, fock

    return predict