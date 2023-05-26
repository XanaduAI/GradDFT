from typing import Callable, Optional, Sequence, Tuple, Union, Dict
from functools import partial

from jax import numpy as jnp
from jax import value_and_grad
from jax.profiler import annotate_function

from utils import Scalar, Array, PyTree
from functional import Functional
from molecule import Molecule, coulomb_potential

def molecule_predictor(
    functional: Functional,
    feature_fn: Optional[Callable] = None,
    **kwargs,
) -> Callable:
    
    """Generate a function that predicts the energy
    energy of a `Molecule` and a corresponding Fock matrix

    Parameters
    ----------
    fxc : Functional
        A callable or a `flax.linen.Module` that predicts the
        exchange-correlation energy given some parameters.
        A callable must have the following signature:

        fxc.apply_and_integrate(params: Array, molecule: Molecule, **functional_kwargs) -> Scalar

        where `params` is any parameter pytree, and `molecule`
        is a Molecule class instance.

    feature_fn : Callable, optional
        A function that calculates and/or loads the molecule features.
        If given, it must be a callable with the following signature:

        feature_fn(molecule: Molecule, *args, **kwargs) -> Tuple[Array, Array, Optional[Array]]


    Returns
    -------
    Callable
        A wrapped verison of `fxc` that calculates input/output features and returns
        the predicted exchange-correlation energy with the corresponding Fock matrix.
        Signature:

        (params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Array]

    Notes
    -----
    In a nutshell, this takes any Jax-transformable functional and does two things:
        1.) Wraps it in a way to return the Fock matrix as well as
        the exchange-correlation energy.
        2.) Explicitly takes a function to generate/load precomputed
        features to feed into a parameterized functional for flexible
        feature generation.

    Examples
    --------
    Given a `Molecule`:
    >>> from qdft import FeedForwardFunctional
    >>> Fxc = FeedForwardFunctional(layer_widths=[128, 128])
    >>> params = Fxc.init(jax.random.PRNGKey(42), jnp.zeros(shape=(32, 11)))
    >>> predictor = make_molecule_predictor(Fxc, returns_feature_grads=False, chunk_size=1000)
    `chunk` size is forwarded to the default feature function as a keyword parameter.
    >>> exc, fock = predictor(params, molecule) # `Might take a while for the default_features`
    >>> fock.shape == molecule.density_matrix.shape
    True
    """

    @partial(value_and_grad, argnums=1)
    def energy_and_grads(
        params: PyTree, rdm1: Array, molecule: Molecule,  functional_kwargs: dict = {}, *args
    ) -> Scalar:

        molecule = molecule.replace(rdm1 = rdm1)
        functional_inputs = feature_fn(molecule, *args, **kwargs)

        return functional.apply_and_integrate(params, molecule, *functional_inputs, **functional_kwargs)

    @partial(annotate_function, name="predict")
    def predict(params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Array]:

        """A DFT functional wrapper, returning the predicted exchange-correlation
        energy as well as the corresponding Fock matrix. This function does **not** require
        that the provided `feature_fn` returns derivatives (Jacobian matrix) of provided
        input features.

        Parameters
        ----------
        params : PyTree
            The functional parameters.
        molecule : Molecule
            The `Molecule` object to predict properties of.
        functional_kwargs : dict, optional
            A `dict` of optional additional keyword arguments
            to pass to the functional.

        Returns
        -------
        Tuple[Scalar, Array]
            A tuple of the predicted exchange-correlation energy and the corresponding
            Fock matrix of the same shape as `molecule.density_matrix`:
            (*batch_size, n_spin, n_orbitals, n_orbitals).
        """

        energy, fock = energy_and_grads(params, molecule.rdm1, molecule, *args)
        fock = 1/2*(fock + fock.transpose(0,2,1))

        if functional.is_xc:
            energy += molecule.nonXC()
            fock += coulomb_potential(molecule.rdm1, molecule.rep_tensor) 
            fock += jnp.stack([molecule.h1e, molecule.h1e], axis=0)

        return energy, fock

    return predict