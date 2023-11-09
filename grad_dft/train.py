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

from typing import Callable, Optional, Tuple, Union
from functools import partial
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar, Float, Complex

from jax import numpy as jnp, vmap
from jax import value_and_grad
from jax.profiler import annotate_function
from jax.lax import stop_gradient
from optax import OptState, GradientTransformation, apply_updates

from grad_dft import (
    coulomb_energy,
    DispersionFunctional, 
    Functional,
    Molecule,
    Solid, 
    abs_clip, 
)

def energy_predictor(
    functional: Functional,
    nlc_functional: Optional[DispersionFunctional] = None,
    clip_cte: float = 1e-30,
    **kwargs,
) -> Callable:
    r"""Generate a function that predicts the energy
    energy of a `Molecule` or `Solid` and a corresponding Fock matrix

    Parameters
    ----------
    functional : Functional
        A callable or a `flax.linen.Module` that predicts the
        exchange-correlation energy given some parameters.
        A callable must have the following signature:

        fxc.energy(params: Array, atoms: Union[Molecule, Solid], **functional_kwargs) -> Scalar

        where `params` is any parameter pytree, and `atoms`
        is a `Molecule` or `Solid` class instance.

    Returns
    -------
    Callable
        A wrapped verison of `fxc` that calculates input/output features and returns
        the predicted energy with the corresponding Fock matrix.
        Signature:

        (params: PyTree, atoms: Union[Molecule, Solid], *args) -> Tuple[Scalar, Array]

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
    >>> predictor = energy_predictor(Fxc, chunk_size=1000)
    `chunk` size is forwarded to the default feature function as a keyword parameter.
    >>> e, fock = predictor(params, molecule) # `Might take a while for the default_features`
    >>> fock.shape == molecule.density_matrix.shape
    True
    """

    @partial(value_and_grad, argnums=1)
    def energy_and_grads(
        params: PyTree,
        rdm1: Union[Float[Array, "spin orbitals orbitals"],
                    Complex[Array, "spin kpt orbitals orbitals"]
                ],
        atoms: Union[Molecule, Solid],
        *args,
        **functional_kwargs,
    ) -> Scalar:
        r"""
        Computes the energy and gradients with respect to the density matrix

        Parameters
        ----------
        params: Pytree
            Functional parameters
        rdm1: Float[Array, "spin orbitals orbitals"]
            The 1-body reduced density matrix.
        atoms: Union[Molecule, Solid]
            The collection of atoms.

        Returns
        -----------
        Scalar
            The energy of the atoms when the state of the system is given by rdm1.
        """

        atoms = atoms.replace(rdm1=rdm1)

        e = functional.energy(params, atoms, *args, **functional_kwargs)
        if nlc_functional:
            e = e + nlc_functional.energy(
                {"params": params["dispersion"]}, atoms, **functional_kwargs
            )
        return e

    @partial(annotate_function, name="predict")
    def predict(params: PyTree, atoms: Union[Molecule, Solid], *args) -> Tuple[Scalar, Array]:
        r"""A DFT functional wrapper, returning the predicted exchange-correlation
        energy as well as the corresponding Fock matrix. This function does **not** require
        that the provided `feature_fn` returns derivatives (Jacobian matrix) of provided
        input features.

        Parameters
        ----------
        params : PyTree
            The functional parameters.
        atoms: Union[Molecule, Solid]
            The collection of atoms.
        *args

        Returns
        -------
        Tuple[Scalar, Array]
            A tuple of the predicted exchange-correlation energy and the corresponding
            Fock matrix of the same shape as `atoms.rdm1`:
            (*batch_size, n_spin, n_orbitals, n_orbitals) for a `Molecule` or 
            (*batch_size, n_spin, n_kpt, n_orbitals, n_orbitals) for a `Solid`.
        """
        
        energy, fock = energy_and_grads(params, atoms.rdm1, atoms, *args)
        
        # Improve stability by clipping and symmetrizing
        if isinstance(atoms, Molecule):
            transpose_dims = (0, 2, 1)
        elif isinstance(atoms, Solid):
            transpose_dims = (0, 1, 3, 2)
        fock = abs_clip(fock, clip_cte)
        fock = 1 / 2 * (fock + fock.transpose(transpose_dims).conj())
        fock = abs_clip(fock, clip_cte)
        
        # Compute the features that should be autodifferentiated
        if functional.energy_densities and functional.densitygrads:
            grad_densities = functional.energy_densities(atoms, *args, **kwargs)
            nograd_densities = stop_gradient(functional.nograd_densities(atoms, *args, **kwargs))
            densities = functional.combine_densities(grad_densities, nograd_densities)
        elif functional.energy_densities:
            grad_densities = functional.energy_densities(atoms, *args, **kwargs)
            nograd_densities = None
            densities = grad_densities
        elif functional.densitygrads:
            grad_densities = None
            nograd_densities = stop_gradient(functional.nograd_densities(atoms, *args, **kwargs))
            densities = nograd_densities
        else:
            densities, grad_densities, nograd_densities = None, None, None

        if functional.coefficient_input_grads and functional.coefficient_inputs:
            grad_cinputs = functional.coefficient_inputs(atoms, *args, **kwargs)
            nograd_cinputs = stop_gradient(
                functional.nograd_coefficient_inputs(atoms, *args, **kwargs)
            )
            cinputs = functional.combine_inputs(grad_cinputs, nograd_cinputs)
        elif functional.coefficient_inputs:
            grad_cinputs = functional.coefficient_inputs(atoms, *args, **kwargs)
            nograd_cinputs = None
            cinputs = grad_cinputs
        elif functional.coefficient_input_grads:
            grad_cinputs = None
            nograd_cinputs = stop_gradient(
                functional.nograd_coefficient_inputs(atoms, *args, **kwargs)
            )
            cinputs = nograd_cinputs
        else:
            cinputs, grad_cinputs, nograd_cinputs = None, None, None

        # Compute the derivatives with respect to nograd_densities
        if functional.densitygrads:
            vxc_expl = functional.densitygrads(
                functional, params, atoms, nograd_densities, cinputs, grad_densities
            )
            print(vxc_expl.shape)
            fock += vxc_expl + vxc_expl.transpose(transpose_dims)  # Sum over omega
            fock = abs_clip(fock, clip_cte)

        if functional.coefficient_input_grads:
            vxc_expl = functional.coefficient_input_grads(
                functional, params, atoms, nograd_cinputs, grad_cinputs, densities
            )
            fock += vxc_expl + vxc_expl.transpose(transpose_dims)  # Sum over omega
            fock = abs_clip(fock, clip_cte)

        fock = abs_clip(fock, clip_cte)

        return energy, fock

    return predict

def Harris_energy_predictor(
    functional: Functional,
    **kwargs
):
    r""""
    Generate a function that predicts the Harris energy, according to the function
    
    .. math::
        E_{\rm Harris}[n_0] = \sum_i \epsilon_i - \int \mathrm{d}r^3 v_{\rm xc}[n_0](r) n_0(r) - \tfrac{1}{2} \int \mathrm{d}r^3 v_{\rm H}[n_0](r) n_0(r) + E_{\rm xc}[n_0

    Parameters
    ----------
    functional : Functional
        The functional to use for the Harris energy.
    **kwargs
        Other keyword arguments to the functional.

    Returns
    -------
    Callable
    """
    @partial(value_and_grad, argnums=1)
    def xc_energy_and_grads(
        params: PyTree, 
        rdm1: Float[Array, "spin orbitals orbitals"], 
        atoms: Union[Molecule, Solid], 
        *args, 
        **kwargs
    ) -> Scalar:
        r"""
        Computes the energy and gradients with respect to the density matrix.

        Parameters
        ----------
        params: Pytree
            Functional parameters
        rdm1: Float[Array, "spin orbitals orbitals"]
            The reduced density matrix.
        atoms: Union[Molecule, Solid]
            The collection of atoms.
        *args
        **kwargs

        Returns
        -----------
        Tuple[Scalar, Float[Array, "spin orbitals orbitals"]]
        """
        atoms = atoms.replace(rdm1=rdm1)
        densities = functional.compute_densities(atoms, *args, **kwargs)
        cinputs = functional.compute_coefficient_inputs(atoms, *args)
        return functional.xc_energy(params, atoms.grid, cinputs, densities, **kwargs)

    
    # Works for Molecules only for now
    def Harris_energy(
        params: PyTree,
        molecule: Molecule,
        *args,
        **kwargs,
    ) -> Scalar:
        r"""
        Computes the Harris functional, which is the functional evaluated at the
        ground state density of the molecule.

        Parameters
        ----------
        params : PyTree
            Parameters of the functional.
        molecule : Molecule
            Molecule to compute the Harris functional for.
        *args
            Other arguments to compute_densities or compute_coefficient_inputs.
        **kwargs
            Other key word arguments to densities and self.xc_energy.

        Returns
        -------
        Scalar
        """

        energy = jnp.einsum("sr,sr->", molecule.mo_occ, molecule.mo_energy)

        coulomb_e = -coulomb_energy(molecule.rdm1.sum(axis = 0), molecule.rep_tensor)

        xc_energy, xcfock = xc_energy_and_grads(params, molecule.rdm1, molecule, *args, **kwargs)

        return energy + xc_energy - jnp.einsum("sij,sij->", molecule.rdm1, xcfock) + coulomb_e + molecule.nuclear_repulsion

    return Harris_energy



def train_kernel(tx: GradientTransformation, loss: Callable) -> Callable:
    r"""Generate a training kernel for a given optimizer and loss function.

    Parameters
    ----------
    tx : GradientTransformation
        An optax gradient transformation.
    loss : Callable
        A loss function that takes in the parameters, a `Molecule` or `Solid` object, and the ground truth energy
        and returns a tuple of the loss value and the gradients.

    Returns
    -------
    Callable
    """

    def kernel(
        params: PyTree, opt_state: OptState, atoms: Union[Molecule, Solid], ground_truth_energy: float, *args
    ) -> Tuple[PyTree, OptState, Scalar, Scalar]:
        r""""
        The training kernel updating the parameters according to the loss
        function and the optimizer.

        Parameters
        ----------
        params : PyTree
            The parameters of the functional.
        opt_state : OptState
            The optimizer state.
        atoms: Union[Molecule, Solid]
            The collection of atoms
        ground_truth_energy : float
            The ground truth energy.
        *args

        Returns
        -------
        Tuple[PyTree, OptState, Scalar, Scalar]
            The updated parameters, optimizer state, loss value, and predicted energy.
        """
        (cost_value, predictedenergy), grads = loss(params, atoms, ground_truth_energy)

        updates, opt_state = tx.update(grads, opt_state, params)
        params = apply_updates(params, updates)

        return params, opt_state, cost_value, predictedenergy

    return kernel


##################### Regularization #####################

# Regularization terms only support `Molecule` object for now

def fock_grad_regularization(molecule: Molecule, F: Float[Array, "spin ao ao"]) -> Scalar:
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
    return jnp.sqrt(jnp.einsum("sij->", (F - molecule.fock) ** 2)) / jnp.sqrt(
        jnp.einsum("sij->", molecule.fock**2)
    )


def dm21_grad_regularization(molecule: Molecule, F: Float[Array, "spin ao ao"]) -> Scalar:
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

    # factors = jnp.einsum("sba,sac,scd->sbd", C.transpose(0,2,1), F, C) ** 2
    # factors = jnp.einsum("sab,sac,scd->sbd", C, F, C) ** 2
    # factors = jnp.einsum("sac,sab,scd->sbd", F, C, C) ** 2
    factors = jnp.einsum("sac,sab,scd->sbd", F, C, C) ** 2  # F is symmetric

    numerator = n[:, :, None] - n[:, None, :]
    denominator = e[:, :, None] - e[:, None, :]

    mask = jnp.logical_and(jnp.abs(factors) > 0, jnp.abs(numerator) > 0)

    safe_denominator = jnp.where(mask, denominator, 1.0)

    second_mask = jnp.abs(safe_denominator) > 0
    safe_denominator = jnp.where(second_mask, safe_denominator, 1.0e-20)

    prefactors = numerator / safe_denominator

    dE = jnp.clip(0.5 * jnp.sum(prefactors * factors), a_min=-10, a_max=10)

    return dE**2


def orbital_grad_regularization(molecule: Molecule, F: Float[Array, "spin ao ao"]) -> Scalar:
    """Deprecated"""

    #  Calculate the gradient regularization term
    new_grad = get_grad(molecule.mo_coeff, molecule.mo_occ, F)

    dE = jnp.linalg.norm(new_grad - molecule.training_gorb_grad, ord="fro")

    return dE**2


def get_grad(
    mo_coeff: Float[Array, "spin ao ao"],
    mo_occ: Float[Array, "spin ao"],
    F: Float[Array, "spin ao ao"],
):
    """RHF orbital gradients

    Parameters
    ----------
    mo_coeff: 2D ndarray
        Orbital coefficients
    mo_occ: 1D ndarray
        Orbital occupancy
    F: 2D ndarray
        Fock matrix in AO representation

    Returns:
    --------
    Gradients in MO representation.  It's a num_occ*num_vir vector.

    Notes:
    ------
    # Similar to pyscf/scf/hf.py:
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(jnp.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()
    """

    C_occ = vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(mo_occ > 0, mo_coeff, 0)
    C_vir = vmap(jnp.where, in_axes=(None, 1, None), out_axes=1)(mo_occ == 0, mo_coeff, 0)

    return jnp.einsum("sab,sac,scd->bd", C_vir.conj(), F, C_occ)


##################### Loss Functions #####################


def mse_energy_loss(
    params: PyTree,
    compute_energy: Callable,
    atoms_list: Union[list[Molecule],
                      list[Solid],
                      list[Union[Molecule,Solid]],
                      Molecule, 
                      Solid
                    ],
    truth_energies: Float[Array, "energy"],
    elec_num_norm: Scalar = True,
) -> Scalar:
    r"""
    Computes the mean-squared error between predicted and truth energies.

    This loss function does not yet support parallel execution for the loss contributions. 
    We instead use a simple serial for loop.

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    compute_energy: Callable(molecule, params) -> molecule.
        any non SCF or SCF method in evaluate.py. The output molecule contains the predicted energy.
    atoms_list: Union[list[Molecule], list[Solid], list[Union[Molecule,Solid]], Molecule, Solid]
        A list of `Molecule` or `Solid` objects or a combination of both. Passing
        a single `Molecule` or `Solid` wraps it in a list internally.
    truth_energies: Float[Array, "energy"]
        the truth values of the energy to measure the predictions against
    elec_num_norm: Scalar
        True to normalize the loss function by the number of electrons in a Molecule.

    Returns
    ----------
    Scalar: the mean-squared error between predicted and truth energies
    """
    # Catch the case where a list of atoms was not passed. I.e, dealing with a single
    # instance.
    if isinstance(atoms_list, Molecule) or isinstance(atoms_list, Solid):
        atoms_list = [atoms_list]
    sum = 0
    for i, atoms in enumerate(atoms_list):
        atoms_out = compute_energy(params, atoms)
        E_predict = atoms_out.energy
        diff = E_predict - truth_energies[i]
        # Not jittable because of if.
        num_elec = jnp.sum(atoms.atom_index) - atoms.charge
        if elec_num_norm:
            diff = diff / num_elec
        sum += (diff) ** 2
    cost_value = sum / len(atoms_list)

    return cost_value

@partial(value_and_grad, has_aux=True)
def simple_energy_loss(params: PyTree,
    compute_energy: Callable,
    atoms: Union[Molecule, Solid],
    truth_energy: Float,
    ):
    r"""
    Computes the loss for a single molecule

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    compute_energy: Callable.
        any non SCF or SCF method in evaluate.py
    atoms: Union[Molecule, Solid]
        The collcection of atoms.
    truth_energy: Float
        The energy value we are training against
    """
    atoms_out = compute_energy(params, atoms)
    E_predict = atoms_out.energy
    diff = E_predict - truth_energy
    return diff**2, E_predict


def sq_electron_err_int(
    pred_density: Float[Array, "ngrid nspin"],
    truth_density: Float[Array, "ngrid nspin"],
    atoms: Union[Molecule, Solid],
    clip_cte=1e-30
) -> Scalar:
    r"""
    Computes the integral:

    .. math::
        \epsilon = \int (\rho_{pred}(r) - \rho_{truth})^2 dr
    Parameters
    ----------
    pred_density: Float[Array, "ngrid nspin"]
        Density predicted by a neural functional
    truth_density: Float[Array, "ngrid nspin"]
        A accurate density used as a truth value in training
    atoms: Union[Molecule, Solid]
        The collection of atoms.

    Returns
        Scalar: the value epsilon described above
    ----------
    """
    pred_density = jnp.clip(pred_density, a_min=clip_cte)
    truth_density = jnp.clip(truth_density, a_min=clip_cte)
    diff_up = jnp.clip(jnp.clip(pred_density[:, 0] - truth_density[:, 0], a_min=clip_cte) ** 2, a_min=clip_cte)
    diff_dn = jnp.clip(jnp.clip(pred_density[:, 1] - truth_density[:, 1], a_min=clip_cte) ** 2, a_min=clip_cte)
    err_int = jnp.sum(diff_up * atoms.grid.weights) + jnp.sum(diff_dn * atoms.grid.weights)
    return err_int


def mse_density_loss(
    params: PyTree,
    compute_energy: Callable,
    atoms_list: Union[list[Molecule],
                      list[Solid],
                      list[Union[Molecule,Solid]],
                      Molecule, 
                      Solid
                    ],
    truth_rhos: list[Float[Array, "ngrid nspin"]],
    elec_num_norm: Scalar = True,
) -> Scalar:
    r"""
    Computes the mean-squared error between predicted and truth densities.

    This loss function does not yet support parallel execution for the loss contributions
    and instead implemented a simple for loop.

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    compute_energy: Callable.
        any non SCF or SCF method in evaluate.py
    atoms_list: Union[list[Molecule], list[Solid], list[Union[Molecule,Solid]], Molecule, Solid]
        A list of `Molecule` or `Solid` objects or a combination of both. Passing
        a single `Molecule` or `Solid` wraps it in a list internally.
    truth_densities: list[Float[Array, "ngrid nspin"]]
        the truth values of the density to measure the predictions against
    elec_num_norm: Scalar
        True to normalize the loss function by the number of electrons in a Molecule.

    Returns
    ----------
    Scalar: the mean-squared error between predicted and truth densities
    """
    # Catch the case where a list of atoms was not passed. I.e, dealing with a single
    # instance.
    if isinstance(atoms_list, Molecule) or isinstance(atoms_list, Solid):
        atoms_list = [atoms_list]
    sum = 0
    for i, atoms in enumerate(atoms_list):
        atoms_out = compute_energy(params, atoms)
        rho_predict = atoms_out.density()
        diff = sq_electron_err_int(rho_predict, truth_rhos[i], atoms)
        # Not jittable because of if.
        num_elec = jnp.sum(atoms.atom_index) - atoms.charge
        if elec_num_norm:
            diff = diff / num_elec**2
        sum += diff
    cost_value = sum / len(atoms_list)

    return cost_value


def mse_energy_and_density_loss(
    params: PyTree,
    compute_energy: Callable,
    atoms_list: Union[list[Molecule],
                      list[Solid],
                      list[Union[Molecule,Solid]],
                      Molecule, 
                      Solid
                    ],
    truth_densities: list[Float[Array, "ngrid nspin"]],
    truth_energies: Float[Array, "energy"],
    rho_factor: Scalar = 1.0,
    energy_factor: Scalar = 1.0,
    elec_num_norm: Scalar = True,
) -> Scalar:
    r"""
    Computes the most general loss function using mean-squared error of energies and densities.

    This loss function does not yet support parallel execution for the loss contributions
    and instead implemented a simple for loop.

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    compute_energy: Callable.
        any non SCF or SCF method in evaluate.py
    atoms_list: Union[list[Molecule], list[Solid], list[Union[Molecule,Solid]], Molecule, Solid]
        A list of `Molecule` or `Solid` objects or a combination of both. Passing
        a single `Molecule` or `Solid` wraps it in a list internally.
    truth_densities: list[Float[Array, "ngrid, nspin"]]
        the truth values of the density to measure the predictions against
    truth_energies: Float[Array, "energy"]
        the truth values of the energy to measure the predictions against
    energy_factor: Scalar
        A weighting factor for the energy portion of the loss. Default = 1.0
    density_factor: Scalar
        A weighting factor for the density portion of the loss. Default = 1.0
    elec_num_norm: Scalar
        True to normalize the loss function by the number of electrons in an atoms instance.

    Returns
    ----------
    Scalar: the mean-squared error of both energies and densities each with it's own weight.
    """
    # Catch the case where a list of atoms was not passed. I.e, dealing with a single
    # instance.
    if isinstance(atoms_list, Molecule) or isinstance(atoms_list, Solid):
        atoms_list = [atoms_list]
    sum_energy = 0
    sum_rho = 0
    for i, atoms in enumerate(atoms_list):
        atoms_out = compute_energy(params, atoms)
        rho_predict = atoms_out.density()
        energy_predict = atoms_out.energy
        diff_rho = sq_electron_err_int(rho_predict, truth_densities[i], atoms)
        diff_energy = energy_predict - truth_energies[i]
        # Not jittable because of if.
        num_elec = jnp.sum(atoms.atom_index) - atoms.charge
        if elec_num_norm:
            diff_rho = diff_rho / num_elec**2
            diff_energy = diff_energy / num_elec
        sum_rho += diff_rho
        sum_energy += diff_energy**2
    energy_contrib = energy_factor * sum_energy / len(atoms_list)
    rho_contrib = rho_factor * sum_rho / len(atoms_list)

    return energy_contrib + rho_contrib
