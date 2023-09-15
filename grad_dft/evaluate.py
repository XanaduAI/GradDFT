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
from jax import jit, numpy as jnp
from jax.lax import Precision
from jax.scipy.special import erfc
from jax.scipy.optimize import minimize as scipyminimize
from flax import struct
import optax
from jax.lax import stop_gradient, cond, fori_loop

import sys

from typing import Callable, Tuple, Optional
from functools import partial, reduce
import time
from scipy.optimize import bisect
from typeguard import typechecked

from grad_dft.functional import Functional

from grad_dft.molecule import Molecule, abs_clip, make_rdm1, orbital_grad
from grad_dft.train import molecule_predictor
from grad_dft.interface.pyscf import (
    generate_chi_tensor,
    mol_from_Molecule,
    process_mol,
    mol_from_Molecule,
)

from grad_dft.external import Functional
from grad_dft.utils import Optimizer, safe_fock_solver
from grad_dft.utils.types import Hartree2kcalmol

from jaxtyping import PyTree, Array, Scalar, Float, jaxtyped


######## Test kernel ########


def make_test_kernel(tx: optax.GradientTransformation, loss: Callable) -> Callable:
    r"""
    Creates a kernel object that can be called to evaluate the loss and other metrics.

    Parameters
    ----------
    tx: optax.GradientTransformation
        A gradient transformation object.
    loss: Callable
        A loss function.

    Returns
    -------
    Callable
    """

    def kernel(
        params: PyTree, system: Molecule, ground_truth_energy: float, *args
    ) -> Tuple[PyTree, optax.OptState, Scalar, Scalar]:
        (cost_value, metrics), _ = loss(params, system, ground_truth_energy)

        return metrics, cost_value

    return kernel

######## Non self-consistent iterator ################

def make_non_scf_predictor(
    functional: Functional,
    chunk_size: int = 1024,
    **kwargs,
) -> Callable:
    r"""
    Creates an non_scf_predictor function which when called non-self consistently
    calculates the total energy at a fixed density.

    Main parameters
    ---------------
    functional: Functional

    Returns
    ---------
    Callable
    """
    predict_molecule = molecule_predictor(functional, chunk_size=chunk_size, **kwargs)
    def non_scf_predictor(params: PyTree, molecule: Molecule, *args) -> Molecule:
        r"""Calculates the total energy at a fixed density non-self consistently.

        Main parameters
        ---------------
        params: Pytree
            Parameters of the neural functional
        molecule: Molecule
            A Grad-DFT molecule object

         Returns
        ---------
        Molecule
            A Grad-DFT Molecule object with updated attributes 
        """
        predicted_e, fock = predict_molecule(params, molecule, *args)
        molecule = molecule.replace(fock=fock)
        molecule = molecule.replace(energy=predicted_e)
        return molecule
    
    return non_scf_predictor

# Add Harris-Foulkes predictor here too! 

######## Test scf loop and orbital optimizers ########

def make_simple_scf_loop(
    functional: Functional,
    mixing_factor: float = 0.4,
    chunk_size: int = 1024,
    max_cycles: int = 50,
    start_cycle: int = 0,
    e_conv: float = 1e-5,
    g_conv: float = 1e-5,
    verbose: int = 0,
    **kwargs,
) -> Callable:
    r"""
    Creates an scf_iterator object that can be called to implement a self-consistent loop.

    Main parameters
    ---------------
    functional: Functional

    verbose: int
        Controls the level of printout

    Returns
    ---------
    float
    """

    predict_molecule = molecule_predictor(functional, chunk_size=chunk_size, **kwargs)

    def simple_scf_iterator(params: PyTree, molecule: Molecule, clip_cte = 1e-30, *args) -> Molecule:
        r"""
        Implements a scf loop for a Molecule and a functional implicitly defined predict_molecule with
        parameters params

        Parameters
        ----------
        params: PyTree
        molecule: Molecule
        *args: Arguments to be passed to predict_molecule function
        """

        nelectron = molecule.atom_index.sum() - molecule.charge

        # predicted_e, fock = predict_molecule(params, molecule, *args)
        # fock = abs_clip(fock, clip_cte)
        # fock = molecule.fock
        old_e = 100000 # we should set the energy in a molecule object really
        for cycle in range(max_cycles):
            # Convergence criterion is energy difference (default 1) kcal/mol and norm of gradient of orbitals < g_conv
            start_time = time.time()
            # old_e = molecule.energy
            if cycle == 0:
                mo_energy = molecule.mo_energy
                mo_coeff = molecule.mo_coeff
                fock = molecule.fock
            else:
                # Diagonalize Fock matrix
                overlap = abs_clip(molecule.s1e, clip_cte)
                mo_energy, mo_coeff = safe_fock_solver(fock, overlap)
                molecule = molecule.replace(mo_coeff=mo_coeff)
                molecule = molecule.replace(mo_energy=mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            molecule = molecule.replace(mo_occ=mo_occ)
            if verbose > 2:
                print(
                    f"Cycle {cycle} took {time.time() - start_time:.1e} seconds to compute and diagonalize Fock matrix"
                )

            # Update the density matrix
            if cycle == 0:
                old_rdm1 = molecule.make_rdm1()
            else:
                rdm1 = (1 - mixing_factor)*old_rdm1 + mixing_factor*abs_clip(molecule.make_rdm1(), clip_cte)
                rdm1 = abs_clip(rdm1, clip_cte)
                molecule = molecule.replace(rdm1=rdm1)
                old_rdm1 = rdm1
            

            computed_charge = jnp.einsum(
                "r,ra,rb,sab->", molecule.grid.weights, molecule.ao, molecule.ao, molecule.rdm1
            )
            assert jnp.isclose(
                nelectron, computed_charge, atol=1e-3
            ), "Total charge is not conserved"

            exc_start_time = time.time()

            predicted_e, fock = predict_molecule(params, molecule, *args)
            fock = abs_clip(fock, clip_cte)
            
            exc_time = time.time()

            if verbose > 2:
                print(
                    f"Cycle {cycle} took {exc_time - exc_start_time:.1e} seconds to compute exc and vhf"
                )

            # Compute the norm of the gradient
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

            if verbose > 1:
                print(
                    f"cycle: {cycle}, energy: {predicted_e:.7e}, energy difference: {abs(predicted_e - old_e):.4e}, seconds: {time.time() - start_time:.1e}"
                )
            if verbose > 2:
                print(
                    f"       relative energy difference: {abs((predicted_e - old_e)/predicted_e):.5e}"
                )
            old_e = predicted_e

        if verbose > 1:
            print(
                f"cycle: {cycle}, predicted energy: {predicted_e:.7e}, energy difference: {abs(predicted_e - old_e):.4e}, norm_gradient_orbitals: {norm_gorb:.2e}"
            )
        # Ensure molecule is fully updated
        molecule = molecule.replace(fock=fock)
        molecule = molecule.replace(energy=predicted_e)
        return molecule

    return simple_scf_iterator

def make_jitted_simple_scf_loop(functional: Functional, cycles: int = 25, mixing_factor: float = 0.4, **kwargs) -> Callable:
    r"""
    Creates an scf_iterator object that can be called to implement a self-consistent loop using linear mixing.
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

    predict_molecule = molecule_predictor(functional, chunk_size=None, **kwargs)

    @jit
    def simple_scf_jitted_iterator(
        params: PyTree, 
        molecule: Molecule, 
        *args
    ) -> Molecule:

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

        Returns
        -------
        predicted_e: Scalar
        molecule: Molecule

        Notes:
        ------
        SCF training loop not implemented for (range-separated) exact-exchange functionals.
        Doing so would require a differentiable way of recomputing the chi tensor.
        """

        if molecule.omegas:
            raise NotImplementedError(
                "SCF training loop not implemented for (range-separated) exact-exchange functionals. \
                                    Doing so would require a differentiable way of recomputing the chi tensor."
            )

        old_e = jnp.inf
        norm_gorb = jnp.inf

        predicted_e, fock = predict_molecule(params, molecule, *args)
        
        old_e = jnp.inf
        norm_gorb = jnp.inf

        predicted_e, fock = predict_molecule(params, molecule, *args)
        rho = molecule.density()
        molecule = molecule.replace(rho=rho)
        

        state = (molecule, predicted_e, old_e, norm_gorb)

        def loop_body(cycle, state):
            old_state = state
            molecule, predicted_e, old_e, norm_gorb = old_state
            old_e = predicted_e
            old_rdm1 = molecule.rdm1
            fock = molecule.fock

            # Diagonalize Fock matrix
            mo_energy, mo_coeff = safe_fock_solver(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff=mo_coeff)
            molecule = molecule.replace(mo_energy=mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            molecule = molecule.replace(mo_occ=mo_occ)

            # Update the density matrix with linear mixing
            unmixed_new_rdm1 =  molecule.make_rdm1()
            rdm1 = (1 - mixing_factor)*old_rdm1 + mixing_factor*unmixed_new_rdm1
            molecule = molecule.replace(rdm1=rdm1)
            rho = molecule.density()
            molecule = molecule.replace(rho=rho)

            # Compute the new energy and Fock matrix
            predicted_e, fock = predict_molecule(params, molecule, *args)
            molecule = molecule.replace(fock=fock)

            # Compute the norm of the gradient
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

            state = (molecule, predicted_e, old_e, norm_gorb)

            return state

        # Compute the scf loop
        final_state = fori_loop(0, cycles, body_fun=loop_body, init_val=state)
        molecule, predicted_e, old_e, norm_gorb = final_state
        molecule = molecule.replace(energy=predicted_e)
        return molecule

    return simple_scf_jitted_iterator


def make_scf_loop(
    functional: Functional,
    level_shift_factor: tuple[float, float] = (0.0, 0.0),
    damp_factor: tuple[float, float] = (0.0, 0.0),
    chunk_size: int = 1024,
    max_cycles: int = 50,
    diis_start_cycle: int = 0,
    e_conv: float = 1e-5,
    g_conv: float = 1e-5,
    diis_method="CDIIS",
    smearing: Optional[str] = None,
    smearing_sigma: Optional[float] = 0.0,
    verbose: int = 0,
    **kwargs,
) -> Callable:
    r"""
    Creates an scf_iterator object that can be called to implement a self-consistent loop.

    Main parameters
    ---------------
    functional: Functional

    verbose: int
        Controls the level of printout

    Returns
    ---------
    float
    """

    predict_molecule = molecule_predictor(functional, chunk_size=chunk_size, **kwargs)

    def scf_iterator(params: PyTree, molecule: Molecule, *args) -> Molecule:
        r"""
        Implements a scf loop for a Molecule and a functional implicitly defined predict_molecule with
        parameters params

        Parameters
        ----------
        params: PyTree
        molecule: Molecule
        *args: Arguments to be passed to predict_molecule function
        """

        # Needed to be able to update the chi tensor
        mol = mol_from_Molecule(molecule)
        _, mf = process_mol(
            mol, compute_energy=False, grid_level=int(molecule.grid_level), training=False
        )

        old_e = jnp.inf
        norm_gorb = jnp.inf
        cycle = 0
        nelectron = molecule.atom_index.sum() - molecule.charge

        predicted_e, fock = predict_molecule(params, molecule, *args)

        # Initialize DIIS
        A = jnp.identity(molecule.s1e.shape[0])
        diis = Diis(overlap_matrix=molecule.s1e, A=A, max_diis=10, diis_method=diis_method)
        diis_data = (
            jnp.empty((0, 2, A.shape[0], A.shape[0])),
            jnp.empty((0, 2, A.shape[0], A.shape[0])),
            jnp.empty(0),
            jnp.empty((0, 2, A.shape[0], A.shape[0])),
        )

        while (
            abs(predicted_e - old_e) * Hartree2kcalmol > e_conv or norm_gorb > g_conv
        ) and cycle < max_cycles:
            # Convergence criterion is energy difference (default 1) kcal/mol and norm of gradient of orbitals < g_conv
            start_time = time.time()
            old_e = predicted_e

            if (
                0 <= cycle < diis_start_cycle - 1
                and abs(damp_factor[0]) + abs(damp_factor[1]) > 1e-4
            ):
                fock = (
                    damping(molecule.s1e, molecule.rdm1[0], fock[0], damp_factor[0]),
                    damping(molecule.s1e, molecule.rdm1[1], fock[1], damp_factor[1]),
                )

            # DIIS iteration
            new_data = (molecule.rdm1, fock, predicted_e)
            if cycle >= diis_start_cycle:
                fock, diis_data = diis.run(new_data, diis_data, cycle)

            if abs(level_shift_factor[0]) + abs(level_shift_factor[1]) > 1e-4:
                fock = (
                    level_shift(molecule.s1e, molecule.rdm1[0], fock[0], level_shift_factor[0]),
                    level_shift(molecule.s1e, molecule.rdm1[1], fock[1], level_shift_factor[1]),
                )

            # Diagonalize Fock matrix
            mo_energy, mo_coeff = safe_fock_solver(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff=mo_coeff)
            molecule = molecule.replace(mo_energy=mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            molecule = molecule.replace(mo_occ=mo_occ)
            if verbose > 2:
                print(
                    f"Cycle {cycle} took {time.time() - start_time:.1e} seconds to compute and diagonalize Fock matrix"
                )

            if smearing:

                def gaussian_smearing_occ(m, mo_energy, sigma):
                    return 0.5 * erfc((mo_energy - m) / sigma)

                def fermi_smearing_occ(m, mo_energy, sigma):
                    return 1 / (jnp.exp((mo_energy - m) / sigma) + 1.0)

                if smearing == "gaussian":
                    smearing_occ = gaussian_smearing_occ
                elif smearing == "fermi-dirac":
                    smearing_occ = fermi_smearing_occ

                def nelec_cost_fn(m, mo_es, sigma, _nelectron):
                    mo_occ = smearing_occ(m, mo_es, sigma)
                    res = mo_occ.sum() - _nelectron
                    return res

                sigma = smearing_sigma
                mo_es = jnp.hstack(mo_energy)
                x0 = bisect(
                    nelec_cost_fn,
                    a=min(mo_energy),
                    b=max(mo_energy),
                    xtol=1e-10,
                    rtol=1e-10,
                    maxiter=10000,
                    args=(mo_es, sigma, nelectron),
                )
                mo_occ = smearing_occ(x0, mo_es, sigma)
                molecule = molecule.replace(mo_occ=mo_occ)

            # Update the density matrix
            rdm1 = molecule.make_rdm1()
            molecule = molecule.replace(rdm1=rdm1)

            computed_charge = jnp.einsum(
                "r,ra,rb,sab->", molecule.grid.weights, molecule.ao, molecule.ao, molecule.rdm1
            )
            assert jnp.isclose(
                nelectron, computed_charge, atol=1e-3
            ), "Total charge is not conserved"

            # Update the chi matrix
            if molecule.omegas:
                chi_start_time = time.time()
                chi = generate_chi_tensor(
                    molecule.rdm1,
                    molecule.ao,
                    molecule.grid.coords,
                    mf.mol,
                    omegas=molecule.omegas,
                    chunk_size=chunk_size,
                    *args,
                )
                molecule = molecule.replace(chi=chi)
                if verbose > 2:
                    print(
                        f"Cycle {cycle} took {time.time() - chi_start_time:.1e} seconds to compute chi matrix"
                    )

            exc_start_time = time.time()
            predicted_e, fock = predict_molecule(params, molecule, *args)
            exc_time = time.time()

            if verbose > 2:
                print(
                    f"Cycle {cycle} took {exc_time - exc_start_time:.1e} seconds to compute exc and vhf"
                )

            # Compute the norm of the gradient
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

            if verbose > 1:
                print(
                    f"cycle: {cycle}, energy: {predicted_e:.7e}, energy difference: {abs(predicted_e - old_e):.4e}, norm_gradient_orbitals: {norm_gorb:.2e}, seconds: {time.time() - start_time:.1e}"
                )
            if verbose > 2:
                print(
                    f"       relative energy difference: {abs((predicted_e - old_e)/predicted_e):.5e}"
                )
            cycle += 1

        if abs(predicted_e - old_e) * Hartree2kcalmol < e_conv and norm_gorb < g_conv:
            # We perform an extra diagonalization to remove the level shift
            # Solve eigenvalue problem
            mo_energy, mo_coeff = safe_fock_solver(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff=mo_coeff)
            molecule = molecule.replace(mo_energy=mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            molecule = molecule.replace(mo_occ=mo_occ)

            # Update the density matrix
            rdm1 = molecule.make_rdm1()
            molecule = molecule.replace(rdm1=rdm1)

            # Update the chi matrix
            if molecule.omegas:
                chi = generate_chi_tensor(
                    molecule.rdm1,
                    molecule.ao,
                    molecule.grid.coords,
                    mf.mol,
                    omegas=molecule.omegas,
                    chunk_size=chunk_size,
                    *args,
                )
                molecule = molecule.replace(chi=chi)

            predicted_e, fock = predict_molecule(params, molecule, *args)

            # Compute the norm of the gradient
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

        if verbose > 1:
            print(
                f"cycle: {cycle}, predicted energy: {predicted_e:.7e}, energy difference: {abs(predicted_e - old_e):.4e}, norm_gradient_orbitals: {norm_gorb:.2e}"
            )
        # Ensure molecule is fully updated
        molecule = molecule.replace(fock=fock)
        molecule = molecule.replace(energy=predicted_e)
        return molecule

    return scf_iterator


def make_orbital_optimizer(
    fxc: Functional,
    tx: Optimizer,
    chunk_size: int = 1024,
    max_cycles: int = 500,
    e_conv: float = 1e-7,
    whitening: str = "PCA",
    precision=Precision.HIGHEST,
    verbose: int = 0,
    **kwargs,
) -> Callable:
    r"""
    Creates an orbital_optimizer object that can be called to optimize the density matrix and minimize the energy.
    Follows the description in

    Tianbo Li, Min Lin, Zheyuan Hu, Kunhao Zheng, Giovanni Vignale, Kenji Kawaguchi, A.H. Castro Neto, Kostya S. Novoselov, Shuicheng YAN
    D4FT: A Deep Learning Approach to Kohn-Sham Density Functional Theory
    ICLR 2023, https://openreview.net/forum?id=aBWnqqsuot7

    Note: This only optimizes the rdm1, not the orbitals, also discussed in the article above.
    Note too: The calculation of tensor chi is not implemented self differentiably, so the functional cannot include exact exchange.
    """

    predict_molecule = molecule_predictor(fxc, chunk_size=chunk_size, **kwargs)

    @partial(jax.value_and_grad, argnums=0)
    def molecule_orbitals_iterator(
        W: Array, D: Array, params: PyTree, molecule: Molecule, *args
    ) -> Tuple[Scalar, Scalar]:
        Q0, _ = jnp.linalg.qr(W[0])
        Q1, _ = jnp.linalg.qr(W[1])
        Q = jnp.stack([Q0, Q1])

        # Compute the molecular orbitals
        C = jnp.einsum("sij,jk->ski", Q, D)

        I = jnp.einsum("sji,jk,skl->sil", C, molecule.s1e, C)
        stack = jnp.stack((jnp.identity(I.shape[1]), jnp.identity(I.shape[1])))
        # assert jnp.allclose(I, stack)

        # Compute the density matrix
        rdm1 = make_rdm1(C, molecule.mo_occ)
        molecule = molecule.replace(rdm1=rdm1, mo_coeff=C)

        # todo: differentiably implement the calculation of the chi tensor,
        # which now relies on pyscf's mol.intor("int1e_grids_sph", hermi=hermi, grids=coords)

        nelectron = molecule.atom_index.sum() - molecule.charge

        computed_charge = jnp.einsum(
            "r,ra,rb,sab->", molecule.grid.weights, molecule.ao, molecule.ao, molecule.rdm1
        )
        # assert jnp.isclose(nelectron, computed_charge, atol = 1e-3), "Total charge is not conserved"

        # Predict the energy and the fock matrix
        predicted_e, _ = predict_molecule(params, molecule, *args)
        return predicted_e

    def neural_iterator(params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Scalar]:
        old_e = jnp.inf
        cycle = 0

        # Predict the energy and the fock matrix
        predicted_e, _ = predict_molecule(params, molecule, *args)

        C = molecule.mo_coeff

        if whitening == "PCA":
            w, v = jnp.linalg.eigh(molecule.s1e)
            D = (jnp.diag(jnp.sqrt(1 / w)) @ v.T).real
            S_1 = (v @ jnp.diag(w) @ v.T).real
            diff = S_1 - molecule.s1e
            # assert jnp.isclose(diff, jnp.zeros_like(diff), atol=1e-4).all()
            # assert jnp.isclose(jnp.linalg.norm(jnp.linalg.inv(D) @ D - jnp.identity(D.shape[0])), 0.0, atol=1e-5)
        elif whitening == "Cholesky":
            D = jnp.linalg.cholesky(jnp.linalg.inv(molecule.s1e)).T
        elif whitening == "ZCA":
            w, v = jnp.linalg.eigh(molecule.s1e)
            D = (v @ jnp.diag(jnp.sqrt(1 / w)) @ v.T).real

        Q = jnp.einsum("sji,jk->sik", C, jnp.linalg.inv(D))  # C transposed
        # Q_ = jnp.einsum('sji,jk,kl->sil', C, v, jnp.diag(jnp.sqrt(w))).real # C transposed
        # assert jnp.allclose(Q, Q_)

        I = jnp.einsum("sji,jk,skl->sil", C, molecule.s1e, C)  # The first C is transposed
        # stack = jnp.stack((jnp.identity(I.shape[1]),jnp.identity(I.shape[1])))
        # assert jnp.allclose(I, stack)

        # I = jnp.einsum('sji,sjk->sik', Q, Q) # The first Q is transposed
        # assert jnp.allclose(I, jnp.stack((jnp.identity(I.shape[1]),jnp.identity(I.shape[1]))))

        W = Q

        opt_state = tx.init(W)

        while abs(predicted_e - old_e) * Hartree2kcalmol > e_conv and cycle < max_cycles:
            start_time = time.time()
            old_e = predicted_e

            predicted_e, grads = molecule_orbitals_iterator(W, D, params, molecule, *args)

            updates, opt_state = tx.update(grads, opt_state, W)
            W = optax.apply_updates(W, updates)

            cycle += 1

            if verbose > 1:
                print(
                    f"cycle: {cycle}, predicted energy: {predicted_e:.7e}, energy difference: {abs(predicted_e - old_e):.4e}"
                )

        return predicted_e

    return neural_iterator


######### Jitted versions #########


def make_jitted_orbital_optimizer(
    functional: Functional, tx: Optimizer, cycles: int = 500, **kwargs
) -> Callable:
    r"""
    Creates an orbital_optimizer object that can be called to optimize the density matrix and minimize the energy.
    Follows the description in

    Tianbo Li, Min Lin, Zheyuan Hu, Kunhao Zheng, Giovanni Vignale, Kenji Kawaguchi, A.H. Castro Neto, Kostya S. Novoselov, Shuicheng YAN
    D4FT: A Deep Learning Approach to Kohn-Sham Density Functional Theory
    ICLR 2023, https://openreview.net/forum?id=aBWnqqsuot7

    Note: This only optimizes the rdm1, not the orbitals, also discussed in the article above.
    Note too: The calculation of tensor chi is not implemented self differentiably, so the functional cannot include exact exchange.
    """

    predict_molecule = molecule_predictor(functional, **kwargs)

    @partial(jax.value_and_grad, argnums=0)
    def molecule_orbitals_energy(
        W: Array, D: Array, params: PyTree, molecule: Molecule, *args
    ) -> Tuple[Scalar, Scalar]:
        Q0, _ = jnp.linalg.qr(W[0])
        Q1, _ = jnp.linalg.qr(W[1])
        Q = jnp.stack([Q0, Q1])

        # Compute the molecular orbitals
        C = jnp.einsum("sij,jk->ski", Q, D)

        # Compute the density matrix
        rdm1 = make_rdm1(C, molecule.mo_occ)
        molecule = molecule.replace(rdm1=rdm1, mo_coeff=C)

        # Predict the energy and the fock matrix
        predicted_e, _ = predict_molecule(params, molecule, *args)
        return predicted_e

    @jit
    def neural_iterator(params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Scalar]:
        # Predict the energy and the fock matrix
        predicted_e, _ = predict_molecule(params, molecule, *args)

        w, v = jnp.linalg.eigh(molecule.s1e)
        D = (jnp.diag(jnp.sqrt(1 / w)) @ v.T).real
        Q = jnp.einsum("sji,jk->sik", molecule.mo_coeff, jnp.linalg.inv(D))  # C transposed
        W = Q

        opt_state = tx.init(W)

        def loop_body(cycle, state):
            W, opt_state, predicted_e = state
            predicted_e, grads = molecule_orbitals_energy(W, D, params, molecule, *args)
            updates, opt_state = tx.update(grads, opt_state, W)
            W = optax.apply_updates(W, updates)
            return W, opt_state, predicted_e

        # Compute the scf loop
        state = W, opt_state, predicted_e
        final_state = fori_loop(0, cycles, body_fun=loop_body, init_val=state)
        W, opt_state, predicted_e = final_state

        return predicted_e

    return neural_iterator


def make_jitted_scf_loop(functional: Functional, cycles: int = 25, **kwargs) -> Callable:
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
    scf_jitted_iterator: Callable
    """

    predict_molecule = molecule_predictor(functional, chunk_size=None, **kwargs)

    @jit
    def scf_jitted_iterator(
        params: PyTree, 
        molecule: Molecule, 
        *args
    ) -> Molecule:
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

        Returns
        -------
        predicted_e: Scalar
        molecule: Molecule

        Notes:
        ------
        SCF training loop not implemented for (range-separated) exact-exchange functionals.
        Doing so would require a differentiable way of recomputing the chi tensor.
        """

        #if molecule.omegas:
        #    raise NotImplementedError(
        #        "SCF training loop not implemented for (range-separated) exact-exchange functionals. \
        #                            Doing so would require a differentiable way of recomputing the chi tensor."
        #    )

        old_e = jnp.inf
        norm_gorb = jnp.inf

        predicted_e, fock = predict_molecule(params, molecule, *args)
        molecule = molecule.replace(fock=fock)

        # Initialize DIIS
        A = jnp.identity(molecule.s1e.shape[0])
        diis = JittableDiis(overlap_matrix=molecule.s1e, A=A, max_diis=10)
        diis_data = (
            jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])),
            jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])),
            jnp.zeros(diis.max_diis),
            jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])),
        )

        state = (molecule, predicted_e, old_e, norm_gorb, diis_data)

        def loop_body(cycle, state):
            old_state = state
            molecule, predicted_e, old_e, norm_gorb, diis_data = old_state
            old_e = predicted_e
            fock = molecule.fock

            # DIIS iteration
            new_data = (molecule.rdm1, fock, predicted_e)
            fock, diis_data = diis.run(new_data, diis_data, cycle)
            molecule = molecule.replace(fock=fock)

            # Diagonalize Fock matrix
            mo_energy, mo_coeff = safe_fock_solver(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff=mo_coeff)
            molecule = molecule.replace(mo_energy=mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            molecule = molecule.replace(mo_occ=mo_occ)

            # Update the density matrix
            rdm1 = molecule.make_rdm1()
            molecule = molecule.replace(rdm1=rdm1)

            # Compute the new energy and Fock matrix
            predicted_e, fock = predict_molecule(params, molecule, *args)
            molecule = molecule.replace(fock=fock)

            # Compute the norm of the gradient
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

            state = (molecule, predicted_e, old_e, norm_gorb, diis_data)

            return state

        # Compute the scf loop
        final_state = fori_loop(0, cycles, body_fun=loop_body, init_val=state)
        molecule, predicted_e, old_e, norm_gorb, diis_data = final_state

        # Perform a final diagonalization without diis (reinitializing)
        diis_data = (
            jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])),
            jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])),
            jnp.zeros(diis.max_diis),
            jnp.zeros((diis.max_diis, 2, A.shape[0], A.shape[0])),
        )
        state = (molecule, predicted_e, old_e, norm_gorb, diis_data)
        state = loop_body(0, state)
        molecule, predicted_e, _, _, _ = final_state
        
        # Ensure molecule is fully updated
        molecule = molecule.replace(energy=predicted_e)
        return molecule


    return scf_jitted_iterator


@struct.dataclass
class JittableDiis:
    r"""DIIS extrapolation, intended for training of the resulting energy of a scf loop.
    If you are looking for a more flexible, not differentiable DIIS, see evaluate.py DIIS class
    The implemented CDIIS computes the Fock matrix as a linear combination of the previous Fock matrices, with
    .. math::
        F_{DIIS} = \sum_i x_i F_i,

    where the coefficients are determined by minimizing the error vector
    .. math::
        e_i = A^T (F_i D_i S - S D_i F_i) A,

    with F_i the Fock matrix at iteration i, D_i the density matrix at iteration i,
    and S the overlap matrix. The error vector is then used to compute the
    coefficients as
    .. math::
        B = \begin{pmatrix}
            <e_1|e_1> & <e_1|e_2> & \cdots & <e_1|e_n> & -1 \\
            <e_2|e_1> & <e_2|e_2> & \cdots & <e_2|e_n> & -1 \\
            \vdots & \vdots & \ddots & \vdots & \vdots \\
            <e_n|e_1> & <e_n|e_2> & \cdots & <e_n|e_n> & -1 \\
            -1 & -1 & \cdots & -1 & 0
        \end{pmatrix},

    .. math::
        x = \begin{pmatrix}
            x_1 \\
            x_2 \\
            \vdots \\
            x_n \\
            0
        \end{pmatrix}
    
    and
    .. math::
        C= \begin{pmatrix}
            0 \\
            0 \\
            \vdots \\
            0 \\
            1
        \end{pmatrix}

    where n is the number of stored Fock matrices. The coefficients are then
    computed as
    .. math::
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

        fds = jnp.einsum(
            "ij,sjk,skl,lm,mn->sin",
            self.A,
            fock_matrix,
            density_matrix,
            self.overlap_matrix,
            self.A.T,
        )
        error_matrix = fds - fds.transpose(0, 2, 1).conj()

        error_vector = cond(
            jnp.greater(cycle, self.max_diis),
            lambda error_vector, error_matrix: jnp.concatenate(
                (error_vector, jnp.expand_dims(error_matrix, axis=0)), axis=0
            )[1:],
            lambda error_vector, error_matrix: error_vector.at[cycle].set(error_matrix),
            error_vector,
            error_matrix,
        )
        density_vector = cond(
            jnp.greater(cycle, self.max_diis),
            lambda density_vector, density_matrix: jnp.concatenate(
                (density_vector, jnp.expand_dims(density_matrix, axis=0)), axis=0
            )[1:],
            lambda density_vector, density_matrix: density_vector.at[cycle].set(density_matrix),
            density_vector,
            density_matrix,
        )
        fock_vector = cond(
            jnp.greater(cycle, self.max_diis),
            lambda fock_vector, fock_matrix: jnp.concatenate(
                (fock_vector, jnp.expand_dims(fock_matrix, axis=0)), axis=0
            )[1:],
            lambda fock_vector, fock_matrix: fock_vector.at[cycle].set(fock_matrix),
            fock_vector,
            fock_matrix,
        )
        energy_vector = cond(
            jnp.greater(cycle, self.max_diis),
            lambda energy_vector, energy: jnp.concatenate(
                (energy_vector, jnp.expand_dims(energy, axis=0)), axis=0
            )[1:],
            lambda energy_vector, energy: energy_vector.at[cycle].set(energy),
            energy_vector,
            energy,
        )

        return density_vector, fock_vector, energy_vector, error_vector

    def run(self, new_data, diis_data, cycle=0):
        diis_data = self.update(new_data, diis_data, cycle)
        density_vector, fock_vector, energy_vector, error_vector = diis_data

        x = self.cdiis_minimize(error_vector, cycle)
        F = jnp.einsum("si,isjk->sjk", x, fock_vector)
        return jnp.einsum("ji,sjk,kl->sil", self.A, F, self.A), diis_data

    def cdiis_minimize(self, error_vector, cycle):
        # Find the coefficients x that solve B @ x = C with B and C defined below
        B = jnp.zeros((2, len(error_vector) + 1, len(error_vector) + 1))
        B = B.at[:, 1:, 1:].set(jnp.einsum("iskl,jskl->sij", error_vector, error_vector))

        def assign_values(i, B):
            value = cond(jnp.less_equal(i, cycle), lambda _: 1.0, lambda _: 0.0, operand=None)
            B = B.at[:, 0, i + 1].set(value)  # Make 0 if i > cycle, else 1
            B = B.at[:, i + 1, 0].set(value)  # Make 0 if i > cycle, else 1
            return B

        def assign_values_diag(i, B):
            value = cond(
                jnp.less_equal(i, cycle),
                lambda error_vector: jnp.einsum("iskl,jskl->sij", error_vector, error_vector)[
                    :, i, i
                ],
                lambda _: jnp.array([1.0, 1.0]),
                error_vector,
            )
            B = B.at[:, i + 1, i + 1].set(value)
            return B

        B = fori_loop(0, error_vector.shape[0] + 2, assign_values, B)
        B = fori_loop(0, error_vector.shape[0] + 2, assign_values_diag, B)

        C = jnp.zeros((2, len(error_vector) + 1))
        C = C.at[:, 0].set(1)

        x0 = jnp.linalg.inv(B[0]) @ C[0]
        x1 = jnp.linalg.inv(B[1]) @ C[1]
        x = jnp.stack([x0, x1], axis=0)

        return x[:, 1:]


########################################################


@struct.dataclass
class Diis:
    r"""DIIS extrapolation, with different variants. The vanilla DIIS computes
    the Fock matrix as a linear combination of the previous Fock matrices, with
    .. math::
        F_{DIIS} = \sum_i x_i F_i,

    where the coefficients are determined by minimizing the error vector
    .. math::
        e_i = A^T (F_i D_i S - S D_i F_i) A,

    with F_i the Fock matrix at iteration i, D_i the density matrix at iteration i,
    and S the overlap matrix. The error vector is then used to compute the
    coefficients as
    .. math::
        B = \begin{pmatrix}
            <e_1|e_1> & <e_1|e_2> & \cdots & <e_1|e_n> & -1 \\
            <e_2|e_1> & <e_2|e_2> & \cdots & <e_2|e_n> & -1 \\
            \vdots & \vdots & \ddots & \vdots & \vdots \\
            <e_n|e_1> & <e_n|e_2> & \cdots & <e_n|e_n> & -1 \\
            -1 & -1 & \cdots & -1 & 0
        \end{pmatrix},

    .. math::
        x = \begin{pmatrix}
            x_1 \\
            x_2 \\
            \vdots \\
            x_n \\
            0
        \end{pmatrix}
    
    and
    .. math::
        C= \begin{pmatrix}
            0 \\
            0 \\
            \vdots \\
            0 \\
            1
        \end{pmatrix}

    where n is the number of stored Fock matrices. The coefficients are then
    computed as
    .. math::
        x = B^{-1} C.

    Diis attributes:
        overlap_matrix (jnp.array): Overlap matrix, molecule.s1e. Shape: (n_orbitals, n_orbitals).
        A (jnp.array): Transformation matrix for CDIIS, molecule.A. Shape: (n_orbitals, n_orbitals).
        max_diis (int): Maximum number of DIIS vectors to store.
        diis_method (str): DIIS method to use. One of "DIIS", "EDIIS", "ADIIS", "EDIIS2", "ADIIS2".
        ediis2_threshold (float): Threshold for EDIIS2 to change from EDIIS to DIIS.
        adiis2_threshold (float): Threshold for ADIIS2 to change from ADIIS to DIIS.

    
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
    diis_method: Optional[str] = "EDIIS2"
    ediis2_threshold: Optional[float] = 1e-2
    adiis2_threshold: Optional[float] = 1e-2

    def update(self, new_data, diis_data):
        density_matrix, fock_matrix, energy = new_data
        density_vector, fock_vector, energy_vector, error_vector = diis_data

        fds = jnp.einsum(
            "ij,sjk,skl,lm,mn->sin",
            self.A,
            fock_matrix,
            density_matrix,
            self.overlap_matrix,
            self.A.T,
        )
        error_matrix = fds - fds.transpose(0, 2, 1).conj()

        if len(error_vector) == 0:
            error_vector = jnp.expand_dims(error_matrix, axis=0)
        else:
            error_vector = jnp.concatenate(
                (error_vector, jnp.expand_dims(error_matrix, axis=0)), axis=0
            )
        density_vector = jnp.concatenate(
            (density_vector, jnp.expand_dims(density_matrix, axis=0)), axis=0
        )
        fock_vector = jnp.concatenate((fock_vector, jnp.expand_dims(fock_matrix, axis=0)), axis=0)
        energy_vector = jnp.concatenate((energy_vector, jnp.expand_dims(energy, axis=0)), axis=0)

        if len(error_vector) > self.max_diis:
            error_vector = error_vector[1:]
            density_vector = density_vector[1:]
            fock_vector = fock_vector[1:]
            energy_vector = energy_vector[1:]

        return density_vector, fock_vector, energy_vector, error_vector

    def run(self, new_data, diis_data, cycle=0):
        diis_data = self.update(new_data, diis_data)
        density_vector, fock_vector, energy_vector, error_vector = diis_data

        if len(error_vector) == 0:
            raise RuntimeError("No DIIS vectors available")

        elif len(error_vector) == 1:
            return fock_vector[0], diis_data

        else:
            if (
                self.diis_method == "CDIIS"
                or (
                    self.diis_method == "EDIIS2"
                    and (energy_vector[-1] - energy_vector[-2]) / (energy_vector[-2])
                    < self.ediis2_threshold
                )
                or (
                    self.diis_method == "ADIIS2"
                    and (energy_vector[-1] - energy_vector[-2]) / (energy_vector[-2])
                    < self.adiis2_threshold
                )
            ):
                x = self.cdiis_minimize(error_vector)
                F = jnp.einsum("si,isjk->sjk", x, fock_vector)
                return jnp.einsum("ji,sjk,kl->sil", self.A, F, self.A), diis_data

            elif self.diis_method == "EDIIS" or (
                self.diis_method == "EDIIS2"
                and (energy_vector[-1] - energy_vector[-2]) / (energy_vector[-2])
                >= self.ediis2_threshold
            ):
                x, _ = self.ediis_minimize(density_vector, fock_vector, energy_vector)
            elif self.diis_method == "ADIIS" or (
                self.diis_method == "ADIIS2"
                and (energy_vector[-1] - energy_vector[-2]) / (energy_vector[-2])
                >= self.adiis2_threshold
            ):
                x, _ = self.adiis_minimize(density_vector, fock_vector, cycle % self.max_diis)

            F = jnp.einsum("i,isjk->sjk", x, fock_vector)
            return F, diis_data

    def cdiis_minimize(self, error_vector):
        # Find the coefficients x that solve B @ x = C with B and C defined below
        B = jnp.zeros((2, len(error_vector) + 1, len(error_vector) + 1))
        B = B.at[:, 1:, 1:].set(jnp.einsum("iskl,jskl->sij", error_vector, error_vector))
        B = B.at[:, 0, 1:].set(1)
        B = B.at[:, 1:, 0].set(1)

        C = jnp.zeros((2, len(error_vector) + 1))
        C = C.at[:, 0].set(1)

        w, v = jnp.linalg.eigh(B[0])
        w, v = w.real, v.real
        x0 = jnp.einsum("ij,jk,km,m-> i", v, jnp.diag(1.0 / w), v.T.conj(), C[0])

        w, v = jnp.linalg.eigh(B[1])
        w, v = w.real, v.real
        x1 = jnp.einsum("ij,jk,km,m-> i", v, jnp.diag(1.0 / w), v.T.conj(), C[1])

        x = jnp.stack([x0, x1], axis=0)
        assert not jnp.any(jnp.isnan(x))
        return x[:, 1:]

    def ediis_minimize(self, density_vector, fock_vector, energy_vector):
        r"""SCF-EDIIS
        Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195

        Warning: This implementation of EDIIS uses jax.scipy.optimize.minimize() to minimize the cost function.
        `minimize` supports jit() compilation, but does not yet support differentiation
        or arguments in the form of multi-dimensional arrays. Support for both is planned.

        Code taken from
        https://github.com/pyscf/pyscf/blob/df92512c09c13063a056dbc543e980e1997d21c8/pyscf/scf/diis.py#L149
        """
        nx = energy_vector.size
        nao = density_vector.shape[-1]
        density_vector = density_vector.reshape(nx, -1, nao, nao)
        fock_vector = fock_vector.reshape(nx, -1, nao, nao)
        df = jnp.einsum("ispq,jsqp->ij", density_vector, fock_vector).real
        diag = df.diagonal()
        df = diag[:, None] + diag - df - df.T

        def costf(x):
            c = x**2 / (x**2).sum()
            return jnp.einsum("i,i", c, energy_vector) - jnp.einsum("i,ij,j", c, df, c)

        res = scipyminimize(costf, jnp.ones(energy_vector.size), method="BFGS", tol=1e-9)
        return (res.x**2) / (res.x**2).sum(), res.fun

    def adiis_minimize(self, density_vector, fock_vector, idnewest):
        r"""
        Ref: JCP 132, 054109 (2010); DOI:10.1063/1.3304922

        Warning: This implementation of EDIIS uses jax.scipy.optimize.minimize() to minimize the cost function.
        `minimize` supports jit() compilation, but does not yet support differentiation
        or arguments in the form of multi-dimensional arrays. Support for both is planned.

        Code taken from
        https://github.com/pyscf/pyscf/blob/df92512c09c13063a056dbc543e980e1997d21c8/pyscf/scf/diis.py#L208
        """

        nx = density_vector.shape[0]
        nao = density_vector.shape[-1]
        density_vector = density_vector.reshape(nx, -1, nao, nao)
        fock_vector = fock_vector.reshape(nx, -1, nao, nao)
        df = jnp.einsum("ispq,jsqp->ij", density_vector, fock_vector).real
        d_fn = df[:, idnewest]
        dn_f = df[idnewest]
        dn_fn = df[idnewest, idnewest]
        dd_fn = d_fn - dn_fn
        df = df - d_fn[:, None] - dn_f + dn_fn

        def costf(x):
            c = x**2 / (x**2).sum()
            return jnp.einsum("i,i", c, dd_fn) * 2 + jnp.einsum("i,ij,j", c, df, c)

        res = scipyminimize(costf, jnp.ones(nx), method="BFGS", tol=1e-9)
        return (res.x**2) / (res.x**2).sum(), res.fun


def damping(s, d, f, factor):
    r"""Copied from pyscf.scf.hf.damping"""
    # dm_vir = s - reduce(numpy.dot, (s,d,s))
    # sinv = numpy.linalg.inv(s)
    # f0 = reduce(numpy.dot, (dm_vir, sinv, f, d, s))
    dm_vir = jnp.eye(s.shape[0]) - jnp.dot(s, d)
    f0 = reduce(jnp.dot, (dm_vir, f, d, s))
    f0 = (f0 + f0.conj().T) * (factor / (factor + 1.0))
    return f - f0


def level_shift(s, d, f, factor):
    r"""Copied from pyscf.scf.hf.level_shift
    
    Apply level shift :math:`\Delta` to virtual orbitals

    .. math::
        :nowrap:

        \begin{align}
            FC &= SCE \\
            F &= F + SC \Lambda C^\dagger S \\
            \Lambda_{ij} &=
            \begin{cases}
                \delta_{ij}\Delta & i \in \text{virtual} \\
                0 & \text{otherwise}
            \end{cases}
        \end{align}

    Returns:
        New Fock matrix, 2D ndarray
    """
    dm_vir = s - reduce(jnp.dot, (s, d, s))
    return f + dm_vir * factor
