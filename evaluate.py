import jax
from jax import numpy as jnp
from jax.lax import Precision

import optax
from typing import Callable, Tuple, Sequence, Optional

from functools import partial
from scipy.special import erfc
import time
from scipy.optimize import bisect
from functools import partial

from external import Functional
from utils import PyTree, Array, Scalar, Ansatz
from functional import Functional
Optimizer = optax.GradientTransformation

from molecule import Molecule, eig, Diis, make_rdm1, get_veff, orbital_grad
from functional import molecule_predictor
from utils import PyTree, Array, Scalar
from interface.pyscf import generate_chi_tensor, mol_from_Molecule, process_mol, mol_from_Molecule
from utils.types import Hartree2kcalmol



def make_molecule_scf_loop(fxc: Functional, omegas:Sequence, chunk_size: int = 1024, 
                            max_cycles: int = 50, diis_start_cycle: int = 1,
                            e_conv: float = 1e-5, g_conv: float = 1e-5, diis_method = 'CDIIS',
                            level_shift: tuple[float, float] = (0.,0.), damp: tuple[float, float] = (0.,0.), 
                            smearing: Optional[str] = None, smearing_sigma: Optional[float] = 0.,
                            precision = Precision.HIGHEST, verbose: int = 0, **kwargs) -> Callable:

    predict_molecule = molecule_predictor(fxc, omegas = omegas, chunk_size = chunk_size, **kwargs)

    def scf_iterator(
        params: PyTree, molecule: Molecule, *args
    ) -> Tuple[Scalar, Scalar]:

        # Used only for the chi matrix
        mol = mol_from_Molecule(molecule)
        _, mf = process_mol(mol, compute_energy=False, grid_level = int(molecule.grid_level), training = False)

        old_e = jnp.inf
        norm_gorb = jnp.inf
        cycle = 0
        nelectron = molecule.atom_index.sum() - molecule.charge

        # Predict the energy and the vxc
        exc, vxc = predict_molecule(params, molecule, *args)
        predicted_e, vhf, (h1e_energy, coulomb2e_energy) = get_veff(exc = exc, vxc = vxc, molecule = molecule, dm = molecule.density_matrix, training = False, precision = precision)

        # Initialize DIIS
        A = jnp.identity(molecule.s1e.shape[0])
        diis = Diis(overlap_matrix=molecule.s1e, A = A, max_diis = 10, diis_method = diis_method)
        diis_data = (jnp.empty((0, 2, A.shape[0], A.shape[0])), jnp.empty((0, 2, A.shape[0], A.shape[0])), 
                    jnp.empty(0), jnp.empty((0, 2, A.shape[0], A.shape[0])))

        while (abs(predicted_e - old_e)*Hartree2kcalmol > e_conv or norm_gorb > g_conv) and cycle < max_cycles:
            # Convergence criterion is energy difference (default 1) kcal/mol and norm of gradient of orbitals < g_conv
            start_time = time.time()
            old_e = predicted_e

            # Compute Fock matrix
            fock = molecule.get_fock(vhf)

            # DIIS iteration
            new_data = (molecule.density_matrix, fock, predicted_e)
            fock, diis_data = diis.run(new_data, diis_data, cycle)

            # Diagonalize Fock matrix
            mo_energy, mo_coeff = eig(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff = mo_coeff)
            molecule = molecule.replace(mo_energy = mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            if verbose > 2:
                print("Cycle {} took {:.1e} seconds to compute and diagonalize Fock matrix".format(cycle, time.time() - start_time))

            if smearing:
                def gaussian_smearing_occ(m, mo_energy, sigma):
                    return 0.5 * erfc((mo_energy - m) / sigma)
                
                def fermi_smearing_occ(m, mo_energy, sigma):
                    return 1/(jnp.exp((mo_energy - m) / sigma)+1.)
                
                if smearing == 'gaussian': smearing_occ = gaussian_smearing_occ
                elif smearing == 'fermi-dirac': smearing_occ = fermi_smearing_occ
                
                def nelec_cost_fn(m, mo_es, sigma, _nelectron):
                    mo_occ = smearing_occ(m, mo_es, sigma)
                    res = (mo_occ.sum() - _nelectron)
                    return res

                sigma = smearing_sigma
                mo_es = jnp.hstack(mo_energy)
                x0 = bisect(nelec_cost_fn, a = min(mo_energy), b = max(mo_energy), xtol= 1e-10, rtol =  1e-10, 
                            maxiter = 10000, args = (mo_es, sigma, nelectron))
                mo_occ = smearing_occ(x0, mo_es, sigma)
                molecule = molecule.replace(mo_occ = mo_occ)

            # Update the density matrix
            rdm1 = molecule.make_rdm1()
            molecule = molecule.replace(rdm1 = rdm1)

            #computed_charge = jnp.einsum('r,ra,rb,sab->', molecule.grid.weights, molecule.ao, molecule.ao, dm)
            #assert jnp.isclose(nelectron, computed_charge, atol = 1e-3), "Total charge is not conserved"

            # Update the chi matrix
            if len(omegas) > 0:
                chi_start_time = time.time() 
                chi = generate_chi_tensor(molecule, mf.mol, omegas = omegas, chunk_size=chunk_size, grid_coords=molecule.grid.coords, *args)
                molecule = molecule.replace(chi = chi)
                if verbose > 2:
                    print("Cycle {} took {:.1e} seconds to compute chi matrix".format(cycle, time.time() - chi_start_time))

            exc_start_time = time.time()
            exc, vxc = predict_molecule(params, molecule, *args)
            predicted_e, vhf, (h1e_energy, coulomb2e_energy) = get_veff(exc = exc, vxc = vxc, molecule = molecule, dm = rdm1, training = False, precision = precision)
            exc_time = time.time()

            # Update the one and two body energies
            molecule = molecule.replace(h1e_energy = h1e_energy)
            molecule = molecule.replace(coulomb2e_energy = coulomb2e_energy)

            if verbose > 2:
                print("Cycle {} took {:.1e} seconds to compute exc and vhf".format(cycle, exc_time - exc_start_time))

            # Compute Fock matrix again, without DIIS, and the norm of the gradient
            fock = molecule.get_fock(vhf)
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

            if verbose > 1:
                print("cycle: {}, energy: {:.7e}, energy difference: {:.4e}, norm_gradient_orbitals: {:.2e}, seconds: {:.1e}".format(cycle, predicted_e, abs(predicted_e - old_e), norm_gorb, time.time() - start_time))
            if verbose > 2:
                print("       relative energy difference: {:.5e}".format(abs((predicted_e - old_e)/predicted_e)))
            cycle += 1

        if (abs(predicted_e - old_e)*Hartree2kcalmol < e_conv and norm_gorb < g_conv):
            # We perform an extra diagonalization to remove the level shift
            # Solve eigenvalue problem
            mo_energy, mo_coeff = eig(fock, molecule.s1e)
            molecule = molecule.replace(mo_coeff = mo_coeff)
            molecule = molecule.replace(mo_energy = mo_energy)

            # Update the molecular occupation
            mo_occ = molecule.get_occ()
            molecule = molecule.replace(mo_occ = mo_occ)

            # Update the density matrix
            rdm1 = molecule.make_rdm1()
            molecule = molecule.replace(rdm1 = rdm1)

            # Update the chi matrix
            if len(omegas) > 0:
                chi = generate_chi_tensor(molecule, mf.mol, omegas = omegas, chunk_size=chunk_size, grid_coords=molecule.grid.coords, *args)
                molecule = molecule.replace(chi = chi)

            exc, vxc = predict_molecule(params, molecule, *args)
            predicted_e, vhf, (h1e_energy, coulomb2e_energy) = get_veff(exc = exc, vxc = vxc, molecule = molecule, dm = rdm1, training = False, precision = precision)

            # Update the one and two body energies
            molecule = molecule.replace(h1e_energy = h1e_energy)
            molecule = molecule.replace(coulomb2e_energy = coulomb2e_energy)

            # Compute Fock matrix again, without DIIS, and the norm of the gradient
            fock = molecule.get_fock(vhf)
            norm_gorb = jnp.linalg.norm(orbital_grad(mo_coeff, mo_occ, fock))

        if verbose > 1:
            print("cycle: {}, predicted energy: {:.7e}, energy difference: {:.4e}, norm_gradient_orbitals: {:.2e}".format(cycle, predicted_e, abs(predicted_e - old_e), norm_gorb))

        return predicted_e

    def diis_make(mf): #todo: convert to jax.numpy
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        return mf_diis

    return scf_iterator

def make_orbital_optimizer(fxc: Functional, tx: Optimizer, omegas:Sequence, chunk_size: int = 1024, 
                            max_cycles: int = 500, e_conv: float = 1e-7, whitening: str = "PCA",
                            precision = Precision.HIGHEST, verbose: int = 0, **kwargs) -> Callable:

    predict_molecule = molecule_predictor(fxc, omegas = omegas, chunk_size = chunk_size, **kwargs)

    @partial(jax.value_and_grad, argnums=0)
    def molecule_orbitals_iterator(W: Array, D: Array, params: PyTree, molecule: Molecule, *args) -> Tuple[Scalar, Scalar]:
        Q0, _ = jnp.linalg.qr(W[0])
        Q1, _ = jnp.linalg.qr(W[1])
        Q = jnp.stack([Q0, Q1])

        # Compute the molecular orbitals
        C = jnp.einsum('sij,jk->ski', Q, D)

        I = jnp.einsum('sji,jk,skl->sil', C, molecule.s1e, C)
        stack = jnp.stack((jnp.identity(I.shape[1]),jnp.identity(I.shape[1])))
        assert jnp.allclose(I, stack)

        # Compute the density matrix
        dm = make_rdm1(C, molecule.mo_occ)
        dm_XND = molecule.make_rdm1()

        nelectron = molecule.atom_index.sum() - molecule.charge

        computed_charge = jnp.einsum('r,ra,rb,sab->', molecule.grid.weights, molecule.ao, molecule.ao, dm)
        assert jnp.isclose(nelectron, computed_charge, atol = 1e-3), "Total charge is not conserved"

        # Predict the energy and the vxc
        exc, vxc = predict_molecule(params, molecule, *args)
        predicted_e, _, _ = get_veff(exc = exc, vxc = vxc, molecule = molecule, dm = dm, training = False, precision = precision)
        return predicted_e

    def neural_iterator(
        params: PyTree, molecule: Molecule, *args
    ) -> Tuple[Scalar, Scalar]:

        # Used only for the chi matrix
        #mol = mol_from_Molecule(molecule)
        #_, mf = process_mol(mol, compute_energy=False, grid_level = int(molecule.grid_level), training = False)

        old_e = jnp.inf
        cycle = 0

        # Predict the energy and the vxc
        exc, vxc = predict_molecule(params, molecule, *args)
        predicted_e, vhf, _ = get_veff(exc = exc, vxc = vxc, molecule = molecule, dm = molecule.density_matrix, training = False, precision = precision)

        C = molecule.mo_coeff

        if whitening == "PCA":
            w, v = jnp.linalg.eig(molecule.s1e)
            D = (jnp.diag(jnp.sqrt(1/w)) @ v.T).real
            S_1 = (v @ jnp.diag(w) @ v.T).real
            diff = S_1 - molecule.s1e
            assert jnp.isclose(diff, jnp.zeros_like(diff), atol=1e-4).all()
            assert jnp.isclose(jnp.linalg.norm(jnp.linalg.inv(D) @ D - jnp.identity(D.shape[0])), 0.0, atol=1e-5)
        elif whitening == "Cholesky":
            D = jnp.linalg.cholesky(jnp.linalg.inv(molecule.s1e)).T
        elif whitening == "ZCA":
            w, v = jnp.linalg.eig(molecule.s1e)
            D = (v @ jnp.diag(jnp.sqrt(1/w)) @ v.T).real

        Q = jnp.einsum('sji,jk->sik', C, jnp.linalg.inv(D)) # C transposed
        Q_ = jnp.einsum('sji,jk,kl->sil', C, v, jnp.diag(jnp.sqrt(w))).real # C transposed
        assert jnp.allclose(Q, Q_)

        I = jnp.einsum('sji,jk,skl->sil', C, molecule.s1e, C) # The first C is transposed
        stack = jnp.stack((jnp.identity(I.shape[1]),jnp.identity(I.shape[1])))
        assert jnp.allclose(I, stack)

        I = jnp.einsum('sji,sjk->sik', Q, Q) # The first Q is transposed
        assert jnp.allclose(I, jnp.stack((jnp.identity(I.shape[1]),jnp.identity(I.shape[1]))))

        W = Q

        opt_state = tx.init(W)

        while abs(predicted_e - old_e)*Hartree2kcalmol > e_conv and cycle < max_cycles:
            start_time = time.time()
            old_e = predicted_e

            predicted_e, grads = molecule_orbitals_iterator(W, D, params, molecule, *args)

            updates, opt_state = tx.update(grads, opt_state, W)
            W = optax.apply_updates(W, updates)

            cycle += 1

            if verbose > 1:
                print("cycle: {}, predicted energy: {:.7e}, energy difference: {:.4e}".format(cycle, predicted_e, abs(predicted_e - old_e)))

        return predicted_e

    return neural_iterator
