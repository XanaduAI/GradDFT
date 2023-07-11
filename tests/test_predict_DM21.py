import os
import warnings

from interface import molecule_from_pyscf
from external import NeuralNumInt
from external import Functional
from functional import dm21_combine, dm21_features

# This only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

dirpath = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(dirpath)
config_path = os.path.normpath(dirpath + "/config/config.json")

data_path = os.path.normpath(dirpath + "/data")
model_path = os.path.normpath(dirpath + "/DM21_model")

learning_rate = 1e-3

from interface import molecule_from_pyscf
from evaluate import make_scf_loop, make_orbital_optimizer
from external.density_functional_approximation_dm21.density_functional_approximation_dm21.compute_hfx_density import get_hf_density
from openfermion import geometry_from_pubchem

from pyscf import gto, dft, cc, scf
import numpy as np
from functional import DM21
from utils.types import Hartree2kcalmol

functional = DM21()
params = functional.generate_DM21_weights()


###################### Closed shell ############################

molecule_name = 'water'
geometry = geometry_from_pubchem(molecule_name)
mol = gto.M(atom = geometry,
            basis="def2-tzvp")
mf2 = scf.RHF(mol)
mf2.kernel()
mycc = cc.CCSD(mf2).run()
ccsd_energy = mycc.e_tot
mf = dft.RKS(mol)
mf.kernel()

grid = mf.grids


def test_predict(mf, energy):
    ## Load the molecule, RKS
    warnings.warn('Remember to set the grid level to 3 in the config file!')

    molecule = molecule_from_pyscf(mf, energy = energy, omegas = [0., 0.4])

    #tx = adam(learning_rate = learning_rate)
    #iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    #e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional,
                                    verbose = 2, 
                                    functional_type = 'DM21')
    e_XND = iterator(params, molecule)

    mf = dft.RKS(mol)
    mf._numint = NeuralNumInt(Functional.DM21)
    e_DM = mf.kernel()

    kcalmoldiff = (e_XND-e_DM)*Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol = 1e1)

##################
test_predict(mf, energy = ccsd_energy)



###################### Open shell ############################


molecule_name = 'Co'
mol = gto.Mole()
mol.atom = 'Co 0 0 0'
mol.basis = "def2-tzvp" # alternatively basis_set_exchange.api.get_basis(name='cc-pvdz', fmt='nwchem', elements='Co')
mol.spin = 3
mol.build()
mf = dft.UKS(mol)
energy = mf.kernel()

grid = mf.grids


def test_predict(mf, energy):
    ## Load the molecule, UKS
    warnings.warn('Remember to set the grid level to 3 in the config file!')

    molecule = molecule_from_pyscf(mf, energy = energy, omegas = [0, 0.4])

    #tx = adam(learning_rate = learning_rate)
    #iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    #e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional,
                                    verbose = 2, 
                                    functional_type = 'DM21')
    e_XND = iterator(params, molecule)

    mf = dft.UKS(mol)
    mf._numint = NeuralNumInt(Functional.DM21)
    e_DM = mf.kernel()

    kcalmoldiff = (e_XND-e_DM)*Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol = 1e1)


##################
test_predict(mf, energy = ccsd_energy)



###################### Check against DM21 tests ############################

def test_rks():
    warnings.warn('Remember to set the basis to sto-3g in the config file before running this test, and the grid level to 3.')
    ni = NeuralNumInt(Functional.DM21)

    mol = gto.Mole()
    mol.atom = [['Ne', 0., 0., 0.]]
    mol.basis = 'sto-3g'
    mol.build()

    mf = dft.RKS(mol)
    mf.small_rho_cutoff = 1.e-20
    mf._numint = ni
    e_DM = mf.run().e_tot # Expected -126.898521

    molecule = molecule_from_pyscf(mf, energy = energy, omegas = [0., 0.4])

    #tx = adam(learning_rate = learning_rate)
    #iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    #e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional,
                                    feature_fn=dm21_features,
                                    combine_features_hf=dm21_combine,
                                    omegas = [0., 0.4], 
                                    verbose = 2, 
                                    functional_type = 'DM21')
    e_XND = iterator(params, molecule)

    kcalmoldiff = (e_XND-e_DM)*Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol = 1e1)


def test_uks():
    warnings.warn('Remember to set the basis to sto-3g in the config file before running this test, and the grid level to 3.')
    ni = NeuralNumInt(Functional.DM21)

    mol = gto.Mole()
    mol.atom = [['C', 0., 0., 0.]]
    mol.spin = 2
    mol.basis = 'sto-3g'
    mol.build()

    mf = dft.UKS(mol)
    mf.small_rho_cutoff = 1.e-20
    mf._numint = ni
    e_DM = mf.run().e_tot # Expected -37.34184876

    molecule = molecule_from_pyscf(mf, energy = energy, omegas = [0., 0.4])

    #tx = adam(learning_rate = learning_rate)
    #iterator = make_orbital_optimizer(functional, tx, omegas = [0., 0.4], verbose = 2, functional_type = 'DM21')
    #e_XND_DF4T = iterator(params, molecule)

    iterator = make_scf_loop(functional,
                                    verbose = 2, 
                                    functional_type = 'DM21')
    e_XND = iterator(params, molecule)

    kcalmoldiff = (e_XND-e_DM)*Hartree2kcalmol
    assert np.allclose(kcalmoldiff, 0, atol = 1e1)


##################
#test_rks()
#test_uks()


molecule_name = 'CoC'
mol = gto.Mole()
mol.atom = [['Co', [0, 0, 0]],
            ['C', [1.56, 0, 0]]] # def2-tzvp
mol.basis = "def2-tzvp" #basis_set_exchange.api.get_basis(name='cc-pvdz', fmt='nwchem', elements='Co')
mol.spin = 1
mol.unit = 'angstrom'
mol.build()
mf = dft.UKS(mol)
energy = mf.kernel()

grid = mf.grids

#test_predict(mf, energy)