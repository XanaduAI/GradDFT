from jax import grad, numpy as jnp
from jax.lax import stop_gradient
from flax.core import freeze
from grad_dft.functional import Functional, exchange_polarization_correction
from grad_dft.interface import molecule_from_pyscf
from grad_dft.molecule import Molecule, coulomb_potential, symmetrize_rdm1 

from jax import config
config.update("jax_enable_x64", True)

# In this basic tutorial we want to introduce the concept of a functional.

# We we will prepare a molecule, following the previous tutorial:
from pyscf import gto, dft
from grad_dft.train import molecule_predictor
# Define the geometry of the molecule and mean-field object
mol = gto.M(atom = [['H', (0, 0, 0)]], basis = 'def2-tzvp', charge = 0, spin = 1)
mf = dft.UKS(mol)
mf.kernel()
# Then we can use the following function to generate the molecule object
HF_molecule = molecule_from_pyscf(mf)

########################### Creating an LDA functional #########################################

# Now we create our functional. We need to define at least the following methods:

# First a features method, which takes a molecule and returns an array of features
# It computes what in the article appears as potential e_\theta(r), as well as the
# input to the neural network to compute the density.
def lsda_density(molecule: Molecule, clip_cte: float = 1e-27):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = molecule.density()
    # To avoid numerical issues in JAX we limit too small numbers.
    rho = jnp.clip(rho, a_min = clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = -3/2 * (3/(4*jnp.pi)) ** (1/3) * (rho**(4/3)).sum(axis = 1, keepdims = True)
    # For simplicity we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    return lda_e

# Then we have to define a function that takes the output of features and returns the energy density.
# Its first argument represents the instance of the functional. Note how we sum over the dimensions
# feature in the LDA energy density that we computed above

# Overall, we have the functional
LSDA = Functional(coefficients = lambda self, *_: jnp.array([[1.]]), energy_densities=lsda_density)
params = freeze({'params': {}}) # Since the functional is not neural, we pass frozen dict for the parameters

# We can compute the predicted energy using the following code:
predict_molecule = molecule_predictor(LSDA)
predicted_energy_0, fock = predict_molecule(params = params, molecule = HF_molecule)
# We may use molecule_predictor to compute the energy of any other molecule too.

# Another form of doing the same is first computing the features and then the energy.
predicted_energy_1 = LSDA.energy(params, HF_molecule)

# Under the hood, what is really happening to compute the energy is the following:
# First we compute the densities
densities = LSDA.compute_densities(molecule = HF_molecule)
# Then we compute the coefficient inputs
cinputs = LSDA.compute_coefficient_inputs(molecule = HF_molecule)
# Finally we compute the exchange-correlation energy
predicted_energy_2 = LSDA.apply_and_integrate(params, HF_molecule.grid, cinputs, densities)
# And add the non-exchange-correlation energy component
predicted_energy_2 += stop_gradient(HF_molecule.nonXC())

# We can check that all methods return the same energy
print('Predicted energies', predicted_energy_0, predicted_energy_1, predicted_energy_2)

########################### Computing the Fock matrix using autodiff #########################################

# How do we compute the fock matrix above? We can use the jax value_and_grad method method:
# Let us start defining a function that computes the energy from some reduced density matrix rdm1:
def compute_energy_and_fock(rdm1, molecule):

    molecule = molecule.replace(rdm1 = rdm1)
    return LSDA.energy(params, molecule)

# Now comes the magic of jax. We can compute the energy and the gradient of the energy
# using jax.grad (or alternatively value_and_grad), indicating the argument with respect to which take derivatives.
new_fock = grad(compute_energy_and_fock, argnums = 0)(HF_molecule.rdm1, HF_molecule)
# We need to add the corrections to compute the full fock matrix
new_fock = 1/2*(new_fock + new_fock.transpose(0,2,1))
rdm1 = symmetrize_rdm1(HF_molecule.rdm1)
new_fock += coulomb_potential(rdm1, HF_molecule.rep_tensor)
new_fock = new_fock + jnp.stack([HF_molecule.h1e, HF_molecule.h1e], axis=0)

print('Is the newly computed fock matrix correct?:',jnp.isclose(fock, new_fock).all() )