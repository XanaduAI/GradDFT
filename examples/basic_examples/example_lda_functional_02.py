from jax import grad, numpy as jnp
from jax.lax import stop_gradient
from flax.core import freeze
from functional import Functional, exchange_polarization_correction
from interface import molecule_from_pyscf
from molecule import Molecule, coulomb_potential, symmetrize_rdm1

from jax import config
config.update("jax_enable_x64", True)

# In this basic tutorial we want to introduce the concept of a functional.

# We we will prepare a molecule, following the previous tutorial:
from pyscf import gto, dft
from train import molecule_predictor
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
def lsda_features(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = molecule.density()
    # To avoid numerical issues in JAX we limit too small numbers.
    rho = jnp.clip(rho, a_min = clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = -3./2. * (3. / (4*jnp.pi)) ** (1 / 3) * (rho.sum(axis = 0))**(4/3)
    # For simplicity we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be a list of arrays of dimension n_grid x n_features.
    e = [jnp.expand_dims(lda_e, axis = 1)]
    return e

# Then we have to define a function that takes the output of features and returns the energy density.
# Its first argument represents the instance of the functional. Note how we sum over the dimensions
# feature in the LDA energy density that we computed above
def lsda(instance, e): return jnp.einsum('rf->r',e)

# Overall, we have the functional
LSDA = Functional(function=lsda, features=lsda_features)
params = freeze({'params': {}}) # Since the functional is not neural, we pass frozen dict for the parameters

# We can compute the predicted energy using the following code:
predict_molecule = molecule_predictor(LSDA)
predicted_energy_0, fock = predict_molecule(params = params, molecule = HF_molecule)
# We may use molecule_predictor to compute the energy of any other molecule too.

# Another form of doing the same is first computing the features and then the energy.
features = lsda_features(molecule = HF_molecule)
predicted_energy_1 = LSDA.energy(params, HF_molecule, *features)

# Under the hood, what is really happening to compute the energy is the following
features = lsda_features(molecule = HF_molecule)
# If we want to compute the energy_density that come out of the functional we can do
xc_energy_density = LSDA.apply(params, *features)
# and then integrate them
predicted_energy_2 = LSDA._integrate(xc_energy_density, HF_molecule.grid.weights)
# and add the non-exchange-correlation energy component
predicted_energy_2 += stop_gradient(HF_molecule.nonXC())
# Note that functional.apply(params, *features) implements
# functional.function(functional, *features) with parameters params

# We can check that all methods return the same energy
print('Predicted energies', predicted_energy_0, predicted_energy_1, predicted_energy_2)

########################### Computing the Fock matrix using autodiff #########################################

# How do we compute the fock matrix above? We can use the jax value_and_grad method method:
# Let us start defining a function that computes the energy from some reduced density matrix rdm1:
def compute_energy_and_fock(rdm1, molecule):

    molecule = molecule.replace(rdm1 = rdm1)
    features = lsda_features(molecule = molecule)
    return LSDA.energy(params, molecule, *features)

# Now comes the magic of jax. We can compute the energy and the gradient of the energy
# using jax.grad (or alternatively value_and_grad), indicating the argument with respect to which take derivatives.
new_fock = grad(compute_energy_and_fock, argnums = 0)(HF_molecule.rdm1, HF_molecule)
# We need to add the corrections to compute the full fock matrix
new_fock = 1/2*(new_fock + new_fock.transpose(0,2,1))
rdm1 = symmetrize_rdm1(HF_molecule.rdm1)
new_fock += coulomb_potential(rdm1, HF_molecule.rep_tensor)
new_fock = jnp.stack([new_fock.sum(axis = 0)/2., new_fock.sum(axis = 0)/2.], axis=0) # Only when molecule.spin != 0
new_fock = new_fock + jnp.stack([HF_molecule.h1e, HF_molecule.h1e], axis=0)

print('Is the newly computed fock matrix correct?:',jnp.isclose(fock, new_fock).all() )