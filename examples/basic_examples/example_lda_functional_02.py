from jax import numpy as jnp, value_and_grad
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
# Define the geometry of the molecule
mol = gto.M(atom = [['H', (0, 0, 0)], ['F', (0, 0, 1.1)]], basis = 'def2-tzvp', charge = 0, spin = 0)

# To perform DFT we also need a grid
grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

# And we will also need a mean-field object
mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# Then we can use the following function to generate the molecule object
HF_molecule = molecule_from_pyscf(mf, grad_order=2, name = 'HF', 
                            energy=ground_truth_energy, omegas = [0., 0.4])

# Now we create our functional. We need to define at least the following methods:

# First a features method, which takes a molecule and returns an array of features
# It computes what in the article appears as potential e_\theta(r), as well as the
# input to the neural network to compute the density.
def lsda_features(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    rho = molecule.density()
    rho = jnp.clip(rho, a_min = clip_cte)
    lda_es = -3./4. * (jnp.array([[3.],[6.]]) / jnp.pi) ** (1 / 3) * (rho.sum(axis = 0))**(4/3)
    lda_e = exchange_polarization_correction(lda_es, rho)
    return [jnp.expand_dims(lda_e, axis = 1)]

# Then we have to define a function that takes the output of features and returns the energy density.
# Its first argument represents the instance of the functional.
def lsda(instance, x): return jnp.einsum('ri->r',x)

# Overall, we have the functional
LSDA = Functional(function=lsda, features=lsda_features)

# We can compute the predicted energy using the following code:
predict_molecule = molecule_predictor(LSDA)
# We need to pass an empty params dictionary. If it were a neural functional we would pass the parameters.
predicted_energy_0, fock = predict_molecule(params = freeze({'params': {}}), molecule = HF_molecule)

# Another form of doing the same is
features = lsda_features(molecule = HF_molecule)
predicted_energy_1 = LSDA.energy(freeze({'params': {}}), HF_molecule, *features)

# Under the hood, what is really happening to compute the energy is the following
features = lsda_features(molecule = HF_molecule)
# If we want to compute the local weights that come out of the functional we can do
xc_energy_density = LSDA.apply(freeze({'params': {}}), *features)
# and then integrate them
predicted_energy_2 = LSDA._integrate(xc_energy_density, HF_molecule.grid.weights)
# and add the non-exchange-correlation energy component
predicted_energy_2 += stop_gradient(HF_molecule.nonXC())


# We can check that both energies are the same
print('Predicted energies', predicted_energy_0, predicted_energy_1, predicted_energy_2)

# How do we compute the fock matrix above? We can use the jax value_and_grad method method:
def compute_energy_and_fock(rdm1, molecule):

    molecule = molecule.replace(rdm1 = rdm1)
    features = lsda_features(molecule = molecule)
    return LSDA.energy(freeze({'params': {}}), molecule, *features)

new_energy, new_fock = value_and_grad(compute_energy_and_fock, argnums = 0)(HF_molecule.rdm1, HF_molecule)

new_fock = 1/2*(new_fock + new_fock.transpose(0,2,1))
rdm1 = symmetrize_rdm1(HF_molecule.rdm1)
new_fock += coulomb_potential(rdm1, HF_molecule.rep_tensor)
new_fock = new_fock + jnp.stack([HF_molecule.h1e, HF_molecule.h1e], axis=0)

print('Is the newly computed fock matrix correct?:',jnp.isclose(fock, new_fock).all() )