from jax import numpy as jnp
from jax.random import split, PRNGKey
from optax import adam, apply_updates
from tqdm import tqdm
from evaluate import make_jitted_orbital_optimizer, make_orbital_optimizer, make_scf_loop, make_jitted_scf_loop
from train import molecule_predictor
from functional import NeuralFunctional, default_loss
from interface import molecule_from_pyscf
from molecule import Molecule
from jax.nn import sigmoid, gelu
from flax import linen as nn
from jax import config
config.update("jax_enable_x64", True)

# In this basic tutorial we want to introduce the concept of a functional.

# We we will prepare a molecule, following the previous tutorial:
from pyscf import gto, dft
# Define the geometry of the molecule
mol = gto.M(atom = [['H', (0, 0, 0)], ['H', (0, 0, 1)]], basis = 'def2-tzvp', charge = 0, spin = 0)
mf = dft.UKS(mol)
ground_truth_energy = mf.kernel()

# Then we can use the following function to generate the molecule object
HH_molecule = molecule_from_pyscf(mf)

########################### Creating a neural functional #########################################


# Now we create our neuralfunctional. We need to define at least the following methods: densities and coefficients
# which compute the two vectors that get dot-multiplied and then integrated over space. If the functional is a 
# neural functional we also need to define coefficient_inputs, for which in this case we will reuse the densities function.
def densities(molecule: Molecule, *_, **__):
    rho = jnp.clip(molecule.density(), a_min = 1e-27)
    kinetic = jnp.clip(molecule.kinetic_density(), a_min = 1e-27)
    return jnp.concatenate((rho, kinetic)).T

out_features = 4
def nn_coefficients(instance, rhoinputs, *_, **__):
    r"""
    Instance is an instance of the class Functional or NeuralFunctional.
    rhoinputs is the input to the neural network, in the form of an array.
    localfeatures represents the potentials e_\theta(r).

    The output of this function is the energy density of the system.
    """

    x = nn.Dense(features=out_features)(rhoinputs)
    x = nn.LayerNorm()(x)
    x = gelu(x)
    return sigmoid(x)

neuralfunctional = NeuralFunctional(coefficients=nn_coefficients, 
                                    energy_densities=densities,
                                    coefficient_inputs=densities)

# Now we can initialize the parameters of the neural network
key = PRNGKey(42)
cinputs = densities(HH_molecule)
params = neuralfunctional.init(key, cinputs)

########################### Training the functional #########################################

learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 20
molecule_predict = molecule_predictor(neuralfunctional)
for iteration in tqdm(range(n_epochs), desc='Training epoch'):
    (cost_value, predicted_energy), grads = default_loss(params, molecule_predict, HH_molecule, ground_truth_energy)
    print('Iteration', iteration ,'Predicted energy:', predicted_energy, 'Cost value:', cost_value)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

neuralfunctional.save_checkpoints(params, tx, step = n_epochs)


########################### Loading the functional and scf loop #########################################

# We can load the functional checkpoint with the following code
step = 20
train_state = neuralfunctional.load_checkpoint(tx, ckpt_dir='ckpts/checkpoint_' + str(step) +'/', step = step)
params = train_state.params
tx = train_state.tx
opt_state = tx.init(params)

# The following code takes some execution time

# Create the scf iterator
HH_molecule = molecule_from_pyscf(mf)
scf_iterator = make_scf_loop(neuralfunctional, verbose = 2, max_cycles = 5)
energy = scf_iterator(params, HH_molecule)
print('Energy from the scf loop:', energy)

# We can alternatively use the jit-ed version of the scf loop
HH_molecule = molecule_from_pyscf(mf)
scf_iterator = make_jitted_scf_loop(neuralfunctional, cycles = 5)
jitted_energy, _, _ = scf_iterator(params, HH_molecule)
print('Energy from the jitted scf loop:', jitted_energy)

# We can even use a direct optimizer of the orbitals
HH_molecule = molecule_from_pyscf(mf)
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
orbital_optimizer = make_orbital_optimizer(neuralfunctional, tx, max_cycles=20)
optimized_energy = orbital_optimizer(params, HH_molecule)
print('Energy from the orbital optimizer:', optimized_energy)


# We can even use a direct optimizer of the orbitals
HH_molecule = molecule_from_pyscf(mf)
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
orbital_optimizer = make_jitted_orbital_optimizer(neuralfunctional, tx, max_cycles=20)
jitted_optimized_energy = orbital_optimizer(params, HH_molecule)
print('Energy from the orbital optimizer:', jitted_optimized_energy)


