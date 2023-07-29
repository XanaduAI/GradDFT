from jax import numpy as jnp
from jax.random import split, PRNGKey
from optax import adam, apply_updates
from tqdm import tqdm
from evaluate import make_jitted_orbital_optimizer, make_orbital_optimizer, make_scf_loop, make_jitted_scf_loop
from train import molecule_predictor
from functional import NeuralFunctional, default_loss, dm21_local_features, dm21_molecule_features
from interface import molecule_from_pyscf
from molecule import Molecule
from jax.nn import sigmoid, gelu

# In this basic tutorial we want to introduce the concept of a functional.

# We we will prepare a molecule, following the previous tutorial:
from pyscf import gto, dft
# Define the geometry of the molecule
mol = gto.M(atom = [['H', (0, 0, 0)], ['H', (0, 0, 1)]], basis = 'def2-tzvp', charge = 0, spin = 0)
mf = dft.UKS(mol)
ground_truth_energy = mf.kernel()

# Then we can use the following function to generate the molecule object
HF_molecule = molecule_from_pyscf(mf)

########################### Creating a neural functional #########################################


# Now we create our neuralfunctional. We need to define at least the following methods:

# First a features method, which takes a molecule and returns an array of features
# It computes what in the article appears as potential e_\theta(r), as well as the
# input to the neural network to compute the density.
def features(molecule: Molecule, functional_type: str = 'MGGA', clip_cte: float = 1e-27, *args, **kwargs):
    
    features = dm21_molecule_features(molecule, *args, **kwargs)
    localfeatures = dm21_local_features(molecule, functional_type, clip_cte)

    # We return them first the input to the neural network, and then the local features e_\theta(r)
    return features, localfeatures

# The second key method we have to define is the function method, which takes a molecule
out_features = 4
def function(instance, rhoinputs, localfeatures, *_, **__):
    r"""
    Instance is an instance of the class Functional or NeuralFunctional.
    rhoinputs is the input to the neural network, in the form of an array.
    localfeatures represents the potentials e_\theta(r).

    The output of this function is the energy density of the system.
    """

    x = instance.dense(features=out_features)(rhoinputs)
    x = instance.layer_norm()(x)
    x = gelu(x)
    x = sigmoid(x)

    return jnp.einsum('ri,ri->r', x, localfeatures)

neuralfunctional = NeuralFunctional(function=function, features=features)

# Now we can initialize the parameters of the neural network
key = PRNGKey(42)
neuralfeatures = features(HF_molecule)
params = neuralfunctional.init(key, *neuralfeatures)

########################### Training the functional #########################################

learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 20
molecule_predict = molecule_predictor(neuralfunctional)
for iteration in tqdm(range(n_epochs), desc='Training epoch'):
    (cost_value, predicted_energy), grads = default_loss(params, molecule_predict, HF_molecule, ground_truth_energy)
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
scf_iterator = make_scf_loop(neuralfunctional, verbose = 2, max_cycles = 5)
energy = scf_iterator(params, HF_molecule)
print('Energy from the scf loop:', energy)

# We can alternatively use the jit-ed version of the scf loop
scf_iterator = make_jitted_scf_loop(neuralfunctional, cycles = 5)
jitted_energy, _, _ = scf_iterator(params, HF_molecule)
print('Energy from the jitted scf loop:', jitted_energy)

# We can even use a direct optimizer of the orbitals
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
orbital_optimizer = make_orbital_optimizer(neuralfunctional, tx, max_cycles=50)
optimized_energy = orbital_optimizer(params, HF_molecule)
print('Energy from the orbital optimizer:', optimized_energy)


# We can even use a direct optimizer of the orbitals
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
orbital_optimizer = make_jitted_orbital_optimizer(neuralfunctional, tx, max_cycles=50)
jitted_optimized_energy = orbital_optimizer(params, HF_molecule)
print('Energy from the orbital optimizer:', jitted_optimized_energy)


