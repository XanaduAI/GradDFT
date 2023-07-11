from jax.random import PRNGKey
from optax import adam, apply_updates
from jax.lax import stop_gradient

from interface.pyscf import molecule_from_pyscf
from functional import DM21, default_loss

# In this example we aim to explain how we can train the DM21 functional.

# First we define a molecule, using pyscf:
from pyscf import gto, dft
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1')

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# Then we compute quite a few properties which we pack into a class called Molecule.
# omegas will indicate the values of w in the range-separated Coulomb kernel
#  erf(w|r-r'|)/|r-r'|.
# Note that w = 0 indicates the usual Coulomb kernel 1/|r-r'|.
molecule = molecule_from_pyscf(mf, omegas = [0., 0.4])

# We create the functional and load its weights.
functional = DM21()
params = functional.generate_DM21_weights()

key = PRNGKey(42) # Jax-style random seed

# We generate the input features to the functional from the molecule we created before.
functional_inputs = functional.features(molecule)
nograd_functional_inputs = stop_gradient(functional.nograd_features(molecule))
functional_inputs = functional.combine(functional_inputs, nograd_functional_inputs)

# We can use this features to compute the energy by parts
energy = functional.apply_and_integrate(params, molecule, *functional_inputs)
energy += molecule.nonXC()

# Or alternatively, we can use an already prepared function that does everything for us
predicted_energy = functional.energy(params, molecule, *functional_inputs)
print('Predicted_energy:',predicted_energy)

# Then, we define the optimizer
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 50
for iteration in range(n_epochs):
    # Here we use a default loss without regularizer, but it is easy to adapt it.
    (cost_value, predicted_energy), grads = default_loss(params, functional, molecule, ground_truth_energy, *functional_inputs)
    print('Iteration', iteration ,'Predicted energy:', predicted_energy)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

# We can save the checkpoint back.
functional.save_checkpoints(params, tx, step = n_epochs)