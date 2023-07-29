from jax.random import split, PRNGKey
from jax import numpy as jnp
from optax import adam, apply_updates

from interface.pyscf import molecule_from_pyscf
from functional import NeuralFunctional, canonicalize_inputs, default_loss, dm21_features
from jax.nn import gelu

# In this example we aim to explain how we can implement the self-consistent loop
# with a simple functional that does not contain Hartree-Fock components.

# First we define a molecule, using pyscf:
from pyscf import gto, dft
from train import molecule_predictor
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1')

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# Then we compute quite a few properties which we pack into a class called Molecule
molecule = molecule_from_pyscf(mf)

# Then we define the Functional, via an function whose output we will integrate.
squash_offset = 1e-4
layer_widths = [256]*6
out_features = 4
sigmoid_scale_factor = 2.
activation = gelu

def function(instance, rhoinputs, localfeatures, *_, **__):
    x = canonicalize_inputs(rhoinputs) # Making sure dimensions are correct

    # Initial layer: log -> dense -> tanh
    x = jnp.log(jnp.abs(x) + squash_offset) # squash_offset = 1e-4
    instance.sow('intermediates', 'log', x)
    x = instance.dense(features=layer_widths[0])(x) # features = 256
    instance.sow('intermediates', 'initial_dense', x)
    x = jnp.tanh(x)
    instance.sow('intermediates', 'tanh', x)

    # 6 Residual blocks with 256-features dense layer and layer norm
    for features,i in zip(layer_widths,range(len(layer_widths))): # layer_widths = [256]*6
        res = x
        x = instance.dense(features=features)(x)
        instance.sow('intermediates', 'residual_dense_'+str(i), x)
        x = x + res # nn.Dense + Residual connection
        instance.sow('intermediates', 'residual_residual_'+str(i), x)
        x = instance.layer_norm()(x) #+ res # nn.LayerNorm
        instance.sow('intermediates', 'residual_layernorm_'+str(i), x) 
        x = activation(x) # activation = jax.nn.gelu
        instance.sow('intermediates', 'residual_elu_'+str(i), x)

    x = instance.head(x, out_features, sigmoid_scale_factor)

    return jnp.einsum('ri,ri->r', x, localfeatures)

functional = NeuralFunctional(function, dm21_features, None, None)

key = PRNGKey(42) # Jax-style random seed

# We generate the features from the molecule we created before
rhoinputs, localfeatures = dm21_features(molecule = molecule, functional_type='MGGA')

# We initialize the Functional parameters
key, = split(key, 1)
params = functional.init(key, rhoinputs, localfeatures)

# If we want to compute the local weights that come out of the functional we can do
xc_energy_density = functional.apply(params, rhoinputs, localfeatures)
# and then integrate them
predicted_energy = functional._integrate(xc_energy_density, grids.weights)
# and add the non-exchange-correlation energy component
predicted_energy += molecule.nonXC()

# Alternatively, we can use an already prepared function that does everything
predicted_energy = functional.energy(params, molecule, rhoinputs, localfeatures)
print('Predicted_energy:',predicted_energy)
# If we had a non-local functional, eg whose function f outputs an energy instead of an array,
# we'd just avoid the integrate step.

# Then, we define the optimizer
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 50
molecule_predict = molecule_predictor(functional)
for iteration in range(n_epochs):
    (cost_value, predicted_energy), grads = default_loss(params, molecule_predict, molecule, ground_truth_energy)
    print('Iteration', iteration ,'Predicted energy:', predicted_energy)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

functional.save_checkpoints(params, tx, step = n_epochs)