from functools import partial

import jax
from jax import numpy as jnp
from jax.nn import gelu
from optax import adam, apply_updates

from interface.pyscf import molecule_from_pyscf
from molecule import default_features
from functional import Functional, canonicalize_inputs

# First we define a molecule:
from pyscf import gto, dft
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1')

grids = dft.gen_grid.Grids(mol)
grids.level = 2
grids.build()

mf = dft.UKS(mol)
mf.grids = grids
ground_truth_energy = mf.kernel()

# Then we compute quite a few properties which we pack into a class called Molecule
molecule = molecule_from_pyscf(mf)

# Then we define the (local) Functional, via an (arbitrary) function f whose output we will integrate (or not, if decide so)
squash_offset = 1e-4
layer_widths = [256]*6
activation = gelu
out_features = 4
sigmoid_scale_factor = 2.

def f(instance, rhoinputs, localfeatures, *_, **__):
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
        x = activation(x) # activation = jax.nn.elu
        instance.sow('intermediates', 'residual_elu_'+str(i), x)

    x = instance.head(x, out_features, sigmoid_scale_factor)

    return jnp.einsum('ri,ri->r', x, localfeatures)


functional = Functional(f)

key = jax.random.PRNGKey(42) # Jax-style random seed

# We generate the features from the molecule we created before
rhoinputs, localfeatures = default_features(molecule = molecule, functional_type='MGGA')

# We initialize the Functional parameters
key, = jax.random.split(key, 1)
params = functional.init(key, rhoinputs, localfeatures)

# If we want to compute the local weights that come out of the functional we can do
localfeatureweights = functional.apply(params, rhoinputs, localfeatures)
# and then integrate them
predicted_energy = functional._integrate(localfeatureweights, grids.weights)

# Alternatively, we can use an already prepared function that does both
predicted_energy = functional.energy(params, molecule, rhoinputs, localfeatures)
print('Predicted_energy:',predicted_energy)
# If we had a non-local functional, eg whose function f outputs an energy instead of an array,
# we'd just avoid the integrate step.


# Now we want to optimize the parameters. To do that the first step is defining an (arbitrary) cost function
@partial(jax.value_and_grad, has_aux = True)
def cost(params, molecule, trueenergy, *functioninputs):
    ''' Computes the loss function, here MSE, between predicted and true energy'''

    predictedenergy = functional.energy(params, molecule, *functioninputs)
    cost_value = (predictedenergy - trueenergy) ** 2

    return cost_value, predictedenergy

# Then, we define the optimizer
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate = learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 50
for iteration in range(n_epochs):
    (cost_value, predicted_energy), grads = cost(params, molecule, ground_truth_energy, rhoinputs, localfeatures)
    print('Iteration', iteration ,'Predicted energy:', predicted_energy)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)