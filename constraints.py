import jax.numpy as jnp
from jax.lax import Precision
from functional import Functional
from molecule import Molecule

from train import compute_features
from utils.types import PyTree

###################### Constraints ############################

def constraint_x1(functional: Functional, params: PyTree, molecule: Molecule, precision = Precision.HIGHEST):
    r"""
    ::math::
        \epsilon_x[n] < 0
    """

    # Compute the input features
    features = compute_features(functional, molecule)[0]

    # Mask the correlation features
    features = jnp.einsum('rf,f->rf', features, functional.exchange_mask)

    # Compute the exchange-correlation energy at each point
    ex = functional.apply(params, features)

    return jnp.less_equal(ex, 0.)