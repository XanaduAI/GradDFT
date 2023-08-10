from typing import Callable
from functools import partial, wraps

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_flatten

from .types import PyTree, Key


def tree_size(tree: PyTree) -> int:
    return sum(l.size for l in tree_leaves(tree))


def tree_isfinite(tree: PyTree) -> PyTree:
    return all(jnp.isfinite(l).all() for l in tree_leaves(tree))


def tree_randn_like(tree: PyTree, key: Key) -> PyTree:
    leaves, treedef = tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten([jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)])


def tree_func(f: Callable) -> Callable:
    @wraps(f)
    def tree_f(tree: PyTree, *args, **kwargs):
        f_ = partial(f, *args, **kwargs)
        return tree_map(f_, tree)

    return tree_f


tree_shape = tree_func(jnp.shape)
