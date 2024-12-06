# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Sequence, Union

import jax
from jax import lax
from jax import numpy as jnp

from jax.tree_util import tree_leaves, tree_map
from jax.extend import linear_util as lu
from jax.api_util import argnums_partial

from .types import Array
from .tree import tree_func


@tree_func
def chunk(x: Array, chunk_size: int = 1) -> Array:
    return x.reshape(-1, chunk_size, *x.shape[1:])


@tree_func
def unchunk(x: Array):
    return x.reshape(-1, *x.shape[2:])


def _get_chunks(x, n_bulk, chunk_size):
    bulk = tree_map(
        lambda l: lax.dynamic_slice_in_dim(l, start_index=0, slice_size=n_bulk, axis=0),
        x,
    )

    return chunk(bulk, chunk_size=chunk_size)


def _get_rest(x, n_bulk, n_rest):
    return tree_map(
        lambda l: lax.dynamic_slice_in_dim(l, start_index=n_bulk, slice_size=n_rest, axis=0),
        x,
    )


def map_over_chunks(fun, argnums=0):
    if isinstance(argnums, int):
        argnums = (argnums,)

    def mapped(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)

        return lax.map(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return mapped


def _chunk_vmapped_function(vmapped_fun, chunk_size, argnums=0):
    if chunk_size is None:
        return vmapped_fun

    def out_fun(*args, **kwargs):
        f = lu.wrap_init(vmapped_fun, kwargs)

        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)

        axis_len = tree_leaves(dyn_args)[0].shape[0]
        n_chunks, n_rest = divmod(axis_len, chunk_size)
        n_bulk = chunk_size * n_chunks

        bulk = _get_chunks(dyn_args, n_bulk, chunk_size)

        y = unchunk(lax.map(lambda x: f_partial.call_wrapped(*x), bulk))

        if n_rest > 0:
            rest = _get_rest(dyn_args, n_bulk, n_rest)
            y_rest = f_partial.call_wrapped(*rest)

            y = tree_map(lambda y1, y2: jnp.concatenate((y1, y2), axis=0), y, y_rest)

        return y

    return out_fun


def vmap_chunked(
    f: Callable, in_axes: Union[int, Sequence[int]] = 0, *, chunk_size: Optional[int] = None
):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(i for i, ix in enumerate(in_axes) if ix is not None)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)

    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)
