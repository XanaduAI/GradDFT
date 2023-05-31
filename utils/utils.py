from typing import Optional
from utils import DType,  default_dtype
import jax.numpy as jnp

def to_device_arrays(*arrays, dtype: Optional[DType] = None):

    if dtype is None:
        dtype = default_dtype()

    return [jnp.asarray(array, dtype=dtype) for array in arrays]