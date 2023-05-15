from typing import Union, Callable
import jax
from jax import numpy as jnp

from flax import linen as nn
import chex, optax
import pyscf.dft as dft
import pyscf.scf as scf
from pyscf.gto import Mole
from pyscf import lib

Ansatz = nn.Module
Key = chex.PRNGKey
PyTree = chex.ArrayTree
Array = Union[chex.Array, chex.ArrayNumpy, chex.ArrayBatched]
Scalar = Union[chex.Scalar, chex.Numeric]
Device = chex.Device
DType = chex.ArrayDType
HartreeFock = Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, scf.ghf.GHF, scf.dhf.DHF]
DensityFunctional = Union[dft.uks.UKS, dft.rks.RKS, dft.roks.ROKS]

def default_dtype() -> DType:
    return jnp.float64 if jax.config.x64_enabled else jnp.float32
