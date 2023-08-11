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

from typing import Union
import jax
from jax import numpy as jnp

from flax import linen as nn
import chex, optax
import pyscf.dft as dft
import pyscf.scf as scf

Ansatz = nn.Module
Key = chex.PRNGKey
PyTree = chex.ArrayTree
Array = Union[chex.Array, chex.ArrayNumpy, chex.ArrayBatched]
Scalar = Union[chex.Scalar, chex.Numeric]
Device = chex.Device
DType = chex.ArrayDType
HartreeFock = Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, scf.ghf.GHF, scf.dhf.DHF]
DensityFunctional = Union[dft.uks.UKS, dft.rks.RKS, dft.roks.ROKS]
Optimizer = optax.GradientTransformation

Hartree2kcalmol = 627.50947 #http://www.u.arizona.edu/~stefanb/linkpages/conversions.html
Picometers2Angstroms = 0.01
Bohr2Angstroms = 0.52917721092

def default_dtype() -> DType:
    return jnp.float64 if jax.config.x64_enabled else jnp.float32
