<div align="center">

# Grad-DFT: a software library for machine learning density functional theory

[![arXiv](http://img.shields.io/badge/arXiv-2101.10279-B31B1B.svg "Grad-DFT")](https://arxiv.org/abs/2101.10279)

</div>



Grad-DFT is a JAX-based library enabling the differentiable design and experimentation of exchange-correlation functionals using machine learning techniques. This library supports a parametrization of exchange-correlation functionals based on energy densities and associated coefficient functions; the latter typically constructed using neural networks:

```math
E_{xc} = \int d\mathbf{r} \mathbf{c}_\theta[\rho](\mathbf{r})\cdot\mathbf{e}[\rho](\mathbf{r}).
```

Grad-DFT provides significant functionality, including fully differentiable and just-in-time compilable self-consistent loop, direct optimization of the orbitals, and implementation of many of the known constraints of the exact functional in the form of loss functionals.

## Use example

### Creating a molecule

The first step is to create a `Molecule` object.

```python
from grad_dft.interface import molecule_from_pyscf
from pyscf import gto, dft

# Define a PySCF mol object for the H2 molecule
mol = gto.M(atom = [['H', (0, 0, 0)], ['H', (0.74, 0, 0)]], basis = 'def2-tzvp', spin = 0)
# Create a PySCF mean-field object
mf = dft.UKS(mol)
mf.kernel()
# Create a Molecule from the mean-field object
molecule = molecule_from_pyscf(mf)
```

### Creating a simple functional

Then we can create a `Functional`.

```python
from jax import numpy as jnp
from grad_dft.functional import Functional

def energy_densities(molecule):
    rho = molecule.density()
    lda_e = -3/2 * (3/(4*jnp.pi))**(1/3) * (rho**(4/3)).sum(axis = 0, keepdims = True)
    return lda_e

coefficients = lambda self, rho: jnp.array([[1.]])

LDA = Functional(coefficients, energy_densities)
```

We can use the functional to compute the predicted energy, where `params` stand for the $\theta$ parameters in the equation above.

```python
from flax.core import freeze

params = freeze({'params': {}})
predicted_energy = LDA.energy(params, molecule)
```

### A more complex neural functional

A more complex, neural functional can be created as

```python
from jax.nn import sigmoid, gelu
from flax import linen as nn
from grad_dft.functional import NeuralFunctional

def coefficient_inputs(molecule):
    rho = jnp.clip(molecule.density(), a_min = 1e-27)
    kinetic = jnp.clip(molecule.kinetic_density(), a_min = 1e-27)
    return jnp.concatenate((rho, kinetic))

def coefficients(self, rhoinputs):
    x = nn.Dense(features=1)(rhoinputs)
    x = nn.LayerNorm()(x)
    return gelu(x)

neuralfunctional = NeuralFunctional(coefficients, energy_densities, coefficient_inputs)
```

with the corresponding energy calculation

```python
from jax.random import PRNGKey

key = PRNGKey(42)
cinputs = coefficient_inputs(molecule)
params = neuralfunctional.init(key, cinputs)

predicted_energy = neuralfunctional.energy(params, molecule)
```

## Install

A core dependency of Grad-DFT is [PySCF](https://pyscf.org). To successfully install this package in the forthcoming installion with `pip`, please ensure that `cmake` is installed and that

```bash
which cmake
```

returns the correct path to the `cmake` binary. For instructions on installing `cmake`, visit https://cmake.org.

Now, in a fresh [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment), navigate to the root directory of this repository and issue

```bash
pip install -e .
```

to install the base package. If you wish to run the examples in `~/examples`, you can run

```bash
pip install -e ".[examples]"
```

to install the additional dependencies.

## Bibtex

```
@article{graddft,
  title={Grad-DFT: a software library for machine learning density functional theory},
  author={Casares, Pablo Antonio Moreno and Baker, Jack and Medvidovi{\'c}, Matija and Dos Reis, Roberto, and Arrazola, Juan Miguel},
  journal={arXiv preprint [number]},
  year={2023}
}
```
