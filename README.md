<div align="center">

# Grad DFT: a software library for machine learning enhanced density functional theory

![Light Theme Image](media/README/light_logo.svg#gh-light-mode-only)

![Dark Theme Image](media/README/dark_logo.svg#gh-dark-mode-only)

[![build](https://img.shields.io/badge/build-passing-graygreen.svg "https://github.com/XanaduAI/GradDFT/actions")](https://github.com/XanaduAI/GradDFT/actions)
[![arXiv](http://img.shields.io/badge/arXiv-2309.15127-B31B1B.svg "Grad-DFT")](https://arxiv.org/abs/2309.15127)
[![License](https://img.shields.io/badge/License-Apache%202.0-9F9F9F "https://github.com/XanaduAI/GradDFT/blob/main/LICENSE")](https://github.com/XanaduAI/GradDFT/blob/main/LICENSE)

</div>

Grad DFT is a JAX-based library enabling the differentiable design and experimentation of exchange-correlation functionals using machine learning techniques. The library provides significant functionality, including (but not limited to) training neural functionals with fully differentiable and just-in-time compilable self-consistent-field loops, direct optimization of the Kohn-Sham orbitals, and implementation of many of the known constraints of the exact functional.

## Functionality

The current version of the library has the following capabilities:

* Create any `NeuralFunctional` that follows the expression

```math
E_{xc,\theta} = \int d\mathbf{r} \mathbf{c}_\theta[\rho](\mathbf{r})\cdot\mathbf{e}[\rho](\mathbf{r}),
```

that is, under the locality assumption.

* Include (non-differentiable) range-separated Hartree Fock components.
* Train neural functionals using fully differentiable and just-in-time (jit) compilable self-consistent iterative procedures.
* Perform DFT simulations with neural functionals using differentiable and just-in-time compilable [direct optimization of the Kohn-Sham orbitals](https://openreview.net/forum?id=aBWnqqsuot7).
* Train neural functionals using loss functions that include contributions from the total energy, density, or both.
* Include regularization terms that prevent the divergence of the self-consistent iterative procedure for non-self-consistently trained functionals. This includes the regularization term suggested in the supplementary material of [DM21](https://www.science.org/doi/full/10.1126/science.abj6511).
* Use [15 constraints of the exact functional](https://www.annualreviews.org/doi/abs/10.1146/annurev-physchem-062422-013259) which can be added to existing loss functions.
* Train with the [Harris functional](https://en.wikipedia.org/wiki/Harris_functional) for higher accuracy non-self consistent training.
* Design neural functionals with a library of energy densities used in well-known functionals such as [B3LYP](https://pubs.acs.org/doi/abs/10.1021/j100096a001) and [DM21](https://www.science.org/doi/full/10.1126/science.abj6511).
* Include simple DFT-D dispersion tails with a neural parametrization.

Future capabilities will include [sharding](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) the training on HPC systems and the implementation of periodic boundary conditions for training neural functionals designed for condensed matter systems.

## Install

A core dependency of Grad DFT is [PySCF](https://pyscf.org). To successfully install this package in the forthcoming installation with `pip`, please ensure that `cmake` is installed and that

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

## Use example

Using Grad DFT typically involves the following steps:

1. Specify an instance of `Molecule`, which has methods to compute the electronic density $\rho$ and derived quantities.
2. Define the function `energy_densities`, that computes $\mathbf{e}\[\rho\](\mathbf{r})$.
3. Implement the function `coefficients`, which may include a neural network, and compute $\mathbf{c}_{\theta}\[\rho\](\mathbf{r})$. If the function `coefficients` requires inputs, specify function `coefficient_inputs` too.
4. Build the `Functional`, which has the method `functional.energy(molecule, params)`, which computes the Kohn-Sham total energy according to

```math
E_{KS}[\rho] = \sum_{i=0}^{\text{occ}} \int d\mathbf{r}\; |\nabla \varphi_{i}(\mathbf{r})|^2  + \frac{1}{2}\int d\mathbf{r} d\mathbf{r}'\frac{\rho(\mathbf{r})\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} +\int d\mathbf{r} U(\mathbf{r}) \rho(\mathbf{r}) + E_{II} + E_{xc}[\rho],
```

with

```math
E_{xc,\theta}[\rho] = \int d\mathbf{r} \mathbf{c}_{\theta}[\rho](\mathbf{r})\cdot\mathbf{e}[\rho](\mathbf{r}),
```

and where `params` indicates neural network parameters $\theta$.

5. Train the neural functional using JAX autodifferentiation capabilities, in particular `jax.grad`.

Now let's see how we can complete the above steps with code in Grad DFT.

### Creating a molecule

The first step is to create a `Molecule` object.

```python
from grad_dft import (
	energy_predictor,
	simple_energy_loss,
	NeuralFunctional,
	molecule_from_pyscf
)
from pyscf import gto, dft

# Define a PySCF mol object for the H2 molecule
mol = gto.M(atom = [['H', (0, 0, 0)], ['H', (0.74, 0, 0)]], basis = 'def2-tzvp', spin = 0)
# Create a PySCF mean-field object
mf = dft.UKS(mol)
mf.kernel()
# Create a Molecule from the mean-field object
molecule = molecule_from_pyscf(mf)
```

### Creating a neural functional

A more complex, neural functional can be created as

```python
from jax.nn import sigmoid, gelu
from jax.random import PRNGKey
from flax import linen as nn
from optax import adam, apply_updates
from tqdm import tqdm

def energy_densities(molecule):
    rho = molecule.density()
    lda_e = -3/2 * (3/(4*jnp.pi))**(1/3) * (rho**(4/3)).sum(axis = 1, keepdims = True)
    return lda_e

def coefficient_inputs(molecule):
    rho = molecule.density()
    kinetic = molecule.kinetic_density()
    return jnp.concatenate((rho, kinetic))

def coefficients(self, rhoinputs):
    x = nn.Dense(features=1)(rhoinputs) # features = 1 means it outputs a single weight
    x = nn.LayerNorm()(x)
    return gelu(x)

neuralfunctional = NeuralFunctional(coefficients, energy_densities, coefficient_inputs)
```

with the corresponding energy calculation

```python
key = PRNGKey(42)
cinputs = coefficient_inputs(molecule)
params = neuralfunctional.init(key, cinputs)

predicted_energy = neuralfunctional.energy(params, molecule)
```

### Training the neural functional

```python
learning_rate = 1e-5
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)

# and implement the optimization loop
n_epochs = 20
compute_energy = energy_predictor(neuralfunctional)
for iteration in tqdm(range(n_epochs), desc="Training epoch"):
    (cost_value, predicted_energy), grads = simple_energy_loss(
        params, compute_energy, molecule, ground_truth_energy
    )
    print("Iteration", iteration, "Predicted energy:", predicted_energy, "Cost value:", cost_value)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

# Save checkpoint
neuralfunctional.save_checkpoints(params, tx, step=n_epochs)
```

For more detailed examples, check out the `~/examples` folder.

<p align="center">

<img src="/media/README/light_mode_disodium_animation.gif#gh-light-mode-only" width="45%" height="45%"/>
</p>

<p align="center">

<img src="/media/README/dark_mode_disodium_animation.gif#gh-dark-mode-only#gh-dark-mode-only" width="45%" height="45%"/>

</p>
<p align="center"> Using a scaled-down version of the neural functional used in the main Grad DFT article, we train it using the total energies and densities derived from the experimental equilibrium geometries of Li<sub>2</sub> and K<sub>2</sub> at the Coupled Cluster Singles & Doubles (CCSD) level of accuracy. The animation shows that during this training, the neural functional also generalized to predict the CCSD density of Na<sub>2</sub>. </p>

## Acknowledgements

We thank helpful comments and insights from Alain Delgado, Modjtaba Shokrian Zini, Stepan Fomichev, Soran Jahangiri, Diego Guala, Jay Soni, Utkarsh Azad, Kasra Hejazi, Vincent Michaud-Rioux, Maria Schuld and Nathan Wiebe.

GradDFT often follows similar calculations and naming conventions as PySCF, though adapted for our purposes. Only a few non-jittable DIIS procedures were directly taken from it. Where this happens, it has been conveniently referenced in the documentation. The tests were also implemented against PySCF results. PySCF Notice file is included for these reasons.

## Bibtex

```
@misc{casares2023graddft,
      title={Grad DFT: a software library for machine learning enhanced density functional theory}, 
      author={Pablo A. M. Casares and Jack S. Baker and Matija Medvidovic and Roberto dos Reis and Juan Miguel Arrazola},
      year={2023},
      eprint={2309.15127},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
```
