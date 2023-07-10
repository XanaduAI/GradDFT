# Differentiable DFT

Differentiable DFT is a Jax-based library for researchers to be able to quickly design and experiment with Machine-Learning-based (also called neural) functionals.

## The Functional

The core of the library is the class [Functional](https://github.com/XanaduAI/DiffDFT/blob/main/functional.py#L25), whose design and use is central to Density Functional Theory. In order to implement this functional, two components are key:

First, we need to define the Jax-based function $f_{\mathbf{\theta}}$ that will implements the functional

$$
E_{xc}[\rho] = \int d{\mathbf{r}} f_{\mathbf{\theta}}(\rho(\mathbf{r}), |\nabla \rho(\mathbf{r})|, |\nabla^2 \rho(\mathbf{r})|, \ldots).
$$

And second we need to a function that generates the features - inputs to $f_{\mathbf{\theta}}$ - from an instance of the auxiliary class [Molecule](https://github.com/XanaduAI/DiffDFT/blob/main/molecule.py#L61). These are divided according to whether autodifferentiation is intented to be used to compute their derivatives. Since `Molecule` can store arbitrary order derivatives of the atomic orbitals, the `Functional` may depend on arbitrary order derivatives of the electronic density. `Molecule` class contains not only many properties defining the electronic system, but also several auxiliary functions to compute properties such as the electronic density and its derivatives.

Once provided with these two defining elements, the energy of a system might be computed via method `functional.energy(params, molecule)`, where `molecule` is an instance of `Molecule`, `functional` is an instance of `Functional`, and `params` represent any parameters the functional may need.

The `Functional` class is also the parent class of [NeuralFunctional](https://github.com/XanaduAI/DiffDFT/blob/main/functional.py#L181), which additionally allows to save and load chechpoints, and allows for a straightforward of the usual multi-layer perceptron such as the one used to construct DM21 [1] (also available as an off-the-shelf class and and which allows to load the original parameters),

$$
E_{xc}^{\text{DM21}}[\rho] = \int  f_{\mathbf{\theta}}^{\text{DM21}}(\mathbf{x}(\mathbf{r}))\cdot
    \begin{bmatrix}
    e_x^{\text{LDA}}(\mathbf{r})\\
    e^{\text{HF}}(\mathbf{r})\\
    e^{\omega \text{HF}}(\mathbf{r})\\
    \end{bmatrix}
    d\mathbf{r},
$$

where $\mathbf{x}$ represent 11 features computed from $\rho$. In general, any functional of the form

$$
E_{xc}[\rho] = E_x[\rho] +E_c[\rho] \\
    = \sum_\sigma\int d\mathbf{r} \rho_\sigma(\mathbf{r}) F_{x,\sigma}[\rho(\mathbf{r})]  \epsilon_{x,\sigma}^{UEG}([\rho], \mathbf{r})
    +\sum_\sigma\int d\mathbf{r} \rho_\sigma(\mathbf{r}) F_{c,\sigma}[\rho(\mathbf{r})]  \epsilon_{c,\sigma}^{UEG}([\rho], \mathbf{r}),
$$

where $F_{x/c, \sigma}$ are the so-called inhomogeneous correction factors, composed of polynomials of a-dimensional derivatives of the electronic density, and $\epsilon_{x/c,\sigma}^{UEG}[\rho_\sigma]$ makes reference to the Homogeneous Electron Gass exchange/correlation electronic energy. (Range-separated) exact-exchange and dispersion components may also be introduced in the functional.

### Functionality, examples and tests

Our software library comes with auxiliary function [make_scf_loop](https://github.com/XanaduAI/DiffDFT/blob/main/evaluate.py#L26), which generates a fully differentiable self-consistent loop as long as the functional does not contain Hartree-Fock features, as these require recomputing expensive atomic integrals. A Jax-jit compilable [make_training_scf_loop](https://github.com/XanaduAI/DiffDFT/blob/main/train.py#L288) is also available in this case.   Alternatively, the user may use [make_orbital_optimizer](https://github.com/XanaduAI/DiffDFT/blob/main/evaluate.py#L195) which implements Ref. [2] approach of direct molecular orbital optimizer.

We also provide a number of regularization loss functions in [train.py](https://github.com/XanaduAI/DiffDFT/blob/main/train.py#L184), as well as an implementation of quite a few of the known constraints of the exact functional in [constraints.py](https://github.com/XanaduAI/DiffDFT/blob/main/constraints.py) [3]. The library also provides a number of [examples](https://github.com/XanaduAI/DiffDFT/tree/main/examples) of usage, as well as [tests](https://github.com/XanaduAI/DiffDFT/tree/main/tests) checking the implementation of the self-consistent loop, a our clone of DM21, and classical functionals such as B3LYP.

## Bibliography

1. J. Kirkpatrick, B. McMorrow, D. H. Turban, A. L. Gaunt, J. S. Spencer, A. G. Matthews, A. Obika,
   L. Thiry, M. Fortunato, D. Pfau, et al. [Pushing the frontiers of density functionals by solving the fractional electron problem](https://www.science.org/doi/abs/10.1126/science.abj6511). Science, 374(6573):1385–1389, 2021
2. T. Li, M. Lin, Z. Hu, K. Zheng, G. Vignale, K. Kawaguchi, A. C. Neto, K. S. Novoselov, and S. YAN. [D4FT: A Deep Learning approach to Kohn-Sham Density Functional Theory](https://openreview.net/forum?id=aBWnqqsuot7). In The Eleventh International Conference on Learning Representations, 2023.
3. Kaplan, Aaron D., Mel Levy, and John P. Perdew. [The predictive power of exact constraints and appropriate norms in density functional theory](https://doi.org/10.1146/annurev-physchem-062422-013259). *Annual Review of Physical Chemistry* 74 (2023): 193-218.
