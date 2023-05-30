from utils import Scalar, Array
from typing import Optional, Union, Callable, Dict, Sequence, Tuple, NamedTuple
from dataclasses import fields
from utils import Array, Scalar
from functools import partial
from utils import PyTree, vmap_chunked
from external.eigh_impl import eigh2d

from jax import numpy as jnp
from jax.lax import Precision, stop_gradient
from jax import vmap, grad
from flax import struct
from flax import linen as nn
import itertools
import jax


@struct.dataclass
class Grid:

    coords: Array
    weights: Array

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"

    def __len__(self):
        return self.weights.shape[0]

    def to_dict(self) -> dict:
        return {"coords": self.coords, "weights": self.weights}

    def integrate(self, vals: Array, axis: int = 0) -> Array:

        """A function that performs grid quadrature (integration) in a differentiable way (using jax.numpy).

        This function is glorified tensor contraction but it sometimes helps
        with readability and expresing intent in the rest of the code.

        Parameters
        ----------
        vals : Array
            Local features/ function values to weigh.
            Expected shape: (..., n_lattice, ...)
        axis : int, optional
            Axis to contract. vals.shape[axis] == n_lattice
            has to hold.

        Returns
        -------
        Array
            Integrals of the same as `vals` but with `axis` contracted out.
            If vals.ndim==(1,), then the output is squeezed to a scalar.
        """

        return jnp.tensordot(self.weights, vals, axes=(0, axis))

@struct.dataclass
class Molecule:

    grid: Grid
    atom_index: Array
    nuclear_pos: Array
    ao: Array
    grad_ao: Array
    rdm1: Array
    nuclear_repulsion: Scalar
    h1e: Array
    vj: Array
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Array
    mf_energy: Optional[Scalar] = None
    s1e: Optional[Array] = None # Not used during training
    omegas: Optional[Array] = None
    chi: Optional[Array] = None
    rep_tensor: Optional[Array] = None
    energy: Optional[Scalar] = None
    basis: Optional[str] = None
    name: Optional[str] = None
    spin: Optional[Scalar] = 0
    charge: Optional[Scalar] = 0
    unit_Angstrom: Optional[bool] = True
    grid_level: Optional[int] = 2
    scf_iteration: Optional[int] = 50
    fock: Optional[Array] = None

    def __repr__(self):
        return f"{self.__class__.__name__}(grid_size={self.grid_size})"

    @property
    def grid_size(self):
        return len(self.grid)

    def density(self, *args, **kwargs):
        return density(self.rdm1, self.ao, *args, **kwargs)

    def grad_density(self, *args, **kwargs):
        return grad_density(self.rdm1, self.ao, self.grad_ao, *args, **kwargs)

    def kinetic_density(self, *args, **kwargs):
        return kinetic_density(self.rdm1, self.grad_ao, *args, **kwargs)
    
    def HF_energy_density(self, *args, **kwargs):
        return HF_energy_density(self.rdm1, self.ao, self.chi, *args, **kwargs)

    def HF_density_grad_2_Fock(self, fxc: Callable, features:Array, local_features: Array, ehf: Array, params: PyTree, *args, **kwargs):
        if self.chi is None: raise ValueError("Precomputed chi tensor has not been loaded.")
        return HF_density_grad_2_Fock(fxc, self, features, local_features, ehf, self.chi, self.ao, params = params, *args, **kwargs)

    def nonXC(self, *args, **kwargs):
        return nonXC(self.rdm1, self.h1e, self.vj, self.nuclear_repulsion, *args, **kwargs)
    
    def make_rdm1(self):
        return make_rdm1(self.mo_coeff, self.mo_occ)
    
    def get_occ(self):
        nelecs = [self.mo_occ[i].sum() for i in range(2)]
        return get_occ(self.mo_energy, nelecs)

    def to_dict(self) -> dict:
        grid_dict = self.grid.to_dict()
        rest = {field.name: getattr(self, field.name) for field in fields(self)[1:]}
        return dict(**grid_dict, **rest)



@partial(jax.jit, static_argnames="precision")
def density(rdm1: Array, ao: Array, precision: Precision = Precision.HIGHEST) -> Array:

    """Calculate electronic density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density. Shape: (*batch, n_spin, n_grid_points)
    """

    return jnp.einsum("...ab,ra,rb->...r", rdm1, ao, ao, precision=precision)

@partial(jax.jit, static_argnames="precision")
def grad_density(
    rdm1: Array, ao: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST
) -> Array:

    """Calculate the electronic density gradient from atomic orbitals.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    grad_ao : Array
        Gradients of atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals, 3)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The density gradient. Shape: (*batch, n_spin, n_grid_points, 3)
    """

    return 2 * jnp.einsum("...ab,ra,rbj->...rj", rdm1, ao, grad_ao, precision=precision)

@partial(jax.jit, static_argnames="precision")
def kinetic_density(rdm1: Array, grad_ao: Array, precision: Precision = Precision.HIGHEST) -> Array:

    """Calculate kinetic energy density from atomic orbitals.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    grad_ao : Array
        Gradients of atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals, 3)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jaxx.lax.Precision.HIGHEST.

    Returns
    -------
    Array
        The kinetic energy density. Shape: (*batch, n_spin, n_grid_points)
    """

    return 0.5 * jnp.einsum("...ab,raj,rbj->...r", rdm1, grad_ao, grad_ao, precision=precision)

@partial(jax.jit, static_argnames=["precision"])
def HF_energy_density(rdm1: Array, ao: Array, chi: Array, precision: Precision = Precision.HIGHEST):
    """Calculate the Hartree-Fock energy density.

    Parameters
    ----------
    rdm1 : Array
        The density matrix.
        Expected shape: (*batch, n_spin, n_orbitals, n_orbitals)
    ao : Array
        Atomic orbitals.
        Expected shape: (n_grid_points, n_orbitals)
    chi : Array
        Precomputed chi density.
        Xc^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψc(r') ψd(r') = Γbd^σ ψb(r) v_{cd}(r)
        where v_{ab}(r) = ∫ dr' f(|r-r'|) ψa(r') ψd(r')
        Expected shape: (*batch, n_grid_points, n_omega, n_spin, n_orbitals)
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.

    Returns
    -------
    Array
        The Hartree-Fock energy density. Shape: (*batch, n_omega, n_spin, n_grid_points).
    """

    #return - jnp.einsum("rwsc,sac,ra->wsr", chi, rdm1, ao, precision=precision)/2
    _hf_energy = lambda _chi, _rdm1, _ao: - jnp.einsum("wsc,sac,a->ws", _chi, _rdm1, _ao, precision=precision)/2
    return vmap(_hf_energy, in_axes=(0, None, 0), out_axes=2)(chi, rdm1, ao)

#@partial(jax.jit, static_argnames=["chunk_size", "precision"])
def HF_density_grad_2_Fock(
    functional: nn.Module, #todo: is this correct, or should we use nn.Module?
    molecule: Molecule,
    feat_wout_hf: Array,
    loc_feat_wout_hf: Array,
    ehf: Array,
    chi: Array, 
    ao: Array,
    params: PyTree,
    *,
    chunk_size: Optional[int] = None,
    precision: Precision = Precision.HIGHEST,
    fxc_kwargs: dict = {}
) -> Array:

    """Calculate the Hartree-Fock matrix contribution due to the partial derivative
    with respect to the Hartree Fock energy energy density.
    F = ∂f(x)/∂e_{HF}^{ωσ} * ∂e_{HF}^{ωσ}/∂Γab^σ = ∂f(x)/∂e_{HF}^{ωσ} * ψa(r) Xb^σ.

    Parameters
    ----------
    fxc : Callable
        FeedForwardFunctional object.
    feat_wout_hf : Array
        Contains a list of the features to input to fxc, except the HF density.
        Expected shape: (n_features, n_grid_points)
    loc_feat_wout_hf : Array
        Contains a list of the features to dot multiply with the output of the functional, except the HF density.
        Expected shape: (n_features, n_grid_points)
    ehf : Array
        The Hartree-Fock energy density. 
        Expected shape: (*batch, n_omega, n_spin, n_grid_points).
    chi : Array
        Xa^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψa(r') ψd(r')
        Expected shape: (*batch, n_grid_points, n_omegas, n_spin, n_orbitals)
    ao : Array
        The orbitals.
        Expected shape: (*batch, n_grid_points, n_orbitals)
    grid_weights : Array
        The weights of the grid points.
        Expected shape: (n_grid_points)
    params : PyTree
        The parameters of the neural network.
    chunk_size : int, optional
        The batch size for the number of atomic orbitals the integral
        evaluation is looped over. For a grid of N points, the solution
        formally requires the construction of a N x N matrix in an intermediate
        step. If `chunk_size` is given, the calculation is broken down into
        smaller subproblems requiring construction of only chunk_size x N matrices.
        Practically, higher `chunk_size`s mean faster calculations with larger
        memory requirements and vice-versa.
    precision : jax.lax.Precision, optional
        Jax `Precision` enum member, indicating desired numerical precision.
        By default jax.lax.Precision.HIGHEST.

    Notes
    -----
    There are other contributions to the Fock matrix from the other features,
    computed separately by automatic differentiation and added later on.
    Also, the chunking is done over the atomic orbitals, not the grid points,
    because the resulting Fock matrix calculation sums over the grid points.

    Returns
    -------
    Array
        The Hartree-Fock matrix contribution due to the partial derivative
        with respect to the Hartree Fock energy density.
        Shape: (*batch, n_omegas, n_spin, n_orbitals, n_orbitals).
    """

    def partial_fxc(params, molecule, x_without_hf, y_without_hf, ehf):

        features, local_features = default_combine_features_hf(ehf, x_without_hf, y_without_hf)

        return functional.energy(params, molecule, features, local_features, **fxc_kwargs)

    gr = grad(partial_fxc, argnums = 4)(params, molecule, feat_wout_hf, loc_feat_wout_hf, ehf)

    @partial(vmap_chunked, in_axes=(0, None, None), chunk_size=chunk_size)
    def chunked_jvp(chi_tensor, gr_tensor, ao_tensor):
        return - jnp.einsum("rws,wsr,ra->wsa", chi_tensor, gr_tensor, ao_tensor, precision=precision)/2.

    return (jax.jit(chunked_jvp)(chi.transpose(3,0,1,2), gr, ao)).transpose(1,2,3,0)

def eig(h,x):
    e0, c0 = eigh2d(h[0],x)
    e1, c1 = eigh2d(h[1],x)
    return jnp.stack((e0, e1), axis = 0), jnp.stack((c0, c1), axis = 0)

######################################################################

def nonXC(
    rdm1: Array, h1e: Array, v_coul: Array, nuclear_repulsion: Scalar, precision = Precision.HIGHEST
) -> Scalar:

    """A function that computes the non-XC part of a DFT functional.

    Parameters
    ----------
    rdm1 : Array
        The 1-Reduced Density Matrix.
        Equivalent to mf.make_rdm1() in pyscf.
        Expected shape: (n_spin, n_orb, n_orb)
    h1e : Array
        The 1-electron Hamiltonian.
        Equivalent to mf.get_hcore(mf.mol) in pyscf.
        Expected shape: (n_orb, n_orb)
    rep_tensor : Array
        The repulsion tensor.
        Equivalent to mf.mol.intor('int2e') in pyscf.
        Expected shape: (n_orb, n_orb, n_orb, n_orb)
    nuclear_repulsion : Scalar
        Equivalent to mf.mol.energy_nuc() in pyscf.
        The nuclear repulsion energy.
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
        The non-XC energy of the DFT functional.
    """

    h1e_energy = one_body_energy(rdm1, h1e, precision)
    coulomb2e_energy = two_body_energy(rdm1, v_coul, precision)

    return nuclear_repulsion + h1e_energy + coulomb2e_energy

def two_body_energy(rdm1, v_coul, precision = Precision.HIGHEST):
    coulomb2e_energy = jnp.einsum('sji,sij->', rdm1, v_coul, precision=precision)/2.
    return coulomb2e_energy

def one_body_energy(rdm1, h1e, precision = Precision.HIGHEST):
    h1e_energy = jnp.einsum("sij,ji->", rdm1, h1e, precision=precision)
    return h1e_energy

def coulomb_potential(rdm1, rep_tensor, precision = Precision.HIGHEST):
    """A function that computes the non-XC part of a DFT functional.

    Parameters
    ----------
    rdm1 : Array
        The 1-Reduced Density Matrix.
        Equivalent to mf.make_rdm1() in pyscf.
        Expected shape: (n_spin, n_orb, n_orb)
    rep_tensor : Array
        The repulsion tensor.
        Equivalent to mf.mol.intor('int2e') in pyscf.
        Expected shape: (n_orb, n_orb, n_orb, n_orb)
    precision : Precision, optional
        The precision to use for the computation, by default Precision.HIGHEST

    Returns
    -------
    Scalar
        Coulomb potential matrix.
    """
    return 2 * jnp.einsum("pqrt,srt->spq", rep_tensor, rdm1, precision=precision)

def HF_exact_exchange(
    chi, rdm1, ao, precision = Precision.HIGHEST
) -> Array:
    
        """A function that computes the exact exchange energy of a DFT functional.

        Parameters
        ----------
        chi : Array
            Xc^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψc(r') ψd(r')
            Expected shape: (n_grid, n_omega, n_spin, n_orbitals)
        rdm1 : Array
            The 1-Reduced Density Matrix.
            Equivalent to mf.make_rdm1() in pyscf.
            Expected shape: (n_spin, n_orb, n_orb)
        ao : Array
            The atomic orbital basis.
            Equivalent to pyscf.dft.numint.eval_ao(mf.mol, grids.coords, deriv=0) in pyscf.
            Expected shape: (n_grid, n_orb)
        precision : Precision, optional
            The precision to use for the computation, by default Precision.HIGHEST
    
				Notes
				-------
				n_omega makes reference to different possible kernels, for example using
				the kernel f(|r-r'|) = erf(w |r-r'|)/|r-r'|.

        Returns
        -------
        Array
            The exact exchange energy of the DFT functional at each point of the grid.
        """

        _hf_energy = lambda _chi, _dm, _ao: - jnp.einsum("wsc,sac,a->ws", _chi, _dm, _ao, precision=precision)/2
        return vmap(_hf_energy, in_axes=(0, None, 0), out_axes=2)(chi, rdm1, ao)


def make_rdm1(mo_coeff, mo_occ):
    '''One-particle density matrix in AO representation

    Args:
        mo_coeff : tuple of 2D ndarrays
            Orbital coefficients for alpha and beta spins. Each column is one orbital.
        mo_occ : tuple of 1D ndarrays
            Occupancies for alpha and beta spins.
    Returns:
        A list of 2D ndarrays for alpha and beta spins

    # Pyscf code
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = jnp.dot(mo_a*mo_occ[0], mo_a.conj().T)
    dm_b = jnp.dot(mo_b*mo_occ[1], mo_b.conj().T)

    return jnp.array((dm_a, dm_b))
    '''

    return jnp.einsum('sij,sj,skj -> sik', mo_coeff, mo_occ, mo_coeff.conj())


def get_occ(mo_energies, nelecs):

    def get_occ_spin(mo_energy, nelec_spin):
        # get the lowest energy indices
        unoccupied_idx = jnp.argsort(mo_energy)[nelec_spin:]
        # get the highest occupied mo_energy value
        vir_mo_energy_min = jnp.min(mo_energy[unoccupied_idx])
        # get the mo_occ
        mo_occ = jnp.where(mo_energy >= vir_mo_energy_min, 0, 1)
        return mo_occ

    mo_occ = jnp.stack([get_occ_spin(mo_energies[s], int(nelecs[s])) for s in range(2)], axis=0)

    return mo_occ


######################################################################

class Reaction(NamedTuple):
    reactants: Sequence[Molecule]
    products: Sequence[Molecule]
    reactant_numbers: Sequence[int]
    product_numbers: Sequence[int]
    energy: float
    name: Optional[str] = None


def make_reaction(
    reactants: Union[Molecule, Sequence[Molecule]],
    products: Union[Molecule, Sequence[Molecule]],
    reactant_numbers: Optional[Sequence[int]] = None,
    product_numbers: Optional[Sequence[int]] = None,
    energy: Optional[float] = None,
    name: Optional[str] = None,
) -> Reaction:

    """Parse inputs and make a `Reaction` object.

    Parameters
    ----------
    reactants : Union[Molecule, Sequence[Molecule]]
        A sequence of `Molecule` objects representing reactants.
    products : Union[Molecule, Sequence[Molecule]]
        A sequence of `Molecule` objects representing products.
    reactant_numbers : Sequence[int], optional
        A sequence of reactant multiplicities for the given reaction. Defaults to 1 for each reactant.
    product_numbers : Sequence[int], optional
        A sequence of product multiplicities for the given reaction. Defaults to 1 for each product.

    Returns
    -------
    Reaction
        A `Reaction` NamedTuple.
    """

    reactants, reactant_numbers = _canonicalize_molecules(reactants, reactant_numbers)
    products, product_numbers = _canonicalize_molecules(products, product_numbers)

    return Reaction(reactants, products, reactant_numbers, product_numbers, energy, name)


def _canonicalize_molecules(
    molecules: Union[Molecule, Sequence[Molecule]], numbers: Optional[Sequence[int]]
) -> Tuple[Sequence[Molecule], Sequence[int]]:

    if isinstance(molecules, Molecule):
        molecules = (molecules,)

    if numbers is None:
        numbers = (1,) * len(molecules)
    elif not isinstance(numbers, Sequence):
        numbers = (numbers,)

    return molecules, numbers



#######################################################################

def orbital_grad(mo_coeff, mo_occ, F):
    '''RHF orbital gradients

    Args:
        mo_coeff: 2D ndarray
            Orbital coefficients
        mo_occ: 1D ndarray
            Orbital occupancy
        F: 2D ndarray
            Fock matrix in AO representation

    Returns:
        Gradients in MO representation.  It's a num_occ*num_vir vector.

    # Similar to pyscf/scf/hf.py:
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(jnp.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()
    '''

    C_occ = jax.vmap(jnp.where, in_axes = (None, 1, None), out_axes=1)(mo_occ > 0, mo_coeff, 0)
    C_vir = jax.vmap(jnp.where, in_axes = (None, 1, None), out_axes=1)(mo_occ == 0, mo_coeff, 0)

    return jnp.einsum("sab,sac,scd->bd", C_vir.conj(), F, C_occ)


def default_molecule_features(
        molecule: Molecule, 
        rho_clip_cte: float = 4.5e-11,
        *_, **__
    ):
    '''
    Computes the electronic density and derivatives

    Parameters
    ----------
    molecule:
        class Molecule
    rho_clip_cte: float
        default 4.5e-11 (chosen carefully, take care if decrease)
    
    Returns
    -------
        Array: shape (n_grid, 7) where 7 is the number of features
    '''

    rho = molecule.density()
    # We need to clip rho away from 0 to obtain good gradients.
    rho = jnp.maximum(abs(rho) , rho_clip_cte)*jnp.sign(rho)
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()

    grad_rho_norm = jnp.sum(grad_rho**2, axis=-1)
    grad_rho_norm_sumspin = jnp.sum(grad_rho.sum(axis=0, keepdims=True) ** 2, axis=-1)

    features = jnp.concatenate((rho, grad_rho_norm_sumspin, grad_rho_norm, tau), axis=0)

    return features.T


def default_functionals(molecule: Molecule, functional_type: Optional[Union[str, Dict[str, int]]] = 'LDA', clip_cte: float = 1e-27, *_, **__):
    '''
    Generates and concatenates different functional levels

    Parameters:
    ----------
    molecule:
        class Molecule

    functional_type:
        Either one of 'LDA', 'GGA', 'MGGA' or Dictionary
        {'u_range': range(), 'w_range': range()} that generates 
        a functional
        $$ \sum_{i\in \text{u_range}} \sum_{j\in \text{w_range}} c_{ij} u^i w^j $$
        where
        $$ x = \frac{|\grad \rho|^{1/2}}{\rho^{4/3}} $$
        $$ u = \frac{\beta x}{1 + \beta x} $$
        and
        $$ t = \frac{3(6\pi^2)^{2/3}}{5}\frac{\rho^{5/3}}{\tau} $$
        $$ w = \frac{\beta t^{-1}}{1+ \beta t^{-1}} $$

    Returns:
        Array: shape (n_grid, n_features)
    '''

    beta = 1/1024.

    if type(functional_type) == str:
        if functional_type == 'LDA' or functional_type == 'DM21':  
            u_range, w_range = range(0,1), range(0,1)
        elif functional_type == 'GGA':
            u_range, w_range = range(0,2), range(0,1)
        elif functional_type == 'MGGA': 
            u_range, w_range = range(0,2), range(0,2)
        else: raise ValueError(f'Functional type {functional_type} not recognized, must be one of LDA, GGA, MGGA.')

    # Molecule preprocessing data
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    tau = molecule.kinetic_density()
    grad_rho_norm = jnp.sum(grad_rho**2, axis=-1)

    # LDA preprocessing data
    log_rho = jnp.log2(jnp.clip(rho, a_min = clip_cte))

    # GGA preprocessing data
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm, a_min = clip_cte))
    log_x_sigma = log_grad_rho_norm/2 - 4/3.*log_rho
    log_u_sigma = jnp.where(jnp.greater(log_rho,jnp.log2(clip_cte)), log_x_sigma - jnp.log2(1 + beta*(2**log_x_sigma)) + jnp.log2(beta), 0)

    # MGGA preprocessing data
    log_tau = jnp.log2(jnp.clip(tau, a_min = clip_cte))
    log_1t_sigma = -(5/3.*log_rho - log_tau + 2/3.*jnp.log2(6*jnp.pi**2) + jnp.log2(3/5.))
    log_w_sigma = jnp.where(jnp.greater(log_rho, jnp.log2(clip_cte)), log_1t_sigma - jnp.log2(1 + beta*(2**log_1t_sigma)) + jnp.log2(beta), 0)

    # Compute the local features
    localfeatures = jnp.empty((0, log_rho.shape[-1]))
    for i, j in itertools.product(u_range, w_range):
        mgga_term = jnp.expand_dims((2**(4/3.*log_rho + i * log_u_sigma + j * log_w_sigma)).sum(axis=0), axis = 0) \
                    * jnp.where(jnp.logical_and(i==0, j==0), -2 * jnp.pi * (3 / (4 * jnp.pi)) ** (4 / 3), 1) # to match DM21
        localfeatures = jnp.concatenate((localfeatures, mgga_term), axis=0)

    return localfeatures.T


def default_features(molecule: Molecule, functional_type: Optional[Union[str, Dict]] = 'LDA', clip_cte: float = 1e-27, *args, **kwargs):
    """
    Generates all features except the HF energy features.
    """
    
    features = default_molecule_features(molecule, *args, **kwargs)
    localfeatures = default_functionals(molecule, functional_type, clip_cte)

    # We return them with the first index being the position r and the second the feature.
    return features, localfeatures

def features_w_hf(molecule: Molecule, features: Callable, functional_type: Optional[Union[str, Dict]] = 'LDA', clip_cte: float = 1e-27, *_, **__):
    """
    Generates all features and the HF energy features.

    Paramters
    ----------
    molecule: Molecule
    features: Callable
        Similar to default_features above, it takes as arguments molecule, functional_type and clip_cte.
    functional_type: Optional[Union[str, Dict]]
        Either a dictionary of ranges, {'u_range': range(...), 'w_range': range(...)}
        or one of "DM21", "LDA", "GGA", "MGGA".
    clip_cte: float
        A small constant to clip and avoid numerical instabilities,
        defaults at 1e-27.
    """

    features, local_features = features(ehf, functional_type, clip_cte)
    ehf = stop_gradient(molecule.HF_energy_density())

    features, local_features = default_combine_features_hf(molecule, features, local_features)

    return features, local_features

def default_combine_features_hf(ehf, features, local_features):

    # Remember that DM concatenates the hf density in the x features by spin...
    features = jnp.concatenate([features, ehf[:,0].T, ehf[:,1].T], axis=1)

    # ... and in the y features by omega.
    local_features = jnp.concatenate([local_features] + [ehf[i].sum(axis=0, keepdims=True).T for i in range(len(ehf))], axis=1)
    return features,local_features

default_features_w_hf = partial(features_w_hf, features = default_features)

##########################################################

