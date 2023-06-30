from random import shuffle
from typing import List, Optional, Union, Sequence, Dict
from itertools import chain, combinations_with_replacement
import os

import numpy as np
from jax import numpy as jnp
from jax.lax import Precision
from jax import vmap

from pyscf import scf  # type: ignore
from pyscf.dft import Grids, numint  # type: ignore
from pyscf.gto import Mole
import pyscf.data.elements as elements

#from qdft.reaction import Reaction, make_reaction, get_grad
from molecule import Grid, Molecule, Reaction, make_reaction
from utils import DType, default_dtype
from jax.tree_util import tree_map

import h5py
from pyscf import cc, dft, scf

from utils import Array, Scalar, DensityFunctional, HartreeFock
from external import _nu_chunk

def grid_from_pyscf(grids: Grids, dtype: Optional[DType] = None) -> Grid:

    if grids.coords is None:
        grids.build()

    coords, weights = to_device_arrays(grids.coords, grids.weights, dtype=dtype)

    return Grid(coords, weights)


def molecule_from_pyscf(
    mf: DensityFunctional, dtype: Optional[DType] = None,
    omegas: Optional[List] = [], energy: Optional[Scalar] = None,
    name: Optional[str] = None, training: bool = False, scf_iteration: int = 50,
    chunk_size: Optional[int] = 1024,
    grad_order: Optional[int] = 2
) -> Molecule:

    #mf, grids = _maybe_run_kernel(mf, grids)
    grid = grid_from_pyscf(mf.grids, dtype=dtype)

    ao, grad_ao, grad_n_ao, rdm1, energy_nuc, h1e, vj, mo_coeff, mo_energy, mo_occ, mf_e_tot, s1e, fock, rep_tensor = to_device_arrays(
        *_package_outputs(mf, mf.grids, training, scf_iteration, grad_order), dtype=dtype
    )

    atom_index, nuclear_pos = to_device_arrays(
        [elements.ELEMENTS.index(e) for e in mf.mol.elements], mf.mol.atom_coords(unit='angstrom'), dtype=dtype
    )

    basis = jnp.array([ord(char) for char in mf.mol.basis]) # jax doesn't support strings
    unit_Angstrom = True
    if name: name = jnp.array([ord(char) for char in name])

    if omegas is not None: 
        chi = generate_chi_tensor(rdm1 = rdm1, ao = ao, grid_coords = grid.coords, mol = mf.mol, 
                                omegas = omegas, chunk_size = chunk_size)
        #chi = to_device_arrays(chi, dtype=dtype)
        #omegas = to_device_arrays(omegas, dtype=dtype)
    else:
        chi = None
    spin = mf.mol.spin
    charge = mf.mol.charge

    grid_level = mf.grids.level

    return Molecule(
        grid, atom_index, nuclear_pos, ao, grad_ao, grad_n_ao, rdm1, energy_nuc, h1e, vj, mo_coeff, mo_occ, mo_energy,
        mf_e_tot, s1e, omegas, chi, rep_tensor, energy, basis, name, spin, charge, unit_Angstrom, grid_level, scf_iteration, fock
    )

def mol_from_Molecule(molecule: Molecule):
    r"""Converts a Molecule object to a PySCF Mole object.
    WARNING: the mol returned is not the same as the orginal mol used to create the Molecule object.
    """
    
    mol = Mole()

    charges = np.asarray(molecule.atom_index)
    positions = np.asarray(molecule.nuclear_pos)

    mol.atom = [[int(charge), pos] for charge, pos in zip(charges, positions)]
    mol.basis = ''.join(chr(num) for num in molecule.basis) # The basis will generally be encoded as a jax array of ints
    mol.unit = 'angstrom' if molecule.unit_Angstrom else 'bohr'

    mol.spin = int(molecule.spin)
    mol.charge = int(molecule.charge)

    mol.build()

    return mol

#@partial(jax.jit, static_argnames=["kernel_fn", "chunk_size", "precision"])
def saver(
    fname: str,
    omegas: Union[Scalar, Sequence[Scalar]],
    reactions: Optional[Union[Reaction, Sequence[Reaction]]] = (),
    molecules: Optional[Union[Molecule, Sequence[Molecule]]] = (),
    training: bool = True,
    *,
    chunk_size: Optional[int] = None,
):

    r"""
    Saves the molecule data to a file, and computes and saves the corresponding chi

    Parameters
    ----------
    fname : str
        Name of the file to save the chi object to.
    omegas : Union[Scalar, Sequence[Scalar]], optional
        Range-separation parameter. A value of 0 disables range-separation
        (i.e. uses the kernel v(r,r') = 1/|r-r'| instead of
        v(r,r') = erf(\omega |r-r'|) / |r-r'|)
        If multiple omegas are given, the chi object is calculated for each omega
        and concatenated along the last axis.
    reactions : Union[Reaction, Sequence[Reaction]]
        Reaction object(s) to calculate the chi object for.
    molecules : Union[Molecule, Sequence[Molecule]]
        Molecule object(s) to calculate the chi object for.
    training : bool, optional
        If True, we avoid saving a few objects
    chunk_size : int, optional
        The batch size for the number of lattice points the integral
        evaluation is looped over. For a grid of N points, the solution
        formally requires the construction of a N x N matrix in an intermediate
        step. If `chunk_size` is given, the calculation is broken down into
        smaller subproblems requiring construction of only chunk_size x N matrices.
        Practically, higher `chunk_size`s mean faster calculations with larger
        memory requirements and vice-versa.

    Notes
    -----
    chi: Array
        $$\chi_{bd}(r) = \Gamma_{ac} \psi_a \int dr' (\chi_b(r') v(r, r') \chi_d(r'))$$,
        used to compute chi_a objects in equation S4 in DM21 paper, and save it to a file.
        Uses the extenal _nu_chunk function from the original DM21 paper, in the _hf_density.py file.
        chi will have dimensions (n_grid_points, n_omegas, n_spin, n_orbitals)

    nu: Array
        The density matrix, with dimensions (n_grid_points, n_spin, n_orbitals, n_orbitals)
        $$nu = \int dr' (\chi_b(r') v(r, r') \chi_d(r'))$$

    Saves
    -------
    In hdf5 format, the following datasets are saved:
        |- Reaction (attributes: energy)
            |- Molecule  (attributes: reactant/product, reactant/product_numbers)
                |- All the attributes in Molecule class
                |- chi (attributes: omegas)
        |- Molecule (attributes: energy)
            |- All the attributes in Molecule class
            |- chi (attributes: omegas)


    Raises:
    -------
        ValueError: if omega is negative.
        TypeError: if molecules is not a Molecule or Sequence of Molecules; if reactions is not a Reaction or Sequence of Reactions.
    """

    #######

    fname = fname.replace(".hdf5", "").replace(".h5", "")

    if isinstance(molecules, Molecule):
        molecules = (molecules,)
    if isinstance(reactions, Reaction):
        reactions = (reactions,)
    if isinstance(omegas, (int, float)):
        omegas = (omegas,)

    with h5py.File(f"{fname}.hdf5", "a") as file:

        # First we save the reactions
        for i, reaction in enumerate(reactions):

            if reaction.name: react = file.create_group(f"reaction_{reaction.name}_{i}")
            else: react = file.create_group(f"reaction_{i}")
            react["energy"] = reaction.energy

            for j, molecule in enumerate(list(chain(reaction.reactants, reaction.products))):

                if molecule.name: mol_group = react.create_group(f"molecule_{molecule.name}_{j}")
                else: mol_group = react.create_group(f"molecule_{j}")
                if len(omegas) > 1:
                    save_molecule_chi(molecule, omegas, chunk_size, mol_group)
                else:
                    mol_group.create_dataset(f"chi", data = jnp.empty((1)))
                    mol_group.create_dataset(f"omegas", data = omegas)
                save_molecule_data(mol_group, molecule, training)
                if j<len(reaction.reactants):
                    mol_group.attrs["type"] = "reactant"
                    mol_group["reactant_numbers"] = reaction.reactant_numbers[j]
                else:
                    mol_group.attrs["type"] = "product"
                    mol_group["product_numbers"] = reaction.product_numbers[j-len(reaction.reactant_numbers)]

        # Then we save the molecules
        for j, molecule in enumerate(molecules):

            if molecule.name: mol_group = file.create_group(f"molecule_{molecule.name}_{j}")
            else: mol_group = file.create_group(f"molecule_{j}")
            if len(omegas) > 1:
                save_molecule_chi(molecule, omegas, chunk_size, mol_group)
            else:
                mol_group.create_dataset(f"chi", data = jnp.empty((1)))
                mol_group.create_dataset(f"omegas", data = omegas)
            save_molecule_data(mol_group, molecule, training)

def loader(fpath: str, randomize: Optional[bool] = False, training: Optional[bool] = True, config_omegas: Optional[Union[Scalar, Sequence[Scalar]]] = None):
    r"""
    Reads the molecule, energy and precomputed chi matrix from a file.

    Parameters
    ----------
    fname : str
        Name of the file to read the fxx matrix from.
    key : PRNGKeyArray
        Key to use for randomization of the order of elements output.
    randomize : bool, optional
        Whether to randomize the order of elements output, by default False
    training : bool, optional
        Whether we are training or not, by default True
    omegas : Union[Scalar, Sequence[Scalar]], optional
        Range-separation parameter. Use to select the chi matrix to load, by default None

    Yields
    -------
    type: str
        Whether it is a Molecule or Reaction.
    molecule/reaction : Molecule or Reaction
        The molecule or reaction object.

    todo: randomize input
    """

    fpath = fpath.replace(".hdf5", "").replace(".h5", "")

    with h5py.File(os.path.normpath(f"{fpath}.hdf5"), "r") as file:

        items=list(file.items()) # List of tuples
        if randomize and training: shuffle(items)

        for grp_name, group in items:

            if "molecule" in grp_name:

                args = {}
                if not training: args["name"] = grp_name.split("_")[1]
                omegas = list(group["omegas"])
                for key, value in group.items():
                    if key in ["omegas"]:
                        args[key] = list(value)
                    elif key in ["energy"]:
                        args[key] = jnp.float32(value) if training else jnp.float64(value)
                    elif key in ["s1e", "mf_energy", "rep_tensor"]:  
                        if not training: args[key] = jnp.asarray(value)
                    elif key in ["scf_iteration", "spin", "charge"]:
                        args[key] = jnp.int32(value)
                    elif key == 'chi':
                        # select the indices from the omegas array and load the corresponding chi matrix
                        if config_omegas is None: args[key] = jnp.asarray(value)
                        elif config_omegas == []: args[key] = None
                        else:
                            if isinstance(omegas, (int, float)): omegas = (omegas,)
                            assert all([omega in omegas for omega in config_omegas]), f"chi tensors for omega list {config_omegas} were not all precomputed in the molecule"
                            indices = [omegas.index(omega) for omega in config_omegas]
                            args[key] = jnp.stack([jnp.asarray(value)[:, i] for i in indices], axis=1)
                    else: 
                        args[key] = jnp.asarray(value)

                for key, value in group.attrs.items():
                    if not training:
                        args[key] = str(value)

                grid = Grid(args["coords"], args["weights"])
                del args["coords"], args["weights"]

                molecule = Molecule(grid, **args)

                yield 'molecule', molecule

            if "reaction" in grp_name:

                reactants = []
                products = []
                reactant_numbers = []
                product_numbers = []

                
                if not training:
                    name = grp_name.split("_")[1]
                    energy = jnp.float64(group["energy"])
                else:
                    name = None
                    energy = jnp.float32(group["energy"])

                for molecule_name, molecule in group.items():

                    if molecule_name == 'energy': continue

                    args = {}
                    if not training: args["name"] = molecule_name.split("_")[1]
                    molecule_omegas = list(molecule["omegas"])
                    for key, value in molecule.items():
                        if key in ["energy", "reactant_numbers", "product_numbers"]:
                            continue
                        if key in ["omegas"]:
                            args[key] = list(value)
                        elif key in ["energy"]:
                            args[key] = jnp.float32(value) if training else jnp.float64(value)
                        elif key in ["s1e", "mf_energy"]:  
                            if not training: args[key] = jnp.asarray(value)
                        elif key == 'chi':
                            # select the indices from the omegas array and load the corresponding chi matrix
                            if config_omegas is None: args[key] = jnp.asarray(value)
                            elif config_omegas == []: args[key] = None
                            else:
                                if isinstance(omegas, (int, float)): omegas = (omegas,)
                                assert all([omega in omegas for omega in config_omegas]), f"chi tensors for omega list {config_omegas} were not all precomputed in the molecule"
                                indices = [omegas.index(omega) for omega in config_omegas]
                                args[key] = jnp.stack([jnp.asarray(value)[:, i] for i in indices], axis=1)
                        else: 
                            args[key] = jnp.asarray(value)

                    for key, value in molecule.attrs.items():
                        if not training and key not in ["type"]:
                            args[key] = value

                    grid = Grid(args["coords"], args["weights"])
                    del args["coords"], args["weights"]
                    args.pop("reactant_numbers", None)
                    args.pop("product_numbers", None)

                    if molecule.attrs["type"] == "reactant":
                        reactants.append(Molecule(grid, **args))
                        reactant_numbers.append(jnp.int32(molecule["reactant_numbers"]))
                    else:
                        products.append(Molecule(grid, **args))
                        product_numbers.append(jnp.int32(molecule["product_numbers"]))

                reaction = make_reaction(reactants, products, reactant_numbers, product_numbers, energy, name)

                yield 'reaction', reaction

def save_molecule_data(mol_group: h5py.Group, molecule: Molecule, training: bool = True):
    r"""Auxiliary function to save all data except for chi"""

    to_numpy = lambda arr: arr if (isinstance(arr, str) or isinstance(arr, float)) else np.asarray(arr)
    d = tree_map(to_numpy, molecule.to_dict())

    for name, data in d.items():

        if name in ("chi", "omegas", "basis", "name") or data is None: continue
        elif training and name in ("s1e", "mf_energy"): continue
        elif not training and name in ("training_gorb_grad"): continue
        #elif name in ("nuclear_pos"): mol_group.create_dataset(name, data=data/2, dtype='float32')
        else: mol_group.create_dataset(name, data=data, dtype='float64')

    if not training: 
        mol_group.attrs["basis"] = d["basis"]
        if d["name"] is not None:
            mol_group.attrs["name"] = d["name"]

def save_molecule_chi(molecule: Molecule, omegas: Union[Sequence[Scalar], Scalar], chunk_size: int,
                    mol_group: h5py.Group, precision: Precision = Precision.HIGHEST):
    r"""Auxiliary function to save chi tensor"""

    grid_coords = molecule.grid.coords
    mol = mol_from_Molecule(molecule)

    chunks = (np.ceil(grid_coords.shape[0] / chunk_size).astype(int),1,1,1) if chunk_size else None
    shape = (grid_coords.shape[0], len(omegas), molecule.rdm1.shape[0], molecule.ao.shape[1])
    # Remember that molecule.rdm1.shape[0] represents the spin
    
    if chunk_size is None:
        chunk_size = grid_coords.shape[0]

    chi = generate_chi_tensor(molecule.rdm1, molecule.ao, molecule.grid.coords, mol, omegas = omegas, precision = precision)

    mol_group.create_dataset(f"chi", shape = shape, chunks = chunks, dtype = 'float64', data = chi)
    mol_group.create_dataset(f"omegas", data = omegas)

##############################################################################################################


def to_device_arrays(*arrays, dtype: Optional[DType] = None):

    if dtype is None:
        dtype = default_dtype()

    out = []
    for array in arrays:
        if isinstance(array, dict):
            for k, v in array.items():
                array[k] = jnp.asarray(v)
            out.append(array)
        else:
            out.append(jnp.asarray(array))

    return out


def _maybe_run_kernel(mf: HartreeFock, grids: Optional[Grids] = None):

    if mf.mo_coeff is None:

        # kernel not run yet

        if hasattr(mf, "grids"):  # Is probably DFT

            if grids is not None:
                mf.grids = grids
            elif mf.grids is not None:
                grids = mf.grids
            else:
                raise RuntimeError(
                    "A `Grids` object has to be provided either through `mf` or explicitly!"
                )

        mf.verbose = 0
        mf.kernel()

    return mf, grids


def ao_grads(mol: Mole, coords: Array, order = 2) -> Dict:
    r"""Function to compute nth order atomic orbital grads, for n > 1.
    
    ::math::
            \nabla^n \psi

    Parameters
    ----------
    mf: PySCF Density Functional object.

    Outputs
    ----------
    Dict
    For each order n > 1, result[n] is an array of shape
    (n_grid, n_ao, 3) where the third coordinate indicates 
    ::math::
        \frac{\partial^n \psi}{\partial x_i^n}
    
    for :math:`x_i` is one of the usual cartesian coordinates x, y or z.
    """

    ao_ = numint.eval_ao(mol, coords, deriv=order)
    if order == 0:
        return ao_[0]
    result = {}
    i = 4
    for n in range(2, order+1):
        result[n] = jnp.empty((ao_[0].shape[0], ao_[0].shape[1], 0))
        for c in combinations_with_replacement('xyz', r = n):
            if len(set(c)) == 1:
                result[n] = jnp.concatenate((result[n], jnp.expand_dims(ao_[i], axis = 2)), axis = 2)
            i += 1
    return result


def _package_outputs(mf: DensityFunctional, grids: Optional[Grids] = None, training: bool = False, scf_iteration: int = 50, grad_order: int = 2):

    ao_ = numint.eval_ao(mf.mol, grids.coords, deriv=1)#, non0tab=grids.non0tab)
    if scf_iteration != 0:
        rdm1 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    else:
        rdm1 = mf.get_init_guess(mf.mol, mf.init_guess)

    s1e = mf.get_ovlp(mf.mol)
    h1e = mf.get_hcore(mf.mol)

    if rdm1.ndim == 2:  # Restricted HF

        half_dm = rdm1 / 2
        half_mo_coeff = mf.mo_coeff
        half_mo_energy = mf.mo_energy
        half_mo_occ = mf.mo_occ / 2

        rdm1 = np.stack([half_dm, half_dm], axis=0)
        mo_coeff = np.stack([half_mo_coeff, half_mo_coeff], axis=0)
        mo_energy = np.stack([half_mo_energy, half_mo_energy], axis=0)
        mo_occ = np.stack([half_mo_occ, half_mo_occ], axis=0)

        # Warning: this is for closed shell systems only.

    elif rdm1.ndim == 3:  # Unrestricted HF
        mo_coeff = np.stack(mf.mo_coeff, axis=0)
        mo_energy = np.stack(mf.mo_energy, axis=0)
        mo_occ = np.stack(mf.mo_occ, axis=0)
    else:
        raise RuntimeError(f"Invalid density matrix shape. Got {rdm1.shape} for AO shape {ao.shape}")

    ao = ao_[0]
    grad_ao = ao_[1:4].transpose(1, 2, 0)

    #grad_grad_ao = compute_grad2_ao(ao_)
    #grad_grad_ao = reshape_grad2ao(ao_.transpose(1,2,0))

    grad_n_ao = ao_grads(mf.mol, mf.grids.coords, order = grad_order)

    #h1e_energy = np.einsum("sij,ji->", dm, h1e)
    vj = 2 * mf.get_j(mf.mol, rdm1, hermi = 1) # The 2 is to compensate for the /2 in the definition of the density matrix 

    if not training:
        rep_tensor = mf.mol.intor('int2e')
    else:
        rep_tensor = None
    # v_j = jnp.einsum("pqrt,srt->spq", rep_tensor, dm)
    # v_k = jnp.einsum("ptqr,srt->spq", rep_tensor, dm)

    #coulomb2e_energy = np.einsum("sij,sji->", dm, vj)/2

    mf_e_tot = mf.e_tot
    fock = np.stack([h1e,h1e], axis=0) + mf.get_veff(mf.mol, rdm1)

    energy_nuc = mf.energy_nuc()

    return ao, grad_ao, grad_n_ao, rdm1, energy_nuc, h1e, vj, mo_coeff, mo_energy, mo_occ, mf_e_tot, s1e, fock, rep_tensor

##############################################################################################################

def process_mol(mol, compute_energy=True, grid_level: int = 2, training: bool = False, max_cycle: Optional[int] = None, xc_functional = 'b3lyp'):
    if compute_energy:
        if mol.multiplicity == 1: mf2 = scf.RHF(mol)
        else: mf2 = scf.UHF(mol)
        mf2.kernel()
        mycc = cc.CCSD(mf2).run()
        energy = mycc.e_tot
    else: energy = None
    if mol.multiplicity == 1: mf = dft.RKS(mol)
    else: mf = dft.UKS(mol)
    mf.grids.level = int(grid_level)
    mf.grids.build() # with_non0tab=True
    if training: 
        mf.xc = xc_functional
    if max_cycle is not None:
        mf.max_cycle = max_cycle
    elif not training: 
        mf.max_cycle = 0
    mf.kernel()

    return energy,mf

def generate_chi_tensor(rdm1, ao, grid_coords, mol, omegas, chunk_size = 1024, precision = Precision.HIGHEST):
    
    r"""
    Generates the chi tensor, according to the molecular data and omegas provided.

    Parameters
    ----------
    rdm1: Array
        The molecular reduced density matrix Γbd^σ
        Expected shape: (n_spin, n_orbitals, n_orbitals)

    ao: Array
        The atomic orbitals ψa(r)
        Expected shape: (n_grid_points, n_orbitals)

    grid_coords: Array
        The coordinates of the grid.
        Expected shape: (n_grid_points)

    mol: A Pyscf mol object.

    omegas: List



    Returns
    ----------
    chi : Array
        Xa^σ = Γbd^σ ψb(r) ∫ dr' f(|r-r'|) ψa(r') ψd(r')
        Expected shape: (n_grid_points, n_omegas, n_spin, n_orbitals)
    """

    def chi_make(dm_, ao_, nu):
        return jnp.einsum("...bd,b,da->...a", dm_, ao_, nu, precision=precision)

    chi = []
    for omega in omegas:
        chi_omega = []
        for chunk_index, end_index, nu_chunk in _nu_chunk(mol,grid_coords,omega,chunk_size):
            chi_chunk = vmap(chi_make, in_axes = (None, 0,0), out_axes = 0)(rdm1, ao[chunk_index:end_index], nu_chunk)
            chi_omega.append(chi_chunk)
        chi_omega = jnp.concatenate(chi_omega, axis = 0)
        chi.append(chi_omega)
    if chi:
        return jnp.stack(chi, axis = 1)
    else: return jnp.array(chi)