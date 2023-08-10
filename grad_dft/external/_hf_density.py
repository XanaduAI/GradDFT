from typing import Generator, Tuple, NamedTuple, Optional, Union

import jax
from jax import numpy as jnp

from pyscf.gto.mole import Mole  # type: ignore
from pyscf.lib import logger  # type: ignore
from pyscf.dft import numint  # type: ignore

from chex import Array

# This code is mostly adapted from
# https://github.com/deepmind/deepmind-research/blob/
# master/density_functional_approximation_dm21/
# density_functional_approximation_dm21/compute_hfx_density.py


def _evaluate_nu_slow(mol: Mole, coords: Array, omega: float, hermi: int) -> Array:

    """Computes nu integrals for given coordinates using a slow loop."""

    nu = []

    # Use the Gaussian nuclear model in int1e_rinv_sph to evaluate the screened
    # integrals.

    with mol.with_rinv_zeta(zeta=omega**2):
        for coord in coords:
            with mol.with_rinv_origin(coord):
                nu.append(mol.intor("int1e_rinv_sph", hermi=hermi))

    return jnp.stack(nu, axis=0)


def _evaluate_nu_fast(mol: Mole, coords: Array, omega: float, hermi: int) -> Array:

    """Computes nu integrals for given coordinates using a fast loop within PySCF."""

    with mol.with_range_coulomb(omega=omega):
        # grids keyword argument supported in pyscf >= 2.0.0.

        nu = mol.intor(
            "int1e_grids_sph", hermi=hermi, grids=coords
        )  # pytype: disable=wrong-keyword-args

    return jnp.asarray(nu)


def _evaluate_nu(mol: Mole, coords: Array, omega: float, hermi: bool = True) -> Array:

    """Computes nu integrals for given coordinates.
    \nu_{bd}(r) = \int dr' (\chi_b(r') v(r, r') \chi_d(r')), for \chi the basis functions
    """

    hermi = int(hermi)

    try:
        nu = _evaluate_nu_fast(mol, coords, omega, hermi=hermi)

    except TypeError:

        logger.info(
            mol,
            "Support for int1e_grids not found (requires libcint 4.4.1 and "
            "pyscf 2.0.0a or later. Falling back to slow loop over "
            "individual grid points.",
        )

        nu = _evaluate_nu_slow(mol, coords, omega, hermi=hermi)

    return nu


def _nu_chunk(
    mol: Mole, coords: Array, omega: float, chunk_size: int = 1000
) -> Generator[Tuple[int, int, Array], None, None]:

    """Yields chunks of nu integrals over the grid.
    Args:
      mol: pyscf Mole object.
      coords: coordinates, r', at which to evaluate the nu integrals, shape (N,3).
      omega: range separation parameter. A value of 0 disables range-separation
        (i.e. uses the kernel v(r,r') = 1/|r-r'| instead of
        v(r,r') = erf(\omega |r-r'|) / |r-r'|)
      chunk_size: number of coordinates to evaluate the integrals at a time.
    Yields:
      start_index, end_index, nu_{ab}(r) where
        start_index, end_index are indices into coords,
        nu is an array of shape (end_index-start_index, nao, nao), where nao is
        the number of atomic orbitals and contains
        nu_{ab}(r) = <a(r')|v(r,r')| b(r')>, where a,b are atomic
        orbitals and r' are the grid coordinates in coords[start_index:end_index].
    Raises:
      ValueError: if omega is negative.
    """

    if omega < 0:
        raise ValueError("Range-separated parameter omega must be non-negative!")

    ncoords = len(coords)

    for chunk_index in range(0, ncoords, chunk_size):

        end_index = min(chunk_index + chunk_size, ncoords)
        coords_chunk = coords[chunk_index:end_index]

        nu_chunk = _evaluate_nu(mol, coords_chunk, omega=omega)

        yield chunk_index, end_index, nu_chunk


@jax.jit
def _compute_exx_block(nu: Array, e: Array) -> Tuple[Array, Array]:

    """Computes exx and fxx.
    Args:
      nu: batch of <i|v(r,r_k)|j> integrals, in format (k,i,j) where r_k is the
        position of the k-th grid point, i and j label atomic orbitals.
      e: density matrix in the AO basis at each grid point.
    Returns:
      exx and fxx, where
      fxx_{gb} =\sum_c nu_{gbc} e_{gc} and
      exx_{g} = -0.5 \sum_b e_{gb} fxx_{gb}.
    """

    fxx = jnp.einsum("gbc,gc->gb", nu, e)
    exx = -0.5 * jnp.einsum("gb,gb->g", e, fxx)

    return exx, fxx


@jax.jit
def _compute_jk_block(
    nu: Array, fxx: Array, dm: Array, ao_value: Array, weights: Array
) -> Tuple[Array, Array]:

    """Computes J and K contributions from the given block of nu integrals."""

    batch_size = nu.shape[0]

    vj = jnp.dot(nu.reshape(batch_size, -1), dm.reshape(-1, 1))
    vj = jnp.squeeze(vj)
    vj_ao = jnp.einsum("g,gb->gb", vj * weights, ao_value)

    j = jnp.dot(ao_value.T, vj_ao)

    w_ao = jnp.einsum("g,gb->gb", weights, ao_value)
    k = jnp.dot(fxx.T, w_ao)

    return j, k


class HFDensityResult(NamedTuple):

    r"""Container for results returned by get_hf_density.
    Note that the kernel used in all integrals is defined by the omega input
    argument.
    Attributes:
      exx: exchange energy density at position r on the grid for the alpha, beta
        spin channels.  Each array is shape (N), where N is the number of grid
        points.
      fxx: intermediate for evaluating dexx/dD^{\sigma}_{ab}, where D is the
        density matrix and \sigma is the spin coordinate. See top-level docstring
        for details.  Each array is shape (N, nao), where nao is the number of
        atomic orbitals.
      coulomb: coulomb matrix (restricted calculations) or matrices (unrestricted
        calculations). Each array is shape (nao, nao).
        Restricted calculations: \sum_{} D_{cd} (ab|cd)
        Unrestricted calculations: \sum_{} D^{\sigma}_{cd} (ab|cd)
      exchange: exchange matrix (restricted calculations) or matrices
        (unrestricted calculations). Each array is shape (nao, nao).
        Restricted calculations: \sum_{} D_{cd} (ab|cd)
        Unrestricted calculations: \sum_{} D^{\sigma}_{cd} (ac|bd).
    """

    exx: Tuple[Array, Array]
    fxx: Optional[Tuple[Array, Array]] = None
    coulomb: Optional[Union[Array, Tuple[Array, Array]]] = None
    exchange: Optional[Union[Array, Tuple[Array, Array]]] = None


################################################################################


def hf_density(
    mol: Mole,
    dm: Union[Tuple[Array, Array], Array],
    coords: Array,
    omega: float = 0.0,
    deriv: int = 0,
    ao: Optional[Array] = None,
    chunk_size: int = 1000,
    weights: Optional[Array] = None,
) -> HFDensityResult:

    r"""Computes the (range-separated) HF energy density.
    Args:
      mol: PySCF molecule.
      dm: The density matrix. For restricted calculations, an array of shape
        (M, M), where M is the number of atomic orbitals. For unrestricted
        calculations, either an array of shape (2, M, M) or a tuple of arrays,
        each of shape (M, M), where dm[0] is the density matrix for the alpha
        electrons and dm[1] the density matrix for the beta electrons.
      coords: The coordinates to compute the HF density at, shape (N, 3), where N
        is the number of grid points.
      omega: The inverse width of the error function. An omega of 0. means range
        separation and a 1/|r-R| kernel is used in the nu integrals. Otherwise,
        the kernel erf(\omega|r-R|)/|r-R|) is used. Must be non-negative.
      deriv: The derivative order. Only first derivatives (deriv=1) are currently
        implemented. deriv=0 indicates no derivatives are required.
      ao: The atomic orbitals evaluated on the grid, shape (N, M). These are
        computed if not supplied.
      chunk_size: The number of coordinates to compute the HF density for at once.
        Reducing this saves memory since we don't have to keep as many Nus (nbasis
        x nbasis) in memory at once.
      weights: weight of each grid point, shape (N). If present, the Coulomb and
        exchange matrices are also computed semi-numerically, otherwise only the
        HF density and (if deriv=1) its first derivative are computed.
    Returns:
      HFDensityResult object with the HF density (exx), the derivative of the HF
      density with respect to the density (fxx) if deriv is 1, and the Coulomb and
      exchange matrices if the weights argument is provided.
    Raises:
      NotImplementedError: if a Cartesian basis set is used or if deriv is greater
      than 1.
      ValueError: if omega or deriv are negative.
    """

    if mol.cart:

        raise NotImplementedError(
            "Local HF exchange is not implmented for basis sets with Cartesian functions!"
        )

    if deriv < 0:
        raise ValueError(f"`deriv` must be non-negative, got {deriv}")

    if omega < 0:
        raise ValueError(f"`omega` must be non-negative, got {omega}")

    if deriv > 1:
        raise NotImplementedError("Higher order derivatives are not implemented.")

    if ao is None:
        ao = numint.eval_ao(mol, coords, deriv=0)

    logger.info(mol, "Computing contracted density matrix ...")

    if isinstance(dm, tuple) or dm.ndim == 3:
        dma, dmb = dm
        restricted = False
    else:
        dma = dm / 2
        dmb = dma
        restricted = True

    ea = jnp.dot(ao, dma)
    eb = jnp.dot(ao, dmb)

    exxa = []
    exxb = []
    fxxa = []
    fxxb = []

    ja = jnp.zeros_like(dma)
    jb = jnp.zeros_like(dmb)
    ka = jnp.zeros_like(dma)
    kb = jnp.zeros_like(dmb)

    for start, end, nu in _nu_chunk(mol, coords, omega, chunk_size=chunk_size):

        logger.info(mol, "Computing exx %s / %s ...", end, len(ea))
        exxa_block, fxxa_block = _compute_exx_block(nu, ea[start:end])
        exxa.extend(exxa_block)

        if not restricted:
            exxb_block, fxxb_block = _compute_exx_block(nu, eb[start:end])
            exxb.extend(exxb_block)

        if deriv == 1:
            fxxa.extend(fxxa_block)
            if not restricted:
                fxxb.extend(fxxb_block)

        if weights is not None:

            ja_block, ka_block = _compute_jk_block(
                nu, fxxa_block, dma, ao[start:end], weights[start:end]
            )

            ja += ja_block
            ka += ka_block

            if not restricted:

                jb_block, kb_block = _compute_jk_block(
                    nu, fxxb_block, dmb, ao[start:end], weights[start:end]
                )

                jb += jb_block
                kb += kb_block

    exxa = jnp.stack(exxa, axis=0)

    if fxxa:
        fxxa = jnp.stack(fxxa, axis=0)

    if restricted:
        exxb = exxa
        fxxb = fxxa
    else:
        exxb = jnp.stack(exxb, axis=0)
        fxxb = jnp.stack(fxxb, axis=0)

    if weights is not None:

        return HFDensityResult(
            exx=(exxa, exxb),
            fxx=(fxxa, fxxb) if deriv == 1 else None,
            coulomb=2 * ja if restricted else (ja, jb),
            exchange=2 * ka if restricted else (ka, kb),
        )

    else:
        return HFDensityResult(exx=(exxa, exxb), fxx=(fxxa, fxxb) if deriv == 1 else None)
