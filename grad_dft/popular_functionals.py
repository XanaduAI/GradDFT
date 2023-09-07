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

import jax.numpy as jnp
from grad_dft.molecule import Molecule
from grad_dft.utils import Array
from typing import Dict, List
from flax import linen as nn

from grad_dft.functional import (
    Functional,
    correlation_polarization_correction,
    exchange_polarization_correction,
)


def lsda_x_e(rho, clip_cte):
    # Eq 2.72 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    rho = jnp.clip(rho, a_min=clip_cte)
    lda_es = (
        -3.0
        / 4.0
        * (jnp.array([[3.0, 6.0]]) / jnp.pi) ** (1 / 3)
        * (rho.sum(axis=1, keepdims=True)) ** (4 / 3)
    )
    lda_e = exchange_polarization_correction(lda_es, rho)

    return lda_e


def b88_x_e(rho: Array, grad_rho: Array, clip_cte: float = 1e-27):
    r"""
    B88 exchange functional
    See eq 8 in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.38.3098
    See also https://github.com/ElectronicStructureLibrary/libxc/blob/4bd0e1e36347c6d0a4e378a2c8d891ae43f8c951/maple/gga_exc/gga_x_b88.mpl#L22
    """

    beta = 0.0042

    rho = jnp.clip(rho, a_min=clip_cte)

    # LDA preprocessing data: Note that we duplicate the density to sum and divide in the last eq.
    log_rho = jnp.log2(jnp.clip(rho, a_min=clip_cte))

    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min=clip_cte)) / 2

    # GGA preprocessing data
    log_x_sigma = log_grad_rho_norm - 4 / 3.0 * log_rho

    # assert not jnp.isnan(log_x_sigma).any() and not jnp.isinf(log_x_sigma).any()

    x_sigma = 2**log_x_sigma

    # Eq 2.78 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    b88_e = -(
        beta
        * 2
        ** (
            4 * log_rho / 3
            + 2 * log_x_sigma
            - jnp.log2(1 + 6 * beta * x_sigma * jnp.arcsinh(x_sigma))
        )
    ).sum(axis=1)

    # def fzeta(z): return ((1-z)**(4/3) + (1+z)**(4/3) - 2) / (2*(2**(1/3) - 1))
    # Eq 2.71 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    # b88_e = b88_es[0] + (b88_es[1]-b88_es[0])*fzeta(zeta)

    return b88_e


def pw92_c_e(rho: Array, clip_cte: float = 1e-27):
    r"""
    Eq 10 in
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.13244
    """

    A_ = jnp.array([[0.031091, 0.015545]])
    alpha1 = jnp.array([[0.21370, 0.20548]])
    beta1 = jnp.array([[7.5957, 14.1189]])
    beta2 = jnp.array([[3.5876, 6.1977]])
    beta3 = jnp.array([[1.6382, 3.3662]])
    beta4 = jnp.array([[0.49294, 0.62517]])

    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1, keepdims=True), a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))

    e_PF = -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))

    e_tilde = correlation_polarization_correction(e_PF, rho, clip_cte)

    return e_tilde


def vwn_c_e(rho: Array, clip_cte: float = 1e-27):
    r"""
    VWN correlation functional
    See original paper eq 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also text after eq 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    """

    A = jnp.array([[0.0621814, 0.0621814 / 2]])
    b = jnp.array([[3.72744, 7.06042]])
    c = jnp.array([[12.9352, 18.0578]])
    x0 = jnp.array([[-0.10498, -0.325]])

    rho = jnp.where(rho > clip_cte, rho, 0.0)
    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1, keepdims=True), a_min=clip_cte))
    # assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0
    log_x = log_rs / 2
    rs = 2**log_rs
    x = 2**log_x

    X = 2 ** (2 * log_x) + 2 ** (log_x + jnp.log2(b)) + c
    X0 = x0**2 + b * x0 + c
    # assert not jnp.isnan(X).any() and not jnp.isinf(X0).any()

    Q = jnp.sqrt(4 * c - b**2)

    # check eq with https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/vwn.mpl
    e_PF = (
        A
        / 2
        * (
            2 * jnp.log(x)
            - jnp.log(X)
            + 2 * b / Q * jnp.arctan(Q / (2 * x + b))
            - b
            * x0
            / X0
            * (jnp.log((x - x0) ** 2 / X) + 2 * (2 * x0 + b) / Q * jnp.arctan(Q / (2 * x + b)))
        )
    )

    e_tilde = correlation_polarization_correction(e_PF, rho, clip_cte)

    # We have to integrate e_tilde = e * n as per eq 2.1 in original LYP article
    return e_tilde


def lyp_c_e(rho: Array, grad_rho: Array, grad2rho: Array, clip_cte=1e-27):
    r"""
    LYP correlation functional
    See eq 22 in
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.37.785
    """

    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    CF = (3 / 10) * (3 * jnp.pi**2) ** (2 / 3)

    rho = jnp.clip(rho, a_min=clip_cte)

    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    t = (jnp.where(rho > clip_cte, grad_rho_norm_sq / rho, 0) - grad2rho) / 8.0
    # assert not jnp.isnan(t).any() and not jnp.isinf(t).any()

    frac = jnp.where(
        rho.sum(axis=1) > clip_cte, ((rho**2).sum(axis=1)) / (rho.sum(axis=1)) ** 2, 1
    )
    gamma = 2 * (1 - frac)

    rhos_ts = rho.sum(axis=1) * t.sum(axis=1)
    # assert not jnp.isnan(rhos_ts).any() and not jnp.isinf(rhos_ts).any()

    rho_t = (rho * t).sum(axis=1)
    # assert not jnp.isnan(rho_t).any() and not jnp.isinf(rho_t).any()

    rho_grad2rho = (rho * grad2rho).sum(axis=1)
    # assert not jnp.isnan(rho_grad2rho).any() and not jnp.isinf(rho_grad2rho).any()

    exp_factor = jnp.where(rho.sum(axis=1) > 0, jnp.exp(-c * rho.sum(axis=1) ** (-1 / 3)), 0)
    # assert not jnp.isnan(exp_factor).any() and not jnp.isinf(exp_factor).any()

    rhom1_3 = (rho.sum(axis=1)) ** (-1 / 3.0)
    rho8_3 = (rho ** (8 / 3.0)).sum(axis=1)
    rhom5_3 = (rho.sum(axis=1)) ** (-5 / 3)

    par = 2 ** (2 / 3) * CF * (rho8_3) - rhos_ts + rho_t / 9 + rho_grad2rho / 18

    sum_ = jnp.where(rho.sum(axis=1) > clip_cte, 2 * b * rhom5_3 * par * exp_factor, 0.0)

    return -a * jnp.where(
        rho.sum(axis=1) > clip_cte, gamma / (1 + d * rhom1_3) * (rho.sum(axis=1) + sum_), 0.0
    )


def lsda_density(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    rho = molecule.density()
    lda_e = lsda_x_e(rho, clip_cte)
    return jnp.expand_dims(lda_e, axis=1)


def b88_density(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of B88 functional."""
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    b88_e = b88_x_e(rho, grad_rho, clip_cte)
    lda_e = lsda_x_e(rho, clip_cte)
    # assert not jnp.isnan(b88_e).any() and not jnp.isinf(b88_e).any()
    return jnp.stack((lda_e, b88_e), axis=1)


def vwn_density(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of VWN functional."""
    rho = molecule.density()
    vwn_e = vwn_c_e(rho, clip_cte)
    return jnp.expand_dims(vwn_e, axis=1)


def pw92_densities(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of PW92 functional."""
    rho = molecule.density()
    pw92_e = pw92_c_e(rho, clip_cte)
    return jnp.expand_dims(pw92_e, axis=1)


def lyp_density(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""Auxiliary function to generate the features of LYP functional."""
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()
    lyp_e = lyp_c_e(rho, grad_rho, grad2rho, clip_cte)
    return jnp.expand_dims(lyp_e, axis=1)


def b3lyp_exhf_densities(molecule: Molecule, clip_cte: float = 1e-27, *_, **__):
    r"""
    Auxiliary function to generate the non Hartree-Fock features of B3LYP functional
    See eq 2 in
    https://pubs.acs.org/doi/pdf/10.1021/j100096a001
    """

    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()

    lda_e = lsda_x_e(rho, clip_cte)
    # assert not jnp.isnan(lda_e).any() and not jnp.isinf(lda_e).any()
    b88_e = b88_x_e(rho, grad_rho, clip_cte)
    # assert not jnp.isnan(b88_e).any() and not jnp.isinf(b88_e).any()
    vwn_e = vwn_c_e(rho, clip_cte)
    # assert not jnp.isnan(vwn_e).any() and not jnp.isinf(vwn_e).any()
    lyp_e = lyp_c_e(rho, grad_rho, grad2rho, clip_cte)
    # assert not jnp.isnan(lyp_e).any() and not jnp.isinf(lyp_e).any()

    return jnp.stack((lda_e, b88_e, vwn_e, lyp_e), axis=1)


def b3lyp_combine(features, ehf):
    ehf = jnp.sum(ehf, axis=(0, 1))
    ehf = jnp.expand_dims(ehf, axis=1)
    return jnp.concatenate([features, ehf], axis=1)


def b3lyp_coefficients(instance, *args):
    r"""
    The dot product between the features and the weights in B3LYP.
    """
    a0 = 0.2
    ax = 0.72
    ac = 0.81
    return jnp.array([[1 - a0, ax, 1 - ac, ac, a0]])


def b3lyp_nograd_densities(molecule, *_, **__):
    ehf = molecule.HF_energy_density([0.0])
    # assert not jnp.isnan(ehf).any() and not jnp.isinf(ehf).any()
    return ehf


def b3lyp_hfgrads(
    functional: nn.Module,
    params: Dict,
    molecule: Molecule,
    ehf: Array,
    cinputs: Array,
    grad_densities: Array,
    omegas=jnp.array([0.0]),
):
    vxc_hf = molecule.HF_density_grad_2_Fock(
        functional, params, omegas, ehf, cinputs, grad_densities
    )
    return vxc_hf.sum(axis=0)  # Sum over omega


LSDA = Functional(coefficients=lambda self, *_: jnp.array([[1.0]]), energy_densities=lsda_density)
B88 = Functional(coefficients=lambda self, *_: jnp.array([[1.0]]), energy_densities=b88_density)
VWN = Functional(coefficients=lambda self, *_: jnp.array([[1.0]]), energy_densities=vwn_density)
LYP = Functional(
    coefficients=lambda self, *_: jnp.array([[1.0]]),
    energy_densities=lyp_density,
    exchange_mask=jnp.array([]),
)

B3LYP = Functional(
    coefficients=b3lyp_coefficients,
    energy_densities=b3lyp_exhf_densities,
    nograd_densities=b3lyp_nograd_densities,
    densitygrads=b3lyp_hfgrads,
    combine_densities=b3lyp_combine,
    exchange_mask=jnp.array([1, 1, 0, 0, 1]),
)

PW92 = Functional(coefficients=lambda self, *_: jnp.array([[1.0]]), energy_densities=pw92_densities)
