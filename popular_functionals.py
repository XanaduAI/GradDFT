import jax.numpy as jnp
from molecule import Molecule
from utils import Array

from functional import Functional
from jax.lax import Precision, stop_gradient


def b88(rho: Array, grad_rho: Array, clip_cte: float = 1e-27):
    r"""
    B88 exchange functional
    See eq 8 in https://journals.aps.org/pra/pdf/10.1103/PhysRevA.38.3098
    See also https://github.com/ElectronicStructureLibrary/libxc/blob/4bd0e1e36347c6d0a4e378a2c8d891ae43f8c951/maple/gga_exc/gga_x_b88.mpl#L22
    Remember that the exchange energy of spin polarized densities are E_x(n) = ( E_x(2n_\uparrow) +E_x(2n_\downarrow) )/2, see eq 8.4 in Electronic Structure by Richard Martin. 
    """

    beta = 0.0042

    # LDA preprocessing data: Note that we duplicate the density to sum and divide in the last eq.
    log_rho = jnp.log2(jnp.clip(2*rho, a_min = clip_cte))

    grad_rho_norm_sq = jnp.sum((2*grad_rho)**2, axis=-1)

    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min = clip_cte))/2

    # GGA preprocessing data
    log_x_sigma = log_grad_rho_norm - 4/3.*log_rho

    assert not jnp.isnan(log_x_sigma).any() and not jnp.isinf(log_x_sigma).any()

    x_sigma = 2**log_x_sigma
    return jnp.where(jnp.greater(log_rho,jnp.log2(clip_cte)), 
                log_x_sigma - jnp.log2(1 + 6*beta*x_sigma*jnp.arcsinh(2**log_x_sigma)) + jnp.log2(beta), 
                0).sum(axis = 0)/2.

B88 = Functional(b88)


def vwn(rho: Array, clip_cte: float = 1e-27):

    r"""
    VWN correlation functional
    See original paper eq 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also text after eq 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    """

    A =  0.0621814
    b = 3.72744
    c = 12.9352
    x0 = -0.10498

    log_rho = jnp.log2(jnp.clip(2*rho, a_min = clip_cte))
    assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()

    log_x = jnp.log2((3/(4*jnp.pi))**(1/3))/2 - jnp.log2(jnp.clip(rho, a_min = clip_cte).sum(axis = 0))/6.
    x = 2**log_x

    X = 2**(2*log_x) + 2**(log_x+ jnp.log2(b)) + c
    X0 = x0**2 + b*x0 + c
    assert not jnp.isnan(X).any() and not jnp.isinf(X0).any()

    Q = jnp.sqrt(4*c-b**2)

    return A/2 * ( 2*jnp.log(x)-jnp.log(X) + 2*b/Q * jnp.arctan(Q/(2*x+b)) - b*x0/X0 *
                (jnp.log((x-x0)**2/X) + 2*(2*x0+b)/Q * jnp.arctan(Q/(2*x+b))) )

VWN = Functional(vwn)


def lyp(rho: Array, grad_rho: Array, grad2rho: Array, clip_cte = 1e-27):

    r"""
    LYP correlation functional
    See eq 22 in
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.37.785
    """

    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    CF = (3/10)*(3*jnp.pi**2)**(2/3)

    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    log_rho = jnp.log2(jnp.clip(rho, a_min = clip_cte))
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min = clip_cte))/2

    t = 1/8 * ( 2**(2*log_grad_rho_norm-log_rho) - grad2rho)
    assert not jnp.isnan(t).any() and not jnp.isinf(t).any()

    frac = jnp.where(rho.sum(axis=0) > clip_cte, 2**(jnp.log2((rho**2).sum(axis =0)) - 2*jnp.log2(rho.sum(axis =0))), 1)
    gamma = 2 * (1-frac)

    rhos_ts = jnp.where(jnp.logical_and(t.sum(axis=0) > clip_cte, rho.sum(axis=0) > clip_cte), 
                    2**(jnp.log2(rho.sum(axis=0))+jnp.log2(t.sum(axis=0))), 0)

    assert not jnp.isnan(rhos_ts).any() and not jnp.isinf(rhos_ts).any()

    rho_t = jnp.where(jnp.logical_and(rho.sum(axis=0) > clip_cte, t.sum(axis=0) > clip_cte), 
                    (2**(jnp.log2(rho)+jnp.log2(t))).sum(axis=0), 0)

    assert not jnp.isnan(rho_t).any() and not jnp.isinf(rho_t).any()

    rho_grad2rho = jnp.where(jnp.logical_and(rho.sum(axis=0) > clip_cte, grad2rho.sum(axis=0) > clip_cte), 
                    (2**(jnp.log2(rho)+jnp.log2(grad2rho))).sum(axis=0), 0)

    assert not jnp.isnan(rho_grad2rho).any() and not jnp.isinf(rho_grad2rho).any()

    exp_factor = jnp.where(rho.sum(axis=0)**(-1/3) > clip_cte, jnp.exp(-c*rho.sum(axis=0)**(-1/3)), 1)

    rhom1_3 = 2**(-jnp.log2(rho.sum(axis=0))/3.)
    rhom5_3 = 2**(-5*jnp.log2(rho.sum(axis=0))/3.)
    rho8_3 = (2**(8*jnp.log2(rho)/3)).sum(axis=0)
    return -a * gamma/(1+d*rhom1_3)* (rho.sum(axis=0) + jnp.where(exp_factor > clip_cte, 2*b*rhom5_3*
        (CF*2**(2/3)*(rho8_3) - rhos_ts + rho_t/9 + rho_grad2rho/18)* exp_factor, 0))

LYP = Functional(lyp)

def b3lyp_exhf_features(molecule: Molecule, functional_type: str = 'GGA', clip_cte: float = 1e-27):

    r"""
    See eq 2 in
    https://pubs.acs.org/doi/pdf/10.1021/j100096a001
    """

    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()

    lda_e = -2 * jnp.pi * (3 / (4 * jnp.pi)) ** (4 / 3) * (rho**(4/3)).sum(axis = 0)
    b88_e = b88(rho, grad_rho, clip_cte)
    assert not jnp.isnan(b88_e).any() and not jnp.isinf(b88_e).any()
    vwn_e = vwn(rho, clip_cte)
    assert not jnp.isnan(vwn_e).any() and not jnp.isinf(vwn_e).any()
    lyp_e = lyp(rho, grad_rho, grad2rho, clip_cte)
    assert not jnp.isnan(lyp_e).any() and not jnp.isinf(lyp_e).any()

    return [jnp.stack((lda_e, b88_e, vwn_e, lyp_e), axis = 1)]

def b3lyp_combine(ehf, features):

    ehf = jnp.sum(ehf, axis = (0,1))
    return [jnp.concatenate([features, jnp.expand_dims(ehf, axis = 1)], axis=1)]

def b3lyp(features):
    r"""
    The dot product between the features and the weights
    """
    a0=0.2
    ax=0.72
    ac=0.81
    weights = jnp.array([1-a0, ax, ac, 1-ac, a0])

    return jnp.einsum('rf,f->r',features, weights)

B3LYP = Functional(b3lyp)


