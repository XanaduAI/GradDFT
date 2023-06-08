from jax import grad
import jax.numpy as jnp
from molecule import Molecule
from utils import Array

from functional import Functional
from jax.lax import Precision, stop_gradient


def b88_x_e(rho: Array, grad_rho: Array, clip_cte: float = 1e-27):
    r"""
    B88 exchange functional
    See eq 8 in https://journals.aps.org/pra/pdf/10.1103/PhysRevA.38.3098
    See also https://github.com/ElectronicStructureLibrary/libxc/blob/4bd0e1e36347c6d0a4e378a2c8d891ae43f8c951/maple/gga_exc/gga_x_b88.mpl#L22
    """

    beta = 0.0042

    rho = jnp.clip(rho, a_min = clip_cte)
    zeta = (rho[0] - rho[1])/ rho.sum(axis = 0)

    # LDA preprocessing data: Note that we duplicate the density to sum and divide in the last eq.
    log_rho = jnp.log2(jnp.clip(rho, a_min = clip_cte))

    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min = clip_cte))/2

    # GGA preprocessing data
    log_x_sigma = log_grad_rho_norm - 4/3.*log_rho

    assert not jnp.isnan(log_x_sigma).any() and not jnp.isinf(log_x_sigma).any()

    x_sigma = 2**log_x_sigma
    
    # Eq 2.78 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    b88_e = - (beta*2**(4*log_rho/3 + 2*log_x_sigma - 
                jnp.log2(1 + 6*beta*x_sigma*jnp.arcsinh(x_sigma)))).sum(axis = 0)

    #def fzeta(z): return ((1-z)**(4/3) + (1+z)**(4/3) - 2) / (2*(2**(1/3) - 1))
    # Eq 2.71 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    #b88_e = b88_es[0] + (b88_es[1]-b88_es[0])*fzeta(zeta)

    return b88_e


def vwn_c_e(rho: Array, clip_cte: float = 1e-27):

    r"""
    VWN correlation functional
    See original paper eq 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also text after eq 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    """

    A = jnp.array([[0.0621814],
                    [0.0621814/2]])
    b = jnp.array([[3.72744], 
                    [7.06042]])
    c = jnp.array([[12.9352],
                    [18.0578]])
    x0 = jnp.array([[-0.10498], 
                    [-0.325]])

    rho = jnp.where(rho > clip_cte, rho, 0.)
    log_rho = jnp.log2(jnp.clip(rho.sum(axis = 0), a_min = clip_cte))
    assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()
    log_rs = jnp.log2((3/(4*jnp.pi))**(1/3)) - log_rho/3.
    log_x = log_rs / 2
    rs = 2**log_rs
    x = 2**log_x

    X = 2**(2*log_x) + 2**(log_x+ jnp.log2(b)) + c
    X0 = x0**2 + b*x0 + c
    assert not jnp.isnan(X).any() and not jnp.isinf(X0).any()

    Q = jnp.sqrt(4*c-b**2)

    # check eq with https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/vwn.mpl
    e_PF = A/2 * ( 2*jnp.log(x)-jnp.log(X) + 2*b/Q * jnp.arctan(Q/(2*x+b)) - b*x0/X0 *
                (jnp.log((x-x0)**2/X) + 2*(2*x0+b)/Q * jnp.arctan(Q/(2*x+b))) )
    
    # Spin polarization using eq 2.75 from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    
    e_tilde_PF = jnp.einsum('sr,r->sr', e_PF, rho.sum(axis = 0))

    zeta = jnp.where(rho.sum(axis = 0) > clip_cte, (rho[0] - rho[1]) / (rho.sum(axis = 0)), 0)
    def fzeta(z): return ((1-z)**(4/3) + (1+z)**(4/3) - 2) / (2*(2**(1/3) - 1))

    A_ = 0.016887
    alpha1 = 0.11125
    beta1 = 10.357
    beta2 = 3.6231
    beta3 = 0.88026
    beta4 = 0.49671
    alphac = 2*A_*(1+2*alpha1*rs)*jnp.log(1+(1/(2*A_))/(beta1*jnp.sqrt(rs) + beta2*rs + beta3*rs**(3/2) + beta4*rs**2)) #, 2*A_)
    assert not jnp.isnan(alphac).any() and not jnp.isinf(alphac).any()

    e_tilde = e_tilde_PF[0] + alphac*(fzeta(zeta)/(grad(grad(fzeta))(0.)))*(1-zeta**4) + (e_tilde_PF[1] - e_tilde_PF[0])*fzeta(zeta)*zeta**4
    assert not jnp.isnan(e_tilde).any() and not jnp.isinf(e_tilde).any()

    return e_tilde # We have to integrate e_tilde = e * n as per eq 2.1 in original article

VWN = Functional(vwn_c_e)

def lyp_c_e(rho: Array, grad_rho: Array, grad2rho: Array, clip_cte = 1e-27):

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

    rho = jnp.where(rho > clip_cte, rho, 0)
    grad_rho = jnp.where(abs(grad_rho) > clip_cte, grad_rho, 0)

    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    t = (jnp.where(rho > clip_cte, grad_rho_norm_sq/rho, 0) - grad2rho)/8.
    assert not jnp.isnan(t).any() and not jnp.isinf(t).any()

    frac = jnp.where(rho.sum(axis=0) > clip_cte, 
                    ((rho**2).sum(axis =0))/(rho.sum(axis =0))**2, 1)
    gamma = 2 * (1-frac)

    rhos_ts = rho.sum(axis = 0) * t.sum(axis = 0)
    assert not jnp.isnan(rhos_ts).any() and not jnp.isinf(rhos_ts).any()

    rho_t = (rho*t).sum(axis = 0)
    assert not jnp.isnan(rho_t).any() and not jnp.isinf(rho_t).any()

    rho_grad2rho = (rho*grad2rho).sum(axis = 0)
    assert not jnp.isnan(rho_grad2rho).any() and not jnp.isinf(rho_grad2rho).any()

    exp_factor = jnp.exp(-c*rho.sum(axis=0)**(-1/3))

    rhom1_3 = 2**(-jnp.log2(rho.sum(axis=0))/3.)
    rhom5_3 = 2**(-5*jnp.log2(rho.sum(axis=0))/3.)
    rho8_3 = (2**(8*jnp.log2(rho)/3.)).sum(axis=0)
    return - a * gamma/(1+d*rhom1_3) * (rho.sum(axis=0) + jnp.where(exp_factor > clip_cte, 2*b*rhom5_3*
        (2**(2/3)*CF*(rho8_3) - rhos_ts + rho_t/9 + rho_grad2rho/18)* exp_factor, 0))

LYP = Functional(lyp_c_e)

def lsda_x_e(rho, clip_cte):
    # Eq 2.72 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    rho = jnp.clip(rho, a_min = clip_cte)
    lda_es = -3./4. * (jnp.array([[3.],[6.]]) / jnp.pi) ** (1 / 3) * (rho.sum(axis = 0))**(4/3)
    zeta = (rho[0] - rho[1])/ rho.sum(axis = 0)
    def fzeta(z): return ((1-z)**(4/3) + (1+z)**(4/3) - 2) / (2*(2**(1/3) - 1))
    # Eq 2.71 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    lda_e = lda_es[0] + (lda_es[1]-lda_es[0])*fzeta(zeta)

    return lda_e

def b3lyp_exhf_features(molecule: Molecule, functional_type: str = 'GGA', clip_cte: float = 1e-27):

    r"""
    See eq 2 in
    https://pubs.acs.org/doi/pdf/10.1021/j100096a001
    """

    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()

    lda_e = lsda_x_e(rho, clip_cte)
    assert not jnp.isnan(lda_e).any() and not jnp.isinf(lda_e).any()
    b88_e = b88_x_e(rho, grad_rho, clip_cte)
    assert not jnp.isnan(b88_e).any() and not jnp.isinf(b88_e).any()
    vwn_e = vwn_c_e(rho, clip_cte)
    assert not jnp.isnan(vwn_e).any() and not jnp.isinf(vwn_e).any()
    lyp_e = lyp_c_e(rho, grad_rho, grad2rho, clip_cte)
    assert not jnp.isnan(lyp_e).any() and not jnp.isinf(lyp_e).any()

    return [jnp.stack((lda_e, b88_e, vwn_e, lyp_e), axis = 1)]

def b88_features(molecule: Molecule, functional_type: str = 'GGA', clip_cte: float = 1e-27):
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    b88_e = b88_x_e(rho, grad_rho, clip_cte)
    lda_e = lsda_x_e(rho, clip_cte)
    assert not jnp.isnan(b88_e).any() and not jnp.isinf(b88_e).any()
    return [jnp.stack((lda_e, b88_e), axis = 1)]

def lsda_features(molecule: Molecule, functional_type: str = 'LDA', clip_cte: float = 1e-27):
    rho = molecule.density()
    lda_e = lsda_x_e(rho, clip_cte)
    return [jnp.expand_dims(lda_e, axis = 1)]

def lyp_features(molecule: Molecule, functional_type: str = 'MGGA', clip_cte: float = 1e-27):
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()
    lyp_e = lyp_c_e(rho, grad_rho, grad2rho, clip_cte)
    return [jnp.expand_dims(lyp_e, axis = 1)]

def vwn_features(molecule: Molecule, functional_type: str = 'LDA', clip_cte: float = 1e-27):
    rho = molecule.density()
    lyp_e = vwn_c_e(rho, clip_cte)
    return [jnp.expand_dims(lyp_e, axis = 1)]

def b88_combine(features):
    return [features]

def lsda_combine(features):
    return [features]

def vwn_combine(features):
    return [features]

def lyp_combine(features):
    return [features]

def b3lyp_combine(ehf, features):

    ehf = jnp.sum(ehf, axis = (0,1))
    return [jnp.concatenate([features, jnp.expand_dims(ehf, axis = 1)], axis=1)]

def b3lyp(instance, features):
    r"""
    The dot product between the features and the weights
    """
    a0=0.2
    ax=0.72
    ac=0.81
    weights = jnp.array([1-a0, ax, 1-ac, ac, a0])

    return jnp.einsum('rf,f->r',features, weights)
def b88(instance, x): return jnp.einsum('ri->r',x)
def lsda(instance, x): return jnp.einsum('ri->r',x)
def lyp(instance, x): return jnp.einsum('ri->r',x)
def vwn(instance, x): return jnp.einsum('ri->r',x)

B88 = Functional(b88)
LSDA = Functional(lsda)
B3LYP = Functional(b3lyp)
LYP = Functional(lyp)
VWN = Functional(vwn)


