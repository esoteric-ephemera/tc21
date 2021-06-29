import numpy as np
import multiprocessing
from os.path import isfile

from settings import pi,nproc
from dft.alda import alda,lda_derivs
from utilities.integrators import nquad
from utilities.gauss_quad import gauss_quad
from utilities.roots import bisect,bracket
from utilities.special_funcs import erf
from utilities.interpolators import natural_spline

"""
    Zhixin Qian and Giovanni Vignale,
    ``Dynamical exchange-correlation potentials for an electron liquid'',
    Phys. Rev. B 65, 235121 (2002).
    https://doi.org/10.1103/PhysRevB.65.235121
"""

# [Gamma(1/4)]**2
gamma_14_sq = 13.1450472065968728685447786119766533374786376953125

def density_variables(rs):
    dv = {}
    dv['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dv['rs'] = rs
    dv['n'] = 3.0/(4.0*pi*dv['rs']**3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/rs**3)**(0.5)
    return dv

def mu_xc_per_n(rs,a,b,c):
    return a/rs + (b-a)*rs/(rs**2 + c)

def fit_mu_xc():
    # Data from Table 1 of QV, mu_xc in units of 2 omega_p(0) n
    mul = np.asarray([[1, 0.0073], [2, 0.0077], [3, 0.00801], [4, 0.00837], [5, 0.00851]])
    # convert to mu_xc/n
    mul[:,1] *= 2*(3/mul[:,0]**3)**(0.5)
    from scipy.optimize import curve_fit
    opts,_ = curve_fit(mu_xc_per_n,mul[:,0], mul[:,1])
    for opt in opts:
        print(opt)
    return

def s_3_l(kf):
    """ Eq. 16 of Qian and Vignale """
    lam = (pi*kf)**(0.5) # lambda = 2 k_F/ k_TF
    s3l = 5 - (lam + 5/lam)*np.arctan(lam) - 2/lam*np.arcsin(lam/(1+lam**2)**(0.5))
    s3l += 2/(lam*(2+lam**2)**(0.5))*(pi/2 - np.arctan(1/(lam*(2+lam**2)**(0.5))))
    return -s3l/(45*pi)

def high_freq(dv):
    """
     from Iwamato and Gross, Phys. Rev. B 35, 3003 (1987),
     f(q,omega=infinity) = -4/5 n^(2/3)*d/dn[ eps_xc/n^(2/3)] + 6 n^(1/3) + d/dn[ eps_xc/n^(1/3)]
     eps_xc is XC energy per electron
    """
    # exchange contribution is -1/5 [3/(pi*n^2)]^(1/3)
    finf_x = -1.0/(5.0)*(3.0/(pi*dv['n']**2))**(1.0/3.0)
    # correlation contribution is -[22*eps_c + 26*rs*(d eps_c / d rs)]/(15*n)
    eps_c,d_eps_c_d_rs = lda_derivs(dv,param='PW92')
    finf_c = -(22.0*eps_c + 26.0*dv['rs']*d_eps_c_d_rs)/(15.0*dv['n'])
    fxc_inf = finf_x + finf_c
    return fxc_inf

def qv_static(dv,use_mu_xc=True):
    fxc_0 = alda(dv,x_only=False,param='PW92')

    if use_mu_xc:
        mu_xc_n = mu_xc_per_n(dv['rs'],0.03115158529677855,0.011985054514894128,2.267455018224077)
        fxc_0 += 4/3*mu_xc_n/dv['n']
    return fxc_0

def get_qv_pars(dv,use_mu_xc=True):

    if hasattr(dv['rs'],'__len__'):
        nrs = len(dv['rs'])
        a3l = np.zeros(nrs)
        b3l = np.zeros(nrs)
        g3l = np.zeros(nrs)
        o3l = np.zeros(nrs)
        for irs,ars in enumerate(dv['rs']):
            a3l[irs],b3l[irs],g3l[irs],o3l[irs] = get_qv_pars_single(density_variables(ars),use_mu_xc=use_mu_xc)
    else:
        a3l,b3l,g3l,o3l = get_qv_pars_single(dv,use_mu_xc=use_mu_xc)
    return a3l,b3l,g3l,o3l

def get_qv_pars_single(dv,use_mu_xc=True):

    c3l = 23/15 # just below Eq. 13

    s3l = s_3_l(dv['kF'])
    """     Eq. 28   """
    a3l = 2*(2/(3*pi**2))**(1/3)*dv['rs']**2*s3l
    """     Eq. 29   """
    b3l = 16*(2**10/(3*pi**8))**(1/15)*dv['rs']*(s3l/c3l)**(4/5)

    fxc_0 = alda(dv,x_only=False,param='PW92')

    fxc_inf = high_freq(dv)

    # approx. expression for mu_xc that interpolates metallic data, from Table 1 of QV
    # fitting code located in fit_mu_xc
    if use_mu_xc:
        mu_xc_n = mu_xc_per_n(dv['rs'],0.03115158529677855,0.011985054514894128,2.267455018224077)
        fxc_0 += 4/3*mu_xc_n/dv['n']

    df = fxc_0-fxc_inf

    def solve_g3l(tmp):

        """    Eq. 27    """
        o3l = 1 - 1.5*tmp
        o3l2 = o3l**2
        """    Eq. 30    """
        res = 4*(2*pi/b3l)**(0.5)*a3l/gamma_14_sq
        res += o3l*tmp*np.exp(-o3l2/tmp)/pi + 0.5*(tmp/pi)**(0.5)*(tmp + 2*o3l2)*(1 + erf(o3l/tmp**(0.5)))
        res *= 4*(pi/dv['n'])**(0.5) # 2*omega_p(0)/n
        return df + res
    #import matplotlib.pyplot as plt
    #tmpl = np.linspace(1.e-6,20.0,5000)
    #plt.plot(tmpl,solve_g3l(tmpl))
    #plt.show()
    poss_brack = bracket(solve_g3l,(1.e-6,3.0),nstep=500,vector=True)
    g3l = 1.e-14
    for tbrack in poss_brack:
        tg3l,success = bisect(solve_g3l,tbrack,tol=1.5e-7,maxstep=200)
        if success < 1.5e-7:
            g3l = max(tg3l,g3l)
    o3l = 1 - 1.5*g3l
    return a3l,b3l,g3l,o3l

def im_fxc_longitudinal(omega,dv,pars=()):

    if len(pars)==0:
        a3,b3,g3,om3 = get_qv_pars(dv)
    else:
        a3,b3,g3,om3 = pars

    wp0 = (3/dv['rs']**3)**(0.5)
    wt = omega/(2*wp0)

    imfxc = a3/(1 + b3*wt**2)**(5/4)
    imfxc += wt**2*np.exp(-(np.abs(wt)-om3)**2/g3)
    imfxc *= -omega/dv['n']

    return imfxc

def wrap_kram_kron(to,omega,dv):
    return im_fxc_longitudinal(to,dv)/(to - omega)

def kram_kron(omega,dv):
    return nquad(wrap_kram_kron,('-inf','inf'),'global_adap',{'itgr':'GK','prec':1.e-6,'npts':5,'min_recur':4,'max_recur':1000,'n_extrap':400,'inf_cond':'fun'},pars_ops={'PV':[omega]},args=(omega,dv))

def fxc_longitudinal(dv,omega):

    im_fxc = im_fxc_longitudinal(omega,dv)
    finf=high_freq(dv)
    if hasattr(omega,'__len__'):
        re_fxc = np.zeros(omega.shape)
        for iom,om in enumerate(omega):
            re_fxc[iom],terr = kram_kron(om,dv)
            if terr['code'] == 0:
                print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(om,terr['error']))
    else:
        re_fxc,terr = kram_kron(omega,dv)
        if terr['code'] == 0:
            print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(omega,terr['error']))
    return re_fxc/pi + finf + 1.j*im_fxc


def fxc_longitudinal_multi_proc(dv,omega):

    im_fxc = im_fxc_longitudinal(omega,dv)
    finf=high_freq(dv)
    fxc = np.zeros(omega.shape)
    re_fxc = np.zeros(omega.shape)

    pool = multiprocessing.Pool(processes=min(nproc,len(omega)))
    trefxc = pool.starmap(kram_kron,[(om,dv) for om in omega])
    pool.close()

    for iom,om in enumerate(omega):
        re_fxc[iom],terr = trefxc[iom]
        if terr['code'] == 0:
            print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(om,terr['error']))
    return re_fxc/pi + finf + 1.j*im_fxc

def make_grid(def_pts=200,cut_pt=2.0):

    gfile = './grids/gauss_legendre_{:}_pts.csv'
    if isfile(gfile):
        wg,grid = np.transpose(np.genfromtxt(gfile,delimiter=',',skip_header=1))
    else:
        wg,grid = gauss_quad(def_pts,grid_type='legendre')

    # first step, shift grid and weights from (-1, 1) to (0, 1)
    sgrid = 0.5*(grid + 1)
    swg = 0.5*wg

    """
        ogrid integrates from 0 to cut_pt, then extrapolates from cut_pt to infinity using
        int_0^inf f(x) dx = int_0^x_c f(x) dx + int_0^(1/x_c) f(1/u)/u**2 du,
        with u = 1/x. Then x_c = cut_pt, which by default = 2.
    """
    ogrid = cut_pt*sgrid
    owg = cut_pt*swg
    oextrap = sgrid/cut_pt
    owg_extrap = swg/(cut_pt*oextrap**2)
    ogrid = np.concatenate((ogrid,1/oextrap))
    owg = np.concatenate((owg,owg_extrap))

    return ogrid,owg

def fxc_longitudinal_fixed_grid(omega,dv,grid=[],wg=[]):

    if len(grid) > 0 and len(wg) > 0:
        inf_grid = grid
        inf_wg = wg
    else:
        wpuscl = (3/dv['brs']**3)**(0.5)
        inf_grid,inf_wg = make_grid(def_pts=200,cut_pt=50*wpuscl)

    qvpars = get_qv_pars(dv)
    im_qv = im_fxc_longitudinal(omega,dv,pars=qvpars)
    finf = high_freq(dv)

    w,igrid = np.meshgrid(omega,inf_grid)
    lgrid = -igrid + w - 1.e-10
    im_qv_l = im_fxc_longitudinal(lgrid,dv,pars=qvpars)

    ugrid = w + 1.e-10 + igrid
    im_qv_u = im_fxc_longitudinal(ugrid,dv,pars=qvpars)
    intgd = im_qv_l/(lgrid - w) + im_qv_u/(ugrid - w)

    re_qv = np.einsum('i,ik->k',inf_wg,intgd)
    fxc_qv = re_qv/pi + finf + 1.j*im_qv

    return fxc_qv

def fxc_qv_ifreq_fixed_grid(omega,dv,grid=[],wg=[]):

    if len(grid) > 0 and len(wg) > 0:
        inf_grid = grid
        inf_wg = wg
    else:
        wpuscl = (3/dv['brs']**3)**(0.5)
        inf_grid,inf_wg = make_grid(def_pts=200,cut_pt=50*wpuscl)

    dinf_grid = np.concatenate((inf_grid,-inf_grid))
    dinf_wg = np.concatenate((inf_wg,inf_wg))

    finf = high_freq(dv)

    if hasattr(omega,'__len__'):

        fxc_iu = np.zeros(omega.shape,dtype='complex')

        #if not hasattr(dv['rs'],'__len__'):
        #    fxc_tmp = fxc_longitudinal_fixed_grid(dinf_grid,dv,grid=inf_grid,wg=inf_wg)

        for iw,w in enumerate(omega):
            #if hasattr(dv['rs'],'__len__'):
            fxc_tmp = fxc_longitudinal_fixed_grid(dinf_grid,density_variables(dv['rs'][iw]),grid=inf_grid,wg=inf_wg)
            rintd = (w*(fxc_tmp.real-finf[iw]) + dinf_grid*fxc_tmp.imag)/(dinf_grid**2 + w**2)
            iintd = (-dinf_grid*(fxc_tmp.real-finf[iw]) + w*fxc_tmp.imag)/(dinf_grid**2 + w**2)
            fxc_iu[iw] = np.sum(dinf_wg*(rintd + 1.j*iintd))
            #fxc_iu.real[iw] = np.sum(dinf_wg*rintd)
            #fxc_iu.imag[iw] = np.sum(dinf_wg*iintd)
    else:

        fxc_tmp = fxc_longitudinal_fixed_grid(dinf_grid,dv,grid=inf_grid,wg=inf_wg)
        rintd = (omega*(fxc_tmp.real-finf) + dinf_grid*fxc_tmp.imag)/(dinf_grid**2 + omega**2)
        iintd = (-dinf_grid*(fxc_tmp.real-finf) + omega*fxc_tmp.imag)/(dinf_grid**2 + omega**2)
        fxc_iu = np.sum(dinf_wg*(rintd + 1.j*iintd))

    return finf + fxc_iu/(2*pi)

def qv_fixed_grid(omega,dv,axis='real',ugrid=[],uwg=[]):
    if axis == 'real':
        return fxc_longitudinal_fixed_grid(omega,dv,grid=ugrid,wg=uwg)
    elif axis == 'imag':
        return fxc_qv_ifreq_fixed_grid(omega,dv,grid=ugrid,wg=uwg)
    else:
        raise ValueError('Unknown axis, ',axis)

if __name__=="__main__":

    for rs in [2.00876975652943]:#[0.1,0.5,1,2,3,4,5,10,20,30,40,69,100]:
        dv = density_variables(rs)
        print(exact_constraints(dv,x_only=False,param='PW92'))
        print(exact_constraints(dv,x_only=False,param='PZ81'))
        print(rs,get_qv_pars(rs))
    exit()

    rs = 3
    dvars = density_variables(rs)

    wp = (3/rs**3)**(0.5)

    om = wp*np.linspace(0.0,5.0,50)
    fxc_qv = fxc_longitudinal(dvars,om)*dvars['n']/(2*wp)
    import matplotlib.pyplot as plt
    plt.plot(om/wp,fxc_qv.imag)
    plt.plot(om/wp,fxc_qv.real)
    plt.show()
