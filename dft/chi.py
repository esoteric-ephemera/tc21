import numpy as np

import settings
from dft.alda import alda
from dft.gki_fxc import gki_dynamic
from dft.qv_fxc import fxc_longitudinal, fxc_longitudinal_multi_proc
from dft.mcp07 import mcp07_static,mcp07_dynamic,qv_mcp07

pi = settings.pi

#b = (3.0/(4.0*pi))**(1.0/3.0)

def chi_parser(z,omega,ixn,rs,wfxc,reduce_omega=False,imag_freq=False,ret_eps=False,pars={},LDA='PZ81'):

    """
        + z = q/(2 kF)
        + omega is the frequency, in atomic units
        + ixn is lambda, the coupling constant. Note that "lambda" reserved for
            anonymous functions in Python, so we can't use that name here
        + rs is the jellium density parameter, in bohr
        + wfxc selects the XC kernel
            + LDA (optional) selects either the Perdew-Zunger 1981 (PZ81) or
                Perdew-Wang 1992 (PW92) ALDA
        + reduce_omega (logical, optional) = True uses omega/eps_F instead of
            omega as the frequency variable (useful for integration)
        + imag_freq (logical, optional) selects imaginary frequency axis when needed
        + ret_eps (logical, optional) returns the dielectric function
            1 - [v_c(q) + f_xc(q,omega) ] chi_0(q,omega)
        + pars (dictionary, optional) are parameters for novel kernels
    """

    dvars = {}
    q_ixn_vec = hasattr(ixn,'__len__')
    if q_ixn_vec:
        dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs*np.ones(ixn.shape)
        dvars['rs'] = rs*np.ones(ixn.shape)
    else:
        dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
        dvars['rs'] = rs
    dvars['brs']= rs

    ef = dvars['kF']**2/2.0
    dvars['ef'] = ef
    q = 2*dvars['kF']*z
    if reduce_omega:
        ufreq = omega/(4*z)
    else:
        ufreq = omega/(4*z*ef)
    chi0 = -dvars['kF']/pi**2*lindhard(z,ufreq)

    if wfxc == 'chi0':
        return chi0
    if not q_ixn_vec and ixn == 0.0:
        return chi0

    vc = 4*pi*ixn/q**2

    chi = chi0
    # if interaction strength is positive, appropriately scale
    # also need to know if it is a vector or a scalar
    if q_ixn_vec:
        dvars['rs'][ixn>0.0] *= ixn[ixn>0.0]
        dvars['kF'][ixn>0.0] /= ixn[ixn>0.0]
        q[ixn>0.0] /= ixn[ixn>0.0]
        omega[ixn>0.0] /= ixn[ixn>0.0]**2
    else:
        dvars['rs'] *= ixn
        dvars['kF'] /= ixn
        q /= ixn
        omega /= ixn**2

    dvars['n'] = 3.0/(4.0*pi*dvars['rs']**3)
    dvars['rsh'] = dvars['rs']**(0.5)
    dvars['wp0'] = (3/dvars['rs']**3)**(0.5)

    if imag_freq:
        which_axis = 'imag'
        om = omega
    else:
        which_axis = 'real'
        om = omega.real
    if reduce_omega:
        om *= ef

    if wfxc == 'ALDA':
        fxc = alda(dvars,param=LDA)
    elif wfxc == 'RPA':
        if hasattr(omega,'__len__'):
            fxc = np.zeros(omega.shape)
        else:
            fxc = 0.0
    elif wfxc == 'MCP07':
        fxc = mcp07_dynamic(q,om,dvars,axis=which_axis,param='PZ81')
    elif wfxc == 'static MCP07':
        fxc,_,_ = mcp07_static(q,dvars,param='PZ81')
    elif wfxc == 'TC':
        fxc = mcp07_dynamic(q,om,dvars,axis=which_axis,revised=True,pars=pars,param=LDA)
    elif wfxc == 'GKI':
        fxc = gki_dynamic(dvars,om,axis=which_axis,revised=True,param=LDA,use_par=settings.gki_param)
    elif wfxc == 'QV':
        fxc = fxc_longitudinal(dvars,om)
    elif wfxc == 'QVmulti':
        fxc = fxc_longitudinal_multi_proc(dvars,om)
    elif wfxc == 'QV_MCP07':
        fxc = qv_mcp07(q,om,dvars,axis=which_axis,revised=False)
    elif wfxc == 'QV_TC':
        fxc = qv_mcp07(q,om,dvars,axis=which_axis,revised=True,pars=pars)
    else:
        raise SystemExit('WARNING, unrecognized XC kernel',wfxc)

    if q_ixn_vec:
        fxc[ixn>0.0] /= ixn[ixn>0.0]
    else:
        fxc /= ixn

    eps = 1.0 - (vc + fxc)*chi0
    if ret_eps:
        return eps

    if q_ixn_vec:
        chi[ixn>0.0] = chi0[ixn>0.0]/eps[ixn>0.0]
    else:
        chi = chi0/eps

    return chi


def lindhard(z,uu):

    """
        Eq. 3.6 of Lindhard's paper
    """

    zu1 = z - uu + 0.0j
    zu2 = z + uu + 0.0j

    fx = 0.5 + 0.0j
    fx += (1.0-zu1**2)/(8.0*z)*np.log((zu1 + 1.0)/(zu1 - 1.0))
    fx += (1.0-zu2**2)/(8.0*z)*np.log((zu2 + 1.0)/(zu2 - 1.0))

    return fx


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    q = np.arange(0.01,3.01,0.01)
    rsl = [1,4,10,30,69,100]
    for rs in rsl:

    #rs = 4

        dvars = {}
        dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
        dvars['rs'] = rs
        dvars['n'] = 3.0/(4*pi*rs**3)
        dvars['rsh'] = dvars['rs']**(0.5)
        f0 = alda(dvars,x_only=False,param='PW92')
        bn,finf = exact_constraints(dvars,param='PW92')
        print(rs,f0,finf,f0/finf)
        #fxc,_,_ = mcp07_static(0.5*q,q,dvars,param='PW92')
        #plt.plot(q,-fxc,label='$r_s=$'+str(rs))
    exit()
    plt.xlim([0,3])
    plt.yscale('log')
    plt.xlabel('$q/k_F$',fontsize=12)
    plt.ylabel('$-f_{xc}(q,\omega=0)$',fontsize=12)
    #plt.ylim([-500,-15.310303787092149])
    plt.legend(ncol=(len(rsl)-3))
    plt.show()
    exit()

    bn,finf = exact_constraints(dvars,param='PW92')
    #w,fxcu = np.transpose(np.genfromtxt('./rMCP07_re_fxc.csv',delimiter=',',skip_header=1))
    #fxcu = -cc*bn**(3.0/4.0)*fxcu/gam + finf
    w = np.linspace(0.0001,20.01,2000)
    fxcu = gki_dynamic(dvars,1.j*w,axis='imag',x_only=False,revised=True,param='PW92')
    #np.savetxt('./rMCP07_re_fxc.csv',np.transpose((w,fxcu)),delimiter=',',header='omega, re fxc dimensionless')
    #exit()
    """

    w,fxcu = np.transpose(np.genfromtxt('./rMCP07_re_fxc.csv',delimiter=',',skip_header=1))
    from scipy.optimize import curve_fit
    def interp(x,a,b,c,d,f):
        return( 1.0-d*x**f)/(1.0 + c*x**a)**b
    pars,cov = curve_fit(interp,w,fxcu)
    print(pars,cov)
    exit()

    import matplotlib.pyplot as plt
    fxcu = gki_dynamic(dvars,1.j*w,axis='imag',x_only=False,revised=True,param='PW92')
    plt.plot(w,fxcu)
    plt.plot(w,finf*np.ones(w.shape))
    #plt.ylim(-16,-3)
    plt.xlim(0,10)
    plt.show()
    exit()
    #"""
