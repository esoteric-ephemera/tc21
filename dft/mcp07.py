import numpy as np

from settings import pi,TC_pars,gki_param
from dft.alda import alda,lda_derivs
from dft.gki_fxc import gki_dynamic
from dft.qv_fxc import qv_fixed_grid,qv_static

def mcp07_static(q,dv,param='PZ81'):

    rs = dv['rs']
    n = dv['n']
    kf = dv['kF']
    rsh = dv['rsh']
    cfac = 4*pi/kf**2
    q2 = q**2

    # bn according to the parametrization of Eq. (7) of
    # Massimiliano Corradini, Rodolfo Del Sole, Giovanni Onida, and Maurizia Palummo
    # Phys. Rev. B 57, 14569 (1998)
    # doi: 10.1103/PhysRevB.57.14569
    bn = 1.0 + 2.15*rsh + 0.435*rsh**3
    bn /= 3.0 + 1.57*rsh + 0.409*rsh**3

    f0 = alda(dv,param=param)
    akn = -f0/(4.0*pi*bn)

    ec,d_ec_d_rs = lda_derivs(dv,param=param)
    d_rs_ec_drs = ec + rs*d_ec_d_rs
    # The rs-dependent cn, multiplicative factor of d( r_s eps_c)/d(r_s)
    # eps_c is correlation energy per electron

    cn = -pi/(2.0*kf)*d_rs_ec_drs

    # The gradient term
    cxcn = 1.0 + 3.138*rs + 0.3*rs**2
    cxcd = 1.0 + 3.0*rs + 0.5334*rs**2
    cxc = -0.00238 + 0.00423*cxcn/cxcd
    dd = 2.0*cxc/(n**(4.0/3.0)*(4.0*pi*bn)) - 0.5*akn**2

    # The MCP07 kernel
    vc = 4.0*pi/q2
    cl = vc*bn
    zp = akn*q2
    grad = 1.0 + dd*q2*q2
    cutdown = 1.0 + 1.0/(akn*q2)**2

    qeps = 1.e-6
    fxcmcp07_taylor = f0 + q2 * ( 2*cxc/n**(4/3) + \
        (f0*(akn**2/6 + dd) - cfac*cn*akn**2)*q2 )

    if hasattr(q,'__len__'):
        fxcmcp07 = np.zeros_like(q)
        qm = q < qeps
        fxcmcp07[qm] = fxcmcp07_taylor[qm]
        qm = q >= qeps
        fxcmcp07[qm] = cl*(np.exp(-zp)*grad - 1.0) - cfac*cn/cutdown
    else:
        if q < qeps:
            fxcmcp07 = fxcmcp07_taylor
        else:
            fxcmcp07 = cl*(np.exp(-zp)*grad - 1.0) - cfac*cn/cutdown

    return fxcmcp07,f0,akn

def mcp07_dynamic(q,omega,dv,axis='real',revised=False,pars={},param='PZ81'):

    fxc_q,f0,akn = mcp07_static(q,dv,param=param)
    if revised:
        if len(pars) > 0:
            fp = pars
        elif len(pars) == 0 and len(TC_pars) >0:
            fp = TC_pars
        else:
            raise SystemExit('TC kernel requires fit parameters!')
        kscr = dv['kF']*(fp['a'] + fp['b']*dv['kF']**(1.5))/(1 + dv['kF']**2)
        #kscr = fp['a']*dv['kF']/(1.0 + fp['b']*dv['kF']**(0.5))
        sclfun = (dv['rs']/fp['c'])**2
        #sclfun = dv['rs']**(3/2) *(fp['c'] + fp['d']*dv['rs']**(7/2) )/(1 + dv['rs']**3)
        pscl = sclfun + (1 - sclfun)*np.exp(-fp['d']*(q/kscr)**2)
        fxc_omega = gki_dynamic(dv,omega*pscl,axis=axis,revised=True,param=param,use_par=gki_param)
        fxc = (1.0 + np.exp(-(q/kscr)**2)*(fxc_omega/f0 - 1.0))*fxc_q
    else:
        fxc_omega = gki_dynamic(dv,omega,axis=axis,revised=False,param=param,use_par=gki_param)
        fxc = (1.0 + np.exp(-akn*q**2)*(fxc_omega/f0 - 1.0))*fxc_q

    return fxc

def qv_mcp07(q,omega,dv,axis='real',revised=False,pars={},intgrid=[],intwg=[]):

    fxc_q,f0_gki,akn = mcp07_static(q,dv,param='PW92')
    f0_qv = qv_static(dv)
    if revised:
        if len(pars) > 0:
            fp = pars
        elif len(pars) == 0 and len(TC_pars) >0:
            fp = TC_pars
        else:
            raise SystemExit('TC kernel requires fit parameters!')
        kscr = fp['a']*dv['kF']/(1.0 + fp['b']*dv['kF']**(0.5))
        F1 = fp['c']*dv['rs']**2
        F2 = F1 + (1.0 - F1)*np.exp(-fp['d']*(q/kscr)**2)
        fxc_omega = qv_fixed_grid(F2*omega,dv,axis=axis,ugrid=intgrid,uwg=intwg)
        fxc = (f0_qv + np.exp(-(q/kscr)**2)*(fxc_omega - f0_qv))*fxc_q/f0_gki
    else:
        fxc_omega = qv_fixed_grid(omega,dv,axis=axis,ugrid=intgrid,uwg=intwg)
        fxc = (f0_qv + np.exp(-akn*q**2)*(fxc_omega - f0_qv))*fxc_q/f0_gki

    return fxc
