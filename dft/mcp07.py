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
    vc = 4.0*pi/q**2
    cl = vc*bn
    zp = akn*q**2
    grad = 1.0 + dd*q**4
    cutdown = 1.0 + 1.0/(akn*q**2)**2
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
        kscr = fp['a']*dv['kF']/(1.0 + fp['b']*dv['kF']**(0.5))
        F1 = fp['c']*dv['rs']**2
        F2 = F1 + (1.0 - F1)*np.exp(-fp['d']*(q/kscr)**2)
        fxc_omega = gki_dynamic(dv,F2*omega,axis=axis,revised=True,param=param,use_par=gki_param)
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


def fit_tc21_ifreq():

    from dft.gki_fxc import gki_dynamic_real_freq,gam
    from utilities.integrators import nquad
    from scipy.optimize import leastsq
    from os.path import isfile

    rs = 1
    dv = {}
    dv['rs'] = rs
    dv['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dv['n'] = 3.0/(4.0*pi*dv['rs']**3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/dv['rs']**3)**(0.5)

    def wrap_integrand(tt,freq,rescale=False):
        if rescale:
            alp = 0.1
            to = 2*alp/(tt+1.0)-alp
            d_to_d_tt = 2*alp/(tt+1.0)**2
        else:
            to = tt
            d_to_d_tt = 1.0
        tfxc = gki_dynamic_real_freq(dv,to,x_only=False,revised=True,param='PW92',dimensionless=True)
        num = freq*tfxc.real + to*tfxc.imag
        denom = to**2 + freq**2
        return num/denom*d_to_d_tt

    if isfile('./test_fits/gki_fxc_ifreq.csv'):
        wl,fxciu = np.transpose(np.genfromtxt('./test_fits/gki_fxc_ifreq.csv',delimiter=',',skip_header=1))
    else:
        wl = np.arange(0.005,10.005,0.005)
        fxciu = np.zeros(wl.shape[0])
        for itu,tu in enumerate(wl):
            fxciu[itu],err = nquad(wrap_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(tu,),kwargs={'rescale':True})
            if err['error'] != err['error']:
                fxcu[itu],err = nquad(wrap_integrand,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(tu,))
            if err['code'] == 0:
                print(('WARNING, analytic continuation failed; error {:}').format(err['error']))

        # rescale so that fxc(0) - f_inf = 1 --> factor of gam
        # factor of 1/pi comes from Cauchy principal value integral
        fxciu *= gam/pi
        np.savetxt('./test_fits/gki_fxc_ifreq.csv',np.transpose((wl,fxciu)),delimiter=',',header='b(n)**(0.5)*u, fxc(i u)')

    def fitfun(cp):
        f = (1 - cp[0]*wl + cp[1]*wl**2)/(1 + cp[2]*wl**2 + cp[3]*wl**4 + cp[4]*wl**6 + (cp[1]/gam)**(16/7)*wl**8)**(7/16)
        return f
    def residuals(cp):
        return (fitfun(cp) - fxciu)**2
    pars = leastsq(residuals,np.ones(5))[0]
    with open('./fitting/fxc_ifreq_log.csv','w+') as ofl:
        ofl.write('c1, c2, c3, c4, c5 \n')
        ofl.write(('{:}, {:}, {:}, {:}, {:}\n').format(*pars))
        ofl.write(('Sum of square residuals = {:}').format(np.sum(residuals(pars))))
    tf = fitfun(pars)
    np.savetxt('./test_fits/fxc_ifreq_fit.csv',np.transpose((tf,fxciu,fxciu-tf)),delimiter=',',header='Model, PVI, PVI - Model')

    return
