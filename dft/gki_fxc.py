import numpy as np

from settings import pi
from dft.alda import alda,lda_derivs
from utilities.integrators import nquad


gam = 1.311028777146059809410871821455657482147216796875
# NB: got this value from julia using the following script
# using SpecialFunctions
# BigFloat((gamma(0.25))^2/(32*pi)^(0.5))
cc = 4.81710873550434914847073741839267313480377197265625 # 23.0*pi/15.0

def exact_constraints(dv,x_only=False,param='PZ81'):

    n = dv['n']
    kf = dv['kF']
    rs = dv['rs']

    f0 = alda(dv,x_only=x_only,param=param)
    """
     from Iwamato and Gross, Phys. Rev. B 35, 3003 (1987),
     f(q,omega=infinity) = -4/5 n^(2/3)*d/dn[ eps_xc/n^(2/3)] + 6 n^(1/3) + d/dn[ eps_xc/n^(1/3)]
     eps_xc is XC energy per electron
    """

    # exchange contribution is -1/5 [3/(pi*n^2)]^(1/3)
    finf_x = -1.0/(5.0)*(3.0/(pi*n**2))**(1.0/3.0)

    # correlation contribution is -[22*eps_c + 26*rs*(d eps_c / d rs)]/(15*n)
    if x_only:
        finf = finf_x
    else:
        eps_c,d_eps_c_d_rs = lda_derivs(dv,param=param)
        finf_c = -(22.0*eps_c + 26.0*rs*d_eps_c_d_rs)/(15.0*n)
        finf = finf_x + finf_c

    bfac = (gam/cc)**(4.0/3.0)
    deltaf = finf - f0
    """
    if hasattr(rs,'__len__'):
        deltaf[deltaf < 1.e-14] = 1.e-14
    else:
        if deltaf < 1.e-14:
            deltaf = 1.e-14
    """
    bn = bfac*deltaf**(4.0/3.0)

    return bn,finf

def gki_dynamic_real_freq(dv,u,x_only=False,revised=False,param='PZ81',dimensionless=False):

    if dimensionless:
        xk = u
    else:
        # The exact constraints are the low and high-frequency limits
        bn,finf = exact_constraints(dv,x_only=x_only,param=param)
        bnh = bn**(0.5)

        # for real frequency only! gki_dynamic analytically continues this to
        # purely imaginary frequency
        xk = bnh*u

    """
        Imaginary part from E.K.U. Gross and W. Kohn,
        Phys. Rev. Lett. 55, 2850 (1985),
        https://doi.org/10.1103/PhysRevLett.55.2850,
        and erratum Phys. Rev. Lett. 57, 923 (1986).
    """

    gx = xk/((1.0 + xk**2)**(5.0/4.0))

    if revised:
        cc1,cc2,cc3,cc4 = (0.174724,3.224459,2.221196,1.891998)
        hx = 1/gam*(1 - cc1*xk**2)/(1 + cc2*xk**2 + cc3*xk**4 + cc4*xk**6 + (cc1/gam)**(16/7)*xk**8)**(7/16)
        """
        apar,bpar,cpar = (0.1756,1.0376,2.9787)
        powr = 7.0/(2*cpar)
        hx = 1.0/gam*(1.0 - apar*xk**2)
        hx /= (1.0 + bpar*xk**2 + (apar/gam)**(1.0/powr)*np.abs(xk)**cpar)**powr
        """
    else:
        aj = 0.63
        h0 = 1.0/gam
        hx = h0*(1.0 - aj*xk**2)
        fac = (h0*aj)**(4.0/7.0)
        hx /= (1.0 + fac*xk**2)**(7.0/4.0)

    if dimensionless:
        fxcu = hx + 1.j*gx
    else:
        fxcu = finf - cc*bn**(3/4)*(hx + 1.j*gx)
    return fxcu


def gki_dynamic(dv,u,axis='real',x_only=False,revised=False,param='PZ81',use_par=False):

    if axis == 'real':
        fxcu = gki_dynamic_real_freq(dv,u,x_only=x_only,revised=revised,param=param)

    elif axis == 'imag':

        bn,finf = exact_constraints(dv,x_only=x_only,param=param)
        if use_par:
            if not revised and not x_only and param == 'PZ81':
                cpars = [1.06971,1.52708]#[1.06971136,1.52708142] # higher precision values destabilize the integration
                interp = 1.0/gam/(1.0 + (u.imag*bn**(0.5))**cpars[0])**cpars[1]
            elif revised and not x_only and param == 'PW92':
                y = bn**(0.5)*u.imag
                cp = (1.219946,0.973063,0.42106,1.301184,1.007578)
                interp = 1/gam*(1 - cp[0]*y + cp[1]*y**2)/(1 + cp[2]*y**2 + cp[3]*y**4 + cp[4]*y**6 + (cp[1]/gam)**(16/7)*y**8)**(7/16)
                """
                cpars = [0.99711536, 1.36722527, 0.93805229, 0.0101391,  0.71194338]
                xr = u.imag*bn**(0.5)
                interp = 1.0/gam*(1.0 - cpars[3]*xr**cpars[4])/(1.0 + cpars[2]*xr**cpars[0])**cpars[1]
                """
            fxcu = -cc*bn**(3/4)*interp + finf

        else:

            def wrap_integrand(tt,freq,rescale=False):
                if rescale:
                    alp = 0.1
                    to = 2*alp/(tt+1.0)-alp
                    d_to_d_tt = 2*alp/(tt+1.0)**2
                else:
                    to = tt
                    d_to_d_tt = 1.0
                tfxc = gki_dynamic_real_freq(dv,to,x_only=x_only,revised=revised,param=param,dimensionless=True)
                num = freq*tfxc.real + to*tfxc.imag
                denom = to**2 + freq**2
                return num/denom*d_to_d_tt

            if hasattr(u,'__len__'):
                for itu,tu in enumerate(tmpu):
                    rf = tu.imag*bn[itu]**(0.5)
                    fxcu[itu],err = nquad(wrap_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(rf,),kwargs={'rescale':True})
                    if err['error'] != err['error']:
                        fxcu[itu],err = nquad(wrap_integrand,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(rf,))
                    if err['code'] == 0:
                        print(('WARNING, analytic continuation failed; error {:}').format(err['error']))
                    fxcu[itu] = -cc*bn[itu]**(3.0/4.0)*fxcu[itu]/pi + finf[itu]
            else:
                rf = u.imag*bn**(0.5)
                fxcu,err = nquad(wrap_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(rf,),kwargs={'rescale':True})
                if err['error'] != err['error']:
                    fxcu,err = nquad(wrap_integrand,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(rf,))
                if err['code'] == 0:
                    print(('WARNING, analytic continuation failed; error {:}').format(err['error']))
                fxcu = -cc*bn**(3.0/4.0)*fxcu/pi + finf

    return fxcu


def gki_real_freq_taylor_series(dv,u,uc,revised=False,param='PZ81'):

    bn,finf = exact_constraints(dv,x_only=False,param=param)
    bnh = bn**(0.5)
    xk = bnh*uc

    gx = xk/(1 + xk**2)**(5/4)
    dgx = (1 - 3/2*xk**2)/(1.0 + xk**2)**(9/4)

    if revised:
        cc1,cc2,cc3,cc4 = (0.174724,3.224459,2.221196,1.891998)
        hx_num = (1 - cc1*xk**2)
        hx_den = (1 + cc2*xk**2 + cc3*xk**4 + cc4*xk**6 + (cc1/gam)**(16/7)*xk**8)
        hx_den_pow = hx_den**(7/16)
        hx = 1/gam*hx_num/hx_den_pow

        dhx1 = -2*cc1*xk
        dhx2 = -(7/8)*(cc2*xk + 2*cc3*xk**3 + 3*cc4*xk**5 + 4*(cc1/gam)**(16/7)*xk**7)/hx_den

        dhx = 1/gam*(dhx1 + dhx2*hx_num)/hx_den_pow
        """
        apar,bpar,cpar = (0.1756,1.0376,2.9787)
        powr = 7.0/(2*cpar)
        hx = 1/gam*(1.0 - apar*xk**2)
        hxden = (1 + bpar*xk**2 + (apar/gam)**(1/powr)*np.abs(xk)**cpar)
        hx /= hxden**powr

        dhx1 = -2*apar*xk
        dhx2 = -powr*(2*bpar*xk + cpar*(apar/gam)**(1/powr)*np.sign(xk)*np.abs(xk)**(cpar-1))/hxden
        dhx = 1/gam*(dhx1 + dhx2*(1 - apar*xk**2 ))/hxden**powr
        """
    else:
        aj = 0.63
        h0 = 1/gam
        hx = h0*(1 - aj*xk**2)
        fac = (h0*aj)**(4/7)
        hxden = (1 + fac*xk**2)
        hx /= hxden**(7/4)

        dhx = h0*(-2*aj*xk - 7/2*fac*xk*(1 - aj*xk**2)/hxden)/hxden**(7/4)

    fxcu = finf - cc*bn**(3/4)*(hx + 1.j*gx)
    """
        Cauchy-Riemman conditions for derivative of f(z) = u(x,y) + i v(x,y), for z = x + i y, with u and v real functions, and x and y real numbers

        d u/ dx = dv / dy
        d u / dy = -dv / dx
    """
    #d_fxc_du = - cc*bn**(3/4)*(dhx + 1.j*dgx)*bnh
    #return fxcu + d_fxc_du*(u - uc)
    dhx *= - cc*bn**(3/4)*bnh
    dgx *= - cc*bn**(3/4)*bnh

    dfxcu = dhx*(u.real - uc) - dgx*u.imag
    dfxcu += 1.j*(dgx*(u.real - uc) + dhx*u.imag)

    return fxcu + dfxcu
