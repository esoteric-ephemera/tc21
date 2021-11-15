import numpy as np
from lsda import ec_pw92
from settings import pi

"""
  see J. P. Perdew and Y. Wang,
    Phys. Rev. B 46, 12947 (1992),
    https://doi.org/10.1103/PhysRevB.46.12947
    and erratum Phys. Rev. B 56, 7018 (1997)
    https://doi.org/10.1103/PhysRevB.56.7018
    NB the erratum only corrects the value of the a3
    parameter in gc(rs, zeta, kf R)
"""

def gxc_lsd(r,rs,z):

    kf = (9*pi/4)**(1/3)/rs
    u = kf*r
    gx = gx_lsd_non_osc(u,z)
    gc = gc_lsd(rs,z,u)
    return gx + gc

def gx_lsd_non_osc(u,z):
    """
        INPUTS:
            z = (n_up - n_dn)/n
            u = k_F * R = k_F | r - r'|
        OUTPUTS:
            gx, the (coupling-constant averaged) exchange hole
    """

    opz = min(max(1 + z,0.0),2.0)
    omz = min(max(1 - z,0.0),2.0)
    # Eq. 8
    gx = 1.0 + 0.5*( opz**2*jy_non_osc(opz**(1/3)*u) + omz**2*jy_non_osc(omz**(1/3)*u) )
    return gx

def jy_non_osc(y):
    yeps = 1.e-6
    # Eq. 19
    ca = 0.59
    cb = -0.54354
    cc = 0.027678
    cd = 0.18843

    y2 = y*y

    if hasattr(y,'__len__'):
        ym = y<yeps
        jy[ym] = -0.5 + y2[ym]/10
        ym = y >= yeps
        yym = y2[ym]
        jy[ym] = -ca/(yym*(1 + 4*ca*yym/9)) + (ca/yym + cb + cc*yym)*np.exp(-cd*yym)
    else:
        if y<yeps:
            jy = -0.5 + y2/10
        else:
            jy = -ca/(y2*(1 + 4*ca*y2/9)) + (ca/y2 + cb + cc*y2)*np.exp(-cd*y2)
    return jy


def gc_lsd(rs,z,u):

    """
        INPUTS:
            rs = [ 3/(4 pi n) ]**(1/3)
            z = (n_up - n_dn)/n
            u = k_F * R = k_F | r - r'|
        OUTPUTS:
            gc, the coupling-constant averaged correlation hole for all rs
    """

    if rs <= 10:
        gc = gc_lsd_high_dens(rs,z,u)
    else:
        gc = gc_lsd_low_dens(rs,z,u)

    return gc

def gc_lsd_high_dens(rs,z,u):

    veps = 1.e-6
    # long-range, or high-density parameters, Sec. III
    kappa = 0.8145160769478095863505 # = (16._dp/(3._dp*pi**2))**(1._dp/3._dp)
    a1 = -0.1244
    a2 = 0.027032
    a3 = 0.0024317
    b1 = 0.2199
    b2 = 0.086664
    b3 = 0.012858
    b4 = 0.002

    # short-range, or low-density parameters, Sec. IV
    alpha = 0.193
    beta = 0.525
    gamma = 0.3393
    delta = 0.9
    epsilon = 0.10161

    rs2 = rs*rs
    #Eq. 18
    phi = spinf(z,2/3)
    phi3 = phi**3

    # Eq. 21
    ksr = kappa*rs**(0.5)*u
    v = phi*ksr
    v2 = v*v

    # Eq. 22
    f1v = (a1 + a2*v + a3*v2)/(1.0 + b1*v + v2*(b2 + b3*v + b4*v2))
    # Eq. 45
    dz = 0.305 - 0.136*z**2
    p = dz/(kappa**2*rs*phi**4)
    ph = p**(0.5)
    p2 = p*p

    def gc_taylor_small_v(vv):

        gfac = kappa*phi3*(phi*rs)**2

        cv0 = a3 - a2*b1 + a1*(b1**2 - b2 + p) + c1

        cv1 = -a3*b1 + a2*(b1**2 - b2 + p) - a1*(b1**3 - 2*b1*b2 + b3 + b1*p) + c2

        cv2 = a3*(b1**2 - b2) - a2*(b1**3 - 2*b1*b2 + b3) + a1*(b1**4 - 3*b1**2*b2 + b2**2 + 2*b1*b3 - b4) - a1*p**2/2.0 - p*c1 + c3

        return gfac*(cv0 + vv*(cv1 + vv*cv2))

    # Eq. 38
    oaprs = 1.0 + alpha*rs
    c12 = oaprs/(1.0 + beta*rs*oaprs)
    c1 = -0.0012529 + 0.1244*p + 0.61386*(1.0 - z**2)/(phi**5*rs2)*( c12 - 1.0)
    # Eq. 39
    c2 = 0.0033894 - 0.054388*p + 0.39270*(1.0 - z**2)/(phi**6*rs**(1.5))* (1.0 + gamma*rs)/(2.0 + delta*rs + epsilon*rs2) * c12

    eclsd,_,_ = ec_pw92(rs,z)

    # Eq. 43
    c3 = p2*(0.10847*ph + 1.4604 + 1.0685*np.log(p) + 34.356*eclsd/phi3) + ph*(0.51749*p- 3.5297*c1*ph - 1.903*c2)

    # Eq. 44
    c4 = -p2*(0.081596*p + 0.31677 + ph*(1.081 + 0.71019*np.log(p) + 22.836*eclsd/phi3)) + p*(1.903*c1*ph + 0.76485*c2)

    # Eq. 37
    f2 = (-a1 - (a2 - a1*b1)*v + v2*(c1 + c2*v + v2*(c3 + c4*v)))*np.exp(-dz*(u/phi)**2)

    if hasattr(v,'__len__'):
        vm = v < veps
        gc[vm] = gc_taylor_small_v(v[vm])
        vm = v >= veps
        gc[vm] = phi3*rs*(f1v + f2)/(kappa*u**2)
    else:
        if v<veps:
            gc = gc_taylor_small_v(v)
        else:
            gc = phi3*rs*(f1v + f2)/(kappa*u**2)

    return gc

def gc_lsd_low_dens(rs,z,u):

    """
        INPUTS:
            rs = [ 3/(4 pi n) ]**(1/3)
            z = (n_up - n_dn)/n
            u = k_F * R = k_F | r - r'|
        OUTPUTS:
            gc, the coupling-constant averaged correlation hole for rs > 10
    """

    # short-range, or low-density parameters, Sec. IV
    alpha = 0.193
    beta = 0.525
    # even lower-density parameters, rs > 10
    mu = 1.0891
    nu = -0.1825

    # Eq. B3, for rs > 10 but finite
    oaprs = 1.0 + alpha*rs
    x = 5.591*oaprs/(1.0 + beta*rs*oaprs )

    ec_rs_10,_,_ = ec_pw92(10.0,z)
    ec_lsda,_,_ = ec_pw92(rs,z)
    # Eq. B4
    chi = ( (1.0-x)*(0.8959 - 2*0.2291*spinf(z,4/3))/(-rs*ec_lsda + 10*x*ec_rs_10) )**(0.5)

    # Eq. B1, for lowest densities, rs --> infinity

    y = u*chi
    muy = mu*y
    gxc_rs_inf = 1.0 - (1.0 + muy*(1.0 + 0.5*muy) + nu*y**3)*np.exp(-muy)

    gc_rs_10 = gc_lsd_high_dens(10.0,z,u)
    gx_rs_inf = gx_lsd_non_osc(y,z)

    # Eq. B1
    gc = x*gc_rs_10 + (1.0 - x)*(gxc_rs_inf - gx_rs_inf)

    return gc

def spinf(z,n):
    opz = min(2.0,max(0.0,1+z))
    omz = min(2.0,max(0.0,1-z))
    return (opz**n + omz**n)/2.0


if __name__ == "__main__":

    rsl = [0.01, 0.1, 0.5, 1,2,5,10,20,100]
    for rs in rsl:
        gxc = gxc_lsd(0.0,rs,0.0)
        print('{:}, {:.3f}'.format(rs,gxc))
