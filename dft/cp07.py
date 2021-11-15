import numpy as np
from dft.alda import lda_derivs#, alda

from settings import pi

def fxc_cp07(dv,q,u):

    qeps = 1.e-6
    """
        NB: u = Im(omega), where Re(omega) = 0

        L.A. Constantin and J.M. Pitarke,
            Phys. Rev. B 75, 245127 (2007).
            doi: 10.1103/PhysRevB.75.245127
    """
    ec,d_ec_drs = lda_derivs(dv,param='PW92')
    crs = -pi/(2*dv['kF'])*(ec + dv['rs']*d_ec_drs)

    # brs according to the parametrization of Eq. (7) of
    # Massimiliano Corradini, Rodolfo Del Sole, Giovanni Onida, and Maurizia Palummo
    # Phys. Rev. B 57, 14569 (1998)
    # doi: 10.1103/PhysRevB.57.14569
    rsh = dv['rs']**(0.5)
    brs = 1.0 + 2.15*rsh + 0.435*rsh**3
    brs /= 3.0 + 1.57*rsh + 0.409*rsh**3

    # Eq. A4
    ars = np.exp(10.5/(1 + dv['rs'])**(13/2)) + 0.5

    gam1 = -0.114548231
    gam2 = 0.614523371
    # Eq. 8
    fxc_inf = gam1/dv['n']**gam2

    alp = -0.0255184916
    bet = 0.691590707
    # Eq. 6
    fxc_alda = 4*pi*alp/dv['n']**bet
    #fxc_alda = alda(dv,x_only=False,param='PW92')

    # Eq. 10
    cn = fxc_inf/fxc_alda
    # Eq. A3
    ku = -fxc_alda/(4*pi*brs)*(1 + u*(ars + u*crs))/(1 + u**2)

    q2 = q**2
    kuq2 = ku*q2
    # Eq. 12
    fxc_taylor = 4*pi*brs*ku*(-1 + ku*q2*(1/2 - ku*q2/6) ) \
        - 4*pi*crs*q2*(1 - q2)/dv['kF']**2

    if hasattr(q,'__len__'):
        fxc = np.zeros(q.shape)

        qm = q < qeps
        fxc[qm] = fxc_taylor[qm]

        qm = q >= qeps
        fxc[qm] = 4*pi*brs/q2[qm]*np.expm1(-kuq2[qm]) - 4*pi/dv['kF']**2 * crs/(1 + 1/q2[qm])
    else:
        if q < qeps:
            fxc = fxc_taylor
        else:
            fxc = 4*pi*brs/q2*np.expm1(-kuq2) - 4*pi/dv['kF']**2 * crs/(1 + 1/q2)

    return fxc
