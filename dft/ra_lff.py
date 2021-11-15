import numpy as np

pi = np.pi
#from settings import pi
alpha = (4/(9*pi))**(1/3)

def g0_unp_pw92_pade(rs):
    """
    see Eq. 29 of
      J. P. Perdew and Y. Wang,
        Phys. Rev. B 46, 12947 (1992),
        https://doi.org/10.1103/PhysRevB.46.12947
        and erratum Phys. Rev. B 56, 7018 (1997)
        https://doi.org/10.1103/PhysRevB.56.7018
        NB the erratum only corrects the value of the a3
        parameter in gc(rs, zeta, kf R)
    """

    alpha = 0.193
    beta = 0.525
    return 0.5*(1 + 2*alpha*rs)/(1 + rs*(beta + rs*alpha*beta))**2

def g0_unp_yasuhara(rs):
    """
        H. Yasuhara, Solid State Commun. 11, 1481 (1972)
        NB: need factor of 1/2 to convert from Rydberg to Hartree
    """
    from scipy.special import iv
    lam = alpha*rs/pi
    return 2*lam/iv(1,4*lam**(0.5))**2

def ec_pw92_for_ra(z,rs):

    """
        Richardson-Ashcroft LFF needs some special derivatives of epsc, and moreover, needs them in
        Rydbergs, instead of Hartree.
        This routine gives those special derivatives in Rydberg

        J.P. Perdew and Y. Wang,
        ``Accurate and simple analytic representation of the electron-gas correlation energy'',
        Phys. Rev. B 45, 13244 (1992).
        https://doi.org/10.1103/PhysRevB.45.13244
    """

    rsh = rs**(0.5)
    def g(v):

        q0 = -2*v[0]*(1 + v[1]*rs)
        dq0 = -2*v[0]*v[1]

        q1 = 2*v[0]*(v[2]*rsh + v[3]*rs + v[4]*rs*rsh + v[5]*rs*rs)
        dq1 = v[0]*(v[2]/rsh + 2*v[3] + 3*v[4]*rsh + 4*v[5]*rs)
        ddq1 = v[0]*(-0.5*v[2]/rsh**3 + 3/2*v[4]/rsh + 4*v[5])

        q2 = np.log(1 + 1/q1)
        dq2 = -dq1/(q1**2 + q1)
        ddq2 = (dq1**2*(1 + 2*q1)/(q1**2 + q1) - ddq1)/(q1**2 + q1)

        g = q0*q2
        dg = dq0*q2 + q0*dq2
        ddg = 2*dq0*dq2 + q0*ddq2

        return g,dg,ddg

    unp_pars = [0.031091,0.21370,7.5957,3.5876,1.6382,0.49294]
    pol_pars = [0.015545,0.20548,14.1189,6.1977,3.3662,0.62517]
    alp_pars = [0.016887,0.11125,10.357,3.6231,0.88026,0.49671]

    fz_den = 0.5198420997897464#(2**(4/3)-2)
    fdd0 = 1.7099209341613653#8/9/fz_den

    opz = np.minimum(2,np.maximum(0.0,1+z))
    omz = np.minimum(2,np.maximum(0.0,1-z))
    dxz = (opz**(4/3) + omz**(4/3))/2.0
    d_dxz_dz = 2/3*(opz**(1/3) - omz**(1/3))
    d2_dxz_dz2 = 2/9*(opz**(-2/3) + omz**(-2/3))

    fz = 2*(dxz - 1)/fz_den
    d_fz_dz = 2*d_dxz_dz/fz_den
    d2_fz_dz2 = 2*d2_dxz_dz2/fz_den

    ec0,d_ec0_drs,d_ec0_drs2 = g(unp_pars)
    ec1,d_ec1_drs,d_ec1_drs2 = g(pol_pars)
    ac,d_ac_drs,d_ac_drs2 = g(alp_pars)
    z4 = z**4
    fzz4 = fz*z4

    ec = ec0 - ac/fdd0*(fz - fzz4) + (ec1 - ec0)*fzz4

    d_ec_drs = d_ec0_drs*(1 - fzz4) + d_ec1_drs*fzz4 - d_ac_drs/fdd0*(fz - fzz4)
    d_ec_dz = -ac*d_fz_dz/fdd0 + (4*fz*z**3 + d_fz_dz*z4)*(ac/fdd0 + ec1 - ec0)

    d_ec_drs2 = d_ec0_drs2*(1 - fzz4) + d_ec1_drs2*fzz4 - d_ac_drs2/fdd0*(fz - fzz4)
    d_ec_dz2 = -ac*d2_fz_dz2/fdd0 + (12*fz*z**2 + 8*d_fz_dz*z**3 + d2_fz_dz2*z4) \
        *(ac/fdd0 + ec1 - ec0)

    return 2*ec, 2*d_ec_drs, 2*d_ec_drs2, 2*d_ec_dz2

def fxc_ra(q,w,rs):
    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

        lff_ra_symm and lff_ra_occ return G/q**2

        C.F. Richardson and N.W. Ashcroft,
            Phys. Rev. B 50, 8170 (1994),

        and

        Eq. 32 of M. Lein, E.K.U. Gross, and J.P. Perdew,
            Phys. Rev. B 61, 13431 (2000)
    """
    gs = lff_ra_symm(q,w,rs)
    gn = lff_ra_occ(q,w,rs)
    return -4*pi*(gs + gn)

def lff_ra_symm(q,w,rs):

    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

        There are at least three alpha's in the RA paper
        alpha is determined from exact constraints, and is used in the lambdas (lam_)
        alp is a parameter used to control the parameterization, and is used in a, b and c
    """
    alp = 0.9
    fac = ( 2*(9*pi/4)**(1/3)/rs)**2 # = (2*kF)**2

    ec, d_ec_drs, d_ec_drs2, d_ec_dz2 = ec_pw92_for_ra(0.0,rs)

    # Eq. 40, corrected by Lein, Gross, and Perdew
    lam_s_inf = 3/5 - 2*pi*alpha*rs/5*(rs*d_ec_drs + 2*ec)

    # Eq. 44
    lam_pade = -0.11*rs/(1 + 0.33*rs)
    # Eq. 39, corrected by Lein, Gross, and Perdew
    lam_n_0 = lam_pade*(1 - 3*(2*pi/3)**(2/3)*rs*d_ec_dz2)
    # Eq. 38
    lam_s_0 = 1 + pi/3*alpha*rs**2*(d_ec_drs - rs*d_ec_drs2/2) - lam_n_0

    g0 = g0_unp_pw92_pade(rs)
    omg0 = 1 - g0

    gam_s = 9/16*omg0*lam_s_inf + (1 + 3*(1-1/alp))/4

    # Eq. 56
    a_s = lam_s_inf + (lam_s_0 - lam_s_inf)/(1 + (gam_s*w)**2)
    # Eq. 55
    c_s = 3*lam_s_inf/(4*omg0) - (4/3 - 1/alp + \
        3*lam_s_inf/(4*omg0))/(1 + gam_s*w)
    # Eq. 54
    b_s = a_s/( ( (3*a_s - 2*c_s*omg0)*(1 + w) - 8/3*omg0 )*(1 + w)**3 )

    q2 = q**2
    q6 = q2**3
    # Eq. 53
    g_s = (a_s + 2/3*b_s*omg0*q6)/(1 + q2*(c_s + b_s*q6))

    return g_s/fac

def lff_ra_occ(q,w,rs):

    """
        NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)
    """
    fac = ( 2*(9*pi/4)**(1/3)/rs)**2 # = (2*kF)**2
    gam_n = 0.68
    gnw = gam_n*w
    gnw2 = gnw*gnw
    opgnw = 1 + gnw

    ec, d_ec_drs, d_ec_drs2, d_ec_dz2 = ec_pw92_for_ra(0.0,rs)

    # Eq. 44
    lam_pade = -0.11*rs/(1 + 0.33*rs)
    # Eq. 39, corrected by Lein, Gross, and Perdew
    lam_n_0 = lam_pade*(1 - 3*(2*pi/3)**(2/3)*rs*d_ec_dz2)
    # Eq. 43
    lam_n_inf = 3*pi*alpha*rs*(ec + rs*d_ec_drs)

    """
    Eq. 65. Note that there is a "gamma" instead of "gamma_n" in the printed version of a_n
    assuming this just means gamma_n
    """
    a_n = lam_n_inf + (lam_n_0 - lam_n_inf)/(1 + gnw2)
    """
    Eq. 64
    in this equation, "gam_n(w)" is printed twice. I'm assuming this just means
    gam_n, since this is constant. That seems to give OK agreement with their figure
    """
    c_n = 3*gam_n/(1.18*opgnw) - ( (lam_n_0 + lam_n_inf/3)/(lam_n_0 + 2*lam_n_inf/3) \
        + 3*gam_n/(1.18*opgnw))/(1 + gnw2)
    # Eq. 63
    bt = a_n + lam_n_inf*(1 + 2/3*c_n*opgnw)
    b_n = -3/(2*lam_n_inf*opgnw**2)*( bt + (bt**2 + 4/3*a_n*lam_n_inf)**(0.5) )

    q2 = q**2
    q4 = q2*q2
    # Eq. 62
    g_n = (a_n - lam_n_inf/3 * b_n*q4)/(1 + q2*(c_n + q2*b_n))

    return g_n/fac

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rs = 2.07

    kf = (9*pi/4)**(1/3)/rs
    ql = np.linspace(0.0,9.0,2000)
    fxcst = fxc_ra(ql/(2*kf),0.0,rs)
    plt.plot(ql,fxcst)
    plt.ylim(-4.5,0)
    plt.xlim(0,9)
    plt.show()
    exit()
    lsls = ['-','--']
    for iw,w in enumerate([0.5, 2]):
        gs = lff_ra_symm(ql,w,rs)
        gn = lff_ra_occ(ql,w,rs)
        plt.plot(ql,gs,label="$G_s(q,iw), w= {:}$".format(w),color='darkblue',
            linestyle=lsls[iw])
        plt.plot(ql,gn,label="$G_n(q,iw), w= {:}$".format(w),color='darkorange',
            linestyle=lsls[iw])
    plt.legend(title='$r_s={:}$'.format(rs))
    plt.xlabel('$q/(2k_F)$')
    plt.ylabel('$G(q,iw)$')
    plt.ylim([-1.0,4.0])
    plt.show()
    exit()

    rsl = [0.01, 0.1, 0.5, 1,2,5,10,20,100]
    for rs in rsl:
        gxc = g0_unp_pw92_pade(rs)
        print('{:}, {:.3f}, {:.3f}'.format(rs,gxc,g0_unp_yasuhara(rs)))
