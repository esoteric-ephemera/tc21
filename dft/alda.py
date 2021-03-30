import numpy as np

from settings import pi
# From J. P. Perdew and Alex Zunger,
# Phys. Rev. B 23, 5048, 1981
# doi: 10.1103/PhysRevB.23.5048
# for rs < 1
au = 0.0311
bu = -0.048
cu = 0.0020
du = -0.0116

# for rs > 1
gu = -0.1423
b1u = 1.0529
b2u = 0.3334
gp = -0.0843

# From J. P. Perdew and W. Yang
# Phys. Rev. B 45, 13244 (1992).
# doi: 10.1103/PhysRevB.45.13244
A = 0.0310906908696549008630505284145328914746642112731933593
# from julia BigFloat((1-log(2))/pi^2)
alpha = 0.21370
beta1 = 7.5957
beta2 = 3.5876
beta3 = 1.6382
beta4 = 0.49294

def alda(dv,x_only=False,param='PZ81'):

    n = dv['n']
    kf = dv['kF']
    rs = dv['rs']
    rsh = dv['rsh']

    fx = -pi/kf**2

    # The uniform electron gas adiabatic correlation kernel according to
    if param == 'PZ81':
        # Perdew and Zunger, Phys. Rev. B, 23, 5076 (1981)
        if x_only:
            return fx
        else:
            if hasattr(rs,'__len__'):
                fc = np.zeros(rs.shape)

            fc_lsr = -(3*au + 2*cu*rs*np.log(rs) + (2*du + cu)*rs)/(9*n)

            fc_gtr = 5*b1u*rsh + (7*b1u**2 + 8*b2u)*rs + 21*b1u*b2u*rsh**3 + (4*b2u*rs)**2
            fc_gtr *= gu/(36*n)/(1.0 + b1u*rsh + b2u*rs)**3

            if hasattr(rs,'__len__'):
                fc[rs < 1.0] = fc_lsr[rs < 1.0]
                fc[rs >= 1.0] = fc_gtr[rs >= 1.0]
            else:
                fc = fc_gtr
                if rs < 1.0:
                    fc = fc_lsr

    elif param == 'PW92':
        # J. P. Perdew and W. Yang, Phys. Rev. B 45, 13244 (1992).
        q = 2*A*(beta1*rsh + beta2*rs + beta3*rsh**3 + beta4*rs**2)
        dq = A*(beta1/rsh + 2*beta2 + 3*beta3*rsh + 4*beta4*rs)
        ddq = A*(-beta1/2.0/rsh**3 + 3.0/2.0*beta3/rsh + 4*beta4)

        d_ec_d_rs = 2*A*( -alpha*np.log(1.0 + 1.0/q) + (1.0 + alpha*rs)*dq/(q**2 + q) )
        d2_ec_d_rs2 = 2*A/(q**2 + q)*(  2*alpha*dq + (1.0 + alpha*rs)*( ddq - (2*q + 1.0)*dq**2/(q**2 + q) )  )

        fc = rs/(9.0*n)*(rs*d2_ec_d_rs2 - 2*d_ec_d_rs)

    return fx + fc

def lda_derivs(dv,param='PZ81'):
    rs = dv['rs']
    n = dv['n']
    kf = dv['kF']
    rsh = dv['rsh']

    if param == 'PZ81':
        eps_c = gu/(1.0 + b1u*rsh + b2u*rs)
        eps_c_lsr = au*np.log(rs) + bu + cu*rs*np.log(rs) + du*rs
        if hasattr(rs,'__len__'):
            eps_c[rs < 1.0] = eps_c_lsr[rs < 1.0]
        else:
            if rs < 1.0:
                eps_c = eps_c_lsr[rs < 1.0]

        d_eps_c_d_rs = -gu*(0.5*b1u/rsh + b2u)/(1.0 + b1u*rsh + b2u*rs)**2
        d_ec_drs_lsr = au/rs + cu + cu*np.log(rs) + du
        if hasattr(rs,'__len__'):
            d_eps_c_d_rs[rs < 1.0] = d_ec_drs_lsr[rs < 1.0]
        else:
            if rs < 1.0:
                d_eps_c_d_rs = d_ec_drs_lsr

    elif param == 'PW92':
        q = 2*A*(beta1*rsh + beta2*rs + beta3*rsh**3 + beta4*rs**2)
        dq = A*(beta1/rsh + 2*beta2 + 3*beta3*rsh + 4*beta4*rs)

        eps_c = -2*A*(1.0 + alpha*rs)*np.log(1.0 + 1.0/q)
        d_eps_c_d_rs = 2*A*( -alpha*np.log(1.0 + 1.0/q) + (1.0 + alpha*rs)*dq/(q**2 + q) )

    else:
        raise SystemExit('Unknown LDA, ',param)

    return eps_c,d_eps_c_d_rs
