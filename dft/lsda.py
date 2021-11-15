import numpy as np

from settings import pi

def exc(nu,nd):

    ex,vxu,vxd = ex(nu,nd)
    n = nu + nd
    z = np.minimum(np.maximum((nu - nd)/n,-1.0),1.0)
    rs = (3.0/(4.0*pi*n))**(1.0/3.0)
    ec,vcu,vcd = ec_pw92(rs,z)
    return ex+ec*n, vxu+vcu, vxd+vcd

def ex_lsda(nu,nd):

    if hasattr(nu,'__len__'):
        if len(nu[nu==0.0]) == len(nu): # one-electron systems
            ex,vxd = ueg_x(2*nd)/2.0
            vxu = np.zeros(nu.shape)
        elif len(nd[nd<1.e-14]) == len(nd):# one-electron systems
            ex,vxu = ueg_x(2*nu)/2.0
            vxd = np.zeros(nu.shape)
        else:
            exu,vxu = ueg_x(2*nu)/2.0
            exd,vxd = ueg_x(2*nd)/2.0
    else:
        if nu == 0.0: # one-electron systems
            ex,vxd = ueg_x(2*nd)
            vxu = 0.0
        elif nd == 0.0:# one-electron systems
            ex,vxu = ueg_x(2*nu)
            vxd = 0.0
        else:
            exu,vxu = ueg_x(2*nu)
            exd,vxd = ueg_x(2*nd)
            exu *= 0.5
            exd *= 0.5
            vxu *= 0.5
            vxd *= 0.5

    return exu + exd, vxu, vxd

def ueg_x(n):
    ax = -3.0/(4.0*pi)
    kf = (3*pi**2*n)**(1.0/3.0)
    return ax*kf*n, 4.0/3.0*ax*kf

def eps_x(rs):
    ax = -3.0/(4.0*pi)
    kf = (9*pi/4.0)**(1.0/3.0)/rs
    return ax*kf

def ec_pw92(rs,z):

    # J. P. Perdew and Y. Wang, PRB 45, 13244 (1992).
    # doi: 10.1103/PhysRevB.45.13244
    def g(v,rs):
        q0 = -2.0*v[0]*(1.0 + v[1]*rs)
        q1 = 2.0*v[0]*(v[2]*rs**(0.5) + v[3]*rs + v[4]*rs**(1.5) + v[5]*rs**2)
        q1p = v[0]*(v[2]*rs**(-0.5) + 2*v[3] + 3*v[4]*rs**(0.5) + 4*v[5]*rs)
        dg = -2*v[0]*v[1]*np.log(1.0 + 1.0/q1) - q0*q1p/(q1**2 + q1)
        return q0*np.log(1.0 + 1.0/q1),dg
    if not hasattr(z,'__len__') and z == 0.0:
        ec,d_ec_drs = g([0.031091,0.21370,7.5957,3.5876,1.6382,0.49294],rs)
        vc = ec - rs/3.0*d_ec_drs
        return ec,vc,vc

    fz = (1.0+z)**(4.0/3.0) + (1.0-z)**(4.0/3.0)-2.0
    fz /= 2.0**(4.0/3.0)-2.0
    dfz = 4.0/3.0*( (1.0 +z)**(1.0/3.0) - (1.0 -z)**(1.0/3.0))
    dfz /= 2.0**(4.0/3.0)-2.0
    fdd0 = 8.0/9.0/(2.0**(4.0/3.0)-2.0)

    ec0,d_ec_drs_0 = g([0.031091,0.21370,7.5957,3.5876,1.6382,0.49294],rs)
    ec1,d_ec_drs_1 = g([0.015545,0.20548,14.1189,6.1977,3.3662,0.62517],rs)
    ac,d_ac_drs = g([0.016887,0.11125,10.357,3.6231,0.88026,0.49671],rs)
    ac *= -1.0
    d_ac_drs *= -1.0
    ec = ec0 + ac*fz/fdd0*(1.0 - z**4) + (ec1 - ec0)*fz*z**4

    d_ec_drs = d_ec_drs_0*(1.0 - fz*z**4) + d_ec_drs_1*fz*z**4
    d_ec_drs += d_ac_drs*fz/fdd0*(1.0 - z**4)

    d_ec_dz = 4*z**3*fz*(ec1 - ec0 - ac/fdd0)
    d_ec_dz += dfz*(z**4*ec1 - z**4*ec0 + (1.0 - z**4)*ac/fdd0)

    vcu = ec - rs/3.0*d_ec_drs - (z - 1.0)*d_ec_dz
    vcd = ec - rs/3.0*d_ec_drs - (z + 1.0)*d_ec_dz

    return ec,vcu,vcd

def ec_pz81(rs,z):
    # J. P. Perdew and A. Zunger, PRB 23, 5048 (1981).

    fz = ((1.0+z)**(4.0/3.0) + (1.0-z)**(4.0/3.0) -2.0)/(2.0**(4.0/3.0)-2.0)

    gamma = {'u': -0.1423, 'p': -0.0843}
    beta1 = {'u': 1.0529, 'p': 1.3981}
    beta2 = {'u': 0.3334, 'p': 0.2611}
    A = {'u': 0.0311, 'p': 0.01555}
    B = {'u': -0.048, 'p': -0.0269}
    C = {'u': 0.0020, 'p': 0.0007}
    D = {'u': -0.0116, 'p': -0.0048}
    ec = {}
    for sigma in ['u','p']:
        if hasattr(rs,'__len__'):
            ec[sigma] = np.zeros(rs.shape)
            ec[sigma][rs >= 1.0] = gamma[sigma]/(1.0 + beta1[sigma]*rs[rs >= 1.0]**(0.5) + beta2[sigma]*rs[rs >= 1.0])
            ec[sigma][rs < 1.0] = A[sigma]*np.log(rs[rs < 1.0]) + B[sigma] + C[sigma]*rs[rs < 1.0]*np.log(rs[rs < 1.0]) + D[sigma]*rs[rs < 1.0]
        else:
            if rs >= 1.0:
                ec[sigma] = gamma[sigma]/(1.0 + beta1[sigma]*rs**(0.5) + beta2[sigma]*rs)
            elif rs < 1.0:
                ec[sigma] = A[sigma]*np.log(rs) + B[sigma] + C[sigma]*rs*np.log(rs) + D[sigma]*rs
    return ec['u'] + fz*(ec['p'] - ec['u'])

def ec_rpa_unp(rs):
    # J. P. Perdew and Y. Wang, PRB 45, 13244 (1992).
    # doi: 10.1103/PhysRevB.45.13244
    def g(v,rs):
        q0 = -2.0*v[0]*(1.0 + v[1]*rs)
        q1 = 2.0*v[0]*(v[2]*rs**(0.5) + v[3]*rs + v[4]*rs**(1.5) + v[5]*rs**(1.75))
        return q0*np.log(1.0 + 1.0/q1)
    return g([0.031091,0.082477,5.1486,1.6483,0.23647,0.20614],rs)
