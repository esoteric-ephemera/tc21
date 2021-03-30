import numpy as np
import multiprocessing as mp
from itertools import product

import settings
from mcp07 import chi_mcp07
from gl_grid import gauss_legendre
from integrators import nquad
from lsda import eps_x,ec_pw92,ec_pz81

pi = np.pi
Eh_to_eV = 27.211386245988 # NIST CODATA, https://physics.nist.gov/cgi-bin/cuu/Value?eqhrev
verbose_omega = 0
verbose_lambda = 0
verbose_q = 0

def densvars(rs):
    dvars = {'rs': rs, 'kf': (9*pi/4.0)**(1.0/3.0)/rs, 'n': 3.0/(4.0*pi*rs**3), 'wp': (3.0/rs**3)**(0.5)}
    dvars['ef'] = dvars['kf']**2/2.0
    return dvars

def mcp07_s(omega,z,rs,ur=False):
    return -(chi_mcp07(z,omega*1.j,rs,ixn=0.0,reduce_omega=True,im_omega=True,wfxc=settings.fxc,new_hx=settings.new_hx).real)

def s(q,rs):
    wp = (3.0/rs**3)**(0.5)
    kf = (9.0*pi/4.0)**(1.0/3.0)/rs
    n = kf**3/(3.0*pi**2)
    tmp,err_code = nquad(mcp07_s,(0.0,'inf'),'global_adap',{'itgr':'CC','npts':14},{'inf_cond':'integral'},args=(0.5*q,rs))
    if err_code['code'] == 0:
        print(('WARNING, not converged; last integral error {:.4e}').format(err_code['error']))
    return tmp/(pi*n)*(kf**2/2.0)

def en_integrand(om,lam,z,d,which_eps):
    if which_eps == 'X':
        chi0 = chi_mcp07(z,om*1.0j,d['rs'],ixn=0.0,reduce_omega=True,im_omega=True,wfxc=settings.fxc,new_hx=settings.new_hx)
        s_q_omega_0 = -(chi0.real)
        return s_q_omega_0
    elif which_eps == 'C':
        chi_l = chi_mcp07(z,om*1.0j,d['rs'],ixn=lam,reduce_omega=True,im_omega=True,wfxc=settings.fxc,new_hx=settings.new_hx)
        chi0 = chi_mcp07(z,om*1.0j,d['rs'],ixn=0.0,reduce_omega=True,im_omega=True,wfxc=settings.fxc,new_hx=settings.new_hx)
        s_q_omega_l = -(chi_l.real)
        s_q_omega_0 = -(chi0.real)
        return s_q_omega_l-s_q_omega_0
    elif which_eps == 'XC':
        chi_l = chi_mcp07(z,om*1.0j,d['rs'],ixn=lam,reduce_omega=True,im_omega=True,wfxc=settings.fxc,new_hx=settings.new_hx)
        s_q_omega_l = -(chi_l.real)
        return s_q_omega_l

def ex_wrapper(z,d):
    return nquad(en_integrand,(0.0,'inf'),'global_adap',{'itgr':'CC','npts':14},{'inf_cond':'integral'},args=(None,z,d,'X'))

def ex_parser(z,d):
    if hasattr(z,'__len__'):
        z_l = z
    else:
        z_l = [z]
    if settings.ncore > 1 and len(z_l)>1:
        pool = mp.Pool(processes=min(settings.ncore,len(z_l)))
        tmp_out = pool.starmap(ex_wrapper,product(z_l,[d]))
        pool.close()
        int,err = np.transpose(tmp_out)
        if np.any(err[0]['code']==0.0):
            print('WARNING, not converged')
    else:
        int = []
        for az in z_l:
            tint,err = ex_wrapper(az,d)
            if err['code'] == 0:
                print(('WARNING, not converged; last integral error {:.4e}').format(err['error']))
            int.append(tint)
        int = np.asarray(int)
    return int-(pi*d['n'])/d['ef']#/(pi*n)*ef-1.0

def lam_wrapper(alam,z,d,which_en):
    return nquad(en_integrand,(0.0,'inf'),'global_adap',{'itgr':'CC','npts':14,'max_recur':350},{'inf_cond':'fun_and_deriv','n_extrap':30},args=(alam,z,d,which_en))

def lam_parser(lam,z,d,which_en,do_multi=True):
    if hasattr(lam,'__len__'):
        lam_l = lam
    else:
        lam_l = [lam]
    if do_multi and settings.ncore > 1 and len(lam_l)>1:
        pool = mp.Pool(processes=min(settings.ncore,len(lam_l)))
        tmp_out = pool.starmap(lam_wrapper,product(lam_l,[z],[d],[which_en]))
        pool.close()
        int,err = np.transpose(tmp_out)
        if np.any(err[0]['code']==0.0):
            print('WARNING, omega integration not converged')
            print(('Last errors'+'{:}').format(err[0]['error']))

    else:
        int = []
        for alam in lam_l:
            tint,err = lam_wrapper(alam,z,d,which_en)
            if err['code'] == 0:
                print(('Warning: omega integration failed! Last error {:}').format(err['error']))
            int.append(tint)
        int = np.asarray(int)
    return int

def ec_exc_wrapper(z,d,eps,do_multi=True):
    return nquad(lam_parser,(0.0,1.0),'global_adap',{'itgr':'GK','npts':15,'prec':1.e-6,'min_recur':2,'max_recur':350},args=(z,d,eps,do_multi))

def z_parser(z,d,which_eps):
    if hasattr(z,'__len__'):
        z_l = z
    else:
        z_l = [z]

    if settings.ncore > 1 and len(z_l)>1:
        pool = mp.Pool(processes=min(settings.ncore,len(z_l)))
        tmp_out = pool.starmap(ec_exc_wrapper,product(z_l,[d],[which_eps],[False]))
        pool.close()
        int,err = np.transpose(tmp_out)
        if np.any(err[0]['code']==0.0):
            print('WARNING, not converged')
    else:
        int = []
        for az in z_l:
            tint,err = ec_exc_wrapper(az,d,which_eps)
            if err['code'] == 0:
                print(('WARNING, not converged; last integral error {:.4e}').format(err['error']))
            int.append(tint)
        int = np.asarray(int)

    if which_eps == 'XC':
        int -= (pi*d['n'])/d['ef']

    return int

def int_eps(rs_l,which_eps):
    eps_d = {}
    for rs in rs_l:
        dvars = densvars(rs)
        if which_eps == 'X':
            int,err = nquad(ex_parser,(0.0,'inf'),'global_adap',{'npts':5,'itgr':'GK'},{'inf_cond':'integral'},args=(dvars,))
        else:
            int,err = nquad(z_parser,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-6,'max_recur':350},{'inf_cond':'integral','n_extrap':30},args=(dvars,which_eps))
        eps_d[rs] = 2*dvars['kf']/pi*int/(pi*dvars['n'])*dvars['ef']
        if err['code'] == 0:# and verbose_q:
            #verbose_q+=1
            print(("WARNING, integration failed for at least one q value at rs = {:}; integral error estimate {:.4e}").format(rs,err['error']))
    return eps_d

if __name__=="__main__":
    """
    q_l = np.arange(0.01,3.02,0.5)
    s0 = q_l/4.0*(3 - q_l**2/4.0)
    s0[q_l>2.0]=1.0
    for rs in [1.0,2.0,3.0,4.0,69.0,100.0]:
        print('-------------------')
        print('rs=',rs)
        for iq,q in enumerate(q_l):
            print(q,s(q,rs),s0[iq])
    exit()
    """
    stime=time()
    rs = 4.0
    dv = densvars(rs)
    y=0.0
    yy=0.0
    wg,mesh,wg_err = np.transpose(np.genfromtxt('./grids/gauss_kronrod_11_pts.csv',delimiter=',',skip_header=1))
    mesh = 0.5*20*(mesh+1.0)
    wg *= 0.5*20
    wg_err*= 0.5*20
    for i,z in enumerate(mesh):
        x,err=ec_exc_wrapper(z,dv,'C',do_multi=True)
        y+=wg[i]*x
        yy+=wg_err[i]*x
        print(z,x,err)
    print('eps_c~~',2*dv['kf']/pi*y/(pi*dv['n'])*dv['ef'],ec_pz81(rs,0.0))
    print('GK error',np.abs(y-yy))
    print('Runtime',time()-stime,'s')
    exit()


    rs=4.0
    stime=time()
    tx = int_eps([rs],'X')[rs]
    tc = int_eps([rs],'C')[rs]
    #txc = int_eps([rs],'XC')[rs]
    print(tx,eps_x(rs),100*(1.0 - tx/eps_x(rs)))
    print(tc,ec_pz81(rs,0.0),100*(1.0 - tc/ec_pz81(rs,0.0)))
    #print(txc,eps_x(rs)+ec_pz81(rs,0.0))
    print(time()-stime,'seconds')
    exit()


    trs = [1.0,2.0,3.0,4.0,10.0,20.0,30.0,40.0,50.0,60.0,69.0,100.0]
    stime=time()
    tx = int_eps(trs,'X')
    print(time()-stime,' seconds for X')
    print(tx)
    for rs in trs:
        print(rs,eps_x(rs),100*(1.0 - tx[rs]/eps_x(rs)))
    exit()
    stime=time()
    tc = int_eps(trs,'C')
    print(time()-stime,' seconds for C')
    with open('test_integration.csv','w+') as ofl:
        #ofl.write('rs, eps_x approx, eps_x exact, x % error, eps_xc approx, eps_c approx, eps_c PW92, c % error\n')
        ofl.write('rs, eps_x approx, eps_x exact, x % error, eps_c approx, eps_c PZ81, c % error\n')
        for rs in trs:
            eps_x = -3.0/(4.0*pi)*(9*pi/4.0)**(1.0/3.0)/rs
            n = 3.0/(4.0*pi*rs**3)
            eps_c = ec_pz81(rs,0.0)
            x_pe = 100*(1.0-tx[rs]/eps_x)
            c_appr = tc[rs]#txc[rs]-tx[rs]
            c_pe = 100*(1.0 - c_appr/eps_c)
            #ofl.write(('{:},{:.4f},{:.4f},{:.2f},{:.4f},{:.4f},{:.4f},{:.2f}\n').format(int(rs),tx[rs],eps_x,x_pe,txc[rs],c_appr,eps_c,c_pe))
            ofl.write(('{:},{:.4f},{:.4f},{:.2f},{:.4f},{:.4f},{:.2f}\n').format(int(rs),tx[rs],eps_x,x_pe,c_appr,eps_c,c_pe))
