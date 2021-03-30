import numpy as np
from os import path
from math import floor,ceil
import matplotlib.pyplot as plt
import multiprocessing as mp

import settings
from utilities.gauss_quad import gauss_quad
from dft.chi import chi_parser
from utilities.integrators import nquad

pi = settings.pi

default_grid = './grids/gauss_legendre_2000_pts.csv'
if not path.isfile(default_grid) or path.getsize(default_grid) == 0:
    gauss_quad(2000)
wg,pts = np.transpose(np.genfromtxt(default_grid,delimiter=',',skip_header=1))
pts = 0.5*(pts + 1.0)
wg *= 0.5

def s_q_adap(rs,q_l,prec):
    kf = (9.0*pi/4.0)**(1.0/3.0)/rs
    wp = (3.0/rs**3)**(0.5)
    if not hasattr(q_l,'__len__'):
        ql = q_l*np.ones(1)
    else:
        ql = q_l
    s = np.zeros(ql.shape)
    for iq,q in enumerate(ql):
        def raise_error():
            print(('WARNING: omega integration not coverged to {:} for rs = {:} and q/kF = {:}; last error {:}').format(prec,rs,q,err_code['error']))
        def wrap_chi(tt,rescale=False,scl_omega=True):
            if rescale:
                alp = .1
                to = (1-tt)/tt#2*alp/(tt+1.0)-alp
                d_to_d_tt = 1/tt**2#2*alp/(tt+1.0)**2
            else:
                to = tt
                d_to_d_tt = 1.0
            x = chi_parser(0.5*q,to*1.j,1.0,rs,settings.fxc,reduce_omega=scl_omega,imag_freq=True,LDA=settings.LDA)
            return -x.real*d_to_d_tt
        tmp,err_code = nquad(wrap_chi,(0.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'err_meas':'abs_diff'},kwargs={'rescale':True})
        if err_code['code'] <=0 or tmp < 0.0:
            tmp,err_code = nquad(wrap_chi,(0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'error monitoring':False})
            if err_code['code'] <=0 or tmp < 0.0:
                s[iq] = 0.0
                """
                sometimes S(q,i omega) changes sign abruptly
                to remedy this, we now search for a critical frequency to
                subdivide the integration region 0 < omega < infinity
                """
                sgn = np.sign(wrap_chi(0.0))
                fac_l = [10.0**(-i) for i in range(8)]
                for iafac,afac in enumerate(fac_l):
                    if iafac == 0:
                        dlist = range(1,20)
                    else:
                        if iafac%2==1: # hit the upper wall, go backwards
                            dlist = np.arange(wc,wc-fac_l[iafac-1],-afac)
                        else: # hit the lower wall, go forwards
                            dlist = np.arange(wc,wc+fac_l[iafac-1],afac)
                    for wc in dlist:
                        nsgn = np.sign(wrap_chi(wp*wc))
                        if nsgn != sgn:
                            sgn = nsgn
                            break
                hml = 20
                blist = [i/hml*wc*wp for i in range(2*hml)]#
                blist.append('inf')
                for ival in range(1,len(blist)):
                    for var_scl in [True,False]:
                        tmp,err_code = nquad(wrap_chi,(blist[ival-1],blist[ival]),'global_adap',{'itgr':'GK','npts':5,'prec':prec},kwargs={'scl_omega':var_scl})
                        if err_code['code'] <=0:
                            break
                    if err_code['code']<=0:
                        raise_error()
                        print('third attempt',(blist[ival-1],blist[ival]),tmp)
                    else:
                        if var_scl:
                            fac = (3*pi)/(2*kf)
                        else:
                            fac = (3*pi)/kf**3
                    if not np.isnan(err_code['error']):
                        s[iq] += tmp*fac
            else:
                if err_code['code']<=0:
                    raise_error()
                    print('second attempt')
                else:
                    s[iq]=tmp*(3*pi)/(2*kf)
        else:
            if err_code['code']<=0:
                raise_error()
            else:
                s[iq]=tmp*(3*pi)/(2*kf)
    return s

def grid_augmentor(q,lbd,sgrid,swg):
    spc = .02

    omega_c_max = 80.0 # cutoff in units of omega_p(0) for q = 3, estimated from converged rs = 69 values
    omega_c_min = 3.0 # cutoff for q --> 0, estimated in same way
    ubd = (omega_c_max - omega_c_min)/3.0*q + omega_c_min # interpolate along these

    if not hasattr(sgrid,'__len__') and sgrid == None:
        sgrid = np.zeros(0)
        swg = np.zeros(0)
        ivls = np.arange(0.0,ubd,spc)
    else:
        ivls = np.arange(lbd,ubd,spc)

    for ivl in ivls:
        sgrid = np.append(sgrid, spc*pts + ivl)
        swg = np.append(swg,spc*wg)
    return sgrid,swg,ubd

def frequency_moment(rs,q_l,order,prec):

    n = 3.0/(4.0*pi*rs**3)
    kf = (3*pi**2*n)**(1/3)
    wp = (3.0/rs**3)**(0.5)
    spc = 0.02
    twg = wp*spc*wg

    if not hasattr(q_l,'__len__'):
        q_l = q_l*np.ones(1)
    moment = np.zeros(q_l.shape[0])
    for iq,q in enumerate(q_l):
        omega = wp*spc*pts
        for ivl in np.arange(spc,100.0,spc):
            chi = chi_parser(0.5*q,omega+wp*1.e-10j,1.0,rs,settings.fxc,LDA=settings.LDA)
            s = -chi.imag/(pi*n)
            tsum = np.sum(s*twg*(omega/wp)**order)
            conv = abs(tsum) < prec*abs(moment[iq])
            moment[iq] += tsum
            if conv and ivl > 2.0 + q:
                break
            else:
                omega += wp*spc
    return moment

def gk_freq_moment(rs,q_l,order,prec):
    n = 3.0/(4.0*pi*rs**3)
    kf = (3*pi**2*n)**(1/3)
    ef = kf**2/2.0
    wp = (3.0/rs**3)**(0.5)
    if not hasattr(q_l,'__len__'):
        q_l = q_l*np.ones(1)
    moment = np.zeros(q_l.shape[0])
    for iq,q in enumerate(q_l):
        def wrap_integrand(to,lower):
            wcp = .1
            if lower:
                omega = to
                wg = 1.0
            else:
                omega =(1.0-to)/to
                wg = 1.0/to**2
            chi = chi_parser(0.5*q,omega+1.e-10j*wp,1.0,rs,settings.fxc,reduce_omega=True,imag_freq=False,LDA=settings.LDA)
            s = -chi.imag
            return wg*s*omega**order
        #tmp,err = nquad(wrap_integrand,(0.0,1.0),'global_adap',{'itgr':'GK','npts':7,'prec':0.5*prec,'err_meas':'abs_diff'},args=(True,))
        #moment[iq],err2 = nquad(wrap_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'err_meas':'quadpack','inc_grid':5},args=(False,))
        moment[iq],err2 = nquad(wrap_integrand,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'err_meas':'abs_diff','inc_grid':0},args=(True,))
        if err2['code']<=0:
            moment[iq],err = nquad(wrap_integrand,(0.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'err_meas':'abs_diff'},args=(False,))
            if err['code'] <=0:
                print('warning, GK integration order ',order,' not converged for q=',q,'last error=',err['error'],'(',err2['error'],') abs diff; code',err['code'])
        #if err2['code'] == 0:
        #    print('warning, GK integration order',order,' not converged for q=',q,'last error=',err2['error'],'rescaled')
    return moment*ef*(ef/wp)**(order)/(pi*n)


def moment_parser(rs,q,order,prec=1.e-8,method='original'):
    if settings.nproc > 1 and hasattr(q,'__len__'):
        pool = mp.Pool(processes=min(settings.nproc,len(q)))
        if method=='original':
            tout = pool.starmap(frequency_moment,[(rs,aq,order,prec) for aq in q])
        elif method == 'adap' and order == 0.0:
            tout = pool.starmap(s_q_adap,[(rs,aq,prec) for aq in q])
        elif method == 'gk_adap':
            tout = pool.starmap(gk_freq_moment,[(rs,aq,order,prec) for aq in q])
        pool.close()
        moment = np.zeros(q.shape)
        for itout,anout in enumerate(tout):
            moment[itout] = anout
    else:
        if method=='original':
            moment = frequency_moment(rs,q,order,prec)
        elif method == 'adap' and order == 0.0:
            moment = s_q_adap(rs,q,prec)
        elif method == 'gk_adap':
            moment = gk_freq_moment(rs,q,order,prec)
    return moment

def moment_calc(order):

    q_l = np.arange(settings.q_bounds['min'],settings.q_bounds['max'],settings.q_bounds['step'])
    for rs in settings.rs_list:
        momnt = moment_parser(rs,q_l,order,prec=settings.moment_pars['prec'],method=settings.moment_pars['method'])
        np.savetxt('./freq_data/{:}_moment_{:}_rs_{:}.csv'.format(settings.fxc,order,rs),np.transpose((q_l,sq)),delimiter=',',header='q,<omega**M>/wp(0)**M')
    return

if __name__ == "__main__":

    fig,ax = plt.subplots(figsize=(10,6))

    q_l = np.arange(settings.q_bounds['min'],settings.q_bounds['max'],settings.q_bounds['step'])
    for rs in settings.rs_list:
        sq = moment_parser(rs,q_l,0.0,method='gk_adap')
        #print(sq)
        #exit()
        plt.plot(q_l,sq,label='$r_s={:}$'.format(rs))
        np.savetxt('./freq_data/{:}_S(q)_rs_{:}.csv'.format(settings.fxc,rs),np.transpose((q_l,sq)),delimiter=',',header='q,S(q)')
    """
    clist = ['tab:blue','tab:orange']
    for ifnl,fnl in enumerate(['./freq_data/rMCP07_S(q)_rs_4.csv','./freq_data/rMCP07_S(q)_rs_69.csv']):
        dat = np.genfromtxt(fnl,delimiter=',',skip_header=1)
        rs = (fnl.split('_')[-1]).split('.csv')[0]
        plt.plot(dat[:,0],dat[:,1],label='$r_s={:}$'.format(rs),color=clist[ifnl])
    for ifnl,fnl in enumerate(['./freq_data/MCP07_S(q)_rs_4.csv','./freq_data/MCP07_S(q)_rs_69.csv']):
        dat = np.genfromtxt(fnl,delimiter=',',skip_header=1)
        rs = (fnl.split('_')[-1]).split('.csv')[0]
        plt.plot(dat[:,0],dat[:,1],linestyle='--',color=clist[ifnl])
    """
    ax.set_xlim([0.0,3.0])
    ax.set_ylim([0.0,min(20,ax.get_ylim()[1])])
    ax.hlines(1.0,plt.xlim()[0],plt.xlim()[1],linestyle='--',color='gray')
    ax.set_xlabel('$q/k_F$',fontsize=18)
    ax.set_ylabel('$S(q)$',fontsize=18)
    ax.tick_params(axis='both',labelsize=16)
    plt.title('MCP07 (dashed) and rMCP07 (solid) $S(q)$',fontsize=18)
    ax.legend(fontsize=16)
    plt.show()
