import numpy as np
from os import path,mkdir
from math import floor,ceil
import matplotlib.pyplot as plt
import multiprocessing as mp

import settings
from utilities.gauss_quad import gauss_quad
from dft.chi import chi_parser
from dft.qv_fxc import fxc_longitudinal,density_variables
from utilities.integrators import nquad
from utilities.interpolators import spline,natural_spline

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

def qv_moments(rs,q,order,prec=1.e-8):
    if not path.isdir('./freq_data/qv_tab_data'):
        mkdir('./freq_data/qv_tab_data')
    tab_file = './freq_data/qv_tab_data/fxc_qv_rs_{:}.csv'.format(rs)
    dv = density_variables(rs)
    if path.isfile(tab_file):
        ftab,fxctab_re,fxctab_im = np.transpose(np.genfromtxt(tab_file,delimiter=',',skip_header=1))
        fxctab = np.zeros(ftab.shape,dtype='complex')
        fxctab.real = fxctab_re
        fxctab.imag = fxctab_im
    else:
        ftab = np.linspace(1.e-6,50,500)
        fxctab = np.zeros(ftab.shape,dtype='complex')
        if settings.nproc > 1 and hasattr(q,'__len__'):
            pool = mp.Pool(processes=min(settings.nproc,len(ftab)))
            fxct = pool.starmap(fxc_longitudinal,[(dv,om) for om in ftab])
            pool.close()
            for isqt in range(ftab.shape[0]):
                fxctab[isqt] = fxct[isqt]
        else:
            fxctab = fxc_longitudinal(dv,freqs)
        np.savetxt(tab_file,np.transpose((ftab,fxctab.real,fxctab.imag)),delimiter=',',header='omega,Re f_xc_QV(omega), Im f_xc_QV(omega)')
    qv2 = np.zeros(ftab.shape,dtype='complex')
    qv2.real = natural_spline(ftab,fxctab.real)
    qv2.imag = natural_spline(ftab,fxctab.imag)

    if settings.nproc > 1 and hasattr(q,'__len__'):
        pool = mp.Pool(processes=min(settings.nproc,len(q)))
        tout = pool.starmap(qv_spline_integration,[(ftab,fxctab,qv2,rs,aq,order,prec) for aq in q])
        pool.close()
        moment = np.zeros(q.shape[0])
        for iq in range(q.shape[0]):
            moment[iq] = tout[iq]
    else:
        moment = qv_spline_integration(ftab,fxctab,qv2,q,rs,order,prec)
    return moment/dv['wp0']**order

def qv_spline_integration(freq_tab,qv_tab,ddqv,q,rs,order,prec):
    intgrl,error= nquad(qv_spline_integrand,(freq_tab[0],freq_tab[-1]),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'err_meas':'abs_diff','inc_grid':0},args=(freq_tab,qv_tab,ddqv,q,rs,order))
    if error['code']<=0:
        print('warning, GK integration for order ',order,' moment not converged for q=',q,'last error=',error['error'],'; code',error['code'])
    return intgrl

def qv_spline_integrand(freq,freq_tab,qv_tab,ddqv,q,rs,order):
    fxc = np.zeros(freq.shape[0],dtype='complex')
    fxc.real = spline(freq,freq_tab,qv_tab.real,ddqv.real)
    fxc.imag = spline(freq,freq_tab,qv_tab.imag,ddqv.imag)
    wp = (3/rs**3)**(0.5)
    chi0 = chi_parser(0.5*q,freq+1.e-10j*wp,0.0,rs,'chi0',reduce_omega=False)
    kf = (9*pi/4)**(1/3)/rs
    chi = chi0/(1.0 - (4*pi/(q*kf)**2 + fxc)*chi0)
    sqw = -4*rs**3/3*chi.imag
    return freq**order*sqw


def moment_parser(rs,q,order,prec=1.e-8,method='original'):
    if settings.fxc == 'QV':
        moment = qv_moments(rs,q,order,prec=prec)
    else:
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
        moment = moment_parser(rs,q_l,order,prec=settings.moment_pars['prec'],method=settings.moment_pars['method'])
        if order > 0:
            htext = 'q,<w**{:}>/wp(0)**{:}'.format(order,order)
            fname = './freq_data/{:}_moment_{:}_rs_{:}_{:}.csv'.format(settings.fxc,order,rs,settings.moment_pars['method'])
        else:
            htext = 'q,S(q)'
            fname = './freq_data/{:}_Sq_rs_{:}_{:}.csv'.format(settings.fxc,rs,settings.moment_pars['method'])
        np.savetxt(fname,np.transpose((q_l,moment)),delimiter=',',header=htext)
    return


def sq_plots():

    fig,ax = plt.subplots(figsize=(8,6))

    for irs,rs in enumerate(settings.rs_list):
        q_l,sq = np.transpose(np.genfromtxt('./freq_data/{:}_Sq_rs_{:}_original.csv'.format(settings.fxc,rs),delimiter=',',skip_header=1))
        plt.plot(q_l,sq,color=settings.clist[irs],linewidth=2.5,label='$r_s={:}$'.format(rs))
        if settings.fxc == 'TC':
            pos_dict = {4: (.45,.4), 69: (1.55,.3) }
            if rs not in pos_dict:
                def_pos = (q_l[(len(q_l)-len(q_l)%2)//2],sq[(len(q_l)-len(q_l)%2)//2])
                pos_dict[rs] = def_pos
            ax.annotate('$r_s={:}$'.format(rs),pos_dict[rs],color=settings.clist[irs],fontsize=20)

    ax.set_xlim([0.0,settings.q_bounds['max']])
    ax.set_ylim([0.0,min(20,ax.get_ylim()[1])])
    ax.hlines(1.0,plt.xlim()[0],plt.xlim()[1],linestyle='--',color='gray')
    ax.set_xlabel('$q/k_F$',fontsize=24)
    ax.set_ylabel('$S(q)$',fontsize=24)
    ax.tick_params(axis='both',labelsize=20)

    flbl = settings.fxc
    if settings.fxc == 'TC':
        flbl = 'rMCP07'
    ax.annotate(flbl,((0.85 - 0.05*(len(flbl)-5))*settings.q_bounds['max'],0.05),fontsize=20)
    #if settings.fxc != 'TC':
    #    ax.legend(fontsize=20,loc='lower right')
    #plt.show()
    plt.savefig('./figs/Sq_{:}.pdf'.format(settings.fxc),dpi=600,bbox_inches='tight')
    return

def sq_tc_mcp07_comp_plot():

    fig,ax = plt.subplots(figsize=(8,6))

    for irs,rs in enumerate(settings.rs_list):
        q_l,sq = np.transpose(np.genfromtxt('./freq_data/TC_Sq_rs_{:}_original.csv'.format(rs),delimiter=',',skip_header=1))
        plt.plot(q_l,sq,color=settings.clist[irs],linewidth=2.5,label='$r_s={:}$'.format(rs))
        q_l,sq = np.transpose(np.genfromtxt('./freq_data/MCP07_Sq_rs_{:}_gk_adap.csv'.format(rs),delimiter=',',skip_header=1))
        plt.plot(q_l,sq,color=settings.clist[irs],linewidth=2.5,linestyle='--')
    ax.set_xlim([0.0,settings.q_bounds['max']])
    ax.set_ylim([0.0,min(20,ax.get_ylim()[1])])
    ax.hlines(1.0,plt.xlim()[0],plt.xlim()[1],linestyle='--',color='gray')
    ax.set_xlabel('$q/k_F$',fontsize=24)
    ax.set_ylabel('$S(q)$',fontsize=24)
    ax.tick_params(axis='both',labelsize=20)
    plt.title('MCP07 (dashed) and rMCP07 (solid) $S(q)$',fontsize=20)
    ax.legend(fontsize=20)
    #plt.show()
    plt.savefig('./figs/Sq_comparison.pdf',dpi=600,bbox_inches='tight')
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
    for ifnl,fnl in enumerate(['./freq_data/TC_S(q)_rs_4.csv','./freq_data/TC_S(q)_rs_69.csv']):
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
