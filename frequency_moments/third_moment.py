import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import multiprocessing as mp
from os import path,mkdir
import matplotlib.pyplot as plt
from matplotlib import ticker

import settings
from frequency_moments.frequency_moments import moment_parser
from dft.lsda import ec_pw92
from utilities.gauss_quad import gauss_kronrod
from utilities.integrators import nquad
import utilities.interpolators as interps
#from mcp07 import chi_parser

pi = settings.pi
k_min = settings.q_bounds['min']#0.01
k_max = 14.0

if not path.isdir('./third_moment_sum_rule_data'):
    mkdir('./third_moment_sum_rule_data')

def third_moment_plotter():
    fsz = 16
    clist=settings.clist#['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    m3_sr = {}
    amax = {}
    for rs in settings.rs_list:
        m3_sr[rs] = './third_moment_sum_rule_data/rs_'+str(rs)+'_third_moment_sum_rule_'+settings.fxc+'.csv'
        #m3_sr[rs] = './third_moment_sum_rule_data/rs_'+str(rs)+'_third_moment_sum_rule_'+settings.fxc+'.csv'
    fig,ax = plt.subplots(figsize=(8,6))
    bord = [1e20,-1e20]
    for iter,rs in enumerate(m3_sr):
        #wp = (3.0/rs**3)**(0.5)
        qq,m3,sr,errs = np.transpose(np.genfromtxt(m3_sr[rs],delimiter=',',skip_header=1))
        rel_err = (m3-sr)/(m3+sr)
        bord[0] = min(bord[0],rel_err.min())
        bord[1] = max(bord[1],rel_err.max())
        ax.plot(qq,rel_err,color=clist[iter],linestyle='-',linewidth=2,label='$r_s=$'+str(rs))
        """
        ax.plot(qq,m3,color=clist[iter],linestyle='-',linewidth=2,markersize=0,marker='o',label='$r_s=$'+str(rs))
        ax.plot(qq,sr,color=clist[iter],linestyle='--')
        """
        maxind = np.argmax(np.abs(rel_err))
        amax[rs] = [qq[maxind],rel_err[maxind]]
    print(amax)
    """
    plt.yscale('log')
    ax.set_yticks([10**i for i in range(-8,3)])
    locmin = ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=16)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    """
    ax.set_ylim([1.05*bord[0],1.05*bord[1]])
    ax.set_xlabel('$q/k_F$',fontsize=fsz)
    ax.set_xlim([0.0,3.0])
    ax.tick_params(axis='both',labelsize=fsz*.8)
    ax.legend(fontsize=fsz)
    #plt.title('[3rd moment (solid) or sum rule (dashed)]$/[\omega_p(0)]^3$',fontsize=fsz)
    plt.title('$(\Sigma_3^L - \Sigma_3^R)/(\Sigma_3^L + \Sigma_3^R)$',fontsize=fsz)
    #plt.savefig(base_str+'/m3_sr.pdf',dpi=300,bbox_inches='tight')
    plt.show()
    return

def sq(q_l,rs):
    if settings.fxc == 'MCP07' or settings.fxc == 'TC':
        wmetd = 'gk_adap'
    else:
        wmetd = 'adap'
    return moment_parser(rs,q_l,0.0,method=wmetd)

def set_cutoff(rs):

    conv_par = 1.0e-6
    init_tol = 0.01
    if rs > 10.0:
        init_tol = 0.1
        conv_par = 1.0e-3

    maxl = np.zeros((0,2))
    step_l = [0.5,0.1,0.05,0.01]
    for istep,step in enumerate(step_l):
        os = -1.0
        if istep == 0:
            kbd_l = np.arange(2.0,10,step)
        else:
            kbd_l = np.arange(tkc-step_l[istep-1],10.0,step)
        for tkc in kbd_l:
            s_tmp = sq(tkc,rs)
            maxl = np.vstack((maxl,[tkc,s_tmp[0]]))
            tol = init_tol + (conv_par - init_tol)/(len(step_l)-1)*istep
            if abs(os - s_tmp) < tol:
                if istep+1 == len(step_l):
                    found_kc = True
                break
            else:
                os = s_tmp
    return tkc,found_kc,maxl[np.argsort(maxl[:,1])][-3:]

def extrap_fun(q,pars):
    return 1.0 + pars[0]/q**2 + pars[1]/q**3

def extrap_s(q,s):
    y_extrap = False
    k_fit = q[q<= k_max]
    s_fit = s[q<= k_max]
    km = np.argmax(s_fit)
    ql = k_fit[-1]
    sl = s_fit[-1]
    for ik in range(km,len(k_fit)-2):
        qm = k_fit[ik]
        sm = s_fit[ik]
        oo_det = (ql*qm)**3/(qm - ql)
        apar = ((sm-1.0)/ql**3 - (sl-1.0)/qm**3)*oo_det
        bpar = (-(sm-1.0)/ql**2 + (sl-1.0)/qm**2)*oo_det
        s_ex = extrap_fun(k_fit[ik:],[apar,bpar])
        res = np.sum(np.abs(s_ex - s_fit[ik:]))
        if res < 1.e-3:
            y_extrap = True
            break
    s_extrap = extrap_fun(q[q > k_max],[apar,bpar])
    return s_extrap,y_extrap,[apar,bpar]

def interp_sq():
    q_max = 3.0
    interp_spc = 0.1
    sqd = {}
    kc = {}
    pars = {}
    for rs in settings.rs_list:
        #if rs < 10.0:
        #    kc[rs],found_kc,addn_pts = set_cutoff(rs)
        #else:
        kc[rs] = k_max
        found_kc = True
        #addn_pts = np.array([[0.005,None]])
        if found_kc:
            print(('Determined kc/kF = {:} for rs = {:}').format(kc[rs],rs))
        else:
            raise SystemExit(("Couldn't interpolate S(q) for rs = {:}; no suitable k_c").format(rs))
        k_ubd = ((1.01*q_max)**2 + kc[rs]**2 + 2*(1.01*q_max)*kc[rs])**(0.5)
        k_l = np.arange(k_min,k_ubd+2*interp_spc,interp_spc)
        #for apt in addn_pts[:,0]:
        #    if apt not in k_l:
        #        k_l = np.append(k_l,apt)
        #k_l = np.sort(k_l)
        sqt = np.zeros(k_l.shape[0])
        sqt[k_l<= k_max] = sq(k_l[k_l<= k_max],rs)
        if abs(sqt[k_l<= k_max][-1] - 1.0) < 1.e-3:
            sqt[k_l > k_max] = 1.0
            pars[rs] = []
        else:
            sqt[k_l > k_max],qextrap,pars[rs] = extrap_s(k_l,sqt)
            if not qextrap:
                print('Warning, extrapolation not optimal for rs=',rs)
        tk_l = np.zeros(k_l.shape[0]+1)
        tk_l[1:] = k_l
        tsqt = np.zeros(k_l.shape[0]+1)
        tsqt[1:] = sqt
        sqd[rs] = np.transpose((tk_l,tsqt))
    return kc,sqd,pars

def wrap_u_integrand(u,k,rs,q,sq,sq2,pars,interp,prec):
    #kf = (9.0*pi/4.0)**(1.0/3.0)/rs
    qmk = (q**2 + k**2 - 2*q*k*u)**(0.5)
    if interp=='spline':
        s_qmk = interps.spline(qmk,sq[:,0],sq[:,1],sq2)
        s_k_g = interps.spline(k,sq[:,0],sq[:,1],sq2)
    elif interp == 'linear':
        s_qmk = interps.linear_interpolator(qmk,sq[:,0],sq[:,1])
        s_k_g = interps.linear_interpolator(k,sq[:,0],sq[:,1])
    s_qmk[qmk<k_min]=0.0 # approximate S(q) as vanishing below some minimum
    #s_qmk[qmk>k_max] = extrap_fun(q,pars)
    if k < k_min:
        s_k_g = 0.0
    #elif k > k_max:
    #    s_k_g = extrap_fun(k,pars)
    return u**2*(s_qmk-s_k_g)

def integrate_u(k,rs,q,sq,sq2,interp,prec):
    return nquad(wrap_u_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':prec,'err_meas':'abs_diff','inc_grid':2},args=(k,rs,q,sq,sq2,interp,prec))

def wrap_k_integrand(k,rs,q,sq,sq2,interp,prec):
    u_int = np.zeros(k.shape)
    for iak,ak in enumerate(k):
        u_int[iak],u_err = integrate_u(ak,rs,q,sq,sq2,interp,prec)
        u_int[iak] *= ak**2
        if u_err['code']<=0:
            print('WARNING, u integral not converged, last error',u_err['error'],'code',u_err['code'])
    return u_int

def integrate_k(kc,rs,q,sq,sq2,interp,prec):
    kf = (9.0*pi/4.0)**(1.0/3.0)/rs
    int_opts={'itgr':'GK','npts':5,'prec':prec,'err_meas':'abs_diff','inc_grid':2}
    if rs==69:
        int_opts['reg'] = [[0.0,2.0],[2.0,2.13],[2.13,2.15],[2.15,2.17],[2.17,2.2],[2.2,kc]]
    k_int,k_err = nquad(wrap_k_integrand,(0.0,kc),'global_adap',int_opts,args=(rs,q,sq,sq2,interp,prec))
    k_int *= kf**3/pi
    if k_err['code']<=0:
        print('WARNING, k integral not converged, last error',k_err['error'],'code',k_err['code'])
    return k_int

def third_moment_driver(rs,q,sq,kc,sq2,ex_pars,interp='linear'):

    if not hasattr(q,'__len__'):
        q_l = q*np.ones(1)
    else:
        q_l = np.asarray(q)

    osum = -1.e2
    conv = False

    prec = 1.e-6
    min_recur = 2
    n_recur = 200 # recursions per axis
    sums = np.zeros(q_l.shape)

    kf = (9.0*pi/4.0)**(1.0/3.0)/rs

    grid_ind_l = [5 + 2*i for i in range(3)]
    grid_l = []
    for agrid in grid_ind_l:
        tgrdnm='./grids/gauss_kronrod_'+str(int(2*agrid+1))+'_pts.csv'
        grid_l.append(tgrdnm)
        if not path.isfile(tgrdnm) or path.getsize(tgrdnm) == 0:
            gauss_kronrod(agrid)

    for iq,q in enumerate(q_l):
        """
        sums[iq] = integrate_k(kc,rs,q,sq,sq2,ex_pars,interp,prec)
        continue
        """

        not_conv = 0

        need_denser_grid = False
        good_to_go = False

        sum_l = np.zeros(0)
        err_l = np.zeros(0)
        reg_l = np.zeros((0,4))

        for igrid,grid in enumerate(grid_l):

            wg1,u1,u1wg_gl = np.transpose(np.genfromtxt(grid,delimiter=',',skip_header=1))
            wg,u,uwg_gl = np.transpose(np.genfromtxt(grid,delimiter=',',skip_header=1))

            u1,u = np.meshgrid(u1,u)

            for ir in range(n_recur+1):

                k_sum = 0.0
                k_sum_gl = 0.0

                if ir == 0 and not need_denser_grid:

                    if rs == 69.0:
                        uuc_l = np.linspace(-1.0,1.0,min_recur+1)
                        uc_l = np.zeros((min_recur,2))
                        for iter in range(len(uuc_l)-1):
                            uc_l[iter,0] = uuc_l[iter]
                            uc_l[iter,1] = uuc_l[iter+1]
                        kc_l = [[0.0,2.0],[2.0,2.13],[2.13,2.15],[2.15,2.17],[2.17,2.2],[2.2,kc]]
                    else:
                        kc_l = np.zeros((min_recur,2))
                        uc_l = np.zeros((min_recur,2))
                        uuc_l = np.linspace(-1.0,1.0,min_recur+1)
                        kkc_l = np.linspace(0.0,kc,min_recur+1)
                        for iter in range(len(uuc_l)-1):
                            uc_l[iter,0] = uuc_l[iter]
                            uc_l[iter,1] = uuc_l[iter+1]
                            kc_l[iter,0] = kkc_l[iter]
                            kc_l[iter,1] = kkc_l[iter+1]
                elif ir > 0:
                    kc_l = np.asarray(tc_l)
                    uc_l = np.asarray(tu_l)

                tc_l = []
                tu_l = []

                for tens in product(uc_l,kc_l):

                    uul,kul = tens

                    ulbd,uubd = uul
                    u_g = 0.5*(uubd-ulbd)*u + 0.5*(uubd+ulbd)
                    uwg = 0.5*(uubd-ulbd)*wg
                    twg_gl= 0.5*(uubd-ulbd)*uwg_gl

                    lbd,ku = kul
                    k_g = 0.5*(ku-lbd)*u1 + 0.5*(ku+lbd)
                    wgk = 0.5*(ku-lbd)*kf*wg1
                    wgk_gl = 0.5*(ku-lbd)*kf*u1wg_gl

                    qmk = (q**2 + k_g**2 - 2*q*k_g*u_g)**(0.5)

                    s_qmk = np.zeros(qmk.shape)
                    s_k_g = np.zeros(qmk.shape)
                    for irow,row in enumerate(qmk):
                        #mask = (k_min <= row)# & (row <= k_max)
                        if interp=='spline':
                            s_qmk[irow] = interps.spline(row,sq[:,0],sq[:,1],sq2)
                        elif interp == 'linear':
                            s_qmk[irow] = interps.linear_interpolator(row,sq[:,0],sq[:,1])
                        #s_qmk[irow][row < k_min] = 0.0 # approximate S(q) as vanishing below some minimum
                        #s_qmk[irow][row > k_max] = extrap_fun(row[row > k_max],ex_pars) # use the analytic representation of the extrapolation
                        if irow == 0:
                            krow = k_g[irow]
                            #mask2 = (k_min <= krow)# & (krow <= k_max)
                            if interp=='spline':
                                s_k_g[irow] = interps.spline(krow,sq[:,0],sq[:,1],sq2)
                            elif interp == 'linear':
                                s_k_g[irow] = interps.linear_interpolator(krow,sq[:,0],sq[:,1])
                            #s_k_g[irow][krow < k_min] = 0.0
                            #s_k_g[irow][krow > k_min] = extrap_fun(krow[krow > k_min],ex_pars)
                        else:
                            s_k_g[irow] = s_k_g[0]

                    integrand = (kf*k_g*u_g)**2*(s_qmk-s_k_g)

                    k_sum = 1.0/pi*np.sum(wgk*np.matmul(uwg,integrand))
                    k_sum_gl = 1.0/pi*np.sum(wgk_gl*np.matmul(twg_gl,integrand))
                    err = abs(k_sum_gl-k_sum) # well-known error from Gauss-Kronrod integration

                    # global adaptive bisection of region only with largest error
                    sum_l = np.append(sum_l,k_sum) # to do this, we need to store the info about
                    err_l = np.append(err_l,err) # each region
                    reg_l = np.vstack((reg_l,[uul[0],uul[1],kul[0],kul[1]]))

                if np.sum(err_l) < prec:
                    sums[iq] = np.sum(sum_l)
                else:
                    sorted_inds = np.argsort(err_l)
                    reg_l = reg_l[sorted_inds]# partition region with largest error
                    tu = reg_l[-1][:2]
                    tk = reg_l[-1][2:]
                    reg_l = reg_l[:-1] # remove worst element from current list
                    err_l = err_l[sorted_inds][:-1]
                    sum_l = sum_l[sorted_inds][:-1]
                    tu_l.append([tu[0],0.5*(tu[0]+tu[1])])
                    tu_l.append([0.5*(tu[0]+tu[1]),tu[1]])
                    tc_l.append([tk[0],0.5*(tk[0]+tk[1])])
                    tc_l.append([0.5*(tk[0]+tk[1]),tk[1]])
                if ir == n_recur and igrid<len(grid_l)-1: # but if we're out of iterations, and we still failed,
                    need_denser_grid = True#try to bump up the grid size first, for all badly-behaved regions
                elif ir == n_recur and igrid == len(grid_l)-1:
                    not_conv += 1

                if not_conv > 0 and ir == n_recur and igrid == len(grid_l)-1:
                    print(('WARNING: for rs = {:}, q/kF = {:.2f},').format(rs,q))
                    print(('failed to converge within {:} precision in {:} recursive bisections').format(prec,n_recur**2))
                    print(('Last error {:.4e}').format(np.sum(err_l)))
                if len(tc_l)==0 and len(tu_l) == 0: # if all integrals are reasonably converged, move on to the next q
                    good_to_go = True
                    break
            if good_to_go:
                break
        sums[iq] = np.sum(sum_l) # regardless of convergence, give an estimate of the integral
    return sums

def third_moment_parser(sq,kcd,ex_pars,interp='linear'):

    rs_l = settings.rs_list
    q_l = np.arange(k_min,settings.q_bounds['max'],settings.q_bounds['step'])

    for rs in rs_l:
        if interp == 'spline':
            sq2 = interps.natural_spline(sq[rs][:,0],sq[rs][:,1])
        else:
            sq2 = np.zeros(1)
        if settings.fxc == 'MCP07' or settings.fxc == 'TC':
            wmetd = 'gk_adap'
        else:
            wmetd = 'adap'
        msr3 = moment_parser(rs,q_l,3.0,prec=1.e-6,method='original')

        n = 3.0/(4.0*pi*rs**3)
        kf = (3.0*pi**2*n)**(1.0/3.0)
        tq = q_l*kf
        wp = (3.0/rs**3)**(0.5)
        t0 = 3.0/10.0*kf**2

        ec,vcu,vcd = ec_pw92(rs,0.0)
        tc = -4.0*ec + 1.5*( vcu + vcd )
        t = t0 + tc

        if settings.nproc > 1:
            pool = mp.Pool(processes=min(settings.nproc,len(q_l)))
            tout = pool.starmap(third_moment_driver,[(rs,aq,sq[rs],kcd[rs],sq2,ex_pars[rs],interp) for aq in q_l])
            pool.close()
            sums = np.zeros(q_l.shape)
            for itout,anout in enumerate(tout):
                sums[itout] = anout
        else:
            sums = third_moment_driver(rs,q_l,sq[rs],kcd[rs],sq2,ex_pars[rs],interp=interp)

        sum_rule = tq**2/2.0*(tq**4/4.0 + 4*pi*n + 2*tq**2*t + sums)/wp**3
        with open('./third_moment_sum_rule_data/rs_'+str(rs)+'_third_moment_sum_rule_'+settings.fxc+'.csv','w+') as ofl:
            ofl.write('q/kF, < omega_p(q)**3 >/wp(0)**3, Sum rule/wp(0)**3, < (omega/wp(0))**3*S > - SR\n')
            for iln,ln in enumerate(msr3):
                ofl.write(('{:},{:},{:},{:}\n').format(q_l[iln],ln,sum_rule[iln],ln-sum_rule[iln]))
    return

def third_moment_calculation(interp='linear'):
    kc_d,sq_tab,pars = interp_sq()
    third_moment_parser(sq_tab,kc_d,pars,interp=interp)
    return

if __name__ == "__main__":

    #for rs in [4,10,30,69,100]:
    #    print(rs,sq(k_min,rs),sq(10,rs),sq(14,rs),sq(15,rs))
    #exit()

    #third_moment_calculation(interp='spline')
    third_moment_plotter()
