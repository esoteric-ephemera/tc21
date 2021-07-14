import numpy as np
from itertools import product
import multiprocessing as mp
from os import fsync
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

import settings

gam = 1.311028777146059809410871821455657482147216796875

x,kk = np.transpose(np.genfromtxt('./test_fits/kram_kron_re_fxc.csv',delimiter=',',skip_header=1))

def error_meas(y,yref):
    return np.sum((y - yref)**2)**(0.5)/(1.0*y.shape[0])

def test_fn(cp):
    """
    powr = 7.0/(2*c)
    num = 1.0/gam*(1.0 - a*x**2)
    denom = (1.0 + b*x**2 + (a/gam)**(1.0/powr)*np.abs(x)**c)**powr
    """
    num = 1/gam*(1 - cp[0]*x**2)
    denom = (1 + cp[1]*x**2 + cp[2]*x**4 + cp[3]*x**6 + (cp[0]/gam)**(16/7)*x**8)**(7/16)
    return num/denom

def wrap_err_lsq(var):
    ty = test_fn(var)
    return (ty - kk)**2#error_meas(ty,kk)

def wrap_res_scal(var):
    return np.sum(wrap_err_lsq(var))

def wrap_err(var):
    ty = test_fn(var)
    return error_meas(ty,kk)

def kramers_kronig_plot(pars=None):
    if pars is not None:
        mod = test_fn(pars)
        np.savetxt('./test_fits/new_hx.csv',np.transpose((x,kk,mod)),delimiter=',',header='x, K-K, new h(x)',fmt='%.18f')
    om,fxc_kk,fxc_mod = np.transpose(np.genfromtxt('./test_fits/new_hx.csv',delimiter=',',skip_header=1))
    fig,ax = plt.subplots(figsize=(10,6))

    ax.plot(om[1:],fxc_kk[1:],color='tab:blue')
    ax.plot(om[1:],fxc_mod[1:],color='tab:orange')
    plt.title('Kramers-Kronig (blue) and model (orange), $r_s$ independent',fontsize=20)

    ax.set_xlim([om[1],6.0])
    ax.set_ylabel('$h(b(n)^{1/2}\omega)$',fontsize=20)
    ax.set_xlabel('$b(n)^{1/2}\omega$',fontsize=20)
    ax.tick_params(axis='both',labelsize=16)
    plt.savefig('./test_fits/kramers_kronig_model_comparison.pdf',dpi=600,bbox_inches='tight')
    return

def hx_fit_main():

    npar = 4 # number of fit parameters
    pars = leastsq(wrap_err_lsq,np.ones(npar))[0]
    opt_res = wrap_res_scal(pars)

    # see if we can improve the least squares fit slightly
    nrefine = 50 # number of refinement steps
    adj = 0.25 # effective search radius
    nsearch = 10 # number of candidates per parameter, per step
    old_opt_res = opt_res

    ofl = open('./fitting/hx_fit_log.csv','w+')
    cstr = ''
    for i in range(npar):
        cstr += 'c'+str(i) +', '

    ofl.write('Iteration, '+ cstr + ' SSR \n')
    ofl.write(('LSQ, '+'{:}, '*(npar) + '{:}, \n').format(*pars, opt_res))
    ofl.flush()
    fsync(ofl.fileno())

    dpars = np.zeros(npar+1)

    for irefine in range(nrefine):

        srad = adj**(irefine+1)
        decr = 1 - srad
        incr = 1 + srad

        tmppars = np.zeros((npar,nsearch))
        for ipar in range(npar):
            tmppars[ipar] = np.linspace(pars[ipar]*decr,pars[ipar]*incr,nsearch)
        worklist = product(*tmppars)

        res_l = np.zeros(nsearch**npar)
        if settings.nproc > 1:
            pool = mp.Pool(processes=settings.nproc)
            tout = pool.map(wrap_res_scal,worklist)
            pool.close()
            for itmp,tmp in enumerate(tout):
                res_l[itmp] = tmp
        else:
            for itmp in range(nsearch**npar):
                res_l[itmp] = wrap_res_scal(tpars)

        tres = res_l.min()
        if tres < opt_res:
            opt_res = tres
            tpars = list(product(*tmppars))[np.argmin(res_l)]
            for j in range(npar):
                pars[j] = tpars[j]

        if irefine > 0:
            # check to see how much parameters change, and how much residuals
            # change after each step
            for j in range(npar):
                dpars[j] = abs(lpars[j] - pars[j])/abs(lpars[j])
            dpars[npar] = abs(old_opt_res - opt_res)/abs(old_opt_res)
        lpars = pars[:]

        ofl.write(('{:},'*(npar+1) + '{:}\n').format(irefine,*pars,opt_res))
        ofl.flush()
        fsync(ofl.fileno())

        if irefine > 10 and abs(old_opt_res - opt_res)/abs(old_opt_res) < 1.e-6:
            break
        old_opt_res = opt_res

    ofl.write('==================\n')
    ofl.write(cstr+'\n')
    ofl.write(('{:},'*(npar-1) + '{:}\n').format(*pars))
    true_res = wrap_res_scal(pars)
    ofl.write(('Sum of square residuals =, {:}\n').format(true_res))
    ofl.write('==================\n')
    trunc_pars = [round(apar,6) for apar in pars]
    ofl.write(('{:},'*(npar-1) + '{:} (rounded) \n').format(*trunc_pars))
    round_res = wrap_res_scal(trunc_pars)
    ofl.write(('Sum of square residuals (rounded pars) =, {:}, (abs diff = {:})\n').format(round_res,abs(true_res - round_res)))
    ofl.close()

    np.savetxt('./test_fits/new_hx.csv',np.transpose((x,kk,test_fn(pars))),delimiter=',',header='x, K-K, new h(x)')
    np.savetxt('./test_fits/new_hx_rounded.csv',np.transpose((x,kk,test_fn(trunc_pars))),delimiter=',',header='x, K-K, new h(x)')

    return pars

if __name__ == "__main__":

    mod = test_fn(0.1756,1.0376,3.0,None)
    np.savetxt('./test_fits/new_hx.csv',np.transpose((x,kk,mod)),delimiter=',',header='x, K-K, new h(x)',fmt='%.18f')

    kramers_kronig_plot()
    exit()

    #abs_best = {'a': 0.1756, 'b': 1.0376, 'c': 2.9787} # error 2.887 x 10**(-5)
    abs_best = {'a': 0.1756, 'b': 1.0376, 'c': 3.0} # error 3.673 x 10**(-5)
    #print(wrap_err((abs_best['a'],abs_best['b'],abs_best['c'],None)))
    #exit()
