import numpy as np
from scipy.optimize import leastsq
from os import fsync
from os.path import isfile
from itertools import product
import multiprocessing as mp

from dft.gki_fxc import gam
from settings import pi,nproc

wl,fxciu = np.transpose(np.genfromtxt('./test_fits/gki_fxc_ifreq.csv',delimiter=',',skip_header=1))

def fitfun(cp):
    f = (1 - cp[0]*wl + cp[1]*wl**2)/(1 + cp[2]*wl**2 + cp[3]*wl**4 + cp[4]*wl**6 + (cp[1]/gam)**(16/7)*wl**8)**(7/16)
    return f

def residuals(cp):
    return (fitfun(cp) - fxciu)**2

def resscal(cp):
    return np.sum(residuals(cp))

def fit_tc21_ifreq():

    npar = 5 # number of fit parameters

    pars = leastsq(residuals,np.ones(npar))[0]
    opt_res = resscal(pars)

    # see if we can improve the least squares fit slightly
    nrefine = 50 # number of refinement steps
    adj = 0.75 # effective search radius
    nsearch = 7 # number of candidates per parameter, per step
    old_opt_res = opt_res

    ofl = open('./fitting/fxc_ifreq_log.csv','w+')
    kstr = ''
    for i in range(npar):
        kstr += 'k'+str(i) +', '

    ofl.write('Iteration, '+ kstr + ' SSR \n')
    ofl.write(('LSQ, '+'{:}, '*(npar) + '{:}\n').format(*pars,opt_res))
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
        if nproc > 1:
            pool = mp.Pool(processes=nproc)
            tout = pool.map(resscal,worklist)
            pool.close()
            for itmp,tmp in enumerate(tout):
                res_l[itmp] = tmp
        else:
            for itmp in range(nsearch**npar):
                res_l[itmp] = resscal(tpars)

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

        if irefine > 10 and np.all( dpars < 1.e-6):
            # if params and residuals don't change much, stop
            break
        old_opt_res = opt_res
    ofl.write('==================\n')
    ofl.write(kstr+'\n')
    ofl.write(('{:},'*(npar-1) + '{:}\n').format(*pars))
    true_res = resscal(pars)
    ofl.write(('Sum of square residuals =, {:}\n').format(true_res))
    ofl.write('==================\n')
    trunc_pars = [round(apar,6) for apar in pars]
    ofl.write(('{:},'*(npar-1) + '{:} (rounded) \n').format(*trunc_pars))
    round_res = resscal(trunc_pars)
    ofl.write(('Sum of square residuals (rounded pars) =, {:}, (abs diff = {:})\n').format(round_res,abs(true_res - round_res)))
    ofl.close()

    tf = fitfun(pars)
    np.savetxt('./test_fits/fxc_ifreq_fit.csv',np.transpose((tf,fxciu,fxciu-tf)),delimiter=',',header='Model, PVI, PVI - Model')
    tf = fitfun(trunc_pars)
    np.savetxt('./test_fits/fxc_ifreq_fit_rounded.csv',np.transpose((tf,fxciu,fxciu-tf)),delimiter=',',header='Model, PVI, PVI - Model')

    return

if __name__ == "__main__":

    fit_tc21_ifreq()
