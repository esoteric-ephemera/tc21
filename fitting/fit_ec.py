import numpy as np
import multiprocessing as mp
from itertools import product
from os import path
from scipy.optimize import minimize

import settings
from lsda import ec_pw92
from eps_c import eps_quick,set_up_grid,establish_dependencies
from ec_plot import eps_c_plots

ec_ref = {}
for rs in settings.rs_list:
    ec_ref[rs],_,_ = ec_pw92(rs,0.0)

establish_dependencies()
grid,wg,ws = set_up_grid(settings.z_pts,settings.lambda_pts,settings.u_pts)
fgrid = np.zeros((grid.shape[0],4))
fgrid[:,0]=grid[:,0]
fgrid[:,1]=grid[:,1]
fgrid[:,2]=grid[:,2]
fgrid[:,3]=wg[:]

def residuals(vals,ref,method='uniform'):
    nelt = len(ref)
    if method == 'uniform': # even weighting
        wg = np.ones(nelt)
    elif method == 'sqrt': # w(x) = x^(1/2)
        wg = np.zeros(nelt)
        for ielt,elt in enumerate(vals):
            wg[ielt] = float(elt)**(0.5)
    elif method == 'lin': # w(x) = x
        wg = np.zeros(nelt)
        for ielt,elt in enumerate(vals):
            wg[ielt] = float(elt)
    elif method == 'step': # w(x < 10) = 1, w(x >= 10) = 10
        wg = np.ones(nelt)
        for ielt,elt in enumerate(vals):
            if float(elt) >= 10:
                wg[ielt] = 10.0
    wg /= np.sum(wg) # normalize weights

    err = {}
    err['res'] = 0.0
    for ielt,elt in enumerate(vals):
        err[elt] = abs(ref[elt] - vals[elt])
        err['res'] += wg[ielt]*err[elt]
    return err

def write_to_file(ec,err):
    tout = np.zeros((0,4))
    for rs in ec:
        tout = np.vstack((tout,[rs,ec[rs],ec_ref[rs],err[rs]]))
    np.savetxt('./epsilon_C_rMCP07.csv',tout,delimiter=',',header='rs, eps_c approx, eps_C PW92, absolute difference')
    return

def plot_rMCP07(pars):
    rslist = [i/10.0 for i in range(1,10)]
    for tmp in range(1,121,1):
        rslist.append(tmp)
    for rs in rslist:
        if rs not in ec_ref:
            ec_ref[rs],_,_ = ec_pw92(rs,0.0)
    _,ec,err=get_errors(pars,rsl=rslist)
    write_to_file(ec,err)
    eps_c_plots(targ='./epsilon_C_rMCP07.csv')
    return

def get_errors(pars,rsl=[],multi=False):

    par_d = {}
    par_d['a'],par_d['b'],par_d['c'],par_d['d'] = pars

    if multi:
        rsl = settings.rs_list
    if len(rsl)>0:
        if multi:
            inp = [(par_d,ars) for ars in rsl]
            pool = mp.Pool(processes=min(len(rsl),settings.nproc))
            tout = pool.starmap(wrapper,inp)
            pool.close()
            ecd = {}
            for anout in tout:
                rs,ec = anout
                ecd[rs] = ec
        else:
            ecd = eps_quick('user',pars=par_d,inps=fgrid,rs_l=rsl)
    else:
        ecd = eps_quick('user',pars=par_d,inps=fgrid)
    err = residuals(ecd,ec_ref,method='uniform')

    return par_d,ecd,err

def wrapper(par,rs):
    ecd = eps_quick('user',pars=par,inps=fgrid,rs_l=[rs])
    return rs,ecd[rs]#abs(ecd[rs] - ec_ref[rs])

def fit_optimal():
    a_l = np.arange(0.01,3.5,0.01)
    dat = np.zeros((0,3))
    for rs in ec_ref:
        pool = mp.Pool(processes=settings.nproc)
        tout = pool.starmap(wrapper,product(a_l,[rs]))
        pool.close()
        bind = np.argmin(np.asarray(tout))
        dat = np.vstack((dat,[rs,a_l[bind],tout[bind]]))
        print(dat)
    np.savetxt('./optimal_fit.csv',dat,delimiter=',',header='rs,a*kF,|error|')

def ec_fitting():

    fit_regex = ['rMCP07']

    if settings.fxc in fit_regex:

        if settings.filter_search:
            step_l = [1.0,0.5,0.1,0.5,0.2,0.1,0.05,0.02,0.01]
            init_a_step = step_l[0]
            init_b_step = step_l[0]
            init_c_step = step_l[0]
            init_d_step = step_l[0]
        else:
            step_l = [None]
            init_a_step = settings.a_step
            init_b_step = settings.b_step
            init_c_step = settings.c_step
            init_d_step = settings.d_step
        a_l = np.arange(settings.a_min,settings.a_max,init_a_step)
        b_l = np.arange(settings.b_min,settings.b_max,init_b_step)
        if settings.fit_c:
            c_l = np.arange(settings.c_min,settings.c_max,init_c_step)
        else:
            c_l = np.ones(1)
        if settings.fit_d:
            d_l = np.arange(settings.d_min,settings.d_max,init_d_step)
        else:
            d_l = np.ones(1)


        for iastep,astep in enumerate(step_l):

            if settings.filter_search and iastep > 0:
                a_l = np.arange(a_c-2*astep,a_c + 2*astep+min(step_l),astep)
                a_l = a_l[a_l >= settings.a_min]
                b_l = np.arange(b_c-2*astep,b_c + 2*astep+min(step_l),astep)
                b_l = b_l[b_l >= settings.b_min]
                if settings.fit_c:
                    c_l = np.arange(c_c-2*astep,c_c + 2*astep+min(step_l),astep)
                    c_l = c_l[c_l >= settings.c_min]
                else:
                    c_l = np.zeros(1)
                if settings.fit_d:
                    d_l = np.arange(d_c-2*astep,d_c + 2*astep+min(step_l),astep)
                    d_l = d_l[d_l >= settings.d_min]
                else:
                    d_l = np.zeros(1)

            work_l = product(a_l,b_l,c_l,d_l)

            if settings.nproc > 1:

                pool = mp.Pool(processes=min(settings.nproc,len(settings.rs_list)))
                tout = pool.map(get_errors,work_l)
                pool.close()

            else:
                tout = []
                for vec in work_l:
                    tmp = get_errors(vec)
                    tout.append(tmp)
            dat = np.array(tout)
            pars = dat[:,0]
            ecs = dat[:,1]
            errs = dat[:,2]
            res = np.zeros(0)
            for tmperr in errs:
                res = np.append(res,tmperr['res'])
            best = np.argmin(res)

            if settings.filter_search:
                a_c = pars[best]['a']
                b_c = pars[best]['b']
                c_c = pars[best]['c']
                d_c = pars[best]['d']

        write_to_file(ecs[best],errs[best])

        par = pars[best]
        epsc = ecs[best]
        errors = errs[best]
        logfile = './fitting/ec_fit_log.csv'
        ostr = 'pars\n'
        ostr += ('{:}'*len(par)).format(par) + '\n'
        ostr += 'Residual {:}\n'.format(errors['res'])
        for rs in epsc:
            ostr += '{:}, {:}, {:}\n'.format(rs,epsc[rs],errors[rs])

        return par,epsc,errors

    else:
        return get_errors({'a':None, 'b': None, 'c':None})

def min_func(p):
    _,_,errors=get_errors((p[0],p[1],p[2],p[3]),multi=True)
    return errors['res']

if __name__ == "__main__":
    """
    plot_rMCP07((settings.rMCP07_pars['a'],settings.rMCP07_pars['b'],settings.rMCP07_pars['c']))
    exit()
    """
    pars = settings.rMCP07_pars
    tps = (pars['a'],pars['b'],pars['c'],pars['d'])
    """
    p0 = [pars['a'],pars['b'],pars['c'],pars['d']]
    x = minimize(min_func,p0)
    print(x)
    exit()
    """
    par,epsc,errors=get_errors(tps)
    #print(errors['res'])
    plot_rMCP07(tps)

    exit()

    par,epsc,errors=ec_fitting()
    print(par)
    print(errors['res'])
    #print(epsc)
    for rs in epsc:
        print(rs,epsc[rs],errors[rs])
