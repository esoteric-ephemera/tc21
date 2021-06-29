import numpy as np
import multiprocessing as mp
from itertools import product
from os import path
from scipy.optimize import least_squares


import settings
from dft.lsda import ec_pw92
from eps_c import eps_quick,set_up_grid,establish_dependencies
from plotters.ec_plot import eps_c_plots

ec_ref = {}
for rs in settings.rs_list:
    ec_ref[rs],_,_ = ec_pw92(rs,0.0)

establish_dependencies()
grid,wgg,ws = set_up_grid(settings.z_pts,settings.lambda_pts,settings.u_pts)
fgrid = np.zeros((grid.shape[0],4))
fgrid[:,0]=grid[:,0]
fgrid[:,1]=grid[:,1]
fgrid[:,2]=grid[:,2]
fgrid[:,3]=wgg[:]

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
    wfile = './eps_data/epsilon_C_{:}.csv'.format(settings.fxc)
    np.savetxt(wfile,tout,delimiter=',',header='rs, eps_c approx, eps_C PW92, absolute difference')
    return

def plot_TC(pars):
    rslist = [i/10.0 for i in range(1,10)]
    for tmp in range(1,121,1):
        rslist.append(tmp)
    for rs in rslist:
        if rs not in ec_ref:
            ec_ref[rs],_,_ = ec_pw92(rs,0.0)
    _,ec,err=get_errors(pars,rsl=rslist,multi=True)
    write_to_file(ec,err)
    wfile = './eps_data/epsilon_C_{:}.csv'.format(settings.fxc)
    eps_c_plots(targ=wfile)
    return

def get_errors(pars,rsl=[],multi=False):

    par_d = {}
    par_d['a'],par_d['b'],par_d['c'],par_d['d'] = pars

    if multi and len(rsl)==0:
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

def wrap_lsq(pars):
    _,_,errd = get_errors(pars,multi=True)
    res = np.zeros(len(settings.rs_list))
    for irs,rs in enumerate(settings.rs_list):
        res[irs] = errd[rs]**2
    return res

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
    np.savetxt('./eps_data/optimal_fit.csv',dat,delimiter=',',header='rs,a*kF,|error|')

def ec_fitting():

    fit_regex = ['TC','QV_TC']

    if settings.fxc in fit_regex:

        if settings.ec_fit['method'] in ['lsq','lsq_refine']:
            lsq_fit = least_squares(wrap_lsq,[4,2,.05,2],bounds=((0.,0.,0.,0.),(100.,100.,100.,100.)))
            parv = lsq_fit.x
            par,epsc,errors = get_errors(parv,multi=True)

        logfile = './eps_data/ec_fit_log_{:}.csv'.format(settings.fxc)
        ofl = open(logfile,'w+')

        if settings.ec_fit['method'] in ['filter','fixed','lsq_refine']:

            if settings.ec_fit['method'] == 'filter':
                step_l = [1.0,0.5,0.1,0.5,0.2,0.1,0.05,0.02,0.01]
                nstep = len(step_l)
                init_a_step = step_l[0]
                init_b_step = step_l[0]
                init_c_step = step_l[0]
                init_d_step = step_l[0]
            elif settings.ec_fit['method'] == 'fixed':
                step_l = [None]
                nstep = 1
                init_a_step = settings.a_step
                init_b_step = settings.b_step
                init_c_step = settings.c_step
                init_d_step = settings.d_step

            if settings.ec_fit['method'] == 'lsq_refine':
                best_res = errors['res']
                ofl.write('Iteration, A, B, C, D, Res.\n')
                ofl.write(('LSQ, {:}, {:}, {:}, {:} \n').format(*parv,lsq_fit.cost))
                #step_l = [.1,.05,.02,.01,.005,.002,.001]
                nstep = 4
                a_l = np.linspace(max(settings.a_min,parv[0]*.9),parv[0]*1.1,4)
                b_l = np.linspace(max(settings.b_min,parv[1]*.9),parv[1]*1.1,4)
                c_l = np.linspace(max(settings.c_min,parv[2]*.9),parv[2]*1.1,4)
                d_l = np.linspace(max(settings.d_min,parv[3]*.9),parv[3]*1.1,4)
            else:
                par = {}
                epsc = {}
                errors = {}
                best_res = 1e20
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

            for iastep in range(nstep):

                work_l = product(a_l,b_l,c_l,d_l)

                for vec in work_l:
                    tmp_par,tmp_ec,tmp_err = get_errors(vec,multi=(settings.nproc > 1))
                    if tmp_err['res'] < best_res:
                        par = tmp_par
                        epsc = tmp_ec
                        errors = tmp_err
                        best_res = tmp_err['res']

                if settings.ec_fit['method'] in ['filter','lsq_refine']:
                    a_c = par['a']
                    b_c = par['b']
                    c_c = par['c']
                    d_c = par['d']
                    ofl.write(('{:}, {:}, {:}, {:}, {:}, {:} \n').format(iastep,a_c,b_c,c_c,d_c,best_res))
                    if iastep == nstep-1:
                        ofl.write('==================\n')

                    if settings.ec_fit['method'] == 'filter' and iastep < nstep-1:
                        astep = step_l[iastep]
                        a_l = np.arange(max(settings.a_min,a_c-2*astep),a_c + 2*astep+min(step_l),astep)
                        b_l = np.arange(max(settings.b_min,b_c-2*astep),b_c + 2*astep+min(step_l),astep)
                        if settings.fit_c:
                            c_l = np.arange(max(settings.c_min,c_c-2*astep),c_c + 2*astep+min(step_l),astep)
                        else:
                            c_l = np.zeros(1)
                        if settings.fit_d:
                            d_l = np.arange(max(settings.d_min,d_c-2*astep),d_c + 2*astep+min(step_l),astep)
                        else:
                            d_l = np.zeros(1)
                    elif settings.ec_fit['method'] == 'lsq_refine' and iastep < nstep-1:
                        a_l = np.linspace(max(settings.a_min,a_c*.9),a_c*1.1,4)
                        b_l = np.linspace(max(settings.b_min,b_c*.9),b_c*1.1,4)
                        c_l = np.linspace(max(settings.c_min,c_c*.9),c_c*1.1,4)
                        d_l = np.linspace(max(settings.d_min,d_c*.9),d_c*1.1,4)

        ostr = 'fitted parameters:\n'
        ostr += ('{:}, '*len(par)).format(par['a'],par['b'],par['c'],par['d']) + '\n'
        ostr += 'Residual {:}\n'.format(errors['res'])
        ostr += 'Method {:}\n'.format(settings.ec_fit['method'])
        ostr += '==================\n'
        ostr += 'rs, eps_c (hartree/electron), Abs. rel. error\n'
        for rs in epsc:
            ostr += '{:}, {:}, {:}\n'.format(rs,epsc[rs],errors[rs])
        ofl.write(ostr)
        ofl.close()

        return par,epsc,errors

    else:
        return get_errors({'a':None, 'b': None, 'c':None, 'd':None})

def min_func(p):
    _,_,errors=get_errors((p[0],p[1],p[2],p[3]),multi=True)
    return errors['res']

if __name__ == "__main__":
    """
    plot_TC((settings.TC_pars['a'],settings.TC_pars['b'],settings.TC_pars['c']))
    exit()
    """
    pars = settings.TC_pars
    tps = (pars['a'],pars['b'],pars['c'],pars['d'])
    """
    p0 = [pars['a'],pars['b'],pars['c'],pars['d']]
    x = minimize(min_func,p0)
    print(x)
    exit()
    """
    par,epsc,errors=get_errors(tps)
    #print(errors['res'])
    plot_TC(tps)

    exit()

    par,epsc,errors=ec_fitting()
    print(par)
    print(errors['res'])
    #print(epsc)
    for rs in epsc:
        print(rs,epsc[rs],errors[rs])
