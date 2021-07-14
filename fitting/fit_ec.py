import numpy as np
import multiprocessing as mp
from itertools import product
from os import path,fsync
from scipy.optimize import least_squares,minimize


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
    if method == 'uniform': # even weighting, quick return
        err = {}
        err['res'] = 0.0
        for ielt,elt in enumerate(vals):
            err[elt] = abs(ref[elt] - vals[elt])
            err['res'] += err[elt]
        return err
    elif method == 'sqrt': # w(x) = x^(1/2)
        wg = np.zeros(len(ref))
        for ielt,elt in enumerate(vals):
            wg[ielt] = float(elt)**(0.5)
    elif method == 'lin': # w(x) = x
        wg = np.zeros(len(ref))
        for ielt,elt in enumerate(vals):
            wg[ielt] = float(elt)
    elif method == 'step': # w(x < 10) = 1, w(x >= 10) = 10
        wg = np.ones(len(ref))
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
    # now using fortran libraries for correlation energy
    if settings.eps_c_flib:
        eps_c_plots(use_flib=True)
    else:
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

def get_scalar_error(pars):
    par_d = {}
    par_d['a'],par_d['b'],par_d['c'],par_d['d'] = pars
    ecd = eps_quick('user',pars=par_d,inps=fgrid,rs_l=settings.rs_list)
    err = residuals(ecd,ec_ref,method='uniform')
    return err['res']

def wrap_lsq(pars):
    if settings.fit_c:
        if settings.fit_d:
            wpars = pars
        else:
            wpars = (pars[0],pars[1],pars[2],0.0)
    else:
        wpars = (pars[0],pars[1],0.0,0.0)
    _,_,errd = get_errors(wpars,multi=True)
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

            a_init_l = [settings.a_min]
            b_init_l = [settings.b_min]

            if settings.fit_c:
                c_init_l = [settings.c_min]
                if settings.fit_d:
                    d_init_l = [settings.d_min]
                    work_list = product(a_init_l,b_init_l,c_init_l,d_init_l)
                    bds_list = ((0.,0.,0.,0.),(10.,10.,10.,10.))
                else:
                    work_list = product(a_init_l,b_init_l,c_init_l)
                    bds_list = ((0.,0.,0.),(10.,10.,10.))
            else:
                work_list = product(a_init_l,b_init_l)
                bds_list = ((0.,0.),(10.,10.))

            best_lsq = 1e20
            # this should help reduce the sensitivity of the least squares fit
            # to the starting guess
            for ipars in work_list:
                init_set = [apar for apar in ipars]
                lsq_fit = minimize(get_scalar_error,init_set,method='Nelder-Mead')
                #lsq_fit = least_squares(wrap_lsq,init_set,bounds=bds_list)
                par_tmp = lsq_fit.x
                if len(par_tmp) < 4:
                    for iapp in range(4 - len(par_tmp)):
                        par_tmp.append(0.0)

                par_d_tmp,epsc_tmp,errors_tmp = get_errors(par_tmp,multi=True)
                if errors_tmp['res'] < best_lsq:
                    best_lsq = errors_tmp['res']
                    par = par_d_tmp
                    parv = [apar for apar in par_tmp]
                    epsc = epsc_tmp
                    errors = errors_tmp

        logfile = './fitting/ec_fit_log_{:}.csv'.format(settings.fxc)
        ofl = open(logfile,'w+')

        nstall = 0
        if settings.ec_fit['method'] in ['filter','fixed','lsq_refine']:

            if settings.ec_fit['method'] == 'filter':
                ofl.write('Iteration, A, B, C, D, Res.\n')
                nstep = 50
                a_l = np.linspace(settings.a_min,settings.a_max,5)
                b_l = np.arange(settings.b_min,settings.b_max,5)
                if settings.fit_c:
                    c_l = np.linspace(settings.c_min,settings.c_max,5)
                else:
                    c_l = np.ones(1)
                if settings.fit_d:
                    d_l = np.linspace(settings.d_min,settings.d_max,5)
                else:
                    d_l = np.ones(1)

            elif settings.ec_fit['method'] == 'fixed':
                step_l = [None]
                nstep = 1
                a_l = np.arange(settings.a_min,settings.a_max,settings.a_step)
                b_l = np.arange(settings.b_min,settings.b_max,settings.b_step)
                if settings.fit_c:
                    c_l = np.arange(settings.c_min,settings.c_max,settings.c_step)
                else:
                    c_l = np.zeros(1)
                if settings.fit_d:
                    d_l = np.arange(settings.d_min,settings.d_max,settings.d_step)
                else:
                    c_l = np.zeros(1)

            if settings.ec_fit['method'] == 'lsq_refine':
                best_res = errors['res']
                old_best_res = best_res
                ofl.write('Iteration, A, B, C, D, Res.\n')
                ofl.write(('LSQ, {:}, {:}, {:}, {:}, {:} \n').format(*parv,errors['res']))
                ofl.flush()
                fsync(ofl.fileno())
                #step_l = [.1,.05,.02,.01,.005,.002,.001]
                nstep = 50
                a_l = np.linspace(parv[0]*.9,parv[0]*1.1,4)
                b_l = np.linspace(parv[1]*.9,parv[1]*1.1,4)
                if settings.fit_c:
                    c_l = np.linspace(parv[2]*.9,parv[2]*1.1,4)
                else:
                    c_l = np.zeros(1)
                if settings.fit_d:
                    d_l = np.linspace(parv[3]*.9,parv[3]*1.1,4)
                else:
                    d_l = np.zeros(1)
            else:
                par = {}
                epsc = {}
                errors = {}
                best_res = 1e20
                old_best_res = best_res

            for iastep in range(nstep):

                work_l = product(a_l,b_l,c_l,d_l)

                if settings.nproc > 1:
                    pool = mp.Pool(processes=settings.nproc)
                    tout = pool.map(get_scalar_error,work_l)
                    pool.close()
                    nbetter = 0
                    for itmp,tmp in enumerate(tout):
                        if tmp < best_res:
                            best_res = tmp
                            ibest = itmp
                            nbetter += 1
                    if nbetter > 0:
                        parv = list(product(a_l,b_l,c_l,d_l))[ibest]
                        par,epsc,errors = get_errors(parv,multi=True)
                else:
                    for vec in work_l:
                        tmp_par,tmp_ec,tmp_err = get_errors(vec,multi=(settings.nproc > 1))
                        if tmp_err['res'] < best_res:
                            par = tmp_par
                            epsc = tmp_ec
                            errors = tmp_err
                            best_res = tmp_err['res']

                if settings.ec_fit['method'] in ['filter','lsq_refine']:
                    if iastep > 1:
                        da = abs(a_c - par['a'])/abs(a_c)
                        db = abs(b_c - par['b'])/abs(b_c)
                        dc = abs(c_c - par['c'])/abs(c_c)
                        dd = abs(d_c - par['d'])/abs(d_c)
                        dres = abs(old_best_res-best_res)/abs(old_best_res)
                    a_c = par['a']
                    b_c = par['b']
                    c_c = par['c']
                    d_c = par['d']

                    ofl.write(('{:}, {:}, {:}, {:}, {:}, {:} \n').format(iastep,a_c,b_c,c_c,d_c,best_res))
                    ofl.flush()
                    fsync(ofl.fileno())
                    if iastep == nstep-1:
                        ofl.write('==================\n')

                    if settings.ec_fit['method'] in ['filter','lsq_refine'] and iastep < nstep-1:

                        if iastep > 1:
                            if np.all(np.array([da,db,dc,dres]) < 1.e-6):
                                nstall += 1
                            else:
                                nstall = 0
                            if nstall > 4 and adj <= .1:
                                # if refinement isn't really imroving things, stop and return
                                break
                        old_best_res = best_res

                        adj = 0.75**(iastep+1)
                        decr = 1 - adj
                        incr = 1 + adj
                        a_l = np.linspace(a_c*decr,a_c*incr,4)
                        b_l = np.linspace(b_c*decr,b_c*incr,4)
                        if settings.fit_c:
                            c_l = np.linspace(c_c*decr,c_c*incr,4)
                        else:
                            c_l = np.zeros(1)
                        if settings.fit_d:
                            d_l = np.linspace(d_c*decr,d_c*incr,4)
                        else:
                            d_l = np.zeros(1)

        ofl.write('==================\n')
        ostr = 'fitted parameters:\n'
        ostr += ('{:}, '*len(par)).format(par['a'],par['b'],par['c'],par['d']) + '\n'
        ostr += 'Residual {:}\n'.format(errors['res'])
        ostr += 'Method {:}\n'.format(settings.ec_fit['method'])
        ostr += '==================\n'
        ostr += 'Initialization:\n'
        ostr += 'A: min = {:}, max = {:}, step = {:}\n'.format(settings.a_min,settings.a_max,settings.a_step)
        ostr += 'B: min = {:}, max = {:}, step = {:}\n'.format(settings.b_min,settings.b_max,settings.b_step)
        if settings.fit_c:
            ostr += 'C: min = {:}, max = {:}, step = {:}\n'.format(settings.c_min,settings.c_max,settings.c_step)
        if settings.fit_d:
            ostr += 'D: min = {:}, max = {:}, step = {:}\n'.format(settings.d_min,settings.d_max,settings.d_step)
        ostr += 'z_pts, lambda_pts, u_pts\n {:}, {:}, {:} \n'.format(settings.z_pts,settings.lambda_pts,settings.u_pts)
        ostr += '==================\n'
        trunc_parv = [ round(par[apar],6) for apar in par ]
        trunc_par,trunc_epsc,trunc_errors = get_errors(trunc_parv,multi=True)
        ostr += ('{:}, '*len(par)).format(trunc_par['a'],trunc_par['b'],trunc_par['c'],trunc_par['d']) + '\n'
        ostr += 'Residual (rounded) {:}, abs diff = {:} \n'.format(trunc_errors['res'],abs(trunc_errors['res']-errors['res']))
        ostr += '==================\n'
        ostr += 'rs, eps_c (hartree/electron), Abs. rel. error (using rounded params)\n'
        for rs in epsc:
            ostr += '{:}, {:}, {:}\n'.format(rs,trunc_epsc[rs],trunc_errors[rs])
        ofl.write(ostr)
        ofl.close()

        return trunc_par,trunc_epsc,trunc_errors

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
