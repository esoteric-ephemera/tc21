import numpy as np
from itertools import product
import multiprocessing as mp
import matplotlib.pyplot as plt

pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286198
gam = 1.311028777146059809410871821455657482147216796875

x,kk = np.transpose(np.genfromtxt('./test_fits/kram_kron_re_fxc.csv',delimiter=',',skip_header=1))

def error_meas(y,yref):
    return np.sum((y - yref)**2)**(0.5)/(1.0*y.shape[0])

def test_fn(a,b,c,d):
    #num = 1.0/gam*(1.0 + c*x - (a/gam)*x**2)
    #denom = (1.0 + b*x**(3.0/2.0) + (a/gam)**(4.0/7.0)*x**2)**(7.0/4.0)
    #num = 1.0/gam*(1.0 + b*x**c - a*x)
    #denom = (1.0  + (a/gam)**(4.0/5.0)*x**2)**(5.0/4.0)
    powr = 7.0/(2*c)
    num = 1.0/gam*(1.0 - a*x**2)
    denom = (1.0 + b*x**2 + (a/gam)**(1.0/powr)*x**c)**powr
    return num/denom

def wrap_err(var):
    a,b,c,d = var
    ty = test_fn(a,b,c,d)
    return error_meas(ty,kk)

def kramers_kronig_plot(pars=None):
    if pars is not None:
        mod = test_fn(pars[0],pars[1],pars[2],pars[3])
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
    a_bds = [0.0,2.0]
    b_bds = [0.0,2.0]
    c_bds = [0.0,1.0]

    fling = 10
    step_l = [0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001] # steps in naive grid search

    for istep, step in enumerate(step_l):

        if istep == 0: # for the initial, wide search, use special bounds
            a_min = step
            a_max = 1.5 + 0.5*step
            b_min = step
            b_max = 1.0 + 0.5*step
            c_min = step
            c_max = 1.0
        else:
            a_min = max([a_bds[0],aa - fling*step_l[istep-1]]) # impose restrictions on bounds for coeffs
            a_max = min([a_bds[1],aa + fling*step_l[istep-1]]) # while expanding search region within some

            b_min = max([b_bds[0],bb - fling*step_l[istep-1]]) # basin of previous best parameters
            b_max = min([b_bds[1],bb + fling*step_l[istep-1]])

            c_min = max([c_bds[0],cc - fling*step_l[istep-1]])
            c_max = min([c_bds[1],cc + fling*step_l[istep-1]])

        a_l = np.arange(a_min,a_max,step)
        b_l = np.arange(b_min,b_max,step)
        c_l = np.arange(c_min,c_max,step)
        d_l = np.zeros(1)#np.arange(0.01,1.02,0.01)
        tlist = product(a_l,b_l,c_l,d_l)

        pool = mp.Pool(processes=6)
        tout = pool.map(wrap_err,tlist)
        pool.close()
        ind = np.argmin(np.asarray(tout))
        nlist = list(product(a_l,b_l,c_l,d_l))
        aa,bb,cc,dd=nlist[ind]

    logfile = './fitting/hx_fit_log.csv'
    ostr = 'a param, b param, c param, d param\n'
    ostr += '{:}, {:}, {:}, {:}\n'.format(aa,bb,cc,dd)
    ostr += 'error {:} \n'.format(np.asarray(tout)[ind])
    with open(logfile,'w+') as logf:
        logf.write(ostr)

    np.savetxt('./test_fits/new_hx.csv',np.transpose((x,kk,test_fn(aa,bb,cc,dd))),delimiter=',',header='x, K-K, new h(x)',fmt='%.18f')
    return aa,bb,cc,dd

if __name__ == "__main__":

    mod = test_fn(0.1756,1.0376,3.0,None)
    np.savetxt('./test_fits/new_hx.csv',np.transpose((x,kk,mod)),delimiter=',',header='x, K-K, new h(x)',fmt='%.18f')

    kramers_kronig_plot()
    exit()

    #abs_best = {'a': 0.1756, 'b': 1.0376, 'c': 2.9787} # error 2.887 x 10**(-5)
    abs_best = {'a': 0.1756, 'b': 1.0376, 'c': 3.0} # error 3.673 x 10**(-5)
    #print(wrap_err((abs_best['a'],abs_best['b'],abs_best['c'],None)))
    #exit()
