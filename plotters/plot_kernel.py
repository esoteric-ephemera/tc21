import numpy as np
import matplotlib.pyplot as plt
from os import path,system
from scipy.optimize import minimize_scalar

import settings
from dft.mcp07 import mcp07_dynamic

clist = settings.clist
#clist=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
pi = settings.pi

if not path.isdir('./figs/kernel_plots'):
    system('mkdir -p ./figs/kernel_plots')

def wrap_kernel(q,freq,rs):
    dvars = {}
    dvars['rs'] = rs
    dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dvars['n'] = 3.0/(4.0*pi*rs**3)
    dvars['rsh'] = dvars['rs']**(0.5)

    if settings.fxc == 'MCP07':
        wLDA = 'PZ81'
        qTC = False
    elif settings.fxc == 'TC':
        wLDA = settings.LDA
        qTC = True
    return mcp07_dynamic(q,freq,dvars,axis='real',revised=qTC,pars=settings.TC_pars,param=wLDA)

def get_crossing(q_init,freq,rs):

    def min_func(aq,ret_real=True):
        fxc = wrap_kernel(aq,freq,rs)[0]
        veff_re = 4*pi/aq**2 + fxc.real
        if ret_real:
            return abs(veff_re)/(4*pi/aq**2)
        else:
            return veff_re + 1.j*fxc.imag

    v0 = min_func(q_init,ret_real=False).real
    q0 = q_init
    for step in [1.e-2/10.0**j for j in range(8)]:
        v0_sgn = np.sign(v0.real)
        for aq in [q0 + v0_sgn*step*i for i in range(200)]:
            vnew = min_func(aq,ret_real=False).real
            if np.sign(vnew)*v0_sgn < 0.0:
                v0 = vnew
                q0 = aq
                break
    return aq,min_func(aq,ret_real=False)

    q_l = np.arange(q_init,q)

    lbd = max(q_init-.1,0.001)
    res = minimize_scalar(min_func,bounds=(lbd,q_init+.1),method='bounded')
    qmin = res.x
    veff_cross = min_func(qmin,ret_real=False)
    return qmin,veff_cross

def fxc_plotter(rs):
    wp = (3.0/rs)**(0.5)
    kf = (9.0*pi/4.0)**(1.0/3.0)/rs
    freq_l=[0.0,wp,1e20]

    if rs < 10:
        q_l = np.arange(0.01,4.01,0.01)
    else:
        q_l = np.arange(0.001,1.01,0.001)

    v_bare = 4*pi/q_l**2
    v_eff = np.zeros((len(freq_l),v_bare.shape[0]),dtype='complex')

    fig,ax = plt.subplots(1,2,figsize=(12,6))
    for iw,w in enumerate(freq_l):
        fxc = wrap_kernel(q_l,w,rs)
        v_eff[iw] = v_bare + fxc
        if iw == 0:
            lbl = '= 0$'
        elif iw == 1:
            lbl = '=\omega_p(0)$'
        elif iw == 2:
            lbl = '\\to \infty$'
        ax[0].plot(q_l,kf**2*fxc.real,color=clist[iw],linewidth=2.5,label='$\omega'+lbl)
        ax[1].plot(q_l,kf**2*fxc.imag,color=clist[iw],linewidth=2.5,label='$\omega'+lbl)
    ax[0].legend(fontsize=16)
    for i in range(2):
        ax[i].tick_params(axis='both',labelsize=20)
        ax[i].set_xlim([0.0,q_l[-1]])
        ax[i].set_xlabel('$q/k_{\mathrm{F}}$',fontsize=24)
    ax[0].set_ylabel('$k_{\mathrm{F}}^2 ~\mathrm{Re}~ f_{\mathrm{xc}}(q,\omega)$',fontsize=24)
    ax[1].set_ylabel('$k_{\mathrm{F}}^2 ~\mathrm{Im}~ f_{\mathrm{xc}}(q,\omega)$',fontsize=24)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if settings.fxc == 'TC':
        fig.suptitle('TC21 kernel, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    else:
        fig.suptitle(settings.fxc+' kernel, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    #plt.show()
    plt.savefig('./figs/kernel_plots/'+settings.fxc+'_rs_'+str(rs)+'.pdf',dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    fig,ax = plt.subplots(figsize=(8,6))
    print('-------------')
    print('rs='+str(rs))
    print('q_min, omega, v_eff(q_min,omega)')
    for iw,w in enumerate(freq_l):
        if iw == 0:
            lbl = '= 0$'
        elif iw == 1:
            lbl = '=\omega_p(0)$'
        elif iw == 2:
            lbl = '\\to \infty$'
        ax.plot(q_l,(v_eff[iw]/v_bare).real,linewidth=2.5,label='$\omega'+lbl,color=clist[iw])
        ax.plot(q_l,(v_eff[iw]/v_bare).imag,linewidth=2.5,linestyle='--',color=clist[iw])
        minind = np.argmin(np.abs(v_eff[iw].real))
        q_min_init = q_l[minind]
        qmin,vcross = get_crossing(q_min_init,w,rs)
        print(qmin,w,vcross)

    ax.legend(fontsize=20)
    #for i in range(2):
    ax.set_xlim([0.0,q_l[-1]])
    ax.set_xlabel('$q/k_{\mathrm{F}}$',fontsize=24)
    ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1],linewidth=2.5,linestyle='--',color='gray')
    ax.set_ylabel('$v_{\\mathrm{eff}}(q,\omega)/v_{\\mathrm{bare}}(q)$',fontsize=24)
    #ax.set_ylabel('$\mathrm{Re}\left[\\frac{v_{\\mathrm{eff}}(q,\omega)}{v_{\\mathrm{bare}}(q)}\\right]$',fontsize=16)
    #ax.set_ylabel('$\mathrm{Im}\left[\\frac{v_{\\mathrm{eff}}(q,\omega)}{v_{\\mathrm{bare}}(q)}\\right]$',fontsize=16)
    #plt.tight_layout()
    #fig.subplots_adjust(top=0.9)
    """
    if settings.fxc == 'TC':
        plt.title('TC21 dressed potential, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    else:
        plt.title(settings.fxc+' dressed potential, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    """
    plt.title('$r_{\\mathrm{s}}=$'+str(rs)+' jellium',fontsize=20)
    ax.tick_params(axis='both',labelsize=20)
    #plt.show()
    plt.savefig('./figs/kernel_plots/veff_'+settings.fxc+'_rs_'+str(rs)+'.pdf',dpi=600,bbox_inches='tight')

    return

if __name__ == "__main__":

    for rs in settings.rs_list:
        fxc_plotter(rs)
