import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from os import path,system
from scipy.optimize import minimize_scalar
from utilities.roots import bisect,bracket

import settings
from dft.mcp07 import mcp07_dynamic
from dft.chi import chi_parser

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

    qTC = False
    wLDA = settings.LDA
    if settings.fxc == 'MCP07':
        wLDA = 'PZ81'
    elif settings.fxc == 'TC':
        qTC = True
    else:
        raise SystemExit('Unsupported kernel for plotting,',settings.fxc)
    return mcp07_dynamic(q,freq,dvars,axis='real',revised=qTC,pars=settings.TC_pars,param=wLDA)

def get_crossing(q_init,freq,rs):

    kf = (9*pi/4)**(1/3)/rs
    def min_func(aq,ret_real=True):
        fxc = wrap_kernel(aq,freq,rs)
        veff_re = 4*pi/aq**2 + fxc.real
        if ret_real:
            return veff_re#abs(veff_re)/(4*pi/aq**2)
        else:
            return veff_re + 1.j*fxc.imag
    qbds = bracket(min_func,(0.01,4*kf),vector=True)
    qmin,res = bisect(min_func,(qbds[0][0],qbds[0][1]),tol=1.e-8)
    return qmin,min_func(qmin,ret_real=False)

def fxc_plotter(rs):
    wp = (3.0/rs**3)**(0.5)
    kf = (9.0*pi/4.0)**(1.0/3.0)/rs
    freq_l=[0.0,wp,4*wp]

    # x_l is q/kF
    #if rs < 10:
    x_l = np.arange(0.01,4.01,0.01)
    #else:
    #    x_l = np.arange(0.001,1.01,0.001)
    q_l = kf*x_l

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
            lbl = '=4\omega_p(0)$'
        ax[0].plot(x_l,kf**2*fxc.real,color=clist[iw],linewidth=2.5,label='$\omega'+lbl)
        ax[1].plot(x_l,kf**2*fxc.imag,color=clist[iw],linewidth=2.5,label='$\omega'+lbl)
    ax[0].legend(fontsize=16)
    for i in range(2):
        ax[i].tick_params(axis='both',labelsize=20)
        ax[i].set_xlim([0.0,x_l[-1]])
        ax[i].set_xlabel('$q/k_{\mathrm{F}}$',fontsize=24)
    ax[0].set_ylabel('$k_{\mathrm{F}}^2 ~\mathrm{Re}~ f_{\mathrm{xc}}(q,\omega)$',fontsize=24)
    ax[1].set_ylabel('$k_{\mathrm{F}}^2 ~\mathrm{Im}~ f_{\mathrm{xc}}(q,\omega)$',fontsize=24)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if settings.fxc == 'TC':
        fig.suptitle('rMCP07 kernel, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    else:
        fig.suptitle(settings.fxc+' kernel, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    #plt.show()
    plt.savefig('./figs/kernel_plots/'+settings.fxc+'_rs_'+str(rs)+'.pdf',dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()
    fig,ax = plt.subplots(figsize=(8,6))
    print('-------------')
    print('rs='+str(rs))
    print('q_min/kF, omega, v_eff(q_min,omega)')
    for iw,w in enumerate(freq_l):
        if iw == 0:
            lbl = '= 0$'
        elif iw == 1:
            lbl = '=\omega_p(0)$'
        elif iw == 2:
            lbl = '=4\omega_p(0)$'
        ax.plot(x_l,(v_eff[iw]/v_bare).real,linewidth=2.5,label='$\omega'+lbl,color=clist[iw])
        ax.plot(x_l,(v_eff[iw]/v_bare).imag,linewidth=2.5,linestyle='--',color=clist[iw])
        minind = np.argmin(np.abs(v_eff[iw].real))
        q_min_init = q_l[minind]
        qmin,vcross = get_crossing(q_min_init,w,rs)
        print(qmin/kf,w,vcross)

    ax.legend(fontsize=20)
    #for i in range(2):
    ax.set_xlim([0.0,x_l[-1]])
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('$q/k_{\mathrm{F}}$',fontsize=24)
    ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1],linewidth=2.5,linestyle='--',color='gray')
    ax.set_ylabel('$v_{\\mathrm{eff}}(q,\omega)/v_{\\mathrm{bare}}(q)$',fontsize=24)
    #ax.set_ylabel('$\mathrm{Re}\left[\\frac{v_{\\mathrm{eff}}(q,\omega)}{v_{\\mathrm{bare}}(q)}\\right]$',fontsize=16)
    #ax.set_ylabel('$\mathrm{Im}\left[\\frac{v_{\\mathrm{eff}}(q,\omega)}{v_{\\mathrm{bare}}(q)}\\right]$',fontsize=16)
    #plt.tight_layout()
    #fig.subplots_adjust(top=0.9)
    """
    if settings.fxc == 'TC':
        plt.title('rMCP07 dressed potential, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    else:
        plt.title(settings.fxc+' dressed potential, bulk jellium $r_{\\mathrm{s}}=$'+str(rs),fontsize=24)
    """
    flbl = settings.fxc
    if settings.fxc == 'TC':
        flbl = 'rMCP07'

    plt.title(flbl+', $r_{\\mathrm{s}}=$'+str(rs)+' jellium',fontsize=20)
    ax.tick_params(axis='both',labelsize=20)
    #plt.show()
    plt.savefig('./figs/kernel_plots/veff_'+settings.fxc+'_rs_'+str(rs)+'.pdf',dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    fig,ax = plt.subplots(1,2,figsize=(14,6))


    for iw,w in enumerate(freq_l):
        chi0 = chi_parser(x_l/2,w+1.e-12j*wp,1.0,rs,'chi0',reduce_omega=False)
        epst = 1 - v_eff[iw]*chi0
        epst_rpa = 1 - v_bare*chi0
        if iw == 0:
            lbl = '= 0$'
        elif iw == 1:
            lbl = '=\omega_p(0)$'
        elif iw == 2:
            lbl = '=4\omega_p(0)$'
        ax[1].plot(x_l,epst.real,linewidth=2.5,color=clist[iw])
        ax[1].plot(x_l,epst.imag,linewidth=2.5,linestyle='--',color=clist[iw])
        ax[0].plot(x_l,epst_rpa.real,linewidth=2.5,label='$\omega'+lbl,color=clist[iw])
        ax[0].plot(x_l,epst_rpa.imag,linewidth=2.5,linestyle='--',color=clist[iw])

    for i in range(2):
        ax[i].set_ylim([-2,5])
        ax[i].set_xlim([0.0,x_l[-1]])
        ax[i].xaxis.set_minor_locator(MultipleLocator(0.25))
        ax[i].xaxis.set_major_locator(MultipleLocator(1))
        ax[i].yaxis.set_minor_locator(MultipleLocator(0.5))
        ax[i].yaxis.set_major_locator(MultipleLocator(2))
        ax[i].set_xlabel('$q/k_{\mathrm{F}}$',fontsize=24)
        #ax[i].hlines(1.0,ax[i].get_xlim()[0],ax[i].get_xlim()[1],linewidth=2.5,linestyle='--',color='gray')

        plt.suptitle('$r_{\\mathrm{s}}=$'+str(rs)+' jellium',fontsize=20)
        ax[i].tick_params(axis='both',labelsize=20)

    flbl = settings.fxc
    if settings.fxc == 'TC':
        flbl = 'rMCP07'
    ax[0].annotate('RPA',(0.85*x_l[-1],-1.95),fontsize=20)
    ax[1].annotate(flbl,((0.85-0.05*(len(flbl)-4))*x_l[-1],-1.95),fontsize=20)
    ax[0].set_ylabel('$\\widetilde{\\epsilon}(q,\omega)$',fontsize=24)
    #ax[1].set_ylabel('$\\widetilde{\\epsilon}^{\\mathrm{'+flbl+'}}(q,\omega)$',fontsize=24)
    #ax[0].set_ylabel('$\\widetilde{\\epsilon}^{\\mathrm{RPA}}(q,\omega)$',fontsize=24)
    ax[0].legend(fontsize=20)

    plt.savefig('./figs/kernel_plots/eps_tilde_'+settings.fxc+'_rs_'+str(rs)+'.pdf',dpi=600,bbox_inches='tight')
    #plt.show()


    return

if __name__ == "__main__":

    for rs in settings.rs_list:
        fxc_plotter(rs)
