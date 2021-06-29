import numpy as np
from os.path import isfile
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from dft.chi import chi_parser
from dft.mcp07 import mcp07_static
from dft.gki_fxc import exact_constraints,gki_dynamic_real_freq,gki_real_freq_taylor_series
from utilities.gauss_quad import gauss_quad
from utilities.roots import newton_raphson_2d
import settings

pi = settings.pi

def density_variables(rs):
    dv = {}
    dv['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dv['rs'] = rs
    dv['n'] = 3.0/(4.0*pi*dv['rs']**3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/rs**3)**(0.5)
    return dv


def make_grid(def_pts=200,cut_pt=2.0):

    gfile = './grids/gauss_legendre_{:}_pts.csv'
    if isfile(gfile):
        wg,grid = np.transpose(np.genfromtxt(gfile,delimiter=',',skip_header=1))
    else:
        wg,grid = gauss_quad(def_pts,grid_type='legendre')

    # first step, shift grid and weights from (-1, 1) to (0, 1)
    sgrid = 0.5*(grid + 1)
    swg = 0.5*wg

    """
        ogrid integrates from 0 to cut_pt, then extrapolates from cut_pt to infinity using
        int_0^inf f(x) dx = int_0^x_c f(x) dx + int_0^(1/x_c) f(1/u)/u**2 du,
        with u = 1/x. Then x_c = cut_pt, which by default = 2.
    """
    ogrid = cut_pt*sgrid
    owg = cut_pt*swg
    oextrap = sgrid/cut_pt
    owg_extrap = swg/(cut_pt*oextrap**2)
    ogrid = np.concatenate((ogrid,1/oextrap))
    owg = np.concatenate((owg,owg_extrap))

    return ogrid,owg#,sgrid,swg

def plasmon_dispersion_single(x_l,rs):

    dvs = density_variables(rs)

    wp = np.zeros(x_l.shape[0],dtype='complex')

    #dinf,dinfwg = make_grid(def_pts=200,cut_pt=50*dvs['wp0'])

    def teps(omega,x):
        q = x*dvs['kF']
        to = omega[0]+1.j*omega[1]

        chi0 = chi_parser(x/2,to,0.0,rs,'chi0',reduce_omega=False,imag_freq=False,ret_eps=False,LDA=settings.LDA)
        do_rev = False
        wlda = 'PZ81'
        if settings.fxc == 'TC':
            do_rev = True
            wlda = settings.LDA

        if do_rev:
            fp = settings.TC_pars
            kscr = fp['a']*dvs['kF']/(1.0 + fp['b']*dvs['kF']**(0.5))
            F1 = fp['c']*dvs['rs']**2
            F2 = F1 + (1.0 - F1)*np.exp(-fp['d']*(q/kscr)**2)
            fxcu = gki_real_freq_taylor_series(dvs,F2*to,F2*to.real,revised=do_rev,param=wlda)
            #fxc_dinf = gki_dynamic_real_freq(dvs,F2*dinf,x_only=False,revised=do_rev,param=wlda,dimensionless=False)
        else:
            fxcu = gki_real_freq_taylor_series(dvs,to,to.real,revised=do_rev,param=wlda)
            #fxc_dinf = gki_dynamic_real_freq(dvs,dinf,x_only=False,revised=do_rev,param=wlda,dimensionless=False)
        """
        _,finf = exact_constraints(dvs,x_only=False,param=wlda)
        #fxc_dinf = mcp07_dynamic(x*dvs['kF'],dinf,dvs,axis='real',revised=do_rev,pars=settings.TC_pars,param=wlda)
        rfreq = dinf - to.real
        rintd = ( to.imag*(fxc_dinf.real-finf) + rfreq*fxc_dinf.imag)/(rfreq**2 + to.imag**2)
        iintd = (-rfreq*(fxc_dinf.real-finf) + to.imag*fxc_dinf.imag)/(rfreq**2 + to.imag**2)
        fxcu = finf + np.sum(dinfwg*(rintd+1.j*iintd))/(2*pi)
        """
        fxc_q,f0,akn = mcp07_static(q,dvs,param=wlda)
        if do_rev:
            fxc = (1.0 + np.exp(-(q/kscr)**2)*(fxcu/f0 - 1.0))*fxc_q
        else:
            fxc = (1.0 + np.exp(-akn*q**2)*(fxcu/f0 - 1.0))*fxc_q

        vb = 4*pi/q**2
        epst = 1 - (vb + fxc)*chi0

        return epst

    def abseps(omega,x):
        return np.abs(teps(omega,x))

    brkt = np.array([dvs['wp0'],0.0])

    for ix,x in enumerate(x_l):
        def wrap_nr_2d(omega):
            ttt = teps(omega,x)
            return np.asarray([ttt.real,ttt.imag])
        ow,suc = newton_raphson_2d(wrap_nr_2d,brkt,tol=1.e-6,maxstep=500,h=1.e-6,jacobian=False)

        #wc = ow[0]
        eqc = (0.5*x**2 + x)*dvs['kF']**2

        if suc['code']==0 or abs(ow[0] - eqc) < 1.e-6:
            print(rs,x_l[ix])#,ow[0]/dvs['wp0'],eqc/dvs['wp0'])
            return x_l[:ix],wp[:ix]/dvs['wp0']

        wp[ix] = ow[0]+1.j*ow[1]
        brkt = [wp[ix].real,wp[ix].imag]

    return x_l,wp/dvs['wp0']

def plasmon_dispersion():

    x_l = np.linspace(0.01,2.0,500)

    lsls = ['-','--','-.',':']
    fig,ax = plt.subplots(1,2,figsize=(12,6))

    isl = 0
    if settings.fxc == 'MCP07':
        ax[0].set_ylim([0.6,1.6])
    elif settings.fxc == 'TC':
        ax[0].set_ylim([0.85,1.6])
    ax[1].set_ylim([-0.045,0.0])

    for irs,rs in enumerate(settings.rs_list):

        xn,wpdl_cmplx = plasmon_dispersion_single(x_l,rs)
        fname = './freq_data/wpq_disp_{:}_rs_{:}.csv'.format(settings.fxc,rs)
        np.savetxt(fname,np.transpose((xn,wpdl_cmplx.real,wpdl_cmplx.imag)),delimiter=',',header='q/kF, Re omega_p(q)/ omega_p(0), Im omega_p(q)/ omega_p(0)')
        wpdl = wpdl_cmplx.real
        wpdl_im = wpdl_cmplx.imag
        wind = int(np.ceil(0.7*len(xn)))
        xc = xn[wind]
        if abs(wpdl[wind])>3:
            wind = int(np.ceil(0.4*len(xn)))
            xc = xn[wind]
        if irs > len(settings.clist)-1 and irs%len(settings.clist)==0:
            isl += 1
        clr = settings.clist[irs%len(settings.clist)]
        ax[0].plot(xn,wpdl,linewidth=2.5,color=clr)
        ax[1].plot(xn,wpdl_im,linewidth=2.5,color=clr)
        if irs == len(settings.rs_list)-1:
            for i in range(2):
                ax[i].set_xlim([0,1.05*xn[-1]])

        pos = (xn[-1]-.2,1.01*wpdl[-1])
        if rs ==69:
            if settings.fxc == 'TC':
                pos = (xn[len(xn)//2]-.2,wpdl[len(xn)//2]-.07)
            elif settings.fxc == 'MCP07':
                pos = (xn[len(xn)//2]-.2,wpdl[len(xn)//2]-.1)
        """
        pos = (xc,wpdl[wind]*0.95)
        if rs ==69:
            pos = (xc,wpdl[wind]*0.85)
        elif rs == 30:
            pos = (xc,wpdl[wind]*0.9)
        """
        ax[0].annotate('$r_{\\mathrm{s}}='+str(rs)+'$',pos,color=clr,fontsize=20)

    ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].yaxis.set_major_locator(MultipleLocator(.2))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.005))
    ax[1].yaxis.set_major_locator(MultipleLocator(.01))

    ax[0].hlines(1.0,ax[0].get_xlim()[0],ax[0].get_xlim()[1],linewidth=1,linestyle='-',color='gray')
    for i in range(2):
        ax[i].xaxis.set_minor_locator(MultipleLocator(0.25))
        ax[i].xaxis.set_major_locator(MultipleLocator(0.5))
        ax[i].set_xlabel('$q/k_{\mathrm{F}}$',fontsize=24)
        ax[i].tick_params(axis='both',labelsize=20)
    ax[0].set_ylabel('$\\mathrm{Re}~\\omega_p(q)/\\omega_p(0)$',fontsize=24)
    ax[1].set_ylabel('$\\mathrm{Im}~\\omega_p(q)/\\omega_p(0)$',fontsize=24)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig('./figs/plasmon_dispersion_'+settings.fxc+'.pdf',dpi=600,bbox_inches='tight')
    #plt.show()
    return
