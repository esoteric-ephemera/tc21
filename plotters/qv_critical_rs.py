import numpy as np
from os.path import isfile
import matplotlib.pyplot as plt

from settings import clist
from dft.qv_fxc import density_variables,get_qv_pars

def plot_qv_rs_crit(regen_dat=False):

    wfile = './reference_data/qv_kernel_critical_rs.csv'
    if not isfile(wfile) or regen_dat:
        rsl = np.linspace(0.1,60,5000)
        g = np.zeros(rsl.shape)
        g0 = np.zeros(rsl.shape)
        for irs,rs in enumerate(rsl):
            dv = density_variables(rs)
            _,_,g[irs],_ = get_qv_pars(dv,use_mu_xc=True)
            _,_,g0[irs],_ = get_qv_pars(dv,use_mu_xc=False)
        np.savetxt(wfile,np.transpose((rsl,g,g0)),delimiter=',',header='rs (bohr), Gamma muxc > 0, Gamma muxc = 0')
    else:
        rsl,g,g0 = np.transpose(np.genfromtxt(wfile,delimiter=',',skip_header=1))

    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(rsl,g,color=clist[0],linewidth=2.5)
    ax.plot(rsl,g0,color=clist[1],linewidth=2.5)
    ax.set_xlabel('$r_{\\mathrm{s}}$',fontsize=24)
    ax.set_ylabel('$\\Gamma(r_{\\mathrm{s}})$',fontsize=24)
    ax.set_xlim([0.0,60])
    ax.set_ylim([0.0,1.1*max(g[0],g0[0])])
    ax.tick_params(axis='both',labelsize=20)
    ax.annotate('$\\mu_{\\mathrm{xc}}(r_{\\mathrm{s}})\\neq 0$',(30,.55),fontsize=24,color=clist[0])
    ax.annotate('$\\mu_{\\mathrm{xc}}(r_{\\mathrm{s}})= 0$',(18,.2),fontsize=24,color=clist[1])
    #plt.show()
    plt.savefig('./figs/qv_kernel_crit_rs.pdf',dpi=600,bbox_inches='tight')
    for irs in range(rsl.shape[0]):
        if g[irs]==0:
            print('mu_xc > 0 crit rs',rsl[irs])
            break
    for irs in range(rsl.shape[0]):
        if g0[irs]==0:
            print('mu_xc = 0 crit rs',rsl[irs])
            break

    return
