import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from os.path import isfile

from dft.fhnc_connector import get_sqw_single_rs
from dft.chi import chi_parser
from settings import pi,LDA,clist

def plot_ghost_exciton(regen_qv_dat=False):

    #clist=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    lnsl = ['-','--','-.']

    Eh_to_keV = 0.027211386245988 # https://physics.nist.gov/cgi-bin/cuu/Value?hr

    rs = 8

    n = 3/(4*pi*rs**3)

    q_plot = 2.2 # units of kF
    qmc_q,qmc_o,qmc_sqw = get_sqw_single_rs(rs)
    """
    sq = np.zeros(qmc_q.shape)
    for iq in range(qmc_q.shape[0]):
        sq[iq] = (qmc_o[1]-qmc_o[0])*(np.sum(qmc_sqw[iq,1:-1]) + 0.5*qmc_sqw[iq,0] + 0.5*qmc_sqw[iq,-1])
    plt.plot(qmc_q,sq)
    plt.show()
    exit()
    """
    po = qmc_o*(1000*Eh_to_keV)
    wind = np.argmin(np.abs(qmc_q-q_plot))

    fig,ax = plt.subplots(figsize=(8,6))


    ax.plot(po,qmc_sqw[wind]/Eh_to_keV,color=clist[0],linestyle=lnsl[0],label='$2p2h$',linewidth=2.5)
    maxxn = -1e20
    minn = 1e20
    maxxn = max([maxxn,qmc_sqw[wind].max()/Eh_to_keV])
    minn = min([minn,qmc_sqw[wind].min()/Eh_to_keV])

    for ifxc,fxc in enumerate(['MCP07','TC','QVmulti']):#['RPA','ALDA','MCP07','rMCP07','QVmulti']):

        if fxc == 'QVmulti':
            wfile = './reference_data/qv_rs_8_q_2.2_sq_omega.csv'
            if regen_qv_dat or not isfile(wfile):
                chi = chi_parser(q_plot/2.0,qmc_o+1.e-12j,1.0,rs,fxc,reduce_omega=False,imag_freq=False,ret_eps=False,pars={},LDA=LDA)
                sqw_tddft = -chi.imag/(pi*n)
                np.savetxt(wfile,np.transpose((qmc_o,sqw_tddft)),delimiter=',',header='omega (a.u.), S(2.2 kF, omega) (a.u.)')
            else:
                _,sqw_tddft = np.transpose(np.genfromtxt(wfile,delimiter=',',skip_header=1))
        else:
            chi = chi_parser(q_plot/2.0,qmc_o+1.e-12j,1.0,rs,fxc,reduce_omega=False,imag_freq=False,ret_eps=False,pars={},LDA=LDA)
            sqw_tddft = -chi.imag/(pi*n)
        maxxn = max([maxxn,sqw_tddft.max()/Eh_to_keV])
        minn = min([minn,sqw_tddft.min()/Eh_to_keV])
        lbl = fxc
        if fxc == 'TC':
            lbl = 'rMCP07'
        elif fxc == 'QVmulti':
            lbl = 'QV'
        ax.plot(po,sqw_tddft/Eh_to_keV,color=clist[ifxc+1],linestyle=lnsl[ifxc%3],label=lbl,linewidth=2.5)
    ax.set_xlim([po[0],po[-1]])
    ax.set_ylim([minn,1.05*maxxn])
    ax.set_xlabel('$\\omega$ (eV)',fontsize=24)
    ax.set_ylabel('$S(q,\\omega)$ (1/keV)',fontsize=24)
    ax.tick_params(axis='both',labelsize=20)
    plt.title('$r_{\\mathrm{s}}=8$ Jellium, $q=2.2k_{\\mathrm{F}}$',fontsize=24)
    ax.legend(fontsize=20)
    #plt.show()
    plt.savefig('./figs/ghost_exciton_comp.pdf',dpi=600,bbox_inches='tight')

    return
