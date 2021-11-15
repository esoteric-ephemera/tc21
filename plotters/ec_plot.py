import numpy as np
from os import system
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dft.lsda import ec_pw92

from settings import nproc

clist = ['darkblue','darkorange','darkgreen','darkred','tab:blue','purple','black']

nbs = './eps_data/'#'./reference_data/'
mkrlist=['o','s','d','^','v','x','*','+']

def eps_c_plots(targ=None,use_flib=False):

    to_plot = ['RPA','ALDA','MCP07','TC','CP07','RA']

    fls = {}
    for fxc in to_plot:
        fls[fxc] = nbs+ 'epsilon_C_{:}.csv'.format(fxc)

    if targ is not None:
        fls[targ[0]] = targ[1]
    label = {'MCP07': (57,0.002),
    'static MCP07': (39.73,0.001),
    'RPA': (71.66,-0.02),
    'ALDA': (27,.007),
    'PZ81': (85,-0.002),
    'TC': (80,.001),
    #'CP07': (-1,-1)
    }
    lsl = ['-','-','-','-',':',':']

    if use_flib:
        system('cd dft/dft_fortlib/ ; OMP_NUM_THREADS={:} ; export OMP_NUM_THREADS ; zsh runcalc.sh'.format(nproc))
        dat = np.genfromtxt('./eps_data/jell_eps_c.csv',delimiter=',',skip_header=1)
        ec_d = {}
        ec_d['RPA'] = dat[:,2]
        ec_d['ALDA'] = dat[:,3]
        ec_d['MCP07'] = dat[:,4]
        ec_d['TC'] = dat[:,5]
        ec_d['CP07'] = dat[:,6]
        ec_d['RA'] = dat[:,7]

    ec_out = {}
    fig,ax = plt.subplots(figsize=(8,6))
    axins = ax.inset_axes((0.2,0.1,0.5,0.4) )#width="50%", height="40%", loc=8,borderpad=5)
    rsins = 15
    for ifxc,fxc in enumerate(fls.keys()):
        if use_flib:
            rs = dat[:,0]
            ec = ec_d[fxc]
        else:
            rs,ec,_,_ = np.transpose(np.genfromtxt(fls[fxc],delimiter=',',skip_header=1))

        ec_out[fxc] = ec[rs<=10]

        if fxc in ['ALDA','MCP07','RA']:
            if fxc in ['ALDA']:
                rsc = 30.0
            elif fxc == 'RA':
                rsc = 50.0
            elif fxc == 'MCP07':
                rsc = 69.0
            rsm = rs<=rsc
            ec = ec[rsm]
            rs = rs[rsm]

        lbl = fxc
        if fxc == 'TC':
            lbl = 'rMCP07'

        ax.plot(rs,ec,color=clist[ifxc],label=lbl,linewidth=2.5,linestyle=lsl[ifxc])
        tmprs = rs[rs<=rsins]
        ecpw92,_,_ = ec_pw92(tmprs,0.0)

        axins.plot(tmprs,ec[rs<=rsins]-ecpw92,color=clist[ifxc],label=lbl,linewidth=2.5,linestyle=lsl[ifxc])

        if ifxc == 0:
            ec_out['rs'] = rs[rs<=10]
    rsl = np.linspace(0.01,125.0,3000)
    ec_lda,_,_ = ec_pw92(rsl,0.0)
    ax.plot(rsl,ec_lda,color='black',linestyle='--',label='PW92',linewidth=2.5)
    ax.set_xlim([0,120])
    ax.hlines(0.0,*ax.get_xlim(),linestyle='--',color='gray')

    #axins.plot(rsl[rsl<=rsins],ec_lda[rsl<=rsins],color='black',linestyle='--',label='PW92',linewidth=2.5)
    axins.set_xlim([0,rsins])
    axins.hlines(0.0,*axins.get_xlim(),linestyle='--',color='gray')
    axins.set_ylabel('$\\varepsilon_{\mathrm{c}}(r_s) - \
        \\varepsilon^\mathrm{PW92}_{\mathrm{c}}(r_s)$',fontsize=12)
    axins.set_xlabel('$r_s$',fontsize=12)
    ax.tick_params(axis='both',labelsize=12)

    ax.set_ylabel('$\\varepsilon_{\mathrm{c}}(r_s)$ ($E_h$/electron)',fontsize=24)
    ax.set_xlabel('$r_s$ ($a_0$)',fontsize=24)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylim([-0.1,min(.02,ax.get_ylim()[1])])
    ax.tick_params(axis='both',labelsize=20)
    ax.legend(fontsize=14)
    #plt.show()
    #exit()
    plt.savefig('./figs/ec_plot.pdf',dpi=600,bbox_inches='tight')

    ec_lda,_,_ = ec_pw92(ec_out['rs'],0.0)
    np.savetxt('./eps_data/eps_comp.csv',np.transpose((ec_out['rs'],ec_lda,ec_out['RPA'],ec_out['ALDA'],ec_out['MCP07'],ec_out['TC'])),delimiter=',',header = 'rs, PW92, RPA, ALDA, MCP07, rMCP07')

    texfl = open('./eps_data/eps_comp.tex','w+')
    texfl.write('$\\rs$ & $\\varepsilon_{\\mathrm{c}}$ PW92 (hartree/electron) & RPA & ALDA & MCP07 & rMCP07 \\\\ \hline \n')
    for irs,ars in enumerate(ec_out['rs']):
        if ars < 1:
            texfl.write(('{:.1f} & '+ 4*'{:.4f} & ' + '{:.4f} \\\\ \n').format(ars,ec_lda[irs],ec_out['RPA'][irs],ec_out['ALDA'][irs],ec_out['MCP07'][irs],ec_out['TC'][irs]))
        else:
            if irs == len(ec_out['rs'])-1:
                texfl.write(('{:} & '+ 4*'{:.4f} & ' + '{:.4f} \\\\ \\hline \n').format(int(ars),ec_lda[irs],ec_out['RPA'][irs],ec_out['ALDA'][irs],ec_out['MCP07'][irs],ec_out['TC'][irs]))
            else:
                texfl.write(('{:} & '+ 4*'{:.4f} & ' + '{:.4f} \\\\ \n').format(int(ars),ec_lda[irs],ec_out['RPA'][irs],ec_out['ALDA'][irs],ec_out['MCP07'][irs],ec_out['TC'][irs]))
    texfl.close()

    return


if __name__=="__main__":

    eps_c_plots()
