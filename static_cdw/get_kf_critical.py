import numpy as np
from math import ceil
import multiprocessing as mp
import matplotlib.pyplot as plt
from os import path,system

import settings
from dft.chi import chi_parser
from utilities.roots import bracket,bisect

freq_dependent = ['MCP07','TC']
disperion = False
pi = settings.pi

if not path.isdir('./static_cdw'):
    system('mkdir ./static_cdw')

def wrap_eps(kf,z,fxc):
    rs = (9*pi/4.0)**(1.0/3.0)/kf
    #wp0 = (3.0/rs**3)**(0.5)
    eps=chi_parser(z,0.0,1.0,rs,fxc,ret_eps=True,LDA=settings.LDA)
    return eps.real

def find_kf_critical_driver(z,fxc):

    poss_roots = bracket(wrap_eps,(1.e-6,100),vector=True,args=(z,fxc),nstep=500)
    success = 1.0
    for brack in poss_roots:
        tkfc,success = bisect(wrap_eps,brack,tol=1.e-8,maxstep=1000,args=(z,fxc))
    if abs(success) < 1.e-8:
        return tkfc
    else:
        return 0.0
    step_l = [10**(-i) for i in range(2,10)]
    tmin = 0.0
    tmax = 0.1
    tkfc = 0.0
    omin = 0.1
    for istep,step in enumerate(step_l):
        kf_l = np.arange(tmax,tmin,-step)
        for kf in kf_l[kf_l>0.0]:
            eps = wrap_eps(kf,z)
            if abs(eps) < abs(omin):
                omin = eps
                tkfc = kf
                tmin = kf -step
                tmax = kf+step
        if tkfc == 0.0 and istep == 3:
            break
    return tkfc

def kf_crit_plots():
    clist=settings.clist#['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    lnstls = ['-','--','-.']
    cfac = (9*pi/4.0)**(1.0/3.0)
    base_str = './static_cdw/critical_kf_'
    to_do = ['ALDA','MCP07','TC']#['ALDA','MCP07','MCP07_inf','MCP07_undamp']
    fig,ax = plt.subplots(figsize=(8,6))
    lbl_pos = {}
    for ifxc,fxc in enumerate(to_do):
        dat = np.genfromtxt(base_str+fxc+'.csv',delimiter=',',skip_header=1)
        plt.plot(dat[:,0],dat[:,1],label=fxc,color=clist[ifxc],linewidth=2.5)#,linestyle=lnstls[ifxc%3])
        ipos = ceil(dat.shape[0]*0.6)
        lbl = fxc
        lblx = dat[ipos,0]
        lbly = 1.05*dat[ipos,1]
        if fxc == 'TC':
            lbl = 'TC21'
        elif fxc == 'MCP07':
            lblx = 1.2
            lbly = 0.004
        ax.annotate(lbl,(lblx,lbly),color=clist[ifxc],fontsize=20)
        kfcm = dat[:,1].max()
        print(('fxc_{:}, crit kf = {:} 1/bohr, crit rs = {:} bohr').format(fxc,kfcm,cfac/kfcm))
    ax.set_xlim([0.0,dat[:,0].max()+.01])
    ax.set_ylim([0.0,0.08])
    ax.vlines(1,0.0,0.08,color='gray',linestyle='-.')
    ymin,ymax = ax.get_ylim()
    #ax2 = ax.twinx()
    ticks = ['$\\infty$']
    rsticks = [int(x) for x in np.ceil(cfac/ax.get_yticks()[1:])]
    wticks = [0]
    for i,x in enumerate(rsticks):
        ticks.append('{:}'.format(x))
        wticks.append(cfac/x)
    #ax2.set_ylim(ax.get_ylim())
    #ax2.set_yticks(wticks)
    #ax2.set_yticklabels(ticks)
    #ax2.set_ylabel('$r_{\\mathrm{s,c}}$ (bohr)',fontsize=24)
    ax.set_xlabel('$q/(2k_{\\mathrm{F}})$',fontsize=24)
    ax.set_ylabel('$k_{\\mathrm{F,c}}$ (bohr$^{-1}$)',fontsize=24)
    ax.tick_params(axis='both',labelsize=20)
    #ax2.tick_params(axis='both',labelsize=20)
    #ax.legend(fontsize=12)
    #plt.show()
    plt.savefig('./figs/critical_kf.pdf',dpi=600,bbox_inches='tight')
    return

def kf_crit_search():

    z_l = np.linspace(0.01,2.51,1000)
    for fxc in ['RPA','ALDA','MCP07','TC']:
        if settings.nproc > 1 and len(z_l) > 1:
            pool = mp.Pool(processes=min(settings.nproc,len(z_l)))
            tmp_out = pool.starmap(find_kf_critical_driver,[(zz,fxc) for zz in z_l])
            pool.close()
            kfc = np.asarray(tmp_out)
        else:
            kfc = np.zeros(0)
            for z in z_l:
                kfc = np.append(kfc,find_kf_critical_driver(z,fxc))
        np.savetxt('./static_cdw/critical_kf_'+fxc+'.csv',np.transpose((z_l,kfc)),delimiter=',',header='q/(2k_F),k_F critical')

    return

if __name__=="__main__":

    kf_crit_search()

    kf_crit_plots()
    exit()
