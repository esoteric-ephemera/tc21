import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from os import path,system

import settings
from dft.mcp07 import chi_parser

freq_dependent = ['MCP07','rMCP07']
disperion = False
pi = settings.pi

if not path.isdir('./static_cdw'):
    system('mkdir ./static_cdw')

def wrap_eps(kf,z):
    rs = (9*pi/4.0)**(1.0/3.0)/kf
    #wp0 = (3.0/rs**3)**(0.5)
    eps=chi_parser(z,1.e-14j,1.0,rs,settings.fxc,ret_eps=True,LDA=settings.LDA)
    return eps.real

def find_kf_critical_driver(z):
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

def plots():
    base_str = './static_cdw/critical_kf_'
    to_do = ['ALDA','MCP07','rMCP07']#['ALDA','MCP07','MCP07_inf','MCP07_undamp']
    fig,ax = plt.subplots(figsize=(10,6))
    for fxc in to_do:
        dat = np.genfromtxt(base_str+fxc+'.csv',delimiter=',',skip_header=1)
        plt.plot(dat[:,0],dat[:,1],label=fxc)
    ax.set_xlim([0.0,dat[:,0].max()+.01])
    ax.set_ylim([0.0,0.08])
    ymin,ymax = ax.get_ylim()
    ax2 = ax.twinx()
    cfac = (9*pi/4.0)**(1.0/3.0)
    ticks = ['$\\infty$']
    for i,x in enumerate(cfac/ax.get_yticks()[1:]):
        ticks.append('{:.2f}'.format(x))
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(ticks)
    ax2.set_ylabel('$r_s$ critical',fontsize=12)
    ax.set_xlabel('$q/(2k_F)$',fontsize=12)
    ax.set_ylabel('$k_F$ critical',fontsize=12)
    ax.legend(fontsize=12)
    plt.show()
    return

if __name__=="__main__":

    z_l = np.linspace(0.01,2.51,1000)
    if settings.nproc > 1 and len(z_l) > 1:
        pool = mp.Pool(processes=min(settings.nproc,len(z_l)))
        tmp_out = pool.map(find_kf_critical_driver,z_l)
        pool.close()
        kfc = np.asarray(tmp_out)
    else:
        kfc = np.zeros(0)
        for z in z_l:
            kfc = np.append(kfc,find_kf_critical_driver(z))
    np.savetxt('./static_cdw/critical_kf_'+settings.fxc+'.csv',np.transpose((z_l,kfc)),delimiter=',',header='q/(2k_F),k_F critical')

    plots()
    exit()
