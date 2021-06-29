import numpy as np
import matplotlib.pyplot as plt

import settings

def plot_dens_fluc():
    bas_str = './freq_data/'

    fig,ax = plt.subplots(figsize=(8,6))

    lsls = ['-','--']

    ic = 0
    for irs,rs in enumerate([4,69]):

        wp0 = (3/rs**3)**(0.5)

        for iwfxc,wfxc in enumerate(['TC','MCP07']):

            q,sq = np.transpose(np.genfromtxt(bas_str+'{:}_Sq_rs_{:}_original.csv'.format(wfxc,rs),delimiter=',',skip_header=1))
            _,m1 = np.transpose(np.genfromtxt(bas_str+'{:}_moment_1.0_rs_{:}_original.csv'.format(wfxc,rs),delimiter=',',skip_header=1))
            _,m2 = np.transpose(np.genfromtxt(bas_str+'{:}_moment_2.0_rs_{:}_original.csv'.format(wfxc,rs),delimiter=',',skip_header=1))

            if ic == 0:
                stddev_fluc = np.zeros((4,q.shape[0]))

            avg_fluc = m1/sq
            ax.plot(q,avg_fluc,color=settings.clist[irs],linestyle=lsls[iwfxc],linewidth=2.5)

            stddev_fluc[ic] = (m2/sq - avg_fluc**2)**(0.5)

            if iwfxc == 0:
                ind = np.argmin(np.abs(q-2))
                if rs == 4:
                    lblpos = (q[ind]-.2,avg_fluc[ind]+1)
                elif rs == 69:
                    lblpos = (q[ind]+.15,avg_fluc[ind]+.5)
                ax.annotate('$r_{\\mathrm{s}}='+str(int(rs))+'$',lblpos,fontsize=20,color=settings.clist[irs])
            ic += 1

    ax.set_xlabel('$q/k_{\\mathrm{F}}$',fontsize=24)
    ax.set_xlim([0.0,4.0])
    ax.set_ylim([0.0,5.0])
    ax.tick_params(axis='both',labelsize=20)
    ax.set_ylabel('$\langle \omega_p(q)\\rangle/\omega_p(0)$',fontsize=24)
    #plt.show()
    plt.savefig('./figs/omega_avg_q.pdf',dpi=300,bbox_inches='tight')

    plt.cla()
    plt.clf()

    fig,ax = plt.subplots(figsize=(8,6))

    ic = 0
    for irs,rs in enumerate([4,69]):
        for iwfxc,wfxc in enumerate(['TC','MCP07']):
            ax.plot(q,stddev_fluc[ic],color=settings.clist[irs],linestyle=lsls[iwfxc],linewidth=2.5)
            if iwfxc == 0:
                ind = np.argmin(np.abs(q-3))
                ax.annotate('$r_{\\mathrm{s}}='+str(int(rs))+'$',(q[ind],stddev_fluc[ic,ind]-.15),fontsize=20,color=settings.clist[irs])
            ic += 1
    ax.set_xlabel('$q/k_{\\mathrm{F}}$',fontsize=24)
    ax.set_xlim([0.0,4.0])
    ax.set_ylim([0.0,2.0])
    ax.tick_params(axis='both',labelsize=20)
    ax.set_ylabel('$\langle \Delta \omega_p(q)\\rangle/\omega_p(0)$',fontsize=24)
    #plt.show()
    plt.savefig('./figs/omega_stddev_q.pdf',dpi=300,bbox_inches='tight')

    return

if __name__ == "__main__":

    plot_dens_fluc()
