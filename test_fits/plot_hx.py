import numpy as np
import matplotlib.pyplot as plt
gam = 1.311028777146059809410871821455657482147216796875

def h_mcp07(x):
    aj = 0.63
    h0 = 1.0/gam
    fac = (h0*aj)**(4.0/7.0)
    hx = h0*(1.0 - aj*x**2)/(1.0 + fac*x**2)**(7.0/4.0)
    return hx

def h_tc21(x):
    cc1,cc2,cc3,cc4 = (0.174724,3.224459,2.221196,1.891998)
    hx = 1/gam*(1 - cc1*x**2)/(1 + cc2*x**2 + cc3*x**4 + cc4*x**6 + (cc1/gam)**(16/7)*x**8)**(7/16)
    return hx

def plot_diffs():

    kk_dat = './kram_kron_re_fxc.csv'

    w,fxc_kk = np.transpose(np.genfromtxt(kk_dat,delimiter=',',skip_header=1))
    hm = h_mcp07(w)
    htc = h_tc21(w)

    fig,ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(w,hm-fxc_kk,color='darkblue',linewidth=2.5)
    ax[0].plot(w,htc-fxc_kk,color='darkorange',linewidth=2.5)
    ax[0].set_ylim([1.05*min((hm-fxc_kk).min(),(htc-fxc_kk).min()),2*max((hm-fxc_kk).max(),(htc-fxc_kk).max())])
    ax[0].hlines(0,w.min(),w.max(),color='gray',linewidth=1)
    ax[0].set_ylabel('$h^{\\mathrm{approx}}(y)-h^{\\mathrm{KK}}(y)$',fontsize=20)

    ax[1].plot(w,hm,color='darkblue',linewidth=2.5)
    ax[1].plot(w,htc,color='darkorange',linewidth=2.5)
    ax[1].set_ylim([1.05*min(hm.min(),htc.min()),1.05*max(hm.max(),htc.max())])
    ax[1].set_ylabel('$h^{\\mathrm{approx}}(y)$',fontsize=20)

    for i in range(2):
        ax[i].set_xlim([w.min(),w.max()])
        ax[i].set_xlabel('$y=[b(n)]^{1/2}\omega$',fontsize=20)
        ax[i].tick_params(axis='both',labelsize=18)

    ax[0].annotate('MCP07',(3.9,-.08),color='darkblue',fontsize=18)
    ax[1].annotate('TC21',(3.9,.005),color='darkorange',fontsize=18)

    plt.subplots_adjust(left=.1)
    plt.savefig('./hx_comparison.pdf',dpi=600,bbox_inches='tight')
    return

if __name__ == "__main__":

    plot_diffs()
