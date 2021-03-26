import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm,colors,ticker
from dft.lsda import ec_pw92

nbs = '/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/reference_data/'
mkrlist=['o','s','d','^','v','x','*','+']

def eps_c_plots(targ=None):
    ncl=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    fls = {'MCP07': nbs+ 'MCP07_eps_c_reference.csv',
    #'static MCP07': nbs+ 'epsilon_C_MCP07_static.csv',
    'RPA': nbs+ 'RPA_eps_c_reference.csv',
    'ALDA': nbs+ 'ALDA_eps_c_reference.csv'
     }
    if targ is not None:
        fls['rMCP07'] = targ
    label = {'MCP07': (68.82,0.0),
    'static MCP07': (39.73,0.001),
    'RPA': (71.66,-0.013),
    'ALDA': (33.34,0.01),
    'PZ81': (85,-0.002)
    }
    fig,ax = plt.subplots(figsize=(10,6))
    for ifxc,fxc in enumerate(fls):
        if fxc != 'rMCP07':
            rs,ec = np.transpose(np.genfromtxt(fls[fxc],delimiter=',',skip_header=1))
            if fxc == 'ALDA':
                ec = ec[rs<=30.0]
                rs = rs[rs<=30.0]
            elif fxc == 'MCP07':
                ec = ec[rs<=69.0]
                rs = rs[rs<=69.0]
        else:
            rs,ec,_,_ = np.transpose(np.genfromtxt(fls[fxc],delimiter=',',skip_header=1))
        ax.plot(rs,ec,color=ncl[ifxc],label=fxc,linewidth=2)
        #ax.annotate(fxc,label[fxc],color=ncl[ifxc],fontsize=16)
    rsl = np.linspace(0.01,125.0,3000)
    ec_lda,_,_ = ec_pw92(rsl,0.0)
    ax.plot(rsl,ec_lda,color='black',linestyle='--',label='PW92')
    #ax.annotate('PZ81',label['PZ81'],color='black',fontsize=16)
    ax.set_xlim([0,120])
    ax.hlines(0.0,plt.xlim()[0],plt.xlim()[1],linestyle='--',color='gray')
    ax.set_ylabel('$\\varepsilon_{\mathrm{c}}(r_s)$ ($E_h$/electron)',fontsize=20)
    ax.set_xlabel('$r_s$ ($a_0$)',fontsize=20)
    ax.tick_params(axis='both',labelsize=16)
    ax.set_ylim([-0.1,min(.02,ax.get_ylim()[1])])
    ax.legend(fontsize=14)
    if targ is not None:
        plt.show()
    else:
        plt.savefig(base_str+'/../ec_plot.pdf',dpi=600,bbox_inches='tight')
    return


if __name__=="__main__":

    eps_c_plots()
