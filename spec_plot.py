import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from settings import pi
from dft.chi import chi_parser

color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']

def wrap_mcp07_diff_dsd(rs,q,om=None):
    omega_p0 = (3.0/rs**3)**(0.5)
    olow = (0.05 + 1.e-10j)*omega_p0
    if om is not None:
        olow = om*omega_p0
    n = 3.0/(4*pi*rs**3)
    chi1 = chi_parser(0.5*q,olow,1.0,rs,'MCP07',reduce_omega=False,imag_freq=False,ret_eps=False)
    chi0 = chi_parser(0.5*q,olow,0.0,rs,'MCP07',reduce_omega=False,imag_freq=False,ret_eps=False)
    return (-chi1.imag + chi0.imag)/(pi*n)

qred = np.linspace(1.e-2,3.0,2000)
sdiff = {}
for irs,rs in enumerate([4,69]):
    sdiff[rs] = wrap_mcp07_diff_dsd(rs,qred,om=(0.01))# + 1.e-10j))
"""
fig,ax = plt.subplots(2,1,figsize=(6,6))
for irs,rs in enumerate([4,69]):
    #sdiff[rs] = wrap_mcp07_diff_dsd(rs,qred)
    ax[irs].plot(qred,sdiff[rs])
    ax[irs].set_xlim([0.0,3.0])
    ax[irs].xaxis.set_major_locator(MultipleLocator(.5))
    ax[irs].xaxis.set_minor_locator(MultipleLocator(.25))
    ax[irs].set_ylim([1.05*sdiff[rs].min(),1.05*sdiff[rs].max()])
    ax[irs].set_ylabel('$\\Delta S(r_{\\mathrm{s}}='+str(rs)+')$',fontsize=14)
    ax[irs].tick_params(axis='both',labelsize=12)
ax[0].yaxis.set_major_locator(MultipleLocator(1))
ax[0].yaxis.set_minor_locator(MultipleLocator(.5))
ax[1].yaxis.set_major_locator(MultipleLocator(10000))
ax[1].yaxis.set_minor_locator(MultipleLocator(5000))
ax[1].set_xlabel('$q/k_{\\mathrm{F}}$',fontsize=14)
plt.suptitle('$\\Delta S(r_{\\mathrm{s}}) = S^{\\mathrm{MCP07}}_1(q,\\omega_\\mathrm{low})-S_0(q,\\omega_\\mathrm{low})$, \n $\\omega_\\mathrm{low} = [0.05 + 10^{-10}~i]\\omega_p(0)$',fontsize=14)
#plt.show()
plt.savefig('./figs/MCP07_spec_diff.png',dpi=600,bbox_inches='tight')
plt.cla()
plt.clf()
"""
fign,axl = plt.subplots(figsize=(8,8*2/(1 +5**(0.5))))
axr = axl.twinx()
axl.plot(qred,sdiff[4])
axr.plot(qred,sdiff[69],color=color_list[1])
axl.set_xlim([0.0,3.0])
axr.set_xlim([0.0,3.0])

axl.xaxis.set_major_locator(MultipleLocator(.5))
axl.xaxis.set_minor_locator(MultipleLocator(.25))
axl.set_ylim([1.05*sdiff[4].min(),1.05*sdiff[4].max()])
axr.set_ylim([1.05*sdiff[69].min(),1.05*sdiff[69].max()])
axl.set_ylabel('$\\Delta S(4)$, (blue)',fontsize=14)
axr.set_ylabel('$\\Delta S(69)$, (orange)',fontsize=14)
axl.tick_params(axis='both',labelsize=12)
axr.tick_params(axis='both',labelsize=12)

axl.yaxis.set_major_locator(MultipleLocator(50))
axl.yaxis.set_minor_locator(MultipleLocator(25))
axr.yaxis.set_major_locator(MultipleLocator(50000))
axr.yaxis.set_minor_locator(MultipleLocator(25000))
axl.set_xlabel('$q/k_{\\mathrm{F}}$',fontsize=14)
plt.suptitle('$\\Delta S(r_{\\mathrm{s}}) = S^{\\mathrm{MCP07}}_1(q,\\omega_\\mathrm{low})-S_0(q,\\omega_\\mathrm{low})$,    $\\omega_\\mathrm{low} = 0.01\\omega_p(0)$',fontsize=14)
#plt.suptitle('$\\Delta S(r_{\\mathrm{s}}) = S^{\\mathrm{MCP07}}_1(q,\\omega=0)-S_0(q,\\omega=0)$',fontsize=12)
#plt.show()
plt.savefig('./figs/MCP07_spec_diff_re_olo.pdf',dpi=600,bbox_inches='tight')
