import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from math import ceil
import sys

sys.path.insert(1, '/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/utilities')
from interpolators import natural_spline,spline

do_dimensionless = True

# smoothing = None, binned, or GN for Gaussian noise
smoothing = 'GN'

rs_bd = [60,100]

to_plot = []
rs_l = []
clist=['darkblue','darkorange','darkgreen','darkred','black']
#['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
ln_style_l = ['-','--','-.']

for adir in os.listdir('./data_files/'):
    if adir[0]=='.':
        continue
    rs_tmp = int(adir.split('_')[-1])
    if rs_bd[0] <= rs_tmp <= rs_bd[1]:
        to_plot.append(adir)
        rs_l.append(rs_tmp)

to_plot = np.asarray(to_plot)[np.argsort(rs_l)]

fix,ax = plt.subplots(figsize=(8,6))

pbd = [1e20,-1e20]

for idir,adir in enumerate(to_plot):
    fl_l = glob('./data_files/'+adir+'/sk_*.csv')
    for ifl,fl in enumerate(fl_l):
        nelec = fl.split('_')[-2].split('-')[-1]
        rs = fl.split('_')[-1].split('-')[-1].split('.csv')[0]
        kF = (9.0*np.pi/4.0)**(1.0/3.0)/float(rs)
        lbl =  '$r_s={:}$'.format(rs)#'$r_s={:}$, $N={:}$'.format(rs,nelec)
        k,sk = np.transpose(np.genfromtxt(fl,delimiter=',',skip_header=1))
        if do_dimensionless:
            k/=kF
        if k.min()<pbd[0]:
            pbd[0] = k.min()
        if k.max()>pbd[1]:
            pbd[1] = k.max()

        if smoothing == 'GN':
            skn = np.zeros(sk.shape[0])
            #irange = 10
            for i in range(sk.shape[0]):
                if do_dimensionless:
                    cond = k[i] < 1.0
                else:
                    k[i] < kf
                if cond:
                    irange = 1
                else:
                    irange = 4
                mmin = max([0,i-irange])
                mmax = min([sk.shape[0]-1,i+irange])
                len = mmax - mmin + 1
                skt = sk[mmin:mmax+1]
                mean = np.sum(skt)/len
                var = np.sum(skt**2)/len - mean**2
                wg = np.exp(-(skt - mean)**2/var)
                skn[i] = np.sum(skt*wg)/np.sum(wg)
            ax.plot(k,skn,color=clist[idir],linestyle=ln_style_l[ifl%3],label=lbl,linewidth=2)

        elif smoothing == 'binned':
            for frac in np.arange(0.8,0.01,-0.01):
                kt_l = np.linspace(k.min(),k.max(),ceil(frac*len(k)))
                skt = np.zeros(kt_l.shape)
                skt[-1] = sk[-1]
                norm = np.zeros(kt_l.shape)
                norm[-1] = 1
                lik = 0
                for ikt in range(len(kt_l)-1):
                    for ik in range(lik,len(k)):
                        if kt_l[ikt]<= k[ik] < kt_l[ikt+1]:
                            skt[ikt] += sk[ik]
                            norm[ikt] += 1
                        elif k[ik]>= kt_l[ikt+1]:
                            lik = ik
                            break
                if np.all(norm>0.0):
                    print(('Bin width = {:.4f} ({:.2f}%) for rs = {:}, N = {:}').format((k.max()-k.min())/(ceil(frac*len(k))-1.0),frac,rs,nelec))
                    break
            #sk2 = natural_spline(kt_l,skt/norm)
            #knew = np.linspace(kt_l[0],kt_l[-1],1000)
            #ax.plot(knew,spline(knew,kt_l,skt/norm,sk2),color=clist[idir],linestyle='-',label=lbl)#ln_style_l[(idir+ifl)%3],label=lbl)
            ax.plot(kt_l,skt/norm,color=clist[idir],linestyle=ln_style_l[ifl%3],label=lbl,linewidth=2.5)
        elif smoothing is None:
            ax.plot(k,sk,color=clist[idir],linestyle=ln_style_l[ifl%3],label=lbl,linewidth=2.5)
ax.legend(fontsize=20,loc='lower right')
if do_dimensionless:
    ax.set_xlabel('$q/k_F$',fontsize=24)
    if smoothing is not None:
        ptitle = 'ballone_large_rs_sk_dimensionless_{:}.pdf'.format(smoothing)
    else:
        ptitle = 'ballone_large_rs_sk_dimensionless.pdf'
else:
    ax.set_xlabel('$k$ (1/bohr)',fontsize=24)
    if smoothing is not None:
        ptitle = 'ballone_large_rs_sk_{:}.pdf'.format(smoothing)
    else:
        ptitle = 'ballone_large_rs_sk.pdf'
ax.set_xlim(pbd)
ax.set_ylim([0.0,ax.get_ylim()[1]])
ax.set_ylabel('$S(q)$',fontsize=24)
ax.hlines(1,plt.xlim()[0],plt.xlim()[1],linestyle='-.',color='gray')
if smoothing is not None:
    extratext = smoothing
    if smoothing == 'GN':
        extratext = 'Gaussian noise smoothing'
    plt.title('UEG $S^{\\mathrm{QMC}}(q)$, '+extratext,fontsize=20)
else:
    plt.title('UEG $S^{\\mathrm{QMC}}(q)$',fontsize=20)
ax.tick_params(axis='both',labelsize=20)
#plt.show()
plt.savefig('../figs/'+ptitle,dpi=600,bbox_inches='tight')
