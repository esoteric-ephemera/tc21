import numpy as np
import multiprocessing

import settings
from utilities.integrators import nquad

pi = settings.pi

def densvars(rs):
    dvars = {'rs': rs, 'kf': (9*pi/4.0)**(1.0/3.0)/rs, 'n': 3.0/(4.0*pi*rs**3), 'wp': (3.0/rs**3)**(0.5)}
    dvars['ef'] = dvars['kf']**2/2.0
    dvars['rsh'] = rs**(0.5)
    return dvars

def re_fxc_model(dv,omega):
    return gki_dynamic(dv,omega.real,revised=True).real

def im_fxc_gki(omega):
    # NB: constants only introduced after integration
    return omega/((1.0 + omega**2)**(5.0/4.0))

def wrap_kramers_kronig(to,omega):
    return im_fxc_gki(to)/(to - omega)

def get_kk_re_fxc(om_l):
    if not hasattr(om_l,'__len__'):
        om_l = np.asarray([om_l])
    fxc_re = np.zeros(om_l.shape)

    for iomega,omega in enumerate(om_l):
        fxc_re[iomega],terr = nquad(wrap_kramers_kronig,('-inf','inf'),'global_adap',{'itgr':'GK','prec':5.e-8,'npts':5,'min_recur':4,'max_recur':1000,'n_extrap':400,'inf_cond':'fun'},pars_ops={'PV':[omega]},args=(omega,))
        if terr['code'] == 0:
            print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(omega,terr['error']))
    return fxc_re

def kramers_kronig_re_fxc():

    ol = np.arange(0.0,10.01,0.01)

    if settings.nproc > 1:
        pool = multiprocessing.Pool(processes=min(settings.nproc,ol.shape[0]))
        refxct = pool.map(get_kk_re_fxc,ol)
        pool.close()
        refxc = np.zeros(0)
        for tmp in refxct:
            refxc = np.append(refxc,tmp[0]/pi)
    else:
        refxc = get_kk_re_fxc(ol)

    fname='./test_fits/kram_kron_re_fxc.csv'
    np.savetxt(fname,np.transpose((ol,refxc)),delimiter=',',header='Re omega, K.-K. Re f_xc',fmt='%.18f')
    return

################################################################################

if __name__ == "__main__":

    kramers_kronig_re_fxc()
