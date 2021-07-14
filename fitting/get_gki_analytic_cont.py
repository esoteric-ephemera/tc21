import numpy as np
from settings import pi

from dft.gki_fxc import gki_dynamic_real_freq,gam
from utilities.integrators import nquad

def get_gki_ac():

    rs = 1
    dv = {}
    dv['rs'] = rs
    dv['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dv['n'] = 3.0/(4.0*pi*dv['rs']**3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/dv['rs']**3)**(0.5)

    def wrap_integrand(tt,freq,rescale=False):
        if rescale:
            alp = 0.1
            to = 2*alp/(tt+1.0)-alp
            d_to_d_tt = 2*alp/(tt+1.0)**2
        else:
            to = tt
            d_to_d_tt = 1.0
        tfxc = gki_dynamic_real_freq(dv,to,x_only=False,revised=True,param='PW92',dimensionless=True)
        num = freq*tfxc.real + to*tfxc.imag
        denom = to**2 + freq**2
        return num/denom*d_to_d_tt

    #if not isfile('./test_fits/gki_fxc_ifreq.csv'):
    #    wl,fxciu = np.transpose(np.genfromtxt('./test_fits/gki_fxc_ifreq.csv',delimiter=',',skip_header=1))
    #else:
    wl = np.arange(0.005,10.005,0.005)
    fxciu = np.zeros(wl.shape[0])
    for itu,tu in enumerate(wl):
        fxciu[itu],err = nquad(wrap_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(tu,),kwargs={'rescale':True})
        if err['error'] != err['error']:
            fxcu[itu],err = nquad(wrap_integrand,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(tu,))
        if err['code'] == 0:
            print(('WARNING, analytic continuation failed; error {:}').format(err['error']))

    # rescale so that fxc(0) - f_inf = 1 --> factor of gam
    # factor of 1/pi comes from Cauchy principal value integral
    fxciu *= gam/pi
    np.savetxt('./test_fits/gki_fxc_ifreq.csv',np.transpose((wl,fxciu)),delimiter=',',header='b(n)**(0.5)*u, fxc(i u)')
    return

if __name__ == "__main__":

    get_gki_ac()
