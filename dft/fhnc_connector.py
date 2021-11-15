import numpy as np
from glob import glob
from settings import pi
#from utilities.interpolators import bilinear_interp

"""
    NB: to use the FHNC data of
    M. Panholzer, M. Gatti, L. Reining, Phys. Rev. Lett. 120, 166402 (2018).
    DOI 10.1103/PhysRevLett.120.166402
    https://link.aps.org/doi/10.1103/PhysRevLett.120.166402

    please download it from:
    https://etsf.polytechnique.fr/research/connector/2p2h-kernel

    This code assumes the ``New kernel data with a 0.2 rs spacing'' file is used
    (~ 14 MB)
"""

def import_data():

    tfiles = glob('/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/dft/fxc_fhnc_tab/fxc_*_2p2h_fine_L.txt')

    rss = np.zeros(len(tfiles))

    for ifile,afile in enumerate(tfiles):
        rss[ifile] = float(afile.split('_')[-4])/100.0

    wargs = np.argsort(rss)
    rss = rss[wargs]

    files = []
    for i in range(len(tfiles)):
        files.append(tfiles[wargs[i]])
    return rss,files


def get_sqw_single_rs(rs):

    rshun = int(100*rs)
    rshunstr = '{:}'.format(rshun)
    if rshun < 100:
        rshunstr = '0'+rshunstr
    n = 3/(4*pi*rs**3)
    kf = (9*pi/4)**(1/3)/rs
    ef = kf**2/2
    wp0 = (3/rs**3)**(1/2)

    qblock = 0
    wblock = 0
    dat = []
    fl = '/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/dft/fxc_fhnc_tab/fxc_'+rshunstr+'_2p2h_fine_L.txt'
    with open(fl) as infl:
        for iln,ln in enumerate(infl):
            if iln > 7:
                wln = (ln.strip()).split()
                if len(wln) == 0:
                    if wblock == 0:
                        wblock = iln - 8
                    qblock += 1
                else:
                    # From Eq. 1.3 of H. M. Böhm, R. Holler, E. Krotscheck, and M. Panholzer, Phys. Rev. B 82, 224505(2010),
                    # S(q,omega) = -Im Chi/pi in a.u.
                    # the data tables are in units of the bulk Fermi energy; we convert to atomic units
                    dat.append([float(wln[0]),float(wln[1]),-float(wln[2])/(pi*ef)]) # q/kF, omega/omega_p(0), S(q,omega)
                    #dat.append([kf*float(wln[0]),wp0*float(wln[1]),-float(wln[2])/(np.pi*n)])
    dat = np.asarray(dat)

    q = np.arange(0.1,6.5,0.1) # q/kF
    w = wp0*np.arange(0.0,4.0,0.02) # omega
    sqw = np.reshape(dat[:,2],(qblock,wblock))
    return q,w,sqw


def single_rs(rs,q,omega):

    dat = []
    rs = 4
    rshun = int(100*rs)
    n = 3/(4*np.pi*rs**3)
    kf = (9*np.pi/4)**(1/3)/rs
    wp0 = (3/rs**3)**(1/2)
    qblock = 0
    wblock = 0
    fl = '/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/dft/fxc_fhnc_tab/fxc_{:}_2p2h_fine_L.txt'.format(rshun)
    with open(fl) as infl:
        for iln,ln in enumerate(infl):
            if iln > 7:
                wln = (ln.strip()).split()
                if len(wln) == 0:
                    if wblock == 0:
                        wblock = iln - 8
                    qblock += 1
                else:
                    dat.append([kf*float(wln[0]),wp0*float(wln[1]),-float(wln[2])/(np.pi*n)])
    dat = np.asarray(dat)
    q = np.arange(0.1,6.5,0.1)#dat[:,0]

    w = np.arange(0.0,4.0,0.02)#dat[:,1]

    sqw = np.reshape(dat[:,2],(qblock,wblock)) # sqw[iq,iw] = S(q_iq,omega_iw)
    import matplotlib.pyplot as plt
    plt.plot(w/wp0,sqw[5])
    plt.show()
    exit()

    sq = np.zeros(q.shape)
    for iq in range(q.shape[0]):
        sq[iq] = 0.02*wp0*(np.sum(sqw[iq,1:-1]) + 0.5*sqw[iq,0] + 0.5*sqw[iq,-1])

    import matplotlib.pyplot as plt
    plt.plot(q,sq)
    plt.show()

    return
