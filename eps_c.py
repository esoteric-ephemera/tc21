import numpy as np
from os import system,path
from itertools import product
from time import time
import multiprocessing

import settings
from utilities.gauss_quad import gauss_quad
from utilities.integrators import clenshaw_curtis_grid
from dft.chi import chi_parser

pi = settings.pi
eps = 'C'

def establish_dependencies(): # makes folder structure
    for dependency in ['grids','eps_data','freq_data','reference_data']:
        if not path.isdir('./'+dependency):
            system('mkdir ./'+dependency)
    return
establish_dependencies()

def set_up_grid(n_z_pts,n_L_pts,n_u_pts,zcp=1.0,ucp=2.0,npartition = {'z':4,'L': 4, 'u':4}):

    def mk_axis(ref_grid,ref_wgt,bds,conv_par,npart,rescale=True):
        ax = np.zeros(0)
        axw = np.zeros(0)
        nval = [(bds[1]-bds[0])*ival/npart + bds[0] for ival in range(npart+1)]
        for ival in range(1,npart+1):
            tmp = 0.5*(nval[ival]-nval[ival-1])*ref_grid + 0.5*(nval[ival]+nval[ival-1])
            wtmp = 0.5*(nval[ival]-nval[ival-1])*ref_wgt
            if rescale:
                axw = np.append(axw,wtmp*conv_par*2.0/(1.0 + tmp)**2)
                ax = np.append(ax,conv_par*(2.0/(1.0 + tmp)-1.0))
            else:
                axw = np.append(axw,wtmp)
                ax = np.append(ax,tmp)
        return ax,axw

    z_grid = './grids/gauss_legendre_{:}_pts.csv'.format(n_z_pts)
    L_grid = './grids/gauss_legendre_{:}_pts.csv'.format(n_L_pts)
    u_grid = './grids/gauss_legendre_{:}_pts.csv'.format(n_u_pts)#2*n_u_pts+1)

    for ifl,fl in enumerate([z_grid,L_grid,u_grid]):
        if not path.isfile(fl) or path.getsize(fl)==0:
            if ifl == 2:
                gauss_quad(n_u_pts)
                #clenshaw_curtis_grid(n_u_pts)
            else:
                if ifl == 0:
                    npts = n_z_pts
                elif ifl == 1:
                    npts = n_L_pts
                gauss_quad(npts)
        if ifl == 2:
            wu_ref,u_ref = np.transpose(np.genfromtxt(fl,delimiter=',',skip_header=1))
        else:
            if ifl == 0:
                wz_ref,z_ref = np.transpose(np.genfromtxt(fl,delimiter=',',skip_header=1))
            elif ifl == 1:
                wL_ref,L_ref = np.transpose(np.genfromtxt(fl,delimiter=',',skip_header=1))
    if eps == 'X':
        L = np.ones(1)
        wL = np.ones(1)
    else:
        # 0 < lambda < 1
        L,wL = mk_axis(L_ref,wL_ref,(0.0,1.0),1,npartition['L'],rescale=False)

    # 0 < z < infinity
    # first shift to 0 < z < 1

    z,wz = mk_axis(z_ref,wz_ref,(-1.0,1.0),zcp,npartition['z'])
    """
    oz = z
    owz = wz
    z_ref = 0.5*(z + 1.0)
    wz_ref = 0.5*wz
    z = 0.5*(z + 1.0)
    wz = 0.5*wz

    z_max = 3
    for ival in range(1,z_max): # add intervals
        z = np.append(z,z_ref + ival)
        wz = np.append(wz,wz_ref)
    z = np.append(z,z_max/z_ref) # then append much larger values
    wz = np.append(wz,z_max*wz_ref/z_ref**2)
    """

    # 0 < u < infinity
    u,wu = mk_axis(u_ref,wu_ref,(-1.0,1.0),ucp,npartition['u'])

    """
    u = 0.5*(u + 1.0)
    wu = 0.5*wu

    u_max = 3
    for ival in range(1,u_max): # add intervals
        u = np.append(u,u_ref + ival)
        wu = np.append(wu,wu_ref)
    #u = np.append(u,u_ref[:-1]/(1.0 - u_ref[:-1]) + u_max) # then append much larger values
    #wu = np.append(wu,wu_ref[:-1]/(1.0 - u_ref[:-1])**2)
    u = np.append(u,u_max/z_ref) # then append much larger values
    wu = np.append(wu,u_max*wz_ref/z_ref**2)
    """
    grid = np.asarray(list(product(z,L,u))) # for convenience, going to make the grid
    weight = np.zeros(grid.shape[0]) # and weights 1D arrays
    for iw_vec,w_vec in enumerate(product(wz,wL,wu)):
        twz,twL,twu = w_vec
        weight[iw_vec] = twz*twL*twu

    ws = 0.0
    if eps == 'X' or eps == 'XC':
        ws = np.sum(wz)

    return grid,weight,ws

def eps_quick(gridgen,pars={},rs_l=[],inps=None):

    #establish_dependencies()

    if gridgen == 'auto':
        grid,wg,ws = set_up_grid(settings.z_pts,settings.lambda_pts,settings.u_pts)
        z = grid[:,0]
        lamb = grid[:,1]
        u = grid[:,2]
    elif gridgen == 'user':
        z = inps[:,0]
        lamb = inps[:,1]
        u = inps[:,2]
        wg = inps[:,3]
    need_chi_0 = False
    need_chi_lambda = False
    if eps == 'X' or eps == 'C':
        need_chi_0 = True
    if eps == 'XC' or eps == 'C':
        need_chi_lambda = True

    eps_d = {}

    if len(rs_l)==0:
        rs_l = settings.rs_list

    for rs in rs_l:

        n = 3.0/(4.0*pi*rs**3)
        kf = (9.0*pi/4.0)**(1.0/3.0)/rs
        ef = kf**2/2.0

        eps_d[rs] = 0.0
        if need_chi_lambda:
            chi_lamb = chi_parser(z,u*1.0j,lamb,rs,settings.fxc,imag_freq=True,reduce_omega=True,pars=pars,LDA=settings.LDA)
        if need_chi_0:
            chi_0 = chi_parser(z,u*1.0j,None,rs,'chi0',imag_freq=True,reduce_omega=True,pars=pars,LDA=settings.LDA)
        if eps == 'X':
            integrand = chi_0.real
        elif eps == 'C':
            integrand = chi_lamb.real - chi_0.real
        elif eps == 'XC':
            integrand = chi_lamb.real

        eps_d[rs] = -3*np.sum(wg*integrand)
        #print(eps_d[rs],2*kf/pi*ws)
        if eps == 'X' or eps == 'XC':
            eps_d[rs] -= 2*kf/pi*ws

    return eps_d

def gridtest(var,data):

    tucp,tzcp,nz,nL,nu = var
    grid,wg,ws = set_up_grid(settings.z_pts,settings.lambda_pts,settings.u_pts,ucp=tucp,zcp=tzcp,npartition={'z':nz,'L':nL,'u':nu})
    ogrid = np.zeros((grid.shape[0],grid.shape[1]+1))
    ogrid[:,:3] = grid
    ogrid[:,3] = wg
    epsd = eps_quick('user',rs_l=data[:,0],inps=ogrid)
    abs_err = np.zeros(len(epsd.keys()))
    for irs,rs in enumerate(epsd):
        abs_err[irs] = abs(epsd[rs]-data[:,1][irs])
    mae = np.sum(abs_err)/abs_err.shape[0]
    return mae

def get_conv_grid():
    dat = np.genfromtxt('/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/eps_c_data_28_december_2020/epsilon_C_'+settings.fxc+'.csv',delimiter=',',skip_header=1)

    tucp_l = np.arange(0.5,2.1,0.5)
    tzcp_l = np.arange(0.5,2.1,0.5)
    nz_l = np.arange(2,5,1)
    nL_l = np.arange(2,5,1)
    nu_l = np.arange(2,5,1)

    to_do = product(tucp_l,tzcp_l,nz_l,nL_l,nu_l)
    if settings.nproc > 1:
        wdo = []
        for var in to_do:
            wdo.append([var,dat])
        pool = multiprocessing.Pool(processes=min(settings.nproc,len(wdo)))
        tout = pool.starmap(gridtest,wdo)
        pool.close()
        tout = np.asarray(tout)
        best = np.argmin(tout)
        bmae = tout[best]
        bvar = [x for x in wdo[best][0]]
    else:
        bmae = 1e20
        for var in to_do:
            mae = gridtest(var,dat)
            if mae < bmae:
                bmae = mae
                bvar = [tucp,tzcp,nz,nL,nu]

    return bmae,bvar


if __name__ == "__main__":
    """
    from lsda import ec_rpa_unp
    edict = eps_quick('auto')
    for rs in edict:
        print(edict[rs],ec_rpa_unp(rs))
    exit()
    "" "
    opt = get_conv_grid()
    print(opt)
    exit()
    """
    gg = 'user'
    # NB: ALDA grid calculated with settings.LDA == 'PZ81'
    # MCP07 grid with settings.gki_param == True
    kernel_d = {'RPA': [1.0,0.5,4,4,4], 'ALDA': [1.5,0.5,2,3,4], 'MCP07': [2.0, 1.0, 4, 4, 4]}
    kernel_d['TC'] = kernel_d['MCP07']
    if True:#settings.fxc in kernel_d:
        pars = kernel_d['MCP07']#settings.fxc]
        up = pars[0]
        zp = pars[1]
        grid,wg,ws = set_up_grid(settings.z_pts,settings.lambda_pts,settings.u_pts,ucp=up,zcp=zp,npartition={'z': pars[2], 'L':pars[3], 'u':pars[4]})
        ogrid = np.zeros((grid.shape[0],grid.shape[1]+1))
        ogrid[:,:3] = grid
        ogrid[:,3] = wg
    else:
        gg = 'auto'
        ogrid = np.zeros(0)
    rs_l = np.arange(0.1,1,0.1)
    rs_l = np.append(rs_l,np.arange(1,121,1))
    epsd = eps_quick(gg,rs_l=rs_l,inps=ogrid)
    odata = np.zeros((rs_l.shape[0],2))
    for irs,rs in enumerate(epsd):
        odata[irs,0] = rs
        odata[irs,1] = epsd[rs]
    np.savetxt('./reference_data/'+settings.fxc+'_eps_c_reference.csv',odata,delimiter=',',header='rs,eps_c')
    exit()

    ref_file = '/Users/aaronkaplan/Dropbox/phd.nosync/mcp07_revised/code/eps_c_data_28_december_2020/epsilon_C_'+settings.fxc+'.csv'
    if path.isfile(ref_file):
        dat = np.genfromtxt(ref_file,delimiter=',',skip_header=1)
        epsd =  eps_quick(gg,rs_l=dat[:,0],inps=ogrid)
        for irs,rs in enumerate(epsd):
            print(rs,epsd[rs],dat[:,1][irs],100*(1.0-epsd[rs]/dat[:,1][irs]))#/abs(epsd[rs]+dat[:,1][irs])*200)
    else:
        epsd =  eps_quick(gg,inps=ogrid)
        for irs,rs in enumerate(epsd):
            print(rs,epsd[rs])
