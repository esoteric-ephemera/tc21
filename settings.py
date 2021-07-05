
routine='ECFIT'

"""
    routine options:
        M3SR : third frequency-moment sum rule calculation, options set in third_mom_pars

        FMT : arbitary frequency moment calculation, options set in moment_pars

        HXFIT : fits the real part of the GKI kernel using the Kramers-Kronig relations

        ECFIT : fits the TC kernel to jellium correlation energies per electron

        KFC : finds the critical Fermi wavevector for onset of a static cdw

        QVRSC : plot the critical rs such that no solutions exist to parametrize the QV kernel

        GHPLAS : plots dynamic structure factor S(q,omega) to demonstrate ghost plasmon

        PKER : plots the kernel and effective potential of Fermi liquid theory

        PQMC : plots QMC static structure factor, S(q), data

        PDISP : plots plasmon dispersion curves

        PDFLUC : plots average and std. deviation of a density fluctuation

        UNLOC : calculates the ultranonlocality coefficient for the crystal below
"""

# enter as scalar or vector
rs_list = [4,69]
if not hasattr(rs_list,'__len__'):
    rs_list = [rs_list]

# 'ALDA', 'RPA', 'MCP07', 'static MCP07', 'TC', 'QV', 'QV_MCP07', 'QV_TC'
fxc = 'TC'

TC_par_list = [4.470217788196006, 1.4327137309889693, 0.04466295040605292, 2.918135781120395]
TC_par_names = ['a','b','c','d']
TC_pars = {}
for ipar,apar in enumerate(TC_par_names):
    TC_pars[apar] = TC_par_list[ipar]
#{'a': 4.74, 'b': 1.73, 'c': 0.1, 'd': 0.8}#{'a': 4.01, 'b': 1.21, 'c': 0.11, 'd': 1.07}#{'a':4.01067394,'b': 1.21065643, 'c':0.10975759, 'd': 1.07043728}

# PZ81 or PW92
LDA = 'PW92'

q_bounds = {'min':0.01,'max':4.01,'step':0.01} # bounds and stepsize for wavevectors

moment_pars = {'calc':True,'plot':True,'sq_plots': 'single',
'order':0.0, 'prec':1.e-8,
'method':'gk_adap' # method can be gk_adap (Gauss-Kronrod), original (from PNAS), or adap when order = 0
}
third_mom_pars = {'calc':False,'plot':True,
'interp': 'spline' # interp can be spline or linear
}

"""
    ec_fit_pars['method'] can be either
        filter --> use small grids of increasing fineness to filter a good parameter set
        fixed --> single search over a grid of constant spacing
        lsq --> least squares search with scipy (fastest by far)
        lsq_refine --> initial guess with least squares, refine by grid search
    all methods leverage multicore processing for efficiency
"""
ec_fit = {'method': 'lsq'}

gen_opts = {'calc':False, 'plot': True}

# True: use a tabulated parameterization of the GKI kernel
# False: re-evaluate the Cauchy residue integral for each value of I*omega for omega real
gki_param = True

# number of points/interval in z = q/(2*kF) integration
z_pts = 10#10
# number of points/interval in lambda integration
lambda_pts = 5#10
# number of points/interval in u = omega/eps_F integration
u_pts = 10#20

# True: use three fit parameters (a,b,c); False: use two fit parameters (a,b)
fit_c = True
# True: use a fourth fit parameters
fit_d = True

# optional multicore processing
nproc = 6

# initial bounds for parameters. If filter_search = True, step sizes are ignored
a_min = 0.01
a_max = 3.0
a_step = 0.5

# a and b bounds are required
b_min = 0.0
b_max = 2.0
b_step = 0.5

# if fit_c = False, these don't need to be set
c_min = 0.0
c_max = 2.0
c_step = 0.5

d_min = 0.0
d_max = 2.0
d_step = 0.5

"""
    Some constants shared by other modules
"""
#from julia BigFloat(pi)
pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286198
Eh_to_eV = 27.211386245988 # https://physics.nist.gov/cgi-bin/cuu/Value?hr
bohr_to_ang = 0.529177210903 # https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
crystal = 'Al' # the crystal to do ultranonlocality calculations with
# colors for plots
clist = ['darkblue','darkorange','darkgreen','darkred','black']
