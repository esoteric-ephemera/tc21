
routine='testing'

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
"""

# enter as scalar or vector
rs_list = [1,4,10,30,69]#[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
if not hasattr(rs_list,'__len__'):
    rs_list = [rs_list]

# 'ALDA', 'RPA', 'MCP07', 'static MCP07', 'TC', 'QV', 'QV_MCP07', 'QV_TC'
fxc = 'QV_TC'

TC_pars = {'a': 4.01, 'b': 1.21, 'c': 0.11, 'd': 1.07}#{'a':4.01067394,'b': 1.21065643, 'c':0.10975759, 'd': 1.07043728}

# PZ81 or PW92
LDA = 'PW92'

q_bounds = {'min':0.01,'max':3.01,'step':0.01} # bounds and stepsize for wavevectors

moment_pars = {'calc':False,'sq_plots': 'single',
'order':0.0, 'prec':1.e-8,
'method':'gk_adap' # method can be gk_adap (Gauss-Kronrod), original (from PNAS), or adap when order = 0
}
third_mom_pars = {'calc':False,'plot':True,
'interp': 'spline' # interp can be spline or linear
}
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

# True: fit to eps_c data in sequences of increasing grid fineness
# False: grid search of fixed spacing
filter_search = True

# True: use three fit parameters (a,b,c); False: use two fit parameters (a,b)
fit_c = True
# True: use a fourth fit parameters
fit_d = True

# optional multicore processing
nproc = 4

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
pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286198
#from julia BigFloat(pi)

# colors for plots
clist = ['darkblue','darkorange','darkgreen','darkred','black']
