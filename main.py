import numpy as np

import settings
from dft.lsda import ec_pw92
from frequency_moments.frequency_moments import moment_calc
from frequency_moments.third_moment import third_moment_calculation,plotter
from fitting.fit_re_fxc_omega import hx_fit_main,kramers_kronig_plot
from fitting.fit_ec import ec_fitting

rsl = np.linspace(1.0,10.0,2000)
ec,_,_=ec_pw92(rsl,0.0)
np.savetxt('./eps_c_PW92.csv',np.transpose((rsl,ec)),delimiter=',',header='rs,PW92 eps_c',fmt='%.18f,%.18f')

if settings.routine == 'third moment':

    if settings.third_mom_pars['calc']:
        third_moment_calculation(interp=settings.third_mom_pars['interp'])
    if settings.third_mom_pars['plot']:
        plotter()

elif settings.routine == 'moment':

    moment_calc(settings.moment_pars['order'])

elif settings.routine == 'fit hx':

    apar,bpar,cpar,dpar = hx_fit_main()
    if settings.fit_pars['plot']:
        kramers_kronig_plot(pars=[apar,bpar,cpar,dpar])

elif settings.routine == 'fit ec':

    if settings.fit_pars['fit']:
        pars,epsc,errors=ec_fitting()
    else:
        pars = settings.rMCP07_pars

    if settings.fit_pars['plot']:
        tps = (pars['a'],pars['b'],pars['c'],pars['d'])
        plot_rMCP07(tps)

else:
    raise ValueError('Unknown routine, ',settings.routine)
