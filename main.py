import numpy as np

import settings
from frequency_moments.frequency_moments import moment_calc,sq_plots,sq_tc_mcp07_comp_plot
from frequency_moments.third_moment import third_moment_calculation,third_moment_plotter
from fitting.fit_re_fxc_omega import hx_fit_main,kramers_kronig_plot
from fitting.fit_ec import ec_fitting,plot_TC
from static_cdw.get_kf_critical import kf_crit_search,kf_crit_plots
from plotters.qv_critical_rs import plot_qv_rs_crit
from plotters.ghost_plasmon import plot_ghost_exciton
from plotters.plot_kernel import fxc_plotter
from plotters.qmc_data_plotter import plot_qmc_sq_dat
from plotters.plasmon_dispersion import plasmon_dispersion

def main():

    if settings.routine == 'M3SR':

        if settings.third_mom_pars['calc']:
            third_moment_calculation(interp=settings.third_mom_pars['interp'])
        if settings.third_mom_pars['plot']:
            third_moment_plotter()

    elif settings.routine == 'FMT':

        if settings.moment_pars['calc']:
            moment_calc(settings.moment_pars['order'])
        if settings.moment_pars['sq_plots']=='single':
            sq_plots()
        elif settings.moment_pars['sq_plots']=='comp':
            sq_tc_mcp07_comp_plot()

    elif settings.routine == 'HXFIT':

        apar,bpar,cpar,dpar = hx_fit_main()
        if settings.gen_opts['plot']:
            kramers_kronig_plot(pars=[apar,bpar,cpar,dpar])

    elif settings.routine == 'ECFIT':

        if settings.gen_opts['calc']:
            pars,epsc,errors=ec_fitting()
        else:
            pars = settings.TC_pars

        if settings.gen_opts['plot']:
            tps = (pars['a'],pars['b'],pars['c'],pars['d'])
            plot_TC(tps)

    elif settings.routine == 'KFC':

        if settings.gen_opts['calc']:
            kf_crit_search()
        if settings.gen_opts['plot']:
            kf_crit_plots()

    elif settings.routine == 'QVRSC':

        plot_qv_rs_crit(regen_dat=settings.gen_opts['calc'])

    elif settings.routine == 'GHPLAS':

        plot_ghost_exciton(regen_qv_dat=settings.gen_opts['calc'])

    elif settings.routine == 'PKER':

        for rs in settings.rs_list:
            fxc_plotter(rs)

    elif settings.routine == 'PQMC':

        plot_qmc_sq_dat()

    elif settings.routine == 'PDISP':

        plasmon_dispersion()

    elif settings.routine == 'testing':
        # space just for testing unfinished parts of routines

        return
    else:
        raise ValueError('Unknown routine, ',settings.routine)

    return

if __name__ == "__main__":

    main()
