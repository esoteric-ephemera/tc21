import numpy as np

import settings

def main():

    if settings.routine == 'M3SR':

        from frequency_moments.third_moment import third_moment_calculation,third_moment_plotter

        if settings.third_mom_pars['calc']:
            third_moment_calculation(interp=settings.third_mom_pars['interp'])
        if settings.third_mom_pars['plot']:
            third_moment_plotter()

    elif settings.routine == 'FMT':

        from frequency_moments.frequency_moments import moment_calc,sq_plots,sq_tc_mcp07_comp_plot

        if settings.moment_pars['calc']:
            if hasattr(settings.moment_pars['order'],'__len__'):
                for amom in settings.moment_pars['order']:
                    moment_calc(amom)
            else:
                moment_calc(settings.moment_pars['order'])
        if settings.moment_pars['plot']:
            if settings.moment_pars['sq_plots']=='single':
                sq_plots()
            elif settings.moment_pars['sq_plots']=='comp':
                sq_tc_mcp07_comp_plot()

    elif settings.routine == 'HXFIT':

        from fitting.fit_re_fxc_omega import hx_fit_main,kramers_kronig_plot
        from fitting.kramers_kronig import kramers_kronig_re_fxc

        kramers_kronig_re_fxc()
        hxpars = hx_fit_main()
        if settings.gen_opts['plot']:
            kramers_kronig_plot(pars=hxpars)

    elif settings.routine == 'ECFIT':

        from fitting.fit_ec import ec_fitting,plot_TC

        if settings.gen_opts['calc']:
            pars,epsc,errors=ec_fitting()
        else:
            pars = settings.TC_pars

        if settings.gen_opts['plot']:
            tps = (pars['a'],pars['b'],pars['c'],pars['d'])
            plot_TC(tps)

    elif settings.routine == 'KFC':

        from static_cdw.get_kf_critical import kf_crit_search,kf_crit_plots

        if settings.gen_opts['calc']:
            kf_crit_search()
        if settings.gen_opts['plot']:
            kf_crit_plots()

    elif settings.routine == 'QVRSC':

        from plotters.qv_critical_rs import plot_qv_rs_crit

        plot_qv_rs_crit(regen_dat=settings.gen_opts['calc'])

    elif settings.routine == 'GHPLAS':

        from plotters.ghost_plasmon import plot_ghost_exciton

        plot_ghost_exciton(regen_qv_dat=settings.gen_opts['calc'])

    elif settings.routine == 'PKER':

        from plotters.plot_kernel import fxc_plotter

        for rs in settings.rs_list:
            fxc_plotter(rs)

    elif settings.routine == 'PQMC':

        from plotters.qmc_data_plotter import plot_qmc_sq_dat

        plot_qmc_sq_dat()

    elif settings.routine == 'PDISP':

        from plotters.plasmon_dispersion import plasmon_dispersion

        plasmon_dispersion()

    elif settings.routine == 'UNLOC':

        from ultranonlocality.alpha_sum import calc_alpha,alpha_plotter

        if settings.gen_opts['calc']:
            calc_alpha(['DLDA','MCP07','TC'])
        if settings.gen_opts['plot']:
            alpha_plotter(['DLDA','MCP07','TC'],sign_conv=-1)

    elif settings.routine == 'PDFLUC':

        from plotters.plot_fluctuations import plot_dens_fluc

        plot_dens_fluc()

    elif settings.routine == 'IFREQ':

        from fitting.fit_gki_ac import fit_tc21_ifreq
        from fitting.get_gki_analytic_cont import get_gki_ac

        get_gki_ac()
        fit_tc21_ifreq()

    elif settings.routine == 'testing':
        # space just for testing unfinished parts of routines

        return
    else:
        raise ValueError('Unknown routine, ',settings.routine)

    return

if __name__ == "__main__":

    main()
