import traceback

import sys
import datetime
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics

from SALib.analyze import sobol, fast
from pandas import DataFrame

import benchmark_2_example.benchmark_multi_energy_sim as benchmark_sim
import toolbox_start as benchmark_sa


logger.remove()
logger.add("results.log", level="DEBUG")
logger.add(sys.stderr, level="DEBUG")


def plot_graphs(recipes, folder, format, dpi, scenario_name='test_1'):
    def plot_and_save(store, parameter, title, keys, xlabel='simulation time', ylabel='Power'):

        df_to_plot = DataFrame()
        i = 0

        for key in store.keys():
            df_to_plot['ID_' + str(i)] = store[key][keys]
            i += 1
        df_to_plot.plot(title=title, legend=True, xlabel='simulation time', ylabel=ylabel)
        plt.savefig(f'{folder}/{scenario_name}_{parameter}.{format}', dpi=dpi, format=format)

    # Parameters to be plotted
    parameters = [
        {'parameter': 'heat_pump_power', 'title': 'Heat pump power',
         'keys': ('HeatPumpSim-0.heatpump_0', 'P_effective'), 'ylabel': 'Power'},
        {'parameter': 'voltage_bus1', 'title': 'Voltage Bus 1', 'keys': ('ElNetworkSim-0.Bus_1_0', 'vm_pu'),
         'ylabel': 'Voltage'},
        {'parameter': 'voltage_bus2', 'title': 'Voltage Bus 2', 'keys': ('ElNetworkSim-0.Bus_2_0', 'vm_pu'),
         'ylabel': 'Voltage'},
        {'parameter': 'noise_generator_output', 'title': 'Noise Generator Output',
         'keys': ('NoiseGeneratorSim-0.ng__0', 'output'), 'ylabel': 'Voltage'},
        {'parameter': 'volt_ctrl_setpoint', 'title': 'Vlt Ctrl Setpoint',
         'keys': ('VoltageCtrlSim-0.VoltageController_0', 'hp_p_el_kw_setpoint'), 'ylabel': 'Setpoint'},
        {'parameter': 'HEX_heat_Power', 'title': 'HEX Heat Power',
         'keys': ('HeatExchangerSim-0.HEXConsumer_0', 'P_heat'), 'ylabel': 'Power'},
        {'parameter': 'tank_average_temperature', 'title': 'Tank average temp',
         'keys': ('StorageTankSim-0.StratifiedWaterStorageTank_0', 'T_avg'), 'ylabel': 'Temperature'}
    ]

    # Plotting parameters
    store = pd.HDFStore(benchmark_sim.get_store_filename(scenario_name, ''))
    print('')

    for param in parameters:
        plot_and_save(store, param['parameter'], param['title'], param['keys'], xlabel='simulation time',
                      ylabel=param['ylabel'])

    store.close()


def do_f_test(param1, param2, target_metric, results):
    relevant_results = results[[param1, param2, target_metric]]

    values_param1 = results.groupby(param1).first().index.values.tolist()  # get the variations of param 1
    values_param2 = results.groupby(param2).first().index.values.tolist()  # get the variations of param 2

    param1_option1 = list(results[(results[param1] == values_param1[0])][target_metric])
    param1_option2 = list(results[(results[param1] == values_param1[1])][target_metric])
    param2_option1 = list(results[(results[param2] == values_param2[0])][target_metric])
    param2_option2 = list(results[(results[param2] == values_param2[1])][target_metric])

    value_list_param2 = []
    for value_param2 in values_param2:
        value_list_param2.append(relevant_results[(relevant_results[param2] == value_param2)])

    from scipy import stats

    Fa, pa = stats.f_oneway(param1_option1, param1_option2)
    Fb, pb = stats.f_oneway(param2_option1, param2_option2)
    Fab, pab = stats.f_oneway([results.iloc[0][target_metric], results.iloc[3][target_metric]],
                              [results.iloc[1][target_metric], results.iloc[2][target_metric]])

    return Fa, Fb, Fab, pa, pb, pab


def do_anova_analysis(results, variation_params, target_metric):
    import itertools

    param1 = list(variation_params.keys())[0]
    param2 = list(variation_params.keys())[1]

    for param1, param2 in itertools.product(variation_params.keys(), variation_params.keys()):
        if param1 != param2:
            Fa, Fb, Fab, pa, pb, pab = do_f_test(param1=param1,
                                                 param2=param2,
                                                 target_metric=target_metric,
                                                 results=results)

            # Write results to file
            logger.info(results.to_markdown())
            # logger.info('HP Power: ' + str(sci1) + ', ' + str(sci2) + ', ' + str(sci3) + ', ' + str(sci4))
            logger.info(f'f_oneway test hypothesis a: {param1} - hypothesis b: {param2} - Target metric: {target_metric}')
            logger.info('computed F statistic of the test: Fa: ' + str(Fa) + ', Fb: ' + str(Fb) + ', Fab: ' + str(Fab))
            logger.info(
                'associated p-value from the F distribution: pa: ' + str(pa) + ', pb: ' + str(pb) + ', pab: ' + str(pab))
            logger.info(' (p < 0.05: Hypothesis 0 (same variance) is rejected -> different variances in sample -> '
                  'change in parameter has effect.)')

            # Plot results
            # plt.plot([1, 2], [(sci1 + sci3) / 2, (sci2 + sci4) / 2], 'b-', label="Effect of size of hot water tank")
            # plt.plot([1, 2], [(sci1 + sci2) / 2, (sci3 + sci4) / 2], 'r-', label="Effect of controller deadband")
            # plt.legend()
            # plt.savefig(f'{folder}/{scenario_name}_analysis_of_effect.{format}', dpi=dpi, format=format)
            # plt.show()


def do_oat_analysis(results, variation_params, target_metric, folder_figures, scenario_name='test_1',
                      dpi=300, format='png', plt_show=False):
    results_df = pd.DataFrame()
    variances_df = pd.DataFrame()
    for count, factor in enumerate(variation_params.keys()):
        min = results[target_metric].iloc[2 * count + 1]
        mean = results[target_metric].iloc[0]
        max = results[target_metric].iloc[2 * count + 2]
        # results_df = results_df.append({'factor': factor, 'min': min, 'mean': mean, 'max': max}, ignore_index=True)
        results_df = results_df.append({'factor': factor, 'value': min}, ignore_index=True)
        results_df = results_df.append({'factor': factor, 'value': mean}, ignore_index=True)
        results_df = results_df.append({'factor': factor, 'value': max}, ignore_index=True)
        logger.info(f'Target metric: {target_metric} - Factor: {factor} - changed with min {(min / mean * 100):.2f}%')
        logger.info(f'Target metric: {target_metric} - Factor: {factor} - changed with max {(max / mean * 100):.2f}%')
        variances_df = variances_df.append({'factor': factor, 'variance': statistics.variance([min, mean, max])},
                                           ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.boxplot(by='factor', ax=ax)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title(f'Target metric: {target_metric}')
    if plt_show:
        plt.show()
    plt.savefig(f'{folder_figures}\\{scenario_name}_{target_metric}.{format}',
                dpi=dpi,
                format=format,
                bbox_inches="tight")
    results_df.to_csv(f'{folder_figures}\\{scenario_name}_{target_metric}.csv')
    logger.info(variances_df.sort_values(by=['variance']).to_markdown())

    return variances_df


def do_sobol_analysis(results, variation_params, target_metric, folder_figures, scenario_name='test_1',
                      dpi=300, format='png', plt_show=False):
    from matplotlib import pyplot as plt

    if len(list(variation_params.keys())) == 1:
        param = list(variation_params.keys())[0]
        exog = results[[param]]
        endog = results[[target_metric]]
        all = pd.concat([exog, endog], axis=1).sort_values(by=[param])
        fig = plt.figure()
        ax = plt.axes()
        all.plot(ax=ax, x=param, y=target_metric)
        plt.savefig(f'{folder_figures}/{scenario_name}_sobol_1_factor_{target_metric}.{format}', dpi=dpi, format=format)
        return
    param1 = list(variation_params.keys())[0]
    param2 = list(variation_params.keys())[1]
    # Get the used treatments (battery size and power)
    exog = results[[param1, param2]]
    # Get the self consumption index as response for each treatment
    endog = results[[target_metric]]

    #logger.info(results.to_markdown())
    print(results.to_markdown())

    # We want to fit a metamodel to our system
    # We chose Kriging (Gaussian Process) this time around. You can also choose other metamodels if you want
    # (simplest would be linear interpolation)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    # Kriging needs a kernel. This kernel parameterization should work, but you can also play around with it.
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(exog.values, endog.values)  # Fit the Kriging model to our samples and associated responses

    # Next we need to evaluate our metamodel. Thus, we set up a 50x50 point evaluation grid:
    var_item1 = variation_params[param1]
    var_item2 = variation_params[param2]
    x_vec, y_vec = np.meshgrid(np.linspace(var_item1['min'], var_item1['max'], 50),
                               np.linspace(var_item2['min'], var_item2['max'], 50))

    # x_vec and y_vec is for plotting. We reshape them to get vectors for predicting:
    evalgrid = np.array([x_vec.flatten(), y_vec.flatten()]).T

    # Using our metamodel for predictions (Kriging also gives us a sense of uncertainty via the sigma):
    scipred, sigma = gp.predict(evalgrid, return_std=True)

    # For plotting we need to reshape our prediction array:
    pltsci = np.reshape(scipred, (50, 50))

    from mpl_toolkits.mplot3d import Axes3D
    # matplotlib
    # This gives us the figure in an extra window
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Labelling Axis
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel(target_metric)
    # ax = fig.gca(projection='3d') #Deprecated since Matplotlib 3.4
    # Get 3D surface plot:
    surf = ax.plot_surface(x_vec, y_vec, pltsci)
    # Include our original sample points:
    ax.plot(exog.values[:, 0], exog.values[:, 1], endog.values.ravel(), 'r*')
    plt.savefig(f'{folder_figures}/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}_{scenario_name}_{target_metric}.'
                f'{format}', dpi=dpi, format=format)
    if plt_show:
        plt.show()


def analyze_results(recipes, variations_dict, basic_conf, folder='figures', format='png', dpi=300, doe_type='anova',
                    plots=True, target_metrics=[], folder_figures='figures', scenario_name='test_1', plt_show=False):
    print('')
    if plots:
        plot_graphs(recipes, folder, format, dpi)

    # Store results and get response summaries
    run_store = pd.HDFStore(f'{basic_conf["folder_temp_files"]}\\{basic_conf["summary_filename"]}.h5')
    results = [run_store[k] for k in run_store.keys()]
    run_store.close()
    results = pd.concat(results, axis=0).set_index('ID')

    # for analysis min and max values are needed
    variation_params = {}
    for entity, params in variations_dict.items():
        for param, variation in params.items():
            variation_params[f'{entity}.{param}'] = {}
            if isinstance(variation, dict):
                variation_params[f'{entity}.{param}']['min'] = variation['mean'] - variation['stdvs'] * 3
                variation_params[f'{entity}.{param}']['max'] = variation['mean'] + variation['stdvs'] * 3
            else:
                variation_params[f'{entity}.{param}']['min'] = variation[0]
                variation_params[f'{entity}.{param}']['max'] = variation[1]

    if doe_type == 'sobol' or doe_type == 'LHS':
        for target_metric in target_metrics:
            do_sobol_analysis(results, variation_params, target_metric, folder_figures,
                              scenario_name=scenario_name, dpi=dpi, format=format, plt_show=False)
    elif doe_type == 'extreme_points':
        for target_metric in target_metrics:
            do_anova_analysis(results, variation_params, target_metric)
    elif doe_type == 'OAT':
        variances_df = None
        ranking = {}
        for i, target_metric in enumerate(target_metrics):
            variances_res = do_oat_analysis(results, variation_params, target_metric, folder_figures,
                                            scenario_name=scenario_name, dpi=dpi, format=format, plt_show=False)
            if variances_df is None:
                variances_df = pd.DataFrame(columns=variances_res.iloc[:, 0])

            variance_sorted = variances_res.sort_values(by=['variance'], ascending=False)
            for i, (index, row) in enumerate(variance_sorted.iterrows()):
                if row['factor'] in ranking:
                    ranking[row['factor']] += i
                else:
                    ranking[row['factor']] = i

            variances_res = variances_res.transpose()
            variances_res.columns = variances_res.iloc[0]
            variances_df = variances_df.append(variances_res[1:], ignore_index=True)
        # logger.info(variances_df.to_markdown())
        variance_sum = variances_df.sum(axis=0)
        # logger.info('Sum of variance over all target metrics')
        # logger.info(variance_sum.sort_values(ascending=False))

        fig, ax = plt.subplots(figsize=(12, 4))
        variance_sum.plot.bar(ax=ax)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(f'Variance sum of factor over all targets')
        ax.set_xlabel('Factors')
        if plt_show:
            plt.show()
        plt.savefig(f'{folder_figures}\\{scenario_name}_factor_varianceSum.{format}',
                    dpi=dpi,
                    format=format,
                    bbox_inches="tight")
        variance_sum.to_csv(f'{folder_figures}\\{scenario_name}_factor_varianceSum.csv')

        fig, ax = plt.subplots(figsize=(12, 4))
        ranking_df = pd.DataFrame.from_dict([ranking]).T
        ranking_df.sort_values(by=0).plot.bar(ax=ax, legend=None)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(f'Ranking of factor over all targets (sum of rank)')
        ax.set_xlabel('Factors')
        if plt_show:
            plt.show()
        plt.savefig(f'{folder_figures}\\{scenario_name}_factor_ranking.{format}',
                    dpi=dpi,
                    format=format,
                    bbox_inches="tight")
        variance_sum.to_csv(f'{folder_figures}\\{scenario_name}_factor_ranking.csv')

    elif doe_type == 'sobol_indices' or doe_type == 'fast':
        problem, discrete = benchmark_sa.create_problem(variations_dict)
        # si_results = DataFrame()
        for target_metric in target_metrics:
            logger.info(f'Do sobol indices analysis for target metric {target_metric}:')
            try:
                results_array = results[[target_metric]].iloc[:, 0].to_numpy()
                logger.info(f'Reults_array: {results_array}')
                if doe_type == 'sobol_indices':
                    si = sobol.analyze(problem,
                                       results_array,
                                       calc_second_order=False,
                                       conf_level=0.95,
                                       print_to_console=False)
                elif doe_type == 'fast':
                    si = fast.analyze(problem,
                                      results_array,
                                      conf_level=0.95,
                                      print_to_console=False)
                si_filter = {k: si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
                si_df = pd.DataFrame(si_filter, index=problem["names"])
                si_df.to_csv(f'{folder}/{scenario_name}_si_analysis_{target_metric}.csv')
                # pandas.concat([si_results, pd.DataFrame(si_filter, index=problem["names"])])

                fig, ax = plt.subplots(1)
                # ax.set_ylim([-1, 3])
                indices = si_df[["S1", "ST"]]
                err = si_df[["S1_conf", "ST_conf"]]
                indices.plot.bar(yerr=err.values.T, ax=ax, rot=0)
                ax.tick_params(axis='x', labelrotation=5)
                ax.set_ylabel('Sobol Index')
                ax.set_xlabel('Parameter')
                plt.title(target_metric)
                fig.set_size_inches(8, 4)
                # plt.savefig(f'{folder}/{scenario_name}_si_analysis_2_{target_metric}.{format}', dpi=dpi, format=format)
                fig.savefig(f'{folder}/{scenario_name}_si_analysis_{target_metric}.{format}', dpi=dpi, format=format)
            except Exception as e:
                logger.info(f'Exception for target metric: {target_metric}: {e}')
                traceback.print_exception(*sys.exc_info())
        # logger.info(si_results.to_markdown)
        # si_results.to_csv('si_analysis.csv')
    else:
        logger.info('No analysis type defined, which is matching with the available types.')