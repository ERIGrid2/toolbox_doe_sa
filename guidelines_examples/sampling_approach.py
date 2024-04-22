import random
import os

from SALib.sample import sobol_sequence, sobol, latin, morris, fast_sampler
from SALib import ProblemSpec
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import statistics
from SALib.analyze import fast as fast_analyzer
import pandas as pd
import arrow
from mosaik_pv.pvpanel import PVpanel

FIGURES_FOLDER = 'figures'


def plot_2d_samples(samples, title='No Title Set'):
    logger.info(f'Plot samples for {title} - number of samples: {len(samples)}')
    x = samples[:, 0]
    y = samples[:, 1]
    fig, ax = plt.subplots()
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-1, 5])
    #ax.set_title(title)
    ax.scatter(x, y)
    fig.show()
    # check if folder for figures exists
    if not os.path.isdir(FIGURES_FOLDER):
        os.mkdir(FIGURES_FOLDER)
    fig.savefig(f'{FIGURES_FOLDER}//sampling_{title.replace(" ", "")}.png')


def create_boxplot(data, error, file_name, x_axis='', title=''):
    logger.info(f'data: {data}')
    logger.info(f'error: {error}')

    fig, ax = plt.subplots(1)
    data.plot.bar(yerr=error.values.T, ax=ax, rot=0)
    ax.set_ylabel(x_axis)
    ax.set_xlabel('Parameter')
    plt.title(title)
    #fig.set_size_inches(8, 4)
    plt.show()
    fig.savefig(f'{FIGURES_FOLDER}//{file_name}.png', dpi=300, format='png')


def sample_monte_carlo(problem_def, sampling_size=512):
    samples_list = []
    for i in range(len(problem_def['names'])):
        mean = problem_def['bounds'][i][0]
        std_drv = problem_def['bounds'][i][1]
        sample_range = range(int(1000 * (mean - 3*std_drv)), int(1000 * (mean + 3*std_drv)))
        samples_list.append(np.asarray(random.sample(sample_range, k=sampling_size)) / 1000)
    samples = np.transpose(np.asarray(samples_list))
    return samples


def sample_distribution(problem_def, sampling_size=512):
    samples_list = []
    for i in range(len(problem_def['names'])):
        samples_list.append(np.random.normal(problem_def['bounds'][i][0], problem_def['bounds'][i][1], sampling_size))
    samples = np.transpose(np.asarray(samples_list))
    return samples


def sample_sobol(problem_def, sampling_size=512):
    samples = sobol.sample(problem=problem_def, N=sampling_size, calc_second_order=False)
    return samples


def sample_morris(problem_def, sampling_size=512):
    samples = morris.sample(problem=problem_def, N=sampling_size)
    return samples


def sample_efast(problem_def, sampling_size=512):
    samples = fast_sampler.sample(problem=problem_def, N=sampling_size)
    return samples


def sample_sobol_sequence(problem_def, sampling_size=512):
    samples = sobol_sequence.sample(N=sampling_size, D=len(problem_def['names']))
    for i in range(len(problem_def['names'])):
        mean = problem_def['bounds'][i][0]
        std_drv = problem_def['bounds'][i][1]
        lower_bound = mean - 3*std_drv
        upper_bound = mean + 3*std_drv
        samples[:, i] = samples[:, i] * (upper_bound - lower_bound) + lower_bound
    return samples


def sample_lhs(problem_def, sampling_size=512):
    samples = latin.sample(problem_def, sampling_size)
    return samples


def sampling_example():
    problem = ProblemSpec({'names': ['x1', 'x2'],
                           'groups': None,
                           'dists': ['norm', 'norm'],
                           'bounds': [[1.0, 0.5], [2.0, 1]],
                           })
    logger.info(f'Defined problem: variables {problem["names"]}, dists: {problem["dists"]}, bounds: {problem["bounds"]}')

    n = 64
    logger.info(f'Sampling size: {n}')
    mcs = sample_monte_carlo(problem, sampling_size=n)
    plot_2d_samples(mcs, title='Monte Carlo (Interval)')  #
    distr = sample_distribution(problem, sampling_size=n)
    plot_2d_samples(distr, title='Monte Carlo (Distribution)')  #
    sobol = sample_sobol(problem, sampling_size=n)
    plot_2d_samples(sobol, title='Sobol')  #
    sobol_seq = sample_sobol_sequence(problem, sampling_size=n)
    plot_2d_samples(sobol_seq, title='Sobol Sequence')  #
    lhs = sample_lhs(problem, sampling_size=n)
    plot_2d_samples(lhs, title='Latin Hypercube')  #


def pv_Example(n):
    problem = ProblemSpec({'names': ['dni'],
                           'groups': None,
                           'dists': ['norm'],
                           'bounds': [[0.6, 0.05]],
                           })
    logger.info(f'Defined problem: variables {problem["names"]}, '
                f'dists: {problem["dists"]}, bounds: {problem["bounds"]}')
    logger.info(f'Sampling size: {n}')

    pvpanel = PVpanel(lat=32.117, area=1, efficiency=0.2, el_tilt=45, az_tilt=0)
    date = {'day': 144,
            'hour': 14,
            'minute': 0}

    samples = {}
    # samples['MonteCarlo'] = sample_monte_carlo(problem, sampling_size=n)
    # samples['SobolSequence'] = sample_sobol_sequence(problem, sampling_size=n)
    # samples['Distribution'] = sample_distribution(problem, sampling_size=n)
    # samples['Sobol'] = sample_sobol(problem, sampling_size=n)
    samples['LatinHypercube'] = sample_lhs(problem, sampling_size=n)

    results = {'mean': {}, 'stdev': {}, 'variance': {}}
    for sampling_method, sample in samples.items():
        pv_power = [pvpanel.power(dni, date)[0] for dni in sample]
        logger.info(f'Mean: {statistics.mean(pv_power):.6f} - '
                    f'Standard Derivation: {statistics.stdev(pv_power):.6f} - '
                    f'Variance: {statistics.variance(pv_power):.6f} - '
                    f'{sampling_method}')
        results['mean'][sampling_method] = statistics.mean(pv_power)
        results['stdev'][sampling_method] = statistics.stdev(pv_power)
        results['variance'][sampling_method] = statistics.variance(pv_power)

    results_df = pd.DataFrame(results)
    data = results_df['mean']
    error = results_df['stdev']
    create_boxplot(data, error, file_name='pv_Example', x_axis='Mean', title='PV Example')


def pv_example_2_factors(n):
    problem = ProblemSpec({'names': ['dni', 'el_tilt'],
                           'groups': None,
                           'dists': ['norm', 'norm'],
                           'bounds': [[0.6, 0.05], [45, 3]],
                           })
    logger.info(f'Defined problem: variables {problem["names"]}, '
                f'dists: {problem["dists"]}, bounds: {problem["bounds"]}')
    logger.info(f'Sampling size: {n}')
    date = {'day': 144,
            'hour': 14,
            'minute': 0}

    samples = {}
    samples['MonteCarlo'] = sample_monte_carlo(problem, sampling_size=n)
    samples['SobolSequence'] = sample_sobol_sequence(problem, sampling_size=n)
    samples['Distribution'] = sample_distribution(problem, sampling_size=n)
    samples['Sobol'] = sample_sobol(problem, sampling_size=n)
    samples['LatinHypercube'] = sample_lhs(problem, sampling_size=n)

    results = {'mean': {}, 'stdev': {}, 'variance': {}}
    for sampling_method, sample in samples.items():
        pv_panels = [PVpanel(lat=32.117, area=1, efficiency=0.2, el_tilt=el_tilt, az_tilt=0) for dni, el_tilt in sample]
        pv_power = [pv_panels[count].power(values[0], date) for count, values in enumerate(sample)]
        logger.info(f'Mean: {statistics.mean(pv_power):.6f} - '
                    f'Standard Derivation: {statistics.stdev(pv_power):.6f} - '
                    f'Variance: {statistics.variance(pv_power):.6f} - '
                    f'{sampling_method}')
        results['mean'][sampling_method] = statistics.mean(pv_power)
        results['stdev'][sampling_method] = statistics.stdev(pv_power)
        results['variance'][sampling_method] = statistics.variance(pv_power)

    results_df = pd.DataFrame(results)
    data = results_df['mean']
    error = results_df['stdev']
    create_boxplot(data, error, file_name='pv_example_2_factors', x_axis='Mean', title='PV Example with 2 Factors')


def pv_example_3_factors(n):
    problem = ProblemSpec({'names': ['dni', 'el_tilt', 'az_tilt'],
                            'groups': None,
                            'dists': ['norm', 'norm', 'norm'],
                            'bounds': [[0.6, 0.05], [45, 3], [0, 3]],
                            })
    logger.info(f'Defined problem: variables {problem["names"]}, '
                f'dists: {problem["dists"]}, bounds: {problem["bounds"]}')
    logger.info(f'Sampling size: {n}')
    date = {'day': 144,
            'hour': 14,
            'minute': 0}

    samples = {}
    samples['MonteCarlo'] = sample_monte_carlo(problem, sampling_size=n)
    samples['SobolSequence'] = sample_sobol_sequence(problem, sampling_size=n)
    samples['Distribution'] = sample_distribution(problem, sampling_size=n)
    samples['Sobol'] = sample_sobol(problem, sampling_size=n)
    samples['LatinHypercube'] = sample_lhs(problem, sampling_size=n)

    results = {'mean': {}, 'stdev': {}, 'variance': {}}
    for sampling_method, sample in samples.items():
        pv_panels = [PVpanel(lat=32.117, area=1, efficiency=0.2, el_tilt=el_tilt, az_tilt=az_tilt)
                     for dni, el_tilt, az_tilt in sample]
        pv_power = [pv_panels[count].power(values[0], date) for count, values in enumerate(sample)]
        logger.info(f'Mean: {statistics.mean(pv_power):.6f} - '
                    f'Standard Derivation: {statistics.stdev(pv_power):.6f} - '
                    f'Variance: {statistics.variance(pv_power):.6f} - '
                    f'{sampling_method}')
        results['mean'][sampling_method] = statistics.mean(pv_power)
        results['stdev'][sampling_method] = statistics.stdev(pv_power)
        results['variance'][sampling_method] = statistics.variance(pv_power)

    results_df = pd.DataFrame(results)
    data = results_df['mean']
    error = results_df['stdev']
    create_boxplot(data, error, file_name='pv_example_3_factors', x_axis='Mean', title='PV Example with 3 Factors')


def pv_timeseries(n):
    dni_timeseries = pd.read_csv('data\\dni_savannah_lat32_august.csv')
    dni_timeseries.set_index('Date', inplace=True)
    error_timeseries = pd.read_csv('data\\error_savannah2_August.csv')
    error_timeseries.set_index('Date', inplace=True)
    dni_data = pd.concat([dni_timeseries, error_timeseries], axis=1)
    results = pd.DataFrame(columns=['date', 'mean', 'stdev', 'var'])
    count = 0
    for ind, row in dni_data.iterrows():
        problem = ProblemSpec({'names': ['dni'],
                                'groups': None,
                                'dists': ['norm'],
                                'bounds': [[row['DNI'], row['Error']]],
                                })
        dni_samples = sample_distribution(problem, sampling_size=n)
        pv_panel = PVpanel(lat=32.117, area=1, efficiency=0.2, el_tilt=45, az_tilt=0)
        arrow_date = arrow.get(ind)
        date = {'day': int(arrow_date.format("DDDD")),
                'hour': int(arrow_date.format("HH")),
                'minute': int(arrow_date.format("m"))}
        pv_power = [pv_panel.power(values[0], date) for count, values in enumerate(dni_samples)]
        mean = statistics.mean(pv_power)
        stdev = statistics.stdev(pv_power)
        variance = statistics.variance(pv_power)
        df2 = pd.DataFrame([[ind, mean, stdev, variance]], columns=['date', 'mean', 'stdev', 'var'])
        results = pd.concat([results, df2])
        #logger.info(f'{ind}: Mean: {mean:.6f} - Standard Derivation: {stdev:.6f} - Variance: {variance:.6f} - ')

        count += 1
        if count > 24 * 1:
            break

    # 2 stdev (95%) oder 1 stdev?
    fig, ax = plt.subplots(1)
    results.set_index('date', inplace=True)
    results['error_max'] = results['mean'] + 3 * results['stdev']
    results['error_min'] = results['mean'] - 3 * results['stdev']
    results.plot(y=['mean', 'error_min', 'error_max'], figsize=(9, 6))
    plt.xlim([8, 16])
    plt.ylim([40, 112])
    plt.ylabel('Power')
    plt.xlabel('Time')
    plt.title('Dynamic PV simulation with uncertainty.')
    plt.show()
    fig.savefig(f'{FIGURES_FOLDER}//pv_timeseries.png', dpi=300, format='png')
    logger.info(results)


def pv_sensitivity_analysis():
    problem = ProblemSpec({'names': ['area', 'efficiency', 'el_tilt'],
                           'groups': None,
                           'dists': None,
                           'bounds': [[4.0, 8.0], [0.2, 0.6], [30.0, 60.0]],
                           })
    logger.info(f'Defined problem: variables {problem["names"]}, dists: {problem["dists"]}, bounds: {problem["bounds"]}')
    n = 4096
    morris_samples = sample_morris(problem, n)
    efast_samples = sample_efast(problem, n)
    sobol = sample_sobol(problem, sampling_size=n)
    date = {'day': 144,
            'hour': 14,
            'minute': 0}
    pv_panels = [PVpanel(lat=32.117, area=sample[0], efficiency=sample[1], el_tilt=sample[2], az_tilt=0)
                 for sample in efast_samples]
    dni = 0.6
    pv_power = [pvpanel.power(dni, date) for pvpanel in pv_panels]

    # si = sobol_analyzer.analyze(problem, np.asarray(pv_power), calc_second_order=False, conf_level=0.95,
    #                             print_to_console=True)
    si = fast_analyzer.analyze(problem, np.asarray(pv_power))

    # si_filter = {k: si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
    si_df = pd.DataFrame(si, index=problem["names"])
    # si_df.to_csv(f'{folder}/{scenario_name}_si_analysis_{target_metric}.csv')

    data = si_df[["S1", "ST"]]
    error = si_df[["S1_conf", "ST_conf"]]
    create_boxplot(data, error, file_name='si_analysis', x_axis='Sobol Index', title='Sobol Index Example')



if __name__ == '__main__':
    # 'bounds': [[0.0, 1.0], [1.0, 0.75], [0.0, 0.2], [0.0, 0.2], [-1.0, 1.0], [1.0, 0.25]],
    # 'dists': ['unif', 'triang', 'norm', 'lognorm', 'unif', 'triang']

    sampling_example()
    n = 64
    logger.info(f'First example with one factor.')
    pv_Example(n=n)
    logger.info(f'Second example with two factors.')
    pv_example_2_factors(n=n)
    logger.info(f'Second example with three factors.')
    pv_example_3_factors(n=n)
    pv_sensitivity_analysis()

    pv_timeseries(n=n)
