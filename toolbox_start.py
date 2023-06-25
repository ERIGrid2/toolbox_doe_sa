import copy

import matplotlib.pyplot as plt
import itertools

import pyDOE
import sobol_seq
import json
import multiprocessing as mp
import numpy as np
import os
import sys
import shutil
from loguru import logger

from SALib import ProblemSpec
from SALib.sample import sobol, fast_sampler, latin

import benchmark_2_example.benchmark_multi_energy_sim as benchmark_sim
import benchmark_2_example.benchmark_multi_energy_analysis as benchmark_analysis
import toolbox_analysis

logger.remove()
logger.add("results.log", level="DEBUG")
logger.add(sys.stderr, level="DEBUG")


def check_for_folders(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def read_in_sim_parameters(path):
    f = open(path, "r")
    sim_parameters = json.load(f)
    f.close()
    return sim_parameters


def create_variations_dict(variations_dict, samples):
    variations = {}
    for i in range(samples.shape[0]):
        run_id = 'run_' + str(i).zfill(2)
        variations[run_id] = {'ID': str(i).zfill(2)}
        j = 0
        for entity, factors in list(variations_dict.items()):
            if entity not in variations[run_id]:
                variations[run_id][entity] = {}
            for factor, t in list(factors.items()):
                variations[run_id][entity][factor] = samples[i, j]
                j = j + 1
    return variations


def get_variations_distribution(variations_dict):
    means = []
    stdvs = []
    for entity, factors in list(variations_dict.items()):
        for factor, factor_value in list(factors.items()):
            if type(factor_value) is dict:
                means.append(factor_value['mean'])
                stdvs.append(factor_value['stdvs'])
            else:
                logger.warning(f'Please specifiy mean and stdvs for {entity} - {factor}')
    return means, stdvs


def visualize_variations(variations):
    values = {}
    for run, params in variations.items():
        for entity, factors in params.items():
            try:
                for factor, value in factors.items():
                    if f'{entity}_{factor}' not in values:
                        values[f'{entity}_{factor}'] = []
                    values[f'{entity}_{factor}'].append(value)
            except:
                pass
    plt.figure()
    samples = np.array([list(values.items())[0][1], list(values.items())[1][1]])
    plt.plot(*(samples.T), '*')
    plt.xlabel(list(values.items())[0][0])
    plt.ylabel(list(values.items())[1][0])
    check_for_folders(folder=sim_parameters['folder_figures'])
    plt.savefig(f"{sim_parameters['folder_figures']}/{basic_conf['scenario_name']}_sample_space."
                f"{sim_parameters['format']}", dpi=sim_parameters['dpi'], format=sim_parameters['format'])
    if sim_parameters['show_plots']:
        plt.show()


def create_problem(variations_dict):
    names = []
    dists = []
    bounds = []
    discrete = []
    for entity, factors in variations_dict.items():
        for factor, factor_range in factors.items():
            if isinstance(factor_range, dict):
                if factor_range['type'] == 'norm':
                    names.append(f'{entity}.{factor}')
                    dists.append('norm')
                    bounds.append([factor_range['mean'], factor_range['stdvs']])
                if factor_range['type'] == 'discrete':
                    discrete.append([f'{entity}.{factor}', factor_range['set']])
            elif isinstance(factor_range, list):
                if len(factor_range) == 2:
                    names.append(f'{entity}.{factor}')
                    dists.append('unif')
                    bounds.append([factor_range[0], factor_range[1]])
                else:
                    logger.warning(f'Length of factor range should be 2 (min, max)')
            else:
                logger.info(f'unknown type of range for factor {factor}')
    return ProblemSpec({'names': names,
                        'groups': None,
                        'dists': dists,
                        'bounds': bounds,
                        }), discrete


if __name__ == '__main__':
    config_folder = 'simulation_configurations'
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain.json')
    sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain_oat.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain_designparams.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain_designparams_metamodel_2.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain_scenarioparams_metamodel.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain_scenarioparams.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_inter_domain_sobolIndices.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_demo5.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_tank_scaling.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_tank_scaling_pv.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_tank_scaling_heat.json')
    # sim_parameters = read_in_sim_parameters(f'{config_folder}\simulation_parameters_tank_scaling_lines.json')
    entities_parameters = sim_parameters['entities_parameters']
    basic_conf = sim_parameters['basic_conf']
    variations_dict = sim_parameters['variations_dict']
    scenario_name = basic_conf['scenario_name']
    target_metrics = sim_parameters['target_metrics']

    if sim_parameters['doe_type'] == 'sobol':
        basic_conf['stochastic'] = False
        num_factors = 0
        min = []
        span = []
        for entity, factors in variations_dict.items():
            for factor, factor_range in factors.items():
                min.append(factor_range[0])
                span.append(factor_range[1] - factor_range[0])
                num_factors = num_factors + 1
        samples = sobol_seq.i4_sobol_generate(num_factors, sim_parameters['samples'])

        if sim_parameters['add_extreme_points']:
            min_max = np.array(list(itertools.product([0, 1], repeat=num_factors)))
            samples = np.concatenate((samples, min_max), axis=0)

        # Sobol samples are between 0 and 1, here we scale
        samples = np.array(samples) * np.array(span) + np.array(min)

        variations = create_variations_dict(variations_dict, samples)
        # visualize_variations(variations)

    elif sim_parameters['doe_type'] == 'LHS':
        number_samples = sim_parameters['samples']
        number_factors = len({j for i in variations_dict.values() for j in i})
        samples = pyDOE.lhs(number_factors, samples=number_samples)
        #ToDo: Read data from simulation parameter file
        from scipy.stats.distributions import norm
        means, stdvs = get_variations_distribution(variations_dict)
        for i in range(samples.shape[1]):
            samples[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(samples[:, i])
        variations = create_variations_dict(variations_dict, samples)
        visualize_variations(variations)
        logger.info(variations)

    elif sim_parameters['doe_type'] == 'extreme_points':
        variations_tmp = {}
        for entity, factors in variations_dict.items():
            for factor, factor_range in factors.items():
                variations_tmp[f'{entity}.{factor}'] = [factor_range[0], factor_range[1]]
        keys, values = zip(*variations_tmp.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        samples = [v for v in itertools.product(*values)]
        variations = create_variations_dict(variations_dict, np.array(samples))

    elif sim_parameters['doe_type'] == 'OAT':
        variations_tmp = {}
        variations = {}
        counter = 0
        # First entry with standard parametrization
        run_id = 'run_' + str(0).zfill(2)
        variations[run_id] = {'ID': str(0).zfill(2)}
        counter += 1
        for entity, factors in variations_dict.items():
            for factor, factor_range in factors.items():
                run_id = 'run_' + str(counter).zfill(2)
                variations[run_id] = {'ID': str(counter).zfill(2)}
                if entity not in variations[run_id]:
                    variations[run_id][entity] = {}
                variations[run_id][entity][factor] = factor_range[0]
                counter += 1
                run_id = 'run_' + str(counter).zfill(2)
                variations[run_id] = {'ID': str(counter).zfill(2)}
                if entity not in variations[run_id]:
                    variations[run_id][entity] = {}
                variations[run_id][entity][factor] = factor_range[1]
                counter += 1

    elif sim_parameters['doe_type'] == 'sobol_indices':
        problem, discrete = create_problem(variations_dict)
        samples = sobol.sample(problem,
                               sim_parameters['samples'],
                               calc_second_order=False)
        variations = create_variations_dict(variations_dict, samples)

    elif sim_parameters['doe_type'] == 'fast':
        problem, discrete = create_problem(variations_dict)
        samples = fast_sampler.sample(problem, sim_parameters['samples'])
        variations = create_variations_dict(variations_dict, samples)

    elif sim_parameters['doe_type'] == 'distribution_and_discrete':
        # This currently only works for 1 discrete and 1 distribution
        factor_map = {}
        problem, discrete = create_problem(variations_dict)
        sample_size = sim_parameters['samples']
        set_size = len(discrete[0][1])
        x = sample_size * set_size

        for discrete_factor in discrete:
            tmp_array = None
            for set_elem in discrete_factor[1]:
                tmp2 = np.full((sample_size), int(set_elem))
                if tmp_array is None:
                    tmp_array = tmp2
                else:
                    tmp_array = np.concatenate((tmp_array, tmp2))
            factor_map[discrete_factor[0]] = tmp_array
        tmp_samples = latin.sample(problem, sim_parameters['samples'])
        for i, problem_name in enumerate(problem['names']):
            factor_map[problem_name] = np.concatenate((tmp_samples[:, i], tmp_samples[:, i], tmp_samples[:, i]))

        y = len(list(factor_map.keys()))
        samples = np.full((x, y), 0.0)

        count = 0
        for entity, factors in list(variations_dict.items()):
            for factor, t in list(factors.items()):
                samples[:, count] = factor_map[f'{entity}.{factor}']
                logger.info(f'')
                count += 1

        variations = create_variations_dict(variations_dict, samples)

    else:
        logger.warning('No simulation as doe_type is not sobol, sobol_indices, LHS or extreme_points')

    # Merging of the basic configuration and the variations
    recipes = {key: {**copy.deepcopy(basic_conf), **copy.deepcopy(entities_parameters)} for key in variations}
    for key, data in variations.items():
        for entity, factors in data.items():
            if type(factors) is dict:
                for param, value in factors.items():
                    recipes[key][entity][param] = copy.deepcopy(value)
            else:
                recipes[key][entity] = factors
    # logger.info(recipes)

    # check if folder for figures exists
    if not os.path.isdir('output'):
        os.mkdir('output')
    if not os.path.isdir(sim_parameters['folder_figures']):
        os.mkdir(sim_parameters['folder_figures'])
    # remove temp folder, as old data in it would disturb calculation
    folder_temp_files = basic_conf['folder_temp_files']
    if os.path.isdir(folder_temp_files) and not sim_parameters['skip_simulation']:
        try:
            shutil.rmtree(folder_temp_files)
            logger.info(f"Removed folder {folder_temp_files} to remove old temp files.")
        except OSError as e:
            logger.warning(f"{e.filename} - {e.strerror}")
        #os.remove(folder_temp_files)
    if not os.path.isdir(folder_temp_files):
        os.mkdir(folder_temp_files)

    #store recipes to file for analysis
    with open(f"{folder_temp_files}/recipes.json", "w") as write_file:
        json.dump(recipes, write_file)
    #store variations to file for analysis
    with open(f"{folder_temp_files}/variations.json", "w") as write_file:
        json.dump(variations, write_file)

    logger.info('')
    # logger.info(f'Start simulation with simulation time of {end} seconds')
    logger.info("Start simulation")
    logger.info(f"DoE type: {sim_parameters['doe_type']}")
    logger.info(f"basic_conf: {basic_conf}")
    # logger.info(f"variations_dict: {variations_dict}")
    logger.info(f"number of planned simulation runs: {len(recipes)}")

    if sim_parameters['parallelize']:
        if sim_parameters['skip_simulation']:
            logger.info("Execution of simulation is skipped due to 'skip_simulation' parameter in configuration"
                        "and only analysis script executed.")
        else:
            logger.info('Start parallel execution of scenarios')
            pool = mp.Pool(processes=sim_parameters['num_processes'])
            pool.map(benchmark_sim.run_scenario, list(recipes.values()))
        for recipe_name in recipes:
            logger.info(f'Data processing scenario with recipe {recipe_name}: {recipes[recipe_name]}')
            benchmark_analysis.data_processing(recipes[recipe_name],
                                               variations,
                                               folder_temp_files,
                                               basic_conf['summary_filename'],
                                               sim_parameters['drop_first_day_data'])
    else:
        for recipe_name in recipes:
            if sim_parameters['skip_simulation']:
                logger.info("Execution of simulation is skipped due to 'skip_simulation' parameter in configuration"
                            "and only analysis script executed.")
            else:
                logger.info(f'Run scenario with recipe {recipe_name}: {recipes[recipe_name]}')
                benchmark_sim.run_scenario(recipes[recipe_name])
            benchmark_analysis.data_processing(recipes[recipe_name],
                                               variations,
                                               folder_temp_files,
                                               basic_conf['summary_filename'],
                                               sim_parameters['drop_first_day_data'])

    toolbox_analysis.analyze_results(recipes=recipes,
                                       variations_dict=variations_dict,
                                       basic_conf=basic_conf,
                                       folder=sim_parameters['folder_figures'],
                                       format=sim_parameters['format'],
                                       dpi=sim_parameters['dpi'],
                                       doe_type=sim_parameters['doe_type'],
                                       plots=sim_parameters['plots'],
                                       target_metrics=target_metrics,
                                       folder_figures=sim_parameters['folder_figures'],
                                       scenario_name=basic_conf['scenario_name'],
                                       plt_show=sim_parameters['show_plots'])
    # benchmark_analysis.plot_simulation_results(sim_parameters)
