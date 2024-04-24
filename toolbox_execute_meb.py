import os
import json
from loguru import logger
import multiprocessing as mp
from time import time, ctime
from datetime import timedelta

import benchmark_2_example.benchmark_multi_energy_sim as benchmark_sim
import benchmark_2_example.benchmark_multi_energy_analysis as benchmark_analysis

if __name__ == "__main__":
    import argparse

    # Parse command line options.
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default = 'output/temp_files_tank_scaling_heat', help = 'folder with configuration files')
    args = parser.parse_args()

    # Read configuration files from temp folder to make them available for analysis
    temp_folder = args.folder
    with open(os.path.join(temp_folder, 'recipes.json')) as data: 
        recipes = json.load(data)
    with open(os.path.join(temp_folder, 'variations.json')) as data: 
        variations = json.load(data)
    with open(os.path.join(temp_folder, 'basic_conf.json')) as data: 
        basic_conf = json.load(data)
    with open(os.path.join(temp_folder, 'target_metrics.json')) as data: 
        target_metrics = json.load(data)
    with open(os.path.join(temp_folder, 'sim_parameters.json')) as data: 
        sim_parameters = json.load(data)

    logger.info('')
    # logger.info(f'Start simulation with simulation time of {end} seconds')
    logger.info("Start simulation")
    logger.info(f"DoE type: {sim_parameters['doe_type']}")
    logger.info(f"basic_conf: {basic_conf}")
    # logger.info(f"variations_dict: {variations_dict}")
    logger.info(f"number of planned simulation runs: {len(recipes)}")
    
    if sim_parameters['skip_simulation']:
        logger.info("Execution of simulation is skipped due to 'skip_simulation' parameter in configuration"
                    "and only analysis script executed.")
    else:
        if sim_parameters['parallelize']:
            logger.info(f"Start parallel execution of scenarios in {sim_parameters['num_processes']} processes")
            pool = mp.Pool(processes=sim_parameters['num_processes'])
            pool.map(benchmark_sim.run_scenario, list(recipes.values()))
        else:
            for recipe_name in recipes:
                logger.info(f'Run scenario with recipe {recipe_name}: {recipes[recipe_name]}')
                benchmark_sim.run_scenario(recipes[recipe_name])
    
    results_data_list = []
    for recipe_name in recipes:
        logger.info(f'Data processing scenario with recipe {recipe_name}: {recipes[recipe_name]}')
        results_data_list.append(benchmark_analysis.data_processing(recipes[recipe_name],
                                                                    variations,
                                                                    temp_folder,
                                                                    basic_conf['summary_filename'],
                                                                    sim_parameters['drop_first_day_data']))

    with open(os.path.join(temp_folder, f'{basic_conf["summary_filename"]}.json'),'w') as data: 
        data.write(json.dumps(results_data_list, indent="    "))