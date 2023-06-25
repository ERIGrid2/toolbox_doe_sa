'''
Analysis script for the benchmark results, comparing
results with voltage control enabled and disabled.
'''
import json
import os

import sys
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

import benchmark_2_example.benchmark_multi_energy_sim as benchmark_sim
import toolbox_analysis

logger.remove()
logger.add("results.log", level="DEBUG")
logger.add(sys.stderr, level="DEBUG")

START_TIME = '2019-02-01 00:00:00'

PLOT_DICT = {
    'tank temperatures': [
        'temperature in °C',
        [
            'StratifiedWaterStorageTank_0.T_cold',
            'StratifiedWaterStorageTank_0.T_hot',
            'StratifiedWaterStorageTank_0.T_avg',
            # 'StratifiedWaterStorageTank_0.T_ch_in',
            # 'StratifiedWaterStorageTank_0.T_dis_in',
        ]
    ],
    'tank mass flow': [
        'mass flow in kg/m^3',
        [
            'StratifiedWaterStorageTank_0.mdot_ch_in',
            'StratifiedWaterStorageTank_0.mdot_dis_in',
            'StratifiedWaterStorageTank_0.mdot_ch_out',
            'StratifiedWaterStorageTank_0.mdot_dis_out',
        ]
    ],
    'heatpump': [
        'power in kW',
        [
            'heatpump_0.P_effective',
            'heatpump_0.P_requested',
        ]
    ],
    'flex heat controller state': [
        'state',
        [
            'FHctrl_0.state',
        ]
    ],
    'flex heat controller HP mdot out': [
        'HP mdot out',
        [
            'FHctrl_0.mdot_HP_out',
        ]
    ],
    'voltage controller': [
        'setpoint in kW',
        [
            'VoltageController_0.hp_p_el_kw_setpoint',
        ]
    ],
    'electrical consumption': [
        'electrical consumption in MW',
        [
            'Load_1_0.p_mw',
            'Load_2_0.p_mw',
            'Heat Pump_0.p_mw',
        ]
    ],
    'PV generation': [
        'PV generation in MW',
        [
            'PV_1_0.p_mw',
            'PV_2_0.p_mw',
        ]
    ],
    'voltage levels': [
        'voltage levels in p.u.',
        [
            'Bus_1_0.vm_pu',
            'Bus_2_0.vm_pu',
        ]
    ],
    'line loadings': [
        'line loading in %',
        [
            'LV_Line_0-1_0.loading_percent',
            'LV_Line_1-2_0.loading_percent',
        ]
    ],
    'line losses': [
        'line losses in mw',
        [
            'LV_Line_0-1_0.pl_mw',
            'LV_Line_1-2_0.pl_mw',
        ]
    ],
    'line losses': [
        'line losses in mvar',
        [
            'LV_Line_1-2_0.ql_mvar',
            'LV_Line_0-1_0.ql_mvar',
        ]
    ],
    'DHNetwork': [
        'mass flow (mdot)',
        [
            'DHNetwork_0.mdot_cons1_set',
            'DHNetwork_0.mdot_cons2_set',
            'DHNetwork_0.mdot_grid_set',
            'DHNetwork_0.mdot_tank_in_set',
            'DHNetwork_0.mdot_cons1',
            'DHNetwork_0.mdot_cons2',
            'DHNetwork_0.mdot_grid',
            'DHNetwork_0.mdot_tank_in',
        ]
    ],
}

FIG_TYPE = 'png' # 'pdf'
FIG_SIZE = [15, 8]


def data_processing(recipe, variations, folder_temp_files, summary_filename, drop_first_day_data=True):
    # # Data processing
    sim_results = {}
    results_store = pd.HDFStore(benchmark_sim.get_store_filename(recipe))
    # ToDo: Does the ME benchmark have a specific start time?
    start_time = '2019-02-01 00:00:00'

    for collector in results_store:
        for (simulator, attribute), data in results_store[collector].items():
            # Retrieve short name of data.
            sim_node_name = get_sim_node_name(simulator)
            res_name = '.'.join([sim_node_name, attribute])

            # Convert index to time format.
            data.index = pd.to_datetime(data.index, unit='s', origin=start_time)

            if drop_first_day_data:
                first_day_data = data.first('1D')
                sim_results[res_name] = data.drop(first_day_data.index)
            else:
                sim_results[res_name] = data
    results_store.close()

    # Power Grid
    if recipe['stochastic']:
        grid_voltage_bus_1_vm_pu = sim_results['ng__0.output']
    else:
        grid_voltage_bus_1_vm_pu = sim_results['Bus_1_0.vm_pu']
    grid_voltage_bus_1_max_pu = grid_voltage_bus_1_vm_pu.max()
    grid_voltage_bus_1_min_pu = grid_voltage_bus_1_vm_pu.min()

    grid_voltage_bus_2_vm_pu = sim_results['Bus_2_0.vm_pu']
    grid_voltage_bus_2_max = grid_voltage_bus_2_vm_pu.max()
    grid_voltage_bus_2_min = grid_voltage_bus_2_vm_pu.min()

    line_0_loading = sim_results['LV_Line_0-1_0.loading_percent']
    line_0_loading_max = line_0_loading.max()
    line_1_loading = sim_results['LV_Line_1-2_0.loading_percent']
    line_1_loading_max = line_1_loading.max()

    line_0_losses_pl_mw = sim_results['LV_Line_0-1_0.pl_mw']
    line_1_losses_pl_mw = sim_results['LV_Line_1-2_0.pl_mw']
    line_losses_pl_mw = line_0_losses_pl_mw + line_1_losses_pl_mw
    # line_reactive_consumption_mvar = sim_results['LV_Line_0-1_0.ql_mvar'].sum() + \
    #                                 sim_results['LV_Line_1-2_0.ql_mvar'].sum()


    # use Heat Pump consumption for evaluation!
    hp_p_effective = sim_results['heatpump_0.P_effective']
    # ToDo: rename variable!
    hp_w_effective = sim_results['heatpump_0.Qdot_cond']
    hp_average_COP = hp_w_effective.sum() / hp_p_effective.sum()

    # hp_T_cond_out = sim_results['heatpump_0.T_cond_out']
    # hp_T_evap_out = sim_results['heatpump_0.T_evap_out']

    pv_generation_p_mw = sim_results['PV_1_0.p_mw'] + sim_results['PV_2_0.p_mw']
    el_consumption_p_mw = sim_results['Heat Pump_0.p_mw'] + sim_results['Load_1_0.p_mw'] + sim_results['Load_2_0.p_mw']
    electricity_balance_mw = pv_generation_p_mw - el_consumption_p_mw - line_losses_pl_mw
    electricity_export_time_series_mw = electricity_balance_mw.where(electricity_balance_mw > 0, 0)
    electricity_export_mwh = electricity_export_time_series_mw.sum() / 60  # - line_losses
    electricity_import_time_series_mw = -electricity_balance_mw.where(electricity_balance_mw < 0, 0)
    electricity_import_mwh = electricity_import_time_series_mw.sum() / 60
    self_consumption_mwh = (pv_generation_p_mw - electricity_export_time_series_mw).sum() / 60
    self_consumption_perc = (self_consumption_mwh / (pv_generation_p_mw.sum() / 60)) * 100
    #ToDo: look at summer school example again
    # self_consumption_index = ((pv_generation_p_mw + el_consumption_p_mw) / pv_generation_p_mw).sum()

    # evaluate the percentage of thermal energy provided by the tank
    # https://adgefficiency.com/energy-basics-q-m-cp-dt/
    cp_water = 4.18  # mass heat capacity
    mdot_grid_kg_s = sim_results['FHctrl_0.mdot_grid_set']  # Mass flow injected by the grid
    temp_ext_grid_supply = sim_results['DHNetwork_0.T_supply_grid']  # Supply temperature from the external grid
    temp_ext_grid_return = sim_results['DHNetwork_0.T_return_grid']  # Return temperature to the external grid
    energy_ext_grid_kw_min = mdot_grid_kg_s / 60 * cp_water * (temp_ext_grid_supply - temp_ext_grid_return)

    # temp_hwt = sim_results['StratifiedWaterStorageTank_0.T_hot']
    hp_mdot_cond_out_kg_s = sim_results['heatpump_0.mdot_cond_out']
    hp_T_cond_out = sim_results['heatpump_0.T_cond_out']
    hwt_temp_cold = sim_results['StratifiedWaterStorageTank_0.T_cold']
    energy_tank_charged_kw_min = hp_mdot_cond_out_kg_s / 60 * cp_water * (hp_T_cond_out - hwt_temp_cold)

    hwt_temp_hot = sim_results['StratifiedWaterStorageTank_0.T_hot']
    hwt_temp_return = sim_results['DHNetwork_0.T_return_tank']  # Temperature of water returning from dh network to hwt
    mdot_tank_in_kg_s = sim_results['StratifiedWaterStorageTank_0.mdot_dis_in']  # Mass flow from the tank to dh network
    energy_tank_supplied_kw_min = mdot_tank_in_kg_s / 60 * cp_water * (hwt_temp_hot - hwt_temp_return)

    sum_energy_kw_min = energy_ext_grid_kw_min + energy_tank_charged_kw_min
    #heat_internal_percentage = 100 * energy_tank.sum() / (energy_ext_grid.sum() + energy_tank.sum())

    # evaluate the critical node temperature
    t_supply_1 = sim_results['DHNetwork_0.T_supply_cons1']
    t_supply_2 = sim_results['DHNetwork_0.T_supply_cons2']

    # Look at setpoint of the voltage controller for the heat pump to evaluate the curtailment
    hp_p_rated = sim_results['heatpump_0.P_rated'].max()
    hp_p_el_kw_setpoint = sim_results['VoltageController_0.hp_p_el_kw_setpoint']
    hp_p_el_kw_setpoint_percentage = hp_p_el_kw_setpoint / hp_p_rated * 100
    hp_p_el_kw_setpoint_percentage_mean = hp_p_el_kw_setpoint_percentage.mean()

    sim_data = {'ID': [recipe['ID']],
                #'grid_voltage_bus_1_max_pu': [grid_voltage_bus_1_max_pu],
                'grid_voltage_bus_2_max_pu': [grid_voltage_bus_2_max],
                'line_0_loading_max_perc': [line_0_loading_max],
                #'line_1_loading_max_perc': [line_1_loading_max],
                'hp_electr_energy_gwh': [hp_p_effective.sum() / 60 / 1000],
                'hp_heat_energy_gwh': [hp_w_effective.sum() / 60 / 1000],
                'hp_average_COP': [hp_average_COP],
                'electricity_import_mwh': [electricity_import_mwh],
                'electricity_export_mwh': [electricity_export_mwh],
                #'self_consumption_mwh': [self_consumption_mwh],
                'self_consumption_perc': [self_consumption_perc],
                #'line_losses_mwh': [line_losses_pl_mw.sum() / 60],
                #'hp_p_el_kw_setpoint_perc_mean': [hp_p_el_kw_setpoint_percentage_mean],
                't_supply_min': [min([t_supply_1.min(), t_supply_2.min()])],
                't_supply_max': [max([t_supply_1.max(), t_supply_2.max()])],
                #'energy_sum_gwh': [sum_energy_kw_min.sum() / 60 / 1000],
                'heat_import_gwh': [energy_ext_grid_kw_min.sum() / 60 / 1000],
                'heat_import_perc': [energy_ext_grid_kw_min.sum() / sum_energy_kw_min.sum() / 60 / 1000],
                'hp_perc': [hp_w_effective.sum() / sum_energy_kw_min.sum() / 60 / 1000],
                #'grid_voltage_bus_1_min_pu': [grid_voltage_bus_1_min_pu],
                #'grid_voltage_bus_2_min_pu': [grid_voltage_bus_2_min],
                #'electricity_balance_mwh': [electricity_balance_mw.sum() / 60],
                #'line_reactive_consumption_mvar': [line_reactive_consumption_mvar],
                #'energy_tank_charged_kw_min_gwh': [energy_tank_charged_kw_min.sum() / 60 / 1000],
                #'energy_tank_supplied_kw_min_gwh': [energy_tank_supplied_kw_min.sum() / 60 / 1000],
                #'heat_internal_percentage': [heat_internal_percentage],
                #'self_consumption_index': [self_consumption_index],
                'File ID/dataframe': [
                    '{}'.format(benchmark_sim.get_store_filename(recipe)) + '/' + 'timeseries/sim_{}'.format(recipe['ID'])]}

    # Write variation parameter to sim_data dict (needed for meta model analysis)
    for key in recipe.keys():
        if key in list(variations.values())[0].keys():
            if isinstance(recipe[key], dict):
                for key2, value in recipe[key].items():
                    sim_data[f"{key}.{key2}"] = value
    # print(f"sim_data: {sim_data}")
    sim_data_df = pd.DataFrame(sim_data)
    sim_data_df.to_csv(f"{folder_temp_files}/{summary_filename}.csv")
    run_store = pd.HDFStore(f"{folder_temp_files}/{summary_filename}.h5")
    run_store['run_{}'.format(recipe['ID'])] = sim_data_df
    run_store.close()


def get_sim_node_name(
    full_name
):
    (sim_name, sim_node) = full_name.split('.')
    return sim_node


def retrieve_results(
    store_name,
    start_time,
    drop_first_day_data = True
):
    results_dict = {}
    results_store = pd.HDFStore(store_name)

    for collector in results_store:
        for (simulator, attribute), data in results_store[collector].items():
            # Retrieve short name of data.
            sim_node_name = get_sim_node_name(simulator)
            res_name = '.'.join([sim_node_name, attribute])

            # Convert index to time format.
            data.index = pd.to_datetime(data.index, unit='s', origin=start_time)

            if drop_first_day_data:
                first_day_data = data.first('1D')
                results_dict[res_name] = data.drop(first_day_data.index)
            else:
                results_dict[res_name] = data

    results_store.close()
    return results_dict


def plot_results_compare(
    entity, attr, label, folder_figures,
    recipes,
    dict_results_list,
    fig_id, show=False, fig_type='png'
):
    fig, axes_attr_compare = plt.subplots(figsize=FIG_SIZE)
    for i in range(len(dict_results_list)):
        attr_i = dict_results_list[i][f"{entity}.{attr}"]
        axes_attr_compare.plot(attr_i, label='{} {}'.format(entity, list(recipes.values())[i]['ID']))
    axes_attr_compare.legend(loc='upper right')
    axes_attr_compare.set_xlabel('date')
    axes_attr_compare.set_ylabel(label)

    plt.savefig('{}/fig_{}_{}_{}.{}'.format(folder_figures, fig_id, entity, attr, fig_type))
    if show:
        plt.show()
    plt.close()

    # fig, axes_sorted_attr_compare = plt.subplots(figsize=FIG_SIZE)
    # #df_sorted_attr_compare = pd.DataFrame()
    # for i in range(len(dict_results_list)):
    #     attr_i = dict_results_list[i][f"{entity}.{attr}"]
    #     sorted_attr_i = attr_i.sort_values(ascending=False, ignore_index=True)
    #     axes_sorted_attr_compare.plot(sorted_attr_i, label='{} {}'.format(entity, list(recipes.values())[i]['ID']))
    #     #df_sorted_attr_compare['{} {}'.format(entity, list(recipes.values())[i]['ID'])] = sorted_attr_i
    # axes_sorted_attr_compare.legend(loc='upper right')
    # axes_sorted_attr_compare.set_ylabel(label)
    # axes_sorted_attr_compare.set_title('duration plot of {}'.format(attr))
    # plt.savefig('{}/fig_sorted_{}_{}.{}'.format(FOLDER, fig_id, attr, fig_type))
    # if show:
    #     plt.show()
    # plt.close()

    #axes_sorted_attr_compare_hist = df_sorted_attr_compare.plot.hist(bins=bins, alpha=0.5, figsize=FIG_SIZE)
    #axes_sorted_attr_compare_hist.legend(loc='upper right')
    #axes_sorted_attr_compare_hist.set_xlabel(label)
    #plt.savefig('{}/fig_hist_{}.{}'.format(FOLDER, fig_id, fig_type))
    #if show:
    #    plt.show()
    #plt.close()

    #return (attr_type1.sum(), attr_type2.sum())


def plot_simulation_results(sim_parameters):
    basic_conf = sim_parameters['basic_conf']

    dict_results_list = []
    for filename in os.scandir(basic_conf['folder_temp_files']):
        if filename.is_file():
            if sim_parameters['summary_filename'] not in str(filename):
                if 'recipes.json' not in str(filename):
                    logger.info(filename)
                    dict_results_list.append(retrieve_results(filename,
                                                              START_TIME,
                                                              sim_parameters['drop_first_day_data']))

    #Read in recipes from json file
    with open(f"{basic_conf['folder_temp_files']}/recipes.json", "r") as json_file:
        recipes = json.load(json_file)

    for label in PLOT_DICT:
        x_label = PLOT_DICT[label][0]
        for attr in PLOT_DICT[label][1]:
            entity = attr.split('.')[0]
            attr_name = attr.split('.')[1]
            plot_results_compare(
                entity, attr_name, x_label, sim_parameters['folder_figures'],
                recipes,
                dict_results_list,
                label, sim_parameters['show_plots'], FIG_TYPE
            )


if __name__ == '__main__':
    # sim_parameters = toolbox_analysis.read_in_sim_parameters(r'resources\simulation_parameters.json')
    # sim_parameters = toolbox_analysis.read_in_sim_parameters(r'resources\simulation_parameters.json')
    sim_parameters = toolbox_analysis.read_in_sim_parameters(r'resources\simulation_parameters_inter_domain.json')
    # plot_simulation_results(sim_parameters)

    with open(f"{sim_parameters['basic_conf']['folder_temp_files']}/recipes.json", "r") as read_file:
        recipes = json.loads(read_file)
    with open(f"{sim_parameters['basic_conf']['folder_temp_files']}/variations.json", "r") as read_file:
        variations = json.loads(read_file)

    for recipe_name in recipes:
        logger.info(f'Data processing scenario with recipe {recipe_name}: {recipes[recipe_name]}')
        data_processing(recipes[recipe_name],
                        variations,
                        sim_parameters['basic_conf']['folder_temp_files'],
                        sim_parameters['summary_filename'],
                        sim_parameters['drop_first_day_data'])