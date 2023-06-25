# Copyright (c) 2021 by ERIGrid 2.0. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
'''
This MOSAIK co-simulation setup implements the ERIGrid 2.0 multi-energy benchmark.
'''

# Define default for simulation start time, step size and end (1 MOSAIK time-step = 1 second).

import mosaik
from time import time, ctime
from datetime import timedelta
import sys

import pandapower
from loguru import logger

logger.remove()
logger.add("results.log", level="DEBUG")
logger.add(sys.stderr, level="DEBUG")

START_TIME = '2019-02-01 00:00:00'
# END = 4 * 24 * 60 * 60

# Specify profiles for generation and demand.
HEAT_DEMAND_LOAD_PROFILES = 'resources/heat/heat_demand_load_profiles.csv'
POWER_DEMAND_LOAD_PROFILES = 'resources/power/power_demand_load_profiles.csv'
PV_GENERATION_PROFILES = 'resources/power/pv_generation_profiles.csv'

# MOSAIK simulator configuration.
SIM_CONFIG = {
    'DHNetworkSim': {
        'python': 'benchmark_2_example.simulators:DHNetworkSimulator'
    },
    'ElNetworkSim': {
        'python': 'benchmark_2_example.simulators:ElectricNetworkSimulator'
    },
    'StorageTankSim': {
        'python': 'benchmark_2_example.simulators:StratifiedWaterStorageTankSimulator'
    },
    'HeatPumpSim': {
        'python': 'benchmark_2_example.simulators:ConstantTcondHPSimulator'
    },
    'HeatExchangerSim': {
        'python': 'benchmark_2_example.simulators:HEXConsumerSimulator'
    },
    'FlexHeatCtrlSim': {
        'python': 'benchmark_2_example.simulators:SimpleFlexHeatControllerSimulator'
    },
    'VoltageCtrlSim': {
        'python': 'benchmark_2_example.simulators:VoltageControlSimulator'
    },
    'TimeSeriesSim': {
        'python': 'benchmark_2_example.simulators:TimeSeriesPlayerSim'
    },
    'NoiseGeneratorSim': {
        'python': 'benchmark_2_example.simulators:NoiseGenerator'
    },
    'CollectorSim': {
        'python': 'benchmark_2_example.simulators:Collector'
    },
    'CollectorSimSummerschool': {
        'python': 'benchmark_2_example.simulators:CollectorSummerschool'
    }
}

# Simulation parameters.
HP_TEMP_COND_OUT_TARGET = 75
EXT_SOURCE_SUPPLY_TEMP = 75  # Supply temperature of external DH network.

INIT_HEX_SUPPLY_TEMP = 75  # Return temperature of heat exchanger.
INIT_HEX_RETURN_TEMP = 45  # Return temperature of heat exchanger.
INIT_STORAGE_TANK_TEMP = 70  # Storage tank initial temperature.

def loadProfiles():
    '''
    Load profiles for demand (heat, power) and PV generation.
    '''
    import pandas as pd
    import pathlib

    profiles = {}
    
    here = pathlib.Path(__file__).resolve().parent
    
    profiles['heat_demand'] = pd.read_csv(
        pathlib.Path(here, HEAT_DEMAND_LOAD_PROFILES),
        index_col = 0, parse_dates = True
    )

    profiles['power_demand'] = pd.read_csv(
        pathlib.Path(here, POWER_DEMAND_LOAD_PROFILES),
        index_col = 0, parse_dates = True
    )

    profiles['pv_generation'] = pd.read_csv(
        pathlib.Path(here, PV_GENERATION_PROFILES),
        index_col = 0, parse_dates = True
    )

    return profiles


def initializeSimulators(world, step_size, outfile_name, recipe):
    '''
    Initialize and start all simulators.
    '''   
    simulators = {}

    # Electrical network.
    simulators['el_network'] = world.start(
        'ElNetworkSim',
        step_size=step_size,
        mode='pf'
    )

    # District heating network.
    simulators['dh_network'] = world.start(
        'DHNetworkSim',
        step_size=step_size
    )

    # Heat consumer (heat exchanger).
    simulators['hex_consumer'] = world.start(
        'HeatExchangerSim',
        step_size=step_size
    )

    # Time series player for electrical load profiles and PV generation profiles.
    simulators['load_gen_profiles'] = world.start(
        'TimeSeriesSim',
        eid_prefix='power_demand',
        step_size=step_size
    )

    # Time series player for the consumer heat demand.
    simulators['heat_profiles'] = world.start(
        'TimeSeriesSim',
        eid_prefix = 'heat_demand',
        step_size = step_size
    )

    # Stratified water storage tank.
    simulators['storage_tank'] = world.start(
        'StorageTankSim',
        step_size = step_size
    )

    # Heat pump.
    simulators['heat_pump'] = world.start(
        'HeatPumpSim',
        eid_prefix = 'heatpump',
        step_size = step_size,
    )

    # Flex heat controller.
    simulators['flex_heat_ctrl'] = world.start(
        'FlexHeatCtrlSim',
        step_size = step_size
    )

    # Voltage controller.
    simulators['voltage_ctrl'] = world.start(
        'VoltageCtrlSim',
        step_size = step_size
    )

    # Noise generator.
    simulators['noisegenerator'] = world.start(
        "NoiseGeneratorSim",
        eid_prefix='ng_',
        step_size = step_size,
    )

    # Data collector.
    simulators['collector'] = world.start(
        'CollectorSimSummerschool',
        step_size = step_size,
        print_results = False,
        save_h5 = True,
        h5_storename = outfile_name,
        h5_framename='timeseries/sim_{}'.format(recipe['ID']),
    )

    return simulators


def instantiateEntities(simulators, profiles, recipe):
    '''
    Create instances of simulators.
    '''
    entities = {}

    grid_file = recipe['el_network']['grid_file']
    grid = pandapower.from_json(grid_file)
    grid_tmp_file = grid_file[0:len(grid_file)-5] + f'_{recipe["ID"]}' + grid_file[len(grid_file)-5:len(grid_file)]
    grid['line'].loc[:, 'length_km'][0] = recipe['el_network']['line_0_length']
    grid['line'].loc[:, 'length_km'][1] = recipe['el_network']['line_1_length']
    pandapower.to_json(grid, filename=grid_tmp_file)
    logger.info(f'Wrote new panadapower json file: {grid_tmp_file} with line 0 length '
                f'{recipe["el_network"]["line_0_length"]} and line 1 length {recipe["el_network"]["line_1_length"]}')
    # Electrical network.
    entities['el_network'] = simulators['el_network'].Grid(
        gridfile=grid_tmp_file,
    )

    # Add electrical network components to collection of entities.
    grid = entities['el_network'].children
    entities.update( {element.eid: element for element in grid if element.type in 'Load'} )
    entities.update( {element.eid: element for element in grid if element.type in 'Sgen'} )
    entities.update( {element.eid: element for element in grid if element.type in 'Bus'} )
    entities.update( {element.eid: element for element in grid if element.type in 'Line'} )

    # Time series player for the power consumption profile of load 1.
    entities['consumer_load1'] = simulators['load_gen_profiles'].TimeSeriesPlayer(
        t_start=START_TIME,
        series=profiles['power_demand'].copy(),
        fieldname='Load_1',
        interp_method='pchip',
        scale=recipe['consumer_load']['scale'],
    )

    # Time series player for the power consumption profile of load 2.
    entities['consumer_load2'] = simulators['load_gen_profiles'].TimeSeriesPlayer(
        t_start=START_TIME,
        series=profiles['power_demand'].copy(),
        fieldname='Load_2',
        interp_method='pchip',
        scale=recipe['consumer_load']['scale'],
    )

    # Time series player for generation profile of PV 1.
    entities['gen_pv1'] = simulators['load_gen_profiles'].TimeSeriesPlayer(
        t_start=START_TIME,
        series=profiles['pv_generation'].copy(),
        fieldname='PV_1',
        interp_method='pchip',
        scale=recipe['gen_pv']['scale'],
    )

    # Time series player for generation profile of PV 2.
    entities['gen_pv2'] = simulators['load_gen_profiles'].TimeSeriesPlayer(
        t_start=START_TIME,
        series=profiles['pv_generation'].copy(),
        fieldname='PV_2',
        interp_method='pchip',
        scale=recipe['gen_pv']['scale'],
    )

    # District heating network.
    entities['dh_network'] = simulators['dh_network'].DHNetwork(
        T_supply_grid=recipe['dh_network']['T_supply_grid'],
        P_grid_bar=recipe['dh_network']['P_grid_bar'],
        T_amb=recipe['dh_network']['T_amb'],
        dynamic_temp_flow_enabled=recipe['dh_network']['dynamic_temp_flow_enabled'],
    )

    # Heat exchanger 1.
    entities['hex_consumer1'] = simulators['hex_consumer'].HEXConsumer(
        T_return_target=recipe['hex_consumer1']['T_return_target'],
        P_heat=recipe['hex_consumer1']['P_heat'],
        mdot_hex_in=recipe['hex_consumer1']['mdot_hex_in'],
        mdot_hex_out=recipe['hex_consumer1']['mdot_hex_out'],
    )

    # Heat exchanger 2.
    entities['hex_consumer2'] = simulators['hex_consumer'].HEXConsumer(
        T_return_target=recipe['hex_consumer2']['T_return_target'],
        P_heat=recipe['hex_consumer2']['P_heat'],
        mdot_hex_in=recipe['hex_consumer2']['mdot_hex_in'],
        mdot_hex_out=recipe['hex_consumer2']['mdot_hex_out'],
    )

    # Time series player for heat demand of consumer 1.
    entities['heat_profiles1'] = simulators['heat_profiles'].TimeSeriesPlayer(
        t_start=START_TIME,
        series=profiles['heat_demand'].copy(),
        fieldname='consumer1',
        scale=recipe['heat_profiles']['scale'],
    )


    # Time series player for heat demand of consumer 2.
    entities['heat_profiles2'] = simulators['heat_profiles'].TimeSeriesPlayer(
        t_start=START_TIME,
        series=profiles['heat_demand'].copy(),
        fieldname='consumer2',
        scale=recipe['heat_profiles']['scale'],
    )

    # Stratified water storage tank.
    entities['storage_tank'] = simulators['storage_tank'].WaterStorageTank(
        INNER_HEIGHT=recipe['storage_tank']['INNER_HEIGHT'],
        INNER_DIAMETER=recipe['storage_tank']['INNER_DIAMETER'],
        INSULATION_THICKNESS=recipe['storage_tank']['INSULATION_THICKNESS'],
        STEEL_THICKNESS=recipe['storage_tank']['STEEL_THICKNESS'],
        NB_LAYERS=recipe['storage_tank']['NB_LAYERS'],
        T_volume_initial=recipe['storage_tank']['T_volume_initial'],  # degC
        dt=recipe['step_size']
    )

    # Heat pump.
    entities['heat_pump'] = simulators['heat_pump'].ConstantTcondHP(
        P_rated=recipe['heat_pump']['P_rated'],
        lambda_comp=recipe['heat_pump']['lambda_comp'],
        P_0=recipe['heat_pump']['P_0'],
        eta_sys=recipe['heat_pump']['eta_sys'],
        eta_comp=recipe['heat_pump']['eta_comp'],
        T_evap_out_min=recipe['heat_pump']['T_evap_out_min'],
        dt=recipe['step_size'],
        T_cond_out_target=recipe['heat_pump']['T_cond_out_target'], # degC
        opmode=recipe['heat_pump']['opmode'],  # Constant output power at condenser
    )

    # Flex heat controller.
    entities['flex_heat_ctrl'] = simulators['flex_heat_ctrl'].SimpleFlexHeatController(
        voltage_control_enabled=recipe['voltage_ctrl']['enabled']
    )

    # Voltage controller.
    entities['voltage_ctrl'] = simulators['voltage_ctrl'].VoltageController(
        delta_vm_upper_pu=recipe['voltage_ctrl']['delta_vm_upper_pu'],
        delta_vm_lower_pu_hp_on=recipe['voltage_ctrl']['delta_vm_lower_pu_hp_on'],
        delta_vm_lower_pu_hp_off=recipe['voltage_ctrl']['delta_vm_lower_pu_hp_off'],
        delta_vm_deadband=recipe['voltage_ctrl']['delta_vm_deadband'],
        hp_p_el_mw_rated=recipe['voltage_ctrl']['hp_p_el_mw_rated'],
        hp_p_el_mw_min=recipe['voltage_ctrl']['hp_p_el_mw_min'],
        hp_operation_steps_min=30 * 60 / recipe['step_size'],
        k_p=recipe['voltage_ctrl']['k_p'],
    )

    # Noise generator
    entities['noisegenerator'] = simulators['noisegenerator'].NoiseGenerator(scale=recipe['noise_scale'])

    # Data collector.
    entities['sc_monitor'] = simulators['collector'].Collector()

    return entities


def connectEntities(world, entities, recipe):
    '''
    Add connections between the simulator entities.
    '''
    from benchmark_2_example.simulators.el_network.simulator import make_eid as grid_id
    
    # Connect electrical consumption profiles to electrical loads.
    world.connect(entities['consumer_load1'], entities[grid_id('Load_1',0)], ('out', 'p_mw'))
    world.connect(entities['consumer_load2'], entities[grid_id('Load_2',0)], ('out', 'p_mw'))

    # Connect PV profiles to static generators.
    world.connect(entities['gen_pv1'], entities[grid_id('PV_1',0)], ('out', 'p_mw'))
    world.connect(entities['gen_pv2'], entities[grid_id('PV_2',0)], ('out', 'p_mw'))

    # # Voltage controller.
    # world.connect(entities[grid_id('Bus_1',0)], entities['voltage_ctrl'], ('vm_pu', 'vmeas_pu'))
    # world.connect(entities['voltage_ctrl'], entities['flex_heat_ctrl'], ('hp_p_el_kw_setpoint', 'P_hp_el_setpoint'))
    # world.connect(entities['heat_pump'], entities['flex_heat_ctrl'], ('P_effective', 'P_hp_effective'),
    #     time_shifted=True, initial_data={'P_effective': 0})

    # Noise generator to voltage controller.
    if recipe['stochastic']:
        world.connect(entities[grid_id('Bus_1', 0)], entities['noisegenerator'], ('vm_pu', 'input'))
        world.connect(entities['noisegenerator'], entities['voltage_ctrl'], ('output', 'vmeas_pu'))
    else:
        world.connect(entities[grid_id('Bus_1', 0)], entities['voltage_ctrl'], ('vm_pu', 'vmeas_pu'))

    # Voltage controller to flex_heat_ctrl
    world.connect(entities['voltage_ctrl'], entities['flex_heat_ctrl'], ('hp_p_el_kw_setpoint', 'P_hp_el_setpoint'))
    world.connect(entities['heat_pump'], entities['flex_heat_ctrl'], ('P_effective', 'P_hp_effective'),
                  time_shifted=True, initial_data={'P_effective': 0})

    # District heating network.
    world.connect(entities['flex_heat_ctrl'], entities['dh_network'], ('mdot_1_supply', 'mdot_grid_set'))
    world.connect(entities['flex_heat_ctrl'], entities['dh_network'], ('mdot_3_supply', 'mdot_tank_in_set'))
    world.connect(entities['hex_consumer1'], entities['dh_network'], ('mdot_hex_out', 'mdot_cons1_set'))
    world.connect(entities['hex_consumer2'], entities['dh_network'], ('mdot_hex_out', 'mdot_cons2_set'))

    # Heat demand consumer 1.
    world.connect(entities['heat_profiles1'], entities['dh_network'], ('out', 'Qdot_cons1'))
    world.connect(entities['heat_profiles1'], entities['hex_consumer1'], ('out', 'P_heat'))
    world.connect(entities['hex_consumer1'], entities['flex_heat_ctrl'], ('mdot_hex_out', 'mdot_HEX1'))
    world.connect(entities['dh_network'], entities['hex_consumer1'], ('T_supply_cons1', 'T_supply'),
        time_shifted=True, initial_data={'T_supply_cons1': 70})

    # Heat demand consumer 1.
    world.connect(entities['heat_profiles2'], entities['dh_network'], ('out', 'Qdot_cons2'))
    world.connect(entities['heat_profiles2'], entities['hex_consumer2'], ('out', 'P_heat'))
    world.connect(entities['hex_consumer2'], entities['flex_heat_ctrl'], ('mdot_hex_out', 'mdot_HEX2'))
    world.connect(entities['dh_network'], entities['hex_consumer2'], ('T_supply_cons2', 'T_supply'),
        time_shifted=True, initial_data={'T_supply_cons2': 70})

    # Heat pump.
    world.connect(entities['flex_heat_ctrl'], entities['heat_pump'], ('mdot_2_return', 'mdot_evap_in'))
    world.connect(entities['dh_network'], entities['heat_pump'], ('T_evap_in', 'T_evap_in'),
        time_shifted=True, initial_data={'T_evap_in': 40})
    world.connect(entities['flex_heat_ctrl'], entities['heat_pump'], ('Q_HP_set', 'Q_set'))
    world.connect(entities['heat_pump'], entities['dh_network'], ('Qdot_evap', 'Qdot_evap'))
    world.connect(entities['storage_tank'], entities['heat_pump'], ('T_cold', 'T_cond_in'),
        time_shifted=True, initial_data={'T_cold': INIT_STORAGE_TANK_TEMP})
    world.connect(entities['heat_pump'], entities[grid_id('Heat Pump',0)], ('P_effective_mw', 'p_mw'),
        time_shifted=True, initial_data={'P_effective_mw': 0.})

    # Flex heat control.
    world.connect(entities['flex_heat_ctrl'], entities['heat_pump'], ('mdot_HP_out', 'mdot_cond_in'))
    world.connect(entities['heat_pump'], entities['flex_heat_ctrl'], ('T_cond_out_target', 'T_hp_cond_out'),
        time_shifted=True, initial_data={'T_cond_out_target': HP_TEMP_COND_OUT_TARGET})
    world.connect(entities['heat_pump'], entities['flex_heat_ctrl'], ('T_cond_in', 'T_hp_cond_in'),
        time_shifted=True, initial_data={'T_cond_in': INIT_STORAGE_TANK_TEMP})
    world.connect(entities['heat_pump'], entities['flex_heat_ctrl'], ('T_evap_in', 'T_hp_evap_in'),
        time_shifted=True, initial_data={'T_evap_in': INIT_HEX_RETURN_TEMP})

    # Storage tank inlet.
    world.connect(entities['heat_pump'], entities['storage_tank'], ('mdot_cond_out', 'mdot_ch_in'))
    world.connect(entities['heat_pump'], entities['storage_tank'], ('T_cond_out', 'T_ch_in'))
    world.connect(entities['flex_heat_ctrl'], entities['storage_tank'], ('mdot_3_supply', 'mdot_dis_out'),
        time_shifted=True, initial_data={'mdot_3_supply': 0})
    world.connect(entities['dh_network'], entities['storage_tank'], ('T_return_tank', 'T_dis_in'))

    # Storage tank outlet.
    world.connect(entities['storage_tank'], entities['dh_network'], ('T_hot', 'T_tank_forward'),
        time_shifted=True, initial_data={'T_hot': INIT_STORAGE_TANK_TEMP})
    world.connect(entities['storage_tank'], entities['flex_heat_ctrl'], ('T_hot', 'T_tank_hot'),
        time_shifted=True, initial_data={'T_hot': INIT_STORAGE_TANK_TEMP})


def connectDataCollector(world, entities):
    '''
    Configure and connect the data collector.
    '''
    collector_connections = {}

    collector_connections['storage_tank'] = [
            'T_cold', 'T_hot', 'T_avg', 'mdot_dis_in'
            #'mdot_ch_in', 'mdot_dis_in', 'mdot_ch_out', 'mdot_dis_out',
            #'T_ch_in', 'T_dis_in'
            ]

    #collector_connections['hex_consumer1'] = [
    #        'P_heat', 'mdot_hex_in', 'mdot_hex_out',
    #        'T_supply', 'T_return']

    #collector_connections['hex_consumer2'] = [
    #        'P_heat', 'mdot_hex_in', 'mdot_hex_out',
    #        'T_supply', 'T_return']

    collector_connections['heat_pump'] = [
            'T_cond_out', 'T_cond_in',
            #'T_evap_in', 'T_evap_out',
            'mdot_cond_in', 'mdot_cond_out',
            #'mdot_evap_in', 'mdot_evap_out',
            'Q_set', 'Qdot_cond',
            'W_effective', 'W_requested',
            #'W_max', 'W_evap_max', 'W_cond_max', 'W_rated',
            'P_effective', 'P_requested',
            'P_rated', 'eta_hp'
            ]

    collector_connections['dh_network'] = [
            'T_tank_forward', 'T_supply_cons1', 'T_supply_cons2', 'T_return_cons1', 'T_return_cons2','T_return_tank',
            'T_return_grid', 'T_supply_grid',
            #'mdot_cons1_set', 'mdot_cons2_set', 'mdot_grid_set', 'mdot_tank_in_set',
            #'mdot_cons1', 'mdot_cons2', 'mdot_grid', 'mdot_tank_in',
            #'Qdot_cons1', 'Qdot_cons2', 'Qdot_evap'
            ]

    collector_connections['voltage_ctrl'] = [
            'hp_p_el_kw_setpoint'
            ]

    collector_connections['flex_heat_ctrl'] = [
    #        'hp_on_request', 'hp_off_request',
    #        'mdot_HP_out', 'state', 'Q_HP_set',
            ('mdot_1_supply', 'mdot_grid_set'),
    #        ('mdot_3_supply', 'mdot_tank_in_set')
            ]

    #collector_connections['noisegenerator'] = [
    #    "input", "output"
    #        ]

    collector_connections.update({element.eid: ['p_mw'] for element in entities.values() if element.type in 'Load'})
    collector_connections.update({element.eid: ['p_mw'] for element in entities.values() if element.type in 'Sgen'})
    collector_connections.update({element.eid: ['vm_pu'] for element in entities.values() if element.type in 'Bus'})
    collector_connections.update({element.eid: ['loading_percent',
                                                'pl_mw',
                                                # 'ql_mvar'
                                                ] for element in entities.values() if element.type in 'Line'})
    #collector_connections.update({element.eid: ['p_mw'] for element in entities.values() if element.type in 'Ext_grid'})

    for ent, outputnames in collector_connections.items():
        for outputname in outputnames:
            world.connect(entities[ent], entities['sc_monitor'], outputname)


def get_store_filename(recipe):
    # print(recipe)
    return recipe['folder_temp_files'] + "/" + recipe['scenario_name'] + '_' + recipe['ID'] + '.h5'
    # return scenario_name + '_' + recipe['ID'] + '.h5'


def run_scenario(recipe):
    logger.info(f'run_scenario with recipe: {recipe}')
    outfile_name = get_store_filename(recipe)

    sim_start_time = time()
    # logger.info("CO-SIMULATION STARTED AT:", ctime(sim_start_time))

    # Start MOSAIK orchestrator.
    world = mosaik.World(SIM_CONFIG)

    # Initialize and start all simulators.
    simulators = initializeSimulators(world, recipe['step_size'], outfile_name, recipe)

    # Load profiles for demand (heat, power) and PV generation.
    profiles = loadProfiles()

    # Create instances of simulators.
    entities = instantiateEntities(simulators, profiles, recipe)

    # Add connections between the simulator entities.
    connectEntities(world, entities, recipe)

    # Configure and connect the data collector.
    connectDataCollector(world, entities)

    # Run the simulation.
    world.run(until=recipe['end'])

    sim_elapsed_time = str(timedelta(seconds=time() - sim_start_time))
    logger.info('TOTAL ELAPSED CO-SIMULATION TIME:', sim_elapsed_time)


def get_sim_node_name(
    full_name
):
    (sim_name, sim_node) = full_name.split('.')
    return sim_node
