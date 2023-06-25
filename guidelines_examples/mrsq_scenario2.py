import copy

import mosaik
import moresque.ensembles as eam

# Component initialization commands:
sim_config = {
    'PyPower': {
        'python': 'mosaik_pypower.mosaik:PyPower',
    },
    'ESim': {
        'python': 'moresque.propagator_sim:UQPropagator',
    },
    'CSV': {
        'python': 'mosaik_csv:CSV',
    },
    'PVsim': {
        'python': 'mosaik_pv.mosaik:PvAdapter',
    },
    'OAM': {
        'python': 'moresque.stat_output:StatOut',
    },
    'DB': {
        'cmd': 'mosaik-hdf5 %(addr)s',
    },
}

mosaik_config = {
    'addr': ('127.0.0.1', 5555),
}

# Indendence copula for an arbitrary number of inputs:
def copula_indep(*args):
    out = copy.copy(args[0])
    for i in range(1, len(args)):
        out *= args[i]
    return out

# Simulation time parameters:
START = '2014-03-22 07:00:00'
# END = 24 * 3600          # simulate for five hours
END = 36 * 3600          # simulate for five hours
# END = 120          # simulate for five hours

# Component parameters:
U_GRID = 'data/grid_uncertainty.json' # grid simulator uncertainty overview
GRID_FILE = 'data/demo_lv_grid.json' # grid topology
GRID_CDM = 'data/grid_cdm.json' # child dependency map of simulated grid

U_SUN = 'data/sun_uncertainty.json'  # weather simulation uncertainty overview
DNI_DATA = 'data/dni_savannah_lat32.csv'     # irradiation data
ERROR_DATA = 'data/error_savannah2.csv'      # irradiation data errors

U_HH = 'data/house_uncertainty.json' # household simulation uncertainty overview
PROFILE_FILE = 'data/mprofiles.data' # household consumption data

U_PV = 'data/pv_uncertainty.json'    # PV panel simulator uncertainty overview
DATEFILE = 'data/date_data.csv'      # temporal data

COP = copula_indep      # independence copula

def main():
    world = mosaik.World(sim_config, mosaik_config=mosaik_config)
    create_scenario(world)
    world.run(until=END)

def create_scenario(world):
    # Ensemble for solar irradiation simulator:
    solsim = world.start('CSV', sim_start=START, datafile=DNI_DATA)
    so_config = {'world': world,
                'esim_name': 'ESim',    # Propagator component in sim_config
                'model_name': 'Sun',
                'sim_obj': solsim,
                'ensemble_num': 1,
                'u_config': U_SUN,
                'output_config': {'Sun': ['DNI']},
                'strat_num': 1,         # No external uncertainty expected
                'step_size': 3600,
                'start_date': START,
                'datafile': ERROR_DATA} # Data needed for u-function
    sol_ens = eam.EnsembleSet(so_config)
    # Ensemble output:
    s_outs = [s[0] for s in sol_ens.output_modules]
    # Output aggregation module:
    sun_oam = world.start('OAM', step_size=3600, attrs=['DNI'])
    sun_agg = sun_oam.ItvlOut()    # Interval output is expected
    world.connect(s_outs[0], sun_agg, 'DNI')

    # Ensemble for household simulator:
    hhsim = world.start('CSV', sim_start=START, datafile=PROFILE_FILE)
    hh_config = {'world': world,
                'esim_name': 'ESim',
                'model_name': 'House',
                'sim_obj': hhsim,
                'ensemble_num': 5,      # Ensembles represent five entites
                'u_config': U_HH,
                'output_config': {'House': ['P']},
                'strat_num': 1,         # No external uncertainty expected
                'step_size': 900}
    hh_ens = eam.EnsembleSet(hh_config)
    # Ensemble output:
    hh_outs = [h[0] for h in hh_ens.output_modules]
    # Output aggregation module:
    hh_oam = world.start('OAM', step_size=900, attrs=['P'])
    hh_agg = hh_oam.DstrOut.create(5)   # Distribution output expected
    for i in range(len(hh_outs)):
        world.connect(hh_outs[i], hh_agg[i], 'P')

    # Ensemble for PV panel simulator:
    pvsim = world.start('PVsim', datefile=DATEFILE, start_date=START)
    pv_config = {'world': world,
                'esim_name': 'ESim',
                'model_name': 'PV',
                'sim_obj': pvsim,
                'ensemble_num': 5,
                'u_config': U_PV,
                'input_config': {'PV': {'attrs': {'DNI': 1}, 'name': 'PV'}},
                'output_config': {'PV': ['P']},
                'copula_config': {'input': {'P': COP},
                                 'output': {'P': COP}}, # Copulas needed
                'strat_num': 100,       # arbitrarily picked member number
                'step_size': 3600,
                'model_params': {'lat': 32.117,
                                 'el_tilt': 32.0,
                                 'az_tilt': 0.0}}       # certain parameters
    pv_ens = eam.EnsembleSet(pv_config)
    # Ensemble output:
    pv_outs = [pv[0] for pv in pv_ens.output_modules]
    # Ensemble input:
    pv_ins = [pv[0] for pv in pv_ens.input_modules]
    # Output aggregation module:
    pv_oam = world.start('OAM', step_size=3600, attrs=['P'])
    pv_agg = pv_oam.PboxOut.create(5)   # P-box output expected
    for i in range(len(pv_outs)):
        world.connect(pv_outs[i], pv_agg[i], 'P')

    # Ensemble for power grid simulator (composite):
    pypow = world.start('PyPower', step_size=900)
    pow_config = {'world': world,
                'esim_name': 'ESim',
                'model_name': 'Grid',
                'sim_obj': pypow,
                'ensemble_num': 1,
                'u_config': U_GRID,
                'input_config': {'PQBus': {'attrs': {'P': 2},
                                           'name': 'node'}}, # Nodes receive input
                'output_config': {'PQBus': ['Vm']}, # Branches provide output
                'copula_config': {'input': {'Vm': COP},
                                  'output': {'Vm': COP}},
                # 'output_config': {'Branch': ['P_from']}, # Branches provide output
                # 'copula_config': {'input': {'P_from': COP},
                #                  'output': {'P_from': COP}},
                'strat_num': 600,
                'step_size': 900,
                'child_num': 5,    # a grid entity has 5 nodes and 5 branches
                'model_params': {'gridfile': GRID_FILE},
                # 'cdm': GRID_CDM, # documenting interdendency between children
                'propagation_config': {'input': {'out_grid': 600}}}
    grid_ens = eam.EnsembleSet(pow_config)
    # Ensemble output (at branches):
    branch_outs = [b for b in grid_ens.output_modules[0]]
    # Ensemble input (at nodes/busses):
    node_ins = [n for n in grid_ens.input_modules[0]]
    # Output aggregation module:
    branch_oam = world.start('OAM', step_size=900, attrs=['Vm'])
    branch_agg = branch_oam.PboxOut.create(6)   # P-box output expected
    for i in range(len(branch_outs)):
        world.connect(branch_outs[i], branch_agg[i], 'Vm')

    # Connect solar ensemble entity to PV ensemble entities:
    for pv_in in pv_ins:
        world.connect(s_outs[0], pv_in, 'DNI')

    # Connect PV and household ensemble entities to grid ensemble busses:
    for i in range(len(node_ins)):
        world.connect(hh_outs[i], node_ins[i], 'P')
        world.connect(pv_outs[i], node_ins[i], 'P')

    # Create database::
    db = world.start('DB', step_size=60, duration=END)
    hdf5 = db.Database(filename='results\\mrsq_scenario2.hdf5')
    # Connect aggregation modules to database:
    for branch in branch_agg:
        # Extract mean values, 68.2% intervals and 90% intervals:
        world.connect(branch, hdf5, 'Vm_aveA', 'Vm_aveB',
                      'Vm_q15A', 'Vm_q15B', 'Vm_q84A',
                      'Vm_q84B', 'Vm_q05A', 'Vm_q05B',
                      'Vm_q95A', 'Vm_q95B')
    for pv in pv_agg:
        world.connect(pv, hdf5, 'P_aveA', 'P_aveB', 'P_q15A', 'P_q15B',
                      'P_q84A', 'P_q84B', 'P_q05A', 'P_q05B', 'P_q95A',
                      'P_q95B')
    for house in hh_agg:
        world.connect(house, hdf5, 'P_aveA', 'P_q15A', 'P_q05A', 'P_q95A',
                      'P_q84A')
    # Extract minimum and maximum for solar irradiation:
    world.connect(sun_agg, hdf5, 'DNI_minA', 'DNI_maxA')


if __name__ == '__main__':
    main()
