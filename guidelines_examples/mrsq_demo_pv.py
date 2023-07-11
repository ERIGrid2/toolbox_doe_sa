import os

import mosaik
import moresque.ensembles as eam

# Component initialization commands:
sim_config = {
    'ESim': {
        'python': 'moresque.propagator_sim:UQPropagator',
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
    'CSV': {
        'python': 'mosaik_csv:CSV',
    },
}

mosaik_config = {
    'addr': ('127.0.0.1', 5555),
}

# Simulation time parameters:
START = '2014-05-24 01:00:00'
END = 7 * 24 * 3600          # simulate for one week

# Component parameters:
U_SUN = 'data/sun_uncertainty.json'  # weather simulation uncertainty overview
DNI_DATA = 'data/dni_savannah_lat32.csv'     # irradiation data
ERROR_DATA = 'data/error_savannah2.csv'      # irradiation data errors

U_PV = 'data/pv_uncertainty.json'    # PV panel simulator uncertainty overview
DATEFILE = 'data/date_data.csv'      # temporal data

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

    # Ensemble for PV panel simulator:
    pvsim = world.start('PVsim', datefile=DATEFILE, start_date=START)
    pv_config = {'world': world,
                'esim_name': 'ESim',
                'model_name': 'PV',
                'sim_obj': pvsim,
                'ensemble_num': 1,
                'u_config': U_PV,
                'input_config': {'PV': {'attrs': {'DNI': 1}, 'name': 'PV'}},
                'output_config': {'PV': ['P']},
                'strat_num': 100,       # arbitrarily picked member number
                'step_size': 3600,
                'model_params': {'lat': 32.117,
                                 'az_tilt': 0.0}}       # certain parameters
    pv_ens = eam.EnsembleSet(pv_config)
    # Ensemble output:
    pv_outs = [pv[0] for pv in pv_ens.output_modules]
    # Ensemble input:
    pv_ins = [pv[0] for pv in pv_ens.input_modules]
    # Output aggregation module:
    pv_oam = world.start('OAM', step_size=3600, attrs=['P'])
    pv_agg = pv_oam.ItvlOut()   # P-box output expected
    world.connect(pv_outs[0], pv_agg, 'P')

    # Connect solar ensemble entity to PV ensemble entity:
    world.connect(s_outs[0], pv_ins[0], 'DNI') # only one ensemble given each

    # Create database::
    db = world.start('DB', step_size=60, duration=END)
    if not os.path.isdir('results'):
        os.mkdir('results')
    hdf5 = db.Database(filename='results\\pv_uncertainty_study.hdf5')
    # Connect aggregation modules to database:
    world.connect(pv_agg, hdf5, 'P_minA', 'P_maxA')
    world.connect(sun_agg, hdf5, 'DNI_minA', 'DNI_maxA')


if __name__ == '__main__':
    main()
