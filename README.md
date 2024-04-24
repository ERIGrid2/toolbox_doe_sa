# ERIGrid 2.0: Toolbox with Design of Experiment and Sensitivity Analysis methods

This repository contains a toolbox wit Design of Experiment (DoE) and Sensitivity Analysis (SA) methods
developed in the [ERIGrid 2.0] project.
As example scenario for the usage of the toolbox, a multi-energy networks benchmark model (ME benchmark), which was also 
developed in the [ERIGrid 2.0] project, is used (see folder [```benchmark_2_example```](./benchmark_2_example)).
Additionally, some basic example scenarios for DoE, SA and also Uncertainty Propagation methods are provided in the 
[```guidelines_examples```](./guidelines_examples) folder.

## Toolbox
The toolbox, is divided into two main scripts (toolbox_start and toolbox_analysis) and the simulation script.
In general, this toolbox should be usable with all kinds of simulation and laboratory experiments.
But because a simulation example is used in this repository, only the term simulation will be used in the following.

### toolbox_start
This script has to be called first to read in the JSON file with the simulation parameters.
The path to the file can be provided with the **--config** argument when calling the script.

```
python toolbox_start.py --config .\simulation_configurations\simulation_parameters.json
```

In the configuration file it can be chosen which sampling and analysis method should be used.
Currently, only specific combinations are possible, which are listed in the following table:

| configuration string |  sampling  | analysis |
|:-----|:--------:|------:|
| sobol   | sobol sequence | meta model |
| extreme_points   |  min and max values of interval  |   ANOVA |
| LHS   | LHS |  meta model |
| sobol_indices   |  sobol indices sampling |    sobol indices |
| fast   | eFAST sampling |   fast |
| OAT   | ... |  ... |
| distribution_and_discrete   | ... | ... |

In the JSON file, the value for each simulation parameter (e.g., scenario name, step size, sobol samples, etc.) is 
specified individually.
Some exampled can be found in the **_simulation_configurations_** folder.
Additionally, parameters of the modeled system can be defined in the **_entities_parameters_** part. 
These parameters will be used to create a full parameterization file for each simulation run. 
For example, in the ME benchmark, the basic configuration parameters stand for hot water tank 
dimensions (height, diameter), delta of control voltage for the heat pump and other set points.

To model variations for the experiment, the parameters to be varied have to be defined in the
**_variations_dict_**. For most methods an interval has to be specified, but for the distribution_and_discrete type
also distributions and a list of discrete variations can be defined.
```
    "variations_dict": {
        "storage_tank":{
            "INNER_DIAMETER": [1.0, 4],
        },
        "communication":{
            "delay": {
                "type": "norm",
                "mean": 15.0,
                "stdvs": 2.0
            }
        },
        "grid": {
            "load_step": {
                "type": "discrete",
                "set": [
                  50.0,
                  100.0,
                  200.0
                ]
            }
        }
    },
```
For analysis of the simulation/experiment results, a list of **_target_metrics_** can be defined.
```
"target_metrics": ["electricity_export_mwh", "self_consumption_perc"]
```
Based on the configuration, the scripts creates so-called recipes with a full parameterization set for each 
simulation run, which has to be done based on the chosen method and configuration.
Together with some other configuration files, the recipes are stored in the given scenario folder as
**_recipes.json_**. A shortened example is shown here:
```
{
    "run_00": {
        "ID": "00",
        "scenario_name": "tank_scaling_heat",
        "folder_temp_files": "output\\temp_files_tank_scaling_heat",
        "summary_filename": "runs_summary",
        "end": 172800,
        "step_size": 60,
        "gen_pv": {
            "scale": 1.0
        },
        "storage_tank": {
            "INNER_HEIGHT": 7.9,
            "INNER_DIAMETER": 4.5,
        },
    },
    "run_01": {
        "ID": "01",
        ...
    }
```

### simulation script

To integrate a simulation, the previously written **_recipes.json_** has to be used to run the according
number of simulations with the defined parameterization.

As example with the ME benchmark the **_toolbox_execute_meb.py_** is provided, which calls the 
**_benchmark_sim.run_scenario(recipe)_**.
It can be configured with the **_--folder_** to provide the folder of all scenario files, were the recipes.json 
file was stored by the first script.
```
python.exe toolbox_execute_meb.py --folder .\output\temp_files_tank_scaling_heat\
```

After the simulation run, the results have to be brought in the format needed for further analysis in the toolbox.
For this, **_benchmark_analysis.data_processing()_** is called, which is also part of the ME benchmark and reads in data
from the benchmark specific data structure and stores it as HDF5 file.
Scenario specific adaptions have to be done here, as for example in the ME benchmark the data of the first day is removed
as the simulation needs some time to reach a stable state.
Based on the simulation results the target metrics are calculated.
The data is stored in a pandas DataFrame and written as JSON or HDF5 file. The structure is a list of the following elements:
```
{
    'ID': recipe['ID'],
    '*target metric*': *data*,
    'File ID/dataframe': *path to file data was extracted from*,
    '*parameter name*': *data*,
}
```
With data from the ME benchmark this can look the following:
```
[
    {
        "ID": "00",
        "electricity_export_mwh": 0.449657090307714,
        "self_consumption_perc": 63.45142726914459,
        "File ID/dataframe": "output\\temp_files_tank_scaling_heat/tank_scaling_heat_00.h5/timeseries/sim_00",
        "heat_profiles.scale": 1.0,
        "storage_tank.INNER_HEIGHT": 7.9,
        "storage_tank.INNER_DIAMETER": 4.5,
    },
    {
        "ID": "01",
        "electricity_export_mwh": 0.4496544009240551,
        "self_consumption_perc": 63.45164586490652,
        "File ID/dataframe": "output\\temp_files_tank_scaling_heat/tank_scaling_heat_01.h5/timeseries/sim_01",
        "heat_profiles.scale": 0.95,
        "storage_tank.INNER_HEIGHT": 7.9,
        "storage_tank.INNER_DIAMETER": 6.25,
    },
```

### toolbox_analysis
After the simulation was run and the results were prepared for the toolbox, the toolbox_analysis script can be run.
With the **_--folder_** argument the scenario folder with all configuration and result files has to be defined.
Additionally, the **results_** filename can be specified.

```
python.exe toolbox_analysis.py --folder .\output\temp_files_tank_scaling_heat\ --results runs_summary.json
```

It is composed by a set of functions addressed to process the data and perform the statistical analysis and 
plot analysis results.
For the SA, code from an ERIGrid 1 summer school ist used (https://zenodo.org/record/2837928).

## Multi-Energy Networks Example Scenario

This example is based on the benchmark scenario, which is available at [GitHub](https://github.com/ERIGrid2/benchmark-model-multi-energy-networks).

### benchmark_multi_energy_sim
Here, the mosaik co-simulation setup is stated. 
The simulators and respective entities are initialized as well as the profiles are set. 
Also, the function that run the scenarios is included in the script.

### benchmark_multi_energy_analysis
This file contains some functions to plot figures with the results of the simulation.
Also a function for data processing was added. It reads the simulation results from a HDF5 file and
stores the results, which are relevant for the SA.

## Guideline examples
This code is explained in more detail in the Deliverable D10.2 - 'D-JRA1.2 - Methods for Holistic Test Reproducibility'
of the [ERIGrid 2.0] project.

### Sampling approaches
Some basic examples for DoE and SA methods.

### MoReSQUE examples
Three examples scenarios for the [MoReSQUE](https://gitlab.com/mosaik/tools/mosaik-moresque) Uncertainty Propagation 
tool are shown.

## Funding acknowledgement

The development of [Toolbox with DoE and SA methods](https://github.com/ERIGrid2/toolbox_doe_sa) has been supported by the [ERIGrid 2.0] project of the 
H2020 Programme under [Grant Agreement No. 870620](https://cordis.europa.eu/project/id/870620).

[ERIGrid 2.0]: https://erigrid2.eu
