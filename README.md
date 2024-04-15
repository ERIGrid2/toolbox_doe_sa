# ERIGrid 2.0: Toolbox with Design of Experiment and Sensitivity Analysis methods

This repository contains a toolbox wit Design of Experiment (DoE) and Sensitivity Analysis (SA) methods
developed in the [ERIGrid 2.0] project.
As example scenario for the usage of the toolbox, a multi-energy networks benchmark model (ME benchmark), which was also 
developed in the [ERIGrid 2.0] project, is used (see folder [```benchmark_2_example```](./benchmark_2_example)).
Additionally, some basic example scenarios for DoE, SA and also Uncertainty Propagation methods are provided in the 
[```guidelines_examples```](./guidelines_examples) folder.

## Toolbox
The toolbox, is divided into two main scripts.
The main script is toolbox_start, which will start the simulation and call the methods of toolbox_analysis script 
for analysis of the results.

### toolbox_start
This is main script to run a simulation and analyse the restults
Here, the input parameters (JSON file) are read, and the functions related to the 
scenario simulations, statistical analysis and plotting are called. 
The code starts by reading the folder path hat contains the JSON with the simulation parameters.

In the JSON file, the value for each simulation parameter (e.g., scenario name, step size, sobol samples, etc.) is 
specified individually. 
Additionally, it must be included the "basic configuration" parameters which are sorted as a dictionary. 
These parameters set the values for the variables associated to the system modeled. 
For instance, in the Multi-Energy Network model, the basic configuration parameters stand for hot water tank 
dimensions (height, diameter), delta of control voltage for the heat pump and other set points.

To run it with the multi-energy networks benchmark model, first the requirements have to be installed:
```
> pip install -r benchmark_2_example/requirements.txt
```

To run a simulation the script calls **_benchmark_sim.run_scenario(recipe)_**, which is the ME benchmark,
but can be replaced by any other simulation scipt.

After the simulation run, the results have to be brought in the format needed for further analysis in the toolbox.
For this, **_benchmark_analysis.data_processing()_** is called, which is also part of the ME benchmark and reads in data
from the benchmark specific data structure and stores it as HDF5 file.
Scenario specific adaptions have to be done here, as for example in the ME benchmark the data of the first day is removed
as the simulation needs some time to reach a stable state and the results from each recipe run have to be merged in one 
file.
The data is stored in a pandas DataFrame and written as HDF5 file. The structure is the following:
```
{
    'ID': [recipe['ID']],
    '*parameter name*': [*data*],
    'File ID/dataframe': [*path to file data was extracted from*]}
}
```

Finally, **_toolbox_analysis.analyze_results()_** is called and the results of the simulation analzed.

### toolbox_analysis
It is composed by a set of functions addressed to process the data and perform the statistical analysis 
(incl. Sobol, ANOVA). 
Here is also included functions to plot analysis results.
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
