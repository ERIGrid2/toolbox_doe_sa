# ERIGrid 2.0: Task JRA 1.2 development repository

## Benchmark Model Multi-Energy Networks Example

This example is based on the benchmark scenario, which is available at https://github.com/ERIGrid2/benchmark-model-multi-energy-networks
For the sensitivity analysis, code from an ERIGrid 1 summer school ist used (https://zenodo.org/record/2837928).

## Script Structure
The sensitivity analysis algrotihm which has been developed in EIRGrid 2.0 project under the activitities of JRA 1.2 is divided into three main scripts whcih are explained below.

### benchmark_sensitivity_analysis
This is the script to be run where the input parameters (JSON file) are read, and the functions related to the scenario simulations, statistical analysis and plotting are called. 
The code start by reading the folder path hat contains the JSON with the simulation parameters which has an structure as shown below.

In the JSON file, the value for each simulation parameter (e.g., scenario name, step size, sobol samples, etc.) is specified individually. 
Additionally, it must be included the "basic configuration" parameters which are sorted as a dictionary. 
These parameters set the values for the variables associated to the system modeled. 
For instance, in he Multi-Energy Network model, the basic configuration parameters stand for hot water tank dimensions (height, diameter), delta of control voltage for the heat pump and other set points.

### benchmark_multi_energy_sim
Here, the mosaik co-simulation setup is stated. 
The simulators and respective entities are initialized as well as the profiles are set. 
Also, the function that run the scenarios is included in the script.

### benchmark_multi_energy_analysis
It is composed by a set of functions addressed to process the data and perform the statistical analysis (incl. Sobol, ANOVA). 
Here is also included functions to plot analysis results.

