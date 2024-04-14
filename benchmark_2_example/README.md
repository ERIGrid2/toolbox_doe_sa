# Co-simulation model of JRA1.1 multi-energy benchmark

This version of the JRA1.1 multi-energy benchmark only requires a working installation of [Python](https://www.python.org/).
The recommended version is **Python 3.8**.

Install all required Python packages with the following command:
```
> python -m pip install -r requirements.txt
```

If pandapipes fails to be installed with pip, it might be needed to install it from its [repository](https://github.com/e2nIEE/pandapipes).

## Running the benchmark simulations

Run a benchmark simulation with **voltage control enabled** with the following command:
```
> python benchmark_multi_energy_sim.py --outfile benchmark_results_ctrl_enabled.h5
```

Run a benchmark simulation with **voltage control disabled** with the following command:
```
> python benchmark_multi_energy_sim.py --outfile benchmark_results_ctrl_disabled.h5 --voltage-control-disabled
```

A few comments about running the simulations:

* On a tpyical laptop each simulation takes about 15 minutes to complete.
* You can speed-up the simulation by using a bigger simulation step size or shorter simulation period.
  For more details refer to the usage instructions:
  ```
  > python benchmark_multi_energy_sim.py --help
  ```
* During the initial phase the simulation is still affected by artifacts resulting from the initial conditions.
  In rare cases this causes unrealistic conditions, which results in warnings like the following:
  ```
  UserWarning: Controller not converged: maximum number of iterations per controller is reached at time t=21660.
  ```
  This is to be expected during the first few simulated hours and can be safely ignored.
  (For this reason, the first simulated day is not taken into account in the analysis.)

## Analyzing the benchmark results

After running the simulations, you can produce plots that analyze the benchmark results with the following command:
```
> python benchmark_multi_energy_analysis.py
```

**NOTE**: To exclude simulation data affected by initialization artifacts, data from the first simulated day is by default not included into the analysis.
