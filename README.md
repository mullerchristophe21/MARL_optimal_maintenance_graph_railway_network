# Multi-Agent Reinforcement Learning for Optimal Maintenance of Graph-Based Railway Networks

This repository contains the work for a Master's Thesis project conducted at ETH Zürich, focusing on applying Multi-Agent Reinforcement Learning (MARL) to optimize maintenance strategies for graph-based railway networks.

## Project Overview

- **Author**: Christophe Muller
- **Institution**: ETH Zürich
- **Supervisors**:
  - Giacomo Arcieri (PhD candidate)
  - Gregory Duthé (PhD candidate)
  - Eleni Chatzi (Prof. Doc.)
  - Nicolai Meinshausen (Prof. Doc.)

For a comprehensive understanding of the project, please refer to the [full thesis report](https://drive.google.com/file/d/1OsLInoz9kQ03Af5tQCzU7Uqx5X9NBXom/view?usp=sharing).

## Installation

Follow these steps to set up the project environment:

1. Create a new Conda environment:
   ```
   conda create -n MARL_on_graph python=3.8.19
   ```

2. Clone the repository:
   ```
   git clone https://github.com/mullerchristophe21/MARL_optimal_maintenance_graph_railway_network.git
   ```

3. Install the required packages:
   ```
   conda env create -f environment.yml
   conda activate MARL_on_graph
   ```

## Project Components

The project is divided into three main components:

### 1. Graph Creation

* Visualize the different track lines available in the data in `map_all_lines.html`.
* Adjust the parameters of `scripts/create_graph.py` and run the script.
* Adjust your graph object with the help of the `graph_adjustments.ipynb` notebook:
   * Precise latitude and longitude
   * Start and end dates of the indicator time series
   * Correcting the dates of the maintenance actions
* Save the final version of the graph.

### 2. Bayesian Inference

* Choose parameters of the BI in the `scripts/bayesian_inference.sh` file.
* One obtains a trace of the BI in the `storage_inference` folder.
* This trace can be used for:
   * Generating posterior samples from the joint distribution of the model parameters via the file `scripts/bayesian_inference_ppc.sh`
   * Generating samples of the observations, conditioned on actions, via the fil `scripts/bayesian_samples.sh`
* Analyzes and plots of the resulting files can be processed in the `results_inference.ipynb` notebook.

### 3. Multi-Agent Reinforcement Learning (MARL)

The MARL algorithms consist of two components: the environment and the algorithm.

To run a simulation, some commands are saved in `epymarl/scripts/` looking like:

`python3 -m epymarl.src.main --config=ALGO --env-config=ENV`

Where:

* `ALGO` is a config from `epymarl/src/config/algs/`
* `ENV` is a config from `epymarl/src/config/envs/` 

We describe these here-below.

#### 3.1 RL Environments

From the Bayesian Inference learned parameters, a corresponding MARL environment is created and available in `epymarl/scr/envs/`. This environment requires some configuration:

* The graph on which we want to do the MARL
* The BI trace trained on real data
* The number of timesteps of the episode
* The cost configuration of the environment

Note also that 2 other environments are implemented. One that force the neighbouring track segments to be correlated, and another one also using this correlation but diminishing the overall variance for easier learning.

One can play around with these implementations in the jupyter notebook `epymarl/example_RL_environment.ipynb`.


#### 3.2 RL Algorithms

The RL Algorithms are determined by the configs in `epymarl/src/config/algs/`. Some configs of typical MARL algorithms are prepared together with configs close to the ones used in the report (GNN, Graph Transformer, ...).