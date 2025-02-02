#!/bin/bash


##### SBATCH --gpus=1
# Setting up variables
name_trace="1223_1153___REPORT___1500s_2500t_4c___REPORT_adjusted__lonle_d1lonle_d2__.nc"
name_graph="storage_graph/REPORT_adjusted.pkl"
prc_test=100            # use same data for training and testing
var_d1="lonle_d1"
var_d2="lonle_d2"
n=50
fixed_input=1
use_mean=1

# Calling the Python script with variables as arguments
python3 -m scripts.bayesian_samples "$name_trace" "$name_graph" "$prc_test" "$var_d1" "$var_d2" "$n" "$fixed_input" "$use_mean"