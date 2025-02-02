#!/bin/bash


# Setting up variables
sample_posterior=1500       # Number of samples to draw from the posterior
tune=2500                   # Number of tuning steps
chains=4                    # Number of chains
name_model="REPORT"    # Name of the model (for saving purposes)
path_graph="storage_graph/REPORT_adjusted.pkl"
n_cores=16                   # Number of cores to infer with
note_save=""                # Note to add to the saved file
prc_test=0                  # Optional variable to keep some nodes out of the bayesian inference
target_accept=98            # Target acceptance rate for the NUTS sampler
var_d1="lonle_d1"           # First variable to be modeled
var_d2="lonle_d2"           # Second variable to be modeled

# Calling the Python script with variables as arguments
python3 -m scripts.bayesian_inference "$sample_posterior" "$tune" "$chains" "$name_model" "$path_graph" "$n_cores" "$note_save" "$prc_test" "$target_accept" "$var_d1" "$var_d2"