#!/bin/bash


# Setting up variables
name_trace="1223_1153___REPORT___1500s_2500t_4c___REPORT_adjusted__lonle_d1lonle_d2__.nc"
name_graph="storage_graph/REPORT_adjusted.pkl"
prc_test=0
var_d1="lonle_d1"
var_d2="lonle_d2"

# Calling the Python script with variables as arguments
python3 -m scripts.bayesian_inference_ppc "$name_trace" "$name_graph" "$prc_test" "$var_d1" "$var_d2"