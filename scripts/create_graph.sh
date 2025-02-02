#!/bin/bash

# Setting up variables
lines=752,751,756,753
indicators=lonle_d1,lonle_d2
minimum_track_length=0
track_segment_length=150
threshold_connection_m=3
graph_name="REPORT"
data_path="/mnt/d/ETH_too_big/Thesis_Data/New Data/Data_zurich/RLDataMain"
data_action_path="/mnt/d/ETH_too_big/Thesis_Data/New Data/Data_zurich/RLDataMaintenance"

# Calling the Python script with variables as arguments
python3 -m scripts.create_graph "$lines" "$indicators" "$minimum_track_length" "$track_segment_length" "$threshold_connection_m" "$graph_name" "$data_path" "$data_action_path"