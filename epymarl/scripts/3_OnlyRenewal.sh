#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the PYTHONPATH to include the src directory relative to the script's location
export PYTHONPATH="$SCRIPT_DIR/epymarl/src:$PYTHONPATH"

# Call the Python script with the specified configuration and environment
python3 -m epymarl.src.main --config=3_OnlyRenewal --env-config=MARL_env
