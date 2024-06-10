#!/bin/bash

PROJECT_NAME="bitNet_gradient_free"

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"
  
  # Run the wandb sweep command and store the output in a temporary file
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "hyperparams_opt/$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  
  # Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  wandb agent --count 10 $SWEEP_ID
}

# list of sweeps to call
run_sweep_and_agent "brackets_adam"
run_sweep_and_agent "brackets_mcmc"
run_sweep_and_agent "brackets_sim_annealing"
run_sweep_and_agent "brackets_simple_ga"
run_sweep_and_agent "brackets_zeroth"
