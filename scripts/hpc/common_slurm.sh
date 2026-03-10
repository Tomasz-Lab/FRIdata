#!/bin/bash

# Define the start_computation function
start_computation() {
    # Check if a command was provided
    if [ -z "$1" ]; then
        echo "Error: No Python command provided."
        exit 1
    fi

    export DASK_LOGGING__DISTRIBUTED="WARNING"

    # Store the provided Python command
    local python_command="$1"

    echo 'IP'
    if [[ ! -v IP_INTERFACE ]]; then
	    IP_INTERFACE='ens1f0'
    fi
    ip a sh dev $IP_INTERFACE | grep -oP '(?<=inet )\S+' | cut -d'/' -f1

    cd $DEEPFRI_PATH

    # Robustly try to load GCC and a Conda/Miniconda module (handle varied names)
    LOADED_GCC=false
    LOADED_CONDA=false
    if command -v module >/dev/null 2>&1; then
        GCC_CANDIDATES=(gcc GCC)
        for MOD in "${GCC_CANDIDATES[@]}"; do
            if module load "$MOD" >/dev/null 2>&1; then
                echo "Loaded module: $MOD"
                LOADED_GCC=true
                break
            fi
        done

        CONDA_CANDIDATES=(miniconda3 Miniconda3 miniconda Anaconda3 anaconda3)
        for MOD in "${CONDA_CANDIDATES[@]}"; do
            if module load "$MOD" >/dev/null 2>&1; then
                echo "Loaded module: $MOD"
                LOADED_CONDA=true
                break
            fi
        done
    fi

    if [ "$LOADED_GCC" = false ]; then
        echo "Error: Could not load a GCC module."
        exit 1
    fi

    if [ "$LOADED_CONDA" = false ]; then
        echo "Error: Could not load a Conda module."
        exit 1
    fi

    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_PATH

    echo "Start time: `date`"
    start_time=$(date +%s)

    cd ./FRIdata

    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

    if [ -e $DEEPFRI_PATH/scheduler.json ]; then
         rm $DEEPFRI_PATH/scheduler.json
    fi

    dask scheduler --scheduler-file $DEEPFRI_PATH/scheduler.json --preload ./toolbox/worker_setup.py &

    while [[ ! -e $DEEPFRI_PATH/scheduler.json ]]; do
        sleep 10
    done

    if [[ ! -v LAUNCH_WORKER_SLURM_PATH ]]; then
        LAUNCH_WORKER_SLURM_PATH="$DEEPFRI_PATH/FRIdata/scripts/hpc/launch_workers_slurm.sh"
    fi

    chmod +x $LAUNCH_WORKER_SLURM_PATH

    $LAUNCH_WORKER_SLURM_PATH $SLURM_CPUS_PER_TASK ${nodes_array[0]} &

    echo "Head node workers"

    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun -w "$node_i" -c $SLURM_CPUS_PER_TASK $LAUNCH_WORKER_SLURM_PATH $SLURM_CPUS_PER_TASK $node_i &
        echo "$node_i started srun workers"
    done

    # Record start time
    start_time=$(date +%s)

    echo "eval python command"

    # Execute the provided Python command
    eval "$python_command"

    end_time=$(date +%s)
    echo "End time: `date`"

    duration=$((end_time - start_time))

    # Convert seconds to hours, minutes, and seconds
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))

    # Print the formatted duration
    printf "Computation time: %02d:%02d:%02d\n" $hours $minutes $seconds
}

# Check if this script is being run directly and not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    # Call the function to execute the script with arguments
    start_computation "$@"
fi