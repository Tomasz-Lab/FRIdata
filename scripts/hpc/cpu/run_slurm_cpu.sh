#!/bin/bash

EMBEDDER_TYPE=esm2_t33_650M_UR50D

if [[ ! -v COMMON_SLURM_PATH ]]; then
    COMMON_SLURM_PATH="$DEEPFRI_PATH/FRIdata/scripts/hpc/cpu/common_slurm_cpu.sh"
fi

source $COMMON_SLURM_PATH/common_slurm_cpu.sh

PYTHON_COMMAND="PYTHONPATH='.' python3 -u ${DEEPFRI_PATH}/FRIdata/fridata.py generate_data -t sequences,coordinates -d AFDB -c subset --overwrite --version 1_test_dask -i ${IDS_PATH} --input-path ${AFDB_PATH} -e ${EMBEDDER_TYPE} --slurm --verbose"

start_computation "$PYTHON_COMMAND"