#!/bin/bash

EMBEDDER_TYPE=esm2_t33_650M_UR50D

if [[ ! -v COMMON_SLURM_PATH ]]; then
    COMMON_SLURM_PATH="$DEEPFRI_PATH/FRIdata/scripts/hpc/common_slurm.sh"
fi

source $COMMON_SLURM_PATH

# PYTHONPATH points at src/ so this works without an editable install; if the
# package is installed, `fridata generate_data ...` is equivalent.
PYTHON_COMMAND="PYTHONPATH='${DEEPFRI_PATH}/FRIdata/src' python3 -u -m fridata generate_data -t $1 -d AFDB -c subset --overwrite --version 1_test_dask -i ${IDS_PATH} --input-path ${AFDB_PATH} -e ${EMBEDDER_TYPE} --slurm --verbose"

start_computation "$PYTHON_COMMAND"