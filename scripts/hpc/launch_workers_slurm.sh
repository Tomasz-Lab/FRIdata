#!/bin/bash

export DASK_LOGGING__DISTRIBUTED="WARNING"

WORKERS_COUNT=$(($1))
if [[ ! -v MEMORY_LIMIT ]]; then
  MEMORY_LIMIT='288GiB'
fi

mkdir -p $SCRATCH/slurm_jobdir/$SLURM_JOB_ID/dask-workers

local_dir=$SCRATCH/slurm_jobdir/$SLURM_JOB_ID/dask-workers/$2

mkdir $local_dir

dask worker --scheduler-file $DEEPFRI_PATH/scheduler.json --nworkers $WORKERS_COUNT --nthreads 1 --memory-limit $MEMORY_LIMIT --local-directory $local_dir --preload $DEEPFRI_PATH/FRIdata/src/fridata/worker_setup.py