# FRIdata

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat-square)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![CI](https://github.com/Tomasz-Lab/FRIdata/actions/workflows/ci.yml/badge.svg)](https://github.com/Tomasz-Lab/FRIdata/actions/workflows/ci.yml)
[![Implements Dask](https://img.shields.io/badge/impl-Dask-blue)](https://docs.dask.org/en/stable/)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?style=flat-square)](https://github.com/Tomasz-Lab/FRIdata/)
[![GitHub issues](https://img.shields.io/github/issues/Tomasz-Lab/FRIdata.svg?style=flat-square)](https://github.com/Tomasz-Lab/FRIdata/issues)

## Instalation and activation

1. Download the repo

```
git clone https://github.com/Tomasz-Lab/FRIdata.git
cd FRIdata
```

2. [Install miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

3. Install mamba

```
## prioritize 'conda-forge' channel
conda config --add channels conda-forge

## update existing packages to use 'conda-forge' channel
conda update -n base --all

## install 'mamba'
conda install -n base mamba
```

4. Create a mamba environment

```
mamba create -f toolbox_env_conda.yml
```

5. Activate mamba shell hook

```
# Choose your shell type. Could be one of these: {bash,cmd.exe,dash,fish,nu,posix,powershell,tcsh,xonsh,zsh}
eval "$(mamba shell hook --shell <replace with shell type>)"
```

6. Activate the mamba environment

```
mamba activate tbe
```

## Running tests

```
pytest ./tests
```

## Running on AFDB structures locally

Requires having a directory with AFDB structures and a text file containing list of AFDB IDs with `\n` delimeter.

```
#
# Assuming all steps from `Instalation and activation` succeded
#
FRIDATA_PATH="<repository path>"
AFDB_PATH="<AFDB structures directory path>"
IDS_PATH="<AFDB IDs file path>"

cd ${FRIDATA_PATH}

EMBEDDER_TYPE=esm2_t33_650M_UR50D

# (MACOS only) Fix for OpenMP multiple runtime error
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHONPATH='.' python3 -u ${FRIDATA_PATH}/fridata.py \
 input_generation \
 -t sequences,coordinates,distograms,embeddings \
 -d AFDB \
 -c subset \
 --version test  \
 -i ${IDS_PATH} \
 --input-path ${AFDB_PATH} \
 -e ${EMBEDDER_TYPE}
```

## Running on HPC

Running FRIdata on HPC differs on CPU and GPU nodes. This instruction set is valid for HPC hosted in PLGrid infrastructure. Running on other infrastructures may require additional adjustments.

### CPU

Prerequisites:
- Having active grant valid on the HPC
- Having a full list of mandatory ENV vars set (ideally in .bashrc):
    - `DEEPFRI_PATH`: should always refer to a parent directory of this repo
    - `IDS_PATH`: path to a text file with AFDB indexes listed
    - `AFDB_PATH`: path to AFDB structures (can be empty directory - structures will be fetched there)
    - `DATA_PATH`: path to the parent diretory of all generated output data
    - Optional ENV vars with default values:
        - `COMMON_SLURM_PATH`: path to common_slurn_cpu.sh, defaults to `$DEEPFRI_PATH/FRIdata/scripts/hpc/cpu/common_slurm_cpu.sh`
        - `LAUNCH_WORKER_SLURM_PATH`: path to launch_worker_slurm_cpu.sh, defaults to `$DEEPFRI_PATH/FRIdata/scripts/hpc/cpu/launch_workers_slurm_cpu.sh`
        - `MEMORY_LIMIT`: memory limit per Dask worker, defaults to `288GiB`
        - `IP_INTERFACE`: network unix interface, where dask workers are connected. Defaults to `ens1f0`
        - `CONDA_ENV_PATH`: path to conda environment, defaults to `$DEEPFRI_PATH/conda_dev`
- Have installed module miniconda3
- Have installed module gcc

Steps:

1. Download the repo

```
git clone https://github.com/Tomasz-Lab/FRIdata.git
cd FRIdata
```

2. Update run permissions

```
chmod u+x -R scripts/hpc/cpu
```

3. Run `initialize_slurm_cpu.sh`. As an argument put the path into directory, where `.conda` directory should be installed and specify `--cpu` flag

```
./scripts/hpc/cpu/initialize_slurm_cpu.sh <path to .conda> --cpu
```

4. Schedule SBatch script into the HPC with all the args specified

```
sbatch --cpus-per-task=<cpus> --time=<HH:MM:SS> --nodes=<nodes> --account=<grant name> scripts/hpc/cpu/run_slurm_cpu.sh
```