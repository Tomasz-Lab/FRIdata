# FRIdata

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

PYTHONPATH='.' python3 -u ${FRIDATA_PATH}/toolbox.py \
 input_generation \
 -t sequences,coordinates,distograms,embeddings \
 -d AFDB \
 -c subset \
 --version test  \
 -i ${IDS_PATH} \
 --input-path ${AFDB_PATH} \
 -e ${EMBEDDER_TYPE}
```