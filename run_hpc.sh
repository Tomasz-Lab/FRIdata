#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00

#SBATCH -p plgrid

#SBATCH -A plgdf2storage-cpu


cd $PLG_GROUPS_STORAGE/plggdeepfri2/

cd deepfri

module load miniconda3
conda activate ./dev_env

cd ./deepFRI2-toolbox-dev

#PYTHONPATH='.' python3 -u toolbox/monitoring/scrap-metrics.py &>scrapping_logs.txt &

PYTHONPATH='.' python3 ./fridata.py -d AFDB -c part -t afdb_swissprot_v4
