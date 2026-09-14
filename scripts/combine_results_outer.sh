#!/bin/bash 
#SBATCH -n 1      
#SBATCH -t 1:00:00 
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/

source venv_sksurv_dev/bin/activate

# python results_analysis/aggregate_folds_optuna.py --name_suffix clinical_
python results_analysis/aggregate_folds_optuna.py --cancer KIRP --name_suffix scaledmod_ 