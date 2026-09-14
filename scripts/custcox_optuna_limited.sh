#!/bin/bash
#SBATCH -p normal  
#SBATCH -N 1   
#SBATCH -n 1     
#SBATCH -t 12:00:00 
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python ncv_custcox_optu_arg.py --use_saved_folds --inner_splits 5 --limit_epochs 10 --n_threads 1 --trial_pruning --plot_optuna