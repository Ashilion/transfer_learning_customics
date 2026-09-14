#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 12:00:00 
#SBATCH -c 22
#SBATCH --mem-per-cpu=8G

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python ncv_custcox_optu_arg.py --use_saved_folds --inner_splits 5 --n_trials 20 --n_threads 20