#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 10:00:00 
#SBATCH -c 3
#SBATCH --mem-per-cpu=10G

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python tl_custcox_target.py --use_saved_folds --inner_splits 5