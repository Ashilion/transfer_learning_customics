#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 12:00:00 
#SBATCH -c 3
#SBATCH --mem-per-cpu=4G

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python ncv_custcox_clinical.py --use_saved_folds --inner_splits 5 --supervised --output_name "ncv_custcox_supervised"