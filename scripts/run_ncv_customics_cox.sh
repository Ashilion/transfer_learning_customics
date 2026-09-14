#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 15:00:00 
#SBATCH -c 2 

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python ncv_customics_cox.py