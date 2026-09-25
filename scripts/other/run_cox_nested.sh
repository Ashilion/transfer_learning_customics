#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 00:30:00 
#SBATCH -c 2 

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand

source venv_sksurv_dev/bin/activate

python cox_model/cox_nested_cv.py