#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 2:00:00 
#SBATCH -c 3
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/run_in_customics/sl_%j.out
#SBATCH --error=logs/run_in_customics/sl_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python $1