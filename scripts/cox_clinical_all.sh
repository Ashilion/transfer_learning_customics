#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 3:00:00 
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/

source venv_sksurv_dev/bin/activate

python cox_model/cox_ncv_all_cancer.py --ridge