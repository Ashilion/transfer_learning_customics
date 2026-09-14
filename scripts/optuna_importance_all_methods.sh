#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 4:00:00 
#SBATCH -c 4
#SBATCH --mem-per-cpu=10G
#SBATCH --output=logs/optuna_importance/optuna_importance_%j.out
#SBATCH --error=logs/optuna_importance/optuna_importance_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

python analyse_optuna_all_methods_all_cancer.py --results_dir ../results