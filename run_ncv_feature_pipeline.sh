#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1      
#SBATCH -t 02:00:00 
#SBATCH -c 2 

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv/bin/activate

python ncv_feature_pipeline.py