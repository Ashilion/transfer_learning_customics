#!/bin/bash
#SBATCH -p normal    
#SBATCH -n 1          
#SBATCH -t 00:10:00 

module load python
source venv/bin/activate

python preprocess_tcga.py
python preprocess_tcga_part2.py
python preprocess_tcga_part3.py
