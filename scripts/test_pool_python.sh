#!/bin/bash
#MSUB -r test_pool_python
#MSUB -N 1
#MSUB -n 1
#MSUB -c 36
#MSUB -T 3600
#MSUB -o logs/test_pool_%I.o
#MSUB -e logs/test_pool_%I.e

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

ccc_mprun python thread_limited_custcox_1_outer.py \
            --cancer COAD \
            --outer_fold 0 \
            --n_trials_per_worker 10 \
            --use_saved_folds \
            --inner_splits 5 \
            --ridge \
            --name_suffix test \
            --multiproc 36 \
