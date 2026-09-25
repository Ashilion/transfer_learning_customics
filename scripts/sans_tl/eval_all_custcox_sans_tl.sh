#!/bin/bash
#SBATCH --job-name=custcox_ncv
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=10    
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=20:00:00
#SBATCH --output=logs/combine_outer_optuna/opt_fold_%a_%j.out
#SBATCH --error=logs/combine_outer_optuna/opt_fold_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs

export PYTHONUNBUFFERED=1

CANCER="BLCA"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=6

EXTRA_FLAGS="--use_saved_folds --add_clinical --ridge --supervised --name_suffix supclinridge_"
# EXTRA_FLAGS="--use_saved_folds --add_clinical --ridge --name_suffix clinridge_"
# EXTRA_FLAGS="--use_saved_folds --ridge --supervised --name_suffix supridge_"
# EXTRA_FLAGS="--use_saved_folds --ridge --name_suffix ridge_"


echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"

python eval_ncv.py \
    --cancer                "$CANCER"               \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    --inner_splits          "$INNER_SPLITS"         \
    --n_trials_per_worker   "$N_TRIALS_PER_WORKER"  \
    $EXTRA_FLAGS

exit ${returned_code}