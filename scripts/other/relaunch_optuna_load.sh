#!/bin/bash
#SBATCH --job-name=relaunch_combine_outer_optuna
#SBATCH --array=0-49
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1:00:00
#SBATCH --output=logs/opt_fold_%a_%j.out
#SBATCH --error=logs/opt_fold_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs

export PYTHONUNBUFFERED=1

CANCER="KIRP"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=2
# EXTRA_FLAGS="--use_saved_folds --add_clinical --name_suffix clinical_"
# EXTRA_FLAGS="--use_saved_folds"

# EXTRA_FLAGS="--use_saved_folds --add_clinical --ridge --supervised --name_suffix supclinridge_"
# EXTRA_FLAGS="--use_saved_folds --add_clinical --ridge --name_suffix clinridge_"
# EXTRA_FLAGS="--use_saved_folds --ridge --supervised --name_suffix supridge_"
EXTRA_FLAGS="--use_saved_folds --ridge --name_suffix ridge_"

echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"

python optuna_load_all_study_1_outer.py \
    --cancer                "$CANCER"               \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    --inner_splits          "$INNER_SPLITS"         \
    --n_trials_per_worker   "$N_TRIALS_PER_WORKER"  \
    $EXTRA_FLAGS

exit ${returned_code}