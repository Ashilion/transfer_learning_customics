#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G   
#SBATCH --time=12:00:00
#SBATCH --error=optuna-%j.out
#SBATCH --output=optuna-%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

# Enable the standard and error outputs of Python.
export PYTHONUNBUFFERED=1

CANCER="COAD"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=20

# extra --add_clinical / --supervised / --use_saved_folds / --limit_epochs
EXTRA_FLAGS="--use_saved_folds --limit_epochs 10"

srun python optuna_multi_study.py \
    --cancer         "$CANCER"        \
    --inner_splits   "$INNER_SPLITS"  \
    --n_trials_per_worker       "$N_TRIALS_PER_WORKER"     \
    $EXTRA_FLAGS

returned_code=$?
echo "> script completed with exit code ${returned_code}"

python optuna_load_all_study.py \
    --cancer         "$CANCER"        \
    --inner_splits   "$INNER_SPLITS"  \
    --n_trials_per_worker       "$N_TRIALS_PER_WORKER"     \
    $EXTRA_FLAGS

exit ${returned_code}