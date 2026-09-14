#!/bin/bash
#SBATCH --job-name=relaunch_combine_outer_optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1:00:00
#SBATCH --output=logs/res_opt_fold_%a_%j.out
#SBATCH --error=logs/res_opt_fold_%a_%j.out

module load python
cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs
export PYTHONUNBUFFERED=1

# Lit le fold correspondant à cet array task depuis le fichier
OUTER_FOLD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "../$FOLDS_FILE")

INNER_SPLITS=5
N_TRIALS_PER_WORKER=2
# EXTRA_FLAGS="--use_saved_folds --supervised --name_suffix supervised_"
EXTRA_FLAGS="--use_saved_folds"


echo "========================================"
echo " Job        : $SLURM_JOB_ID"
echo " Array task : $SLURM_ARRAY_TASK_ID"
echo " Outer fold : $OUTER_FOLD"
echo " Node       : $(hostname)"
echo "========================================"

python optuna_load_all_study_1_outer.py \
    --cancer "$CANCER" \
    --outer_fold "$OUTER_FOLD" \
    --inner_splits "$INNER_SPLITS" \
    --n_trials_per_worker "$N_TRIALS_PER_WORKER" \
    $EXTRA_FLAGS