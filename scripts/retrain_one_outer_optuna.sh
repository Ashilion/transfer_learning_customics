#!/bin/bash
#SBATCH --job-name=custcox_ncv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=20:00:00
#SBATCH --output=logs/combine_outer_optuna/opt_fold_%j.out
#SBATCH --error=logs/combine_outer_optuna/opt_fold_%j.out

# Vérification de l'argument
if [ $# -ne 1 ]; then
    echo "Usage: sbatch $0 <outer_fold>"
    exit 1
fi

OUTER_FOLD=$1

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs

export PYTHONUNBUFFERED=1

CANCER="HNSC"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=10
# EXTRA_FLAGS="--use_saved_folds --supervised --name_suffix supervised_"
EXTRA_FLAGS="--use_saved_folds"

echo "========================================"
echo " Job         : $SLURM_JOB_ID"
echo " Outer fold  : $OUTER_FOLD"
echo " Node        : $(hostname)"
echo " Tasks       : $SLURM_NTASKS"
echo "========================================"

srun python optuna_multi_study_1_outer.py \
    --cancer              "$CANCER" \
    --outer_fold          "$OUTER_FOLD" \
    --inner_splits        "$INNER_SPLITS" \
    --n_trials_per_worker "$N_TRIALS_PER_WORKER" \
    $EXTRA_FLAGS

returned_code=$?
exit $returned_code