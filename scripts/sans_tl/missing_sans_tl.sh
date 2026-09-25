#!/bin/bash
#SBATCH --job-name=missing_sans_tl
#SBATCH --array=0-19
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=10    
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=20:00:00
#SBATCH --output=logs/missing_data/sans_tl_%a_%j.out
#SBATCH --error=logs/missing_data/sans_tl_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs

export PYTHONUNBUFFERED=1

CANCER="KIRP"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=4

# EXTRA_FLAGS="--use_saved_folds --add_clinical --ridge --supervised --name_suffix supclinridge_"
# EXTRA_FLAGS="--use_saved_folds --add_clinical --ridge --name_suffix clinridge_"
# EXTRA_FLAGS="--use_saved_folds --ridge --supervised --name_suffix supridge_"
# EXTRA_FLAGS="--use_saved_folds --ridge --name_suffix ridge_"


EXTRA_FLAGS="--use_saved_folds --ridge --name_suffix drop30_"
MISSING_FLAGS="--modality_dropout --missing_rate 0.3 --missing_strategy drop"

echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"


srun python search_ncv.py \
        --cancer                "$CANCER"               \
        --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
        --inner_splits          "$INNER_SPLITS"         \
        --n_trials_per_worker   "$N_TRIALS_PER_WORKER"  \
    $EXTRA_FLAGS $MISSING_FLAGS

returned_code=$?
echo "> srun completed with exit code ${returned_code}"

python eval_ncv.py \
    --cancer                "$CANCER"               \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    --inner_splits          "$INNER_SPLITS"         \
    --n_trials_per_worker   "$N_TRIALS_PER_WORKER"  \
    $EXTRA_FLAGS $MISSING_FLAGS

exit ${returned_code}