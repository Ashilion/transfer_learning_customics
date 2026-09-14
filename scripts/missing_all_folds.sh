#!/bin/bash
#SBATCH --job-name=missing_data
#SBATCH --array=0-9
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=10    
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=20:00:00
#SBATCH --output=logs/missing_data/opt_fold_%a_%j.out
#SBATCH --error=logs/missing_data/opt_fold_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs

export PYTHONUNBUFFERED=1

CANCER="KIRP"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=4

# EXTRA_FLAGS="--use_saved_folds --ridge --missing_rate 0.3  --name_suffix imp30"
# EXTRA_FLAGS="--use_saved_folds --ridge --missing_rate 0.3  --missing_strategy drop --name_suffix drop30"

EXTRA_FLAGS="--use_saved_folds --ridge --missing_rate 0.5  --name_suffix imp50"
# EXTRA_FLAGS="--use_saved_folds --ridge --missing_rate 0.5  --missing_strategy drop --name_suffix drop50"

# EXTRA_FLAGS="--use_saved_folds --ridge --missing_rate 0.7  --name_suffix imp70"
# EXTRA_FLAGS="--use_saved_folds --ridge --missing_rate 0.7  --missing_strategy drop --name_suffix drop70"



echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"


srun python missing_data_ncv.py \
        --cancer                "$CANCER"               \
        --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
        --inner_splits          "$INNER_SPLITS"         \
        --n_trials_per_worker   "$N_TRIALS_PER_WORKER"  \
    $EXTRA_FLAGS

returned_code=$?
echo "> srun completed with exit code ${returned_code}"

python missing_data_load_all.py \
    --cancer                "$CANCER"               \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    --inner_splits          "$INNER_SPLITS"         \
    --n_trials_per_worker   "$N_TRIALS_PER_WORKER"  \
    $EXTRA_FLAGS

exit ${returned_code}