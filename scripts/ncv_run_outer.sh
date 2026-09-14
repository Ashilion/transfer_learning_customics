#!/bin/bash
#SBATCH --job-name=custcox_optuna_ncv
#SBATCH --array=0-49              
#SBATCH --cpus-per-task=1     
#SBATCH --mem-per-cpu=8G    
#SBATCH --time=20:00:00
#SBATCH --output=logs/opt_fold_%a_%j.out
#SBATCH --error=logs/opt_fold_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

mkdir -p logs

CANCER="COAD"
INNER_SPLITS=5
N_TRIALS=20
N_THREADS=1         
# TIMEOUT=3600       

# extra --add_clinical / --supervised / --use_saved_folds / --limit_epochs
EXTRA_FLAGS="--use_saved_folds"

echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " CPUs  : $SLURM_CPUS_PER_TASK"
echo "========================================"

python optuna_outer_fold.py \
    --cancer         "$CANCER"        \
    --outer_fold     "$SLURM_ARRAY_TASK_ID" \
    --inner_splits   "$INNER_SPLITS"  \
    --n_threads      "$N_THREADS"     \
    --n_trials       "$N_TRIALS"     \
    $EXTRA_FLAGS

echo "Fold $SLURM_ARRAY_TASK_ID done."