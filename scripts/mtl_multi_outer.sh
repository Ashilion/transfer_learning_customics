#!/bin/bash
#SBATCH --job-name=mtl_custcox
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G   
#SBATCH --time=10:00:00
#SBATCH --error=logs/tl_target/mtl_%a_%j.out
#SBATCH --output=logs/tl_target/mtl_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

export PYTHONUNBUFFERED=1

EXTRA_FLAGS=""

echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"

srun python mtl_custcox.py \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    $EXTRA_FLAGS

returned_code=$?
echo "> script completed with exit code ${returned_code}"