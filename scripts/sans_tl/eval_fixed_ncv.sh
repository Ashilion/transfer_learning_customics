#!/bin/bash
#SBATCH --job-name=eval_fixed_hp
#SBATCH --array=0-49
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/fixed_hp/opt_fold_%a_%j.out
#SBATCH --error=logs/fixed_hp/opt_fold_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
mkdir -p logs

export PYTHONUNBUFFERED=1
CANCER="KIRP"

# EXTRA_FLAGS="--use_saved_folds --ridge --optimizer adamw --weight_decay 0.2 --fixed_params_file fixed_hp.json --name_suffix adamw_"
EXTRA_FLAGS="--use_saved_folds --ridge --fixed_params_file fixed_hp.json --name_suffix adam_"



echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"

python eval_ncv.py \
    --cancer                "$CANCER"               \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    $EXTRA_FLAGS

exit ${returned_code}