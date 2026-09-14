#!/bin/bash
#SBATCH --job-name=tl_evaluate_zero_shot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G   
#SBATCH --time=01:00:00
#SBATCH --error=logs/tl_source/evaluate_zero_shot-%j.out
#SBATCH --output=logs/tl_source/evaluate_zero_shot-%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

export PYTHONUNBUFFERED=1

CANCER="COAD"

# extra --add_clinical / --supervised / --use_saved_folds / --limit_epochs
EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt vvh_pretrained_model.pt --best_params_out vvh_best_params_source.json"

python evaluate_transfer_zero_shot.py \
    --cancer         "$CANCER"        \
    $EXTRA_FLAGS

exit ${returned_code}