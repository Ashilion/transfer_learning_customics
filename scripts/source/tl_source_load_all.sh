#!/bin/bash
#SBATCH --job-name=tl_source_optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G   
#SBATCH --time=10:00:00
#SBATCH --error=logs/tl_source/load_study-%j.out
#SBATCH --output=logs/tl_source/load_study-%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

export PYTHONUNBUFFERED=1

CANCER="LIHC"
INNER_SPLITS=5

# extra --add_clinical / --supervised / --use_saved_folds / --limit_epochs
# EXTRA_FLAGS="--validation vvh --cv_strategy loco --n_folds_loco 5 --domain_adv"
# EXTRA_FLAGS_VAL="--domain_adv --validation vvh --pretrain_ckpt vvh_pretrained_model.pt --best_params_out vvh_best_params_source.json --name_suffix vvh_"

EXTRA_FLAGS="--validation loss --cv_strategy loco --n_folds_loco 5 --name_suffix loss_"
EXTRA_FLAGS_VAL="--validation loss --pretrain_ckpt loss_pretrained_model.pt --best_params_out loss_best_params_source.json --name_suffix loss_"

python search_source_pretrain.py \
    --cancer         "$CANCER"        \
    $EXTRA_FLAGS_VAL

exit ${returned_code}