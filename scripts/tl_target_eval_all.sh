#!/bin/bash
#SBATCH --job-name=eval_tl_target_optuna
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G   
#SBATCH --time=2:00:00
#SBATCH --error=logs/tl_target/optuna_eval_%a_%j.out
#SBATCH --output=logs/tl_target/optuna_eval_%a_%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

export PYTHONUNBUFFERED=1

CANCER="KIRC"
INNER_SPLITS=5
N_TRIALS_PER_WORKER=10

# EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt vvh_pretrained_model.pt --best_params_in vvh_best_params_source.json"
# EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt loss_pretrained_model.pt --best_params_in loss_best_params_source.json --name_suffix multiarchi_"

EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt loss_pretrained_model.pt --best_params_in loss_best_params_source.json --add_clinical --name_suffix clinridge_ --ridge"
# EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt loss_pretrained_model.pt --best_params_in loss_best_params_source.json --add_clinical --supervised --name_suffix supclinridge_ --ridge"
# EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt loss_pretrained_model.pt --best_params_in loss_best_params_source.json --supervised --name_suffix supridge_ --ridge"
# EXTRA_FLAGS="--use_saved_folds --pretrain_ckpt loss_pretrained_model.pt --best_params_in loss_best_params_source.json --name_suffix ridge_ --ridge"


echo "========================================"
echo " Job   : $SLURM_JOB_ID"
echo " Task  : $SLURM_ARRAY_TASK_ID  (outer fold)"
echo " Node  : $(hostname)"
echo " Tasks : $SLURM_NTASKS"
echo "========================================"

python tl_custcox_target_eval_all_1_outer.py \
    --cancer         "$CANCER"        \
    --outer_fold            "$SLURM_ARRAY_TASK_ID"  \
    $EXTRA_FLAGS

exit ${returned_code}