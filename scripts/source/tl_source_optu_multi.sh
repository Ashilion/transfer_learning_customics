#!/bin/bash
#SBATCH --job-name=tl_source_optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G   
#SBATCH --time=24:00:00
#SBATCH --error=logs/tl_source/optuna-%j.out
#SBATCH --output=logs/tl_source/optuna-%j.out

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate

export PYTHONUNBUFFERED=1

# --- récupération du cancer en argument ---
if [ -z "$1" ]; then
    echo "Usage: sbatch $0 <CANCER>"
    exit 1
fi
CANCER="$1"

INNER_SPLITS=5
N_TRIALS_PER_WORKER=1

# extra --add_clinical / --supervised / --use_saved_folds / --limit_epochs
# EXTRA_FLAGS="--validation vvh --cv_strategy loco --n_folds_loco 5 --domain_adv --name_suffix vvh_"
# EXTRA_FLAGS_VAL="--domain_adv --validation vvh --pretrain_ckpt vvh_pretrained_model.pt --best_params_out vvh_best_params_source.json --name_suffix vvh_"

# EXTRA_FLAGS="--validation loss --cv_strategy loco --n_folds_loco 5 --name_suffix loss_"
# EXTRA_FLAGS_VAL="--validation loss --pretrain_ckpt loss_pretrained_model.pt --best_params_out loss_best_params_source.json --name_suffix loss_"

EXTRA_FLAGS="--validation loss --cv_strategy loco --n_folds_loco 5 --domain_adv --name_suffix lossdann_"
EXTRA_FLAGS_VAL="--domain_adv --validation loss --pretrain_ckpt lossdann_pretrained_model.pt --best_params_out lossdann_best_params_source.json --name_suffix lossdann_"

srun python search_source_pretrain.py \
    --cancer         "$CANCER"        \
    --inner_splits   "$INNER_SPLITS"  \
    --n_trials_per_worker       "$N_TRIALS_PER_WORKER"     \
    $EXTRA_FLAGS

returned_code=$?
echo "> script completed with exit code ${returned_code}"

python eval_source_pretrain.py \
    --cancer         "$CANCER"        \
    $EXTRA_FLAGS_VAL

exit ${returned_code}