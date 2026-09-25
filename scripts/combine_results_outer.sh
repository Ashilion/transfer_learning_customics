#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH --error=logs/combine_results_outer/optuna-%j.out
#SBATCH --output=logs/combine_results_outer/optuna-%j.out

# Usage: sbatch run_aggregate.sh <cancer> <method> <name_suffix>
#   method dans {custcox, finetune, pclsurv, cma}
# Exemples:
#   sbatch run_aggregate.sh HNSC cma cls_
#   sbatch run_aggregate.sh KIRP finetune w
#   sbatch run_aggregate.sh COAD custcox ""

CANCER=$1
METHOD=$2
NAME_SUFFIX=$3

if [ -z "$CANCER" ] || [ -z "$METHOD" ]; then
    echo "Usage: sbatch run_aggregate.sh <cancer> <method> [name_suffix]"
    echo "  method dans {custcox, finetune, pclsurv, cma}"
    exit 1
fi

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/

source venv_sksurv_dev/bin/activate

python results_analysis/cancer_specific/aggregate_folds_optuna.py \
    --cancer "$CANCER" \
    --method "$METHOD" \
    --name_suffix "$NAME_SUFFIX"