#!/bin/bash
# detect_and_launch.sh
# Usage: bash detect_and_launch.sh

CANCER="HNSC"
RESULTS_DIR="/env/cnrgh/proj/math_stats/scratch/hlegrand/results/folds"  
TOTAL_FOLDS=50  
NAME_SUFFIX=""
# Détecte les folds manquants
MISSING=()
for fold in $(seq 0 $((TOTAL_FOLDS - 1))); do
    # Adapte le pattern selon le nom de tes fichiers résultats
    result_file="${RESULTS_DIR}/ncv_custcox_optuna_${NAME_SUFFIX}${CANCER}_fold${fold}.csv"   
    if ! ls $result_file 2>/dev/null | grep -q .; then
        MISSING+=($fold)
    fi
done
if [ ${#MISSING[@]} -eq 0 ]; then
    echo "Tous les folds sont présents, rien à relancer."
    exit 0
fi

echo "Folds manquants : ${MISSING[*]}"
echo "Nombre : ${#MISSING[@]}"

# Écrit la liste dans un fichier temporaire que le job array va lire
FOLDS_FILE="missing_folds_$(date +%Y%m%d_%H%M%S).txt"
printf '%s\n' "${MISSING[@]}" > "$FOLDS_FILE"

# Soumet le job array avec la bonne taille
sbatch --array=0-$((${#MISSING[@]} - 1)) \
       --export=ALL,FOLDS_FILE="$FOLDS_FILE",CANCER="$CANCER" \
       scripts/relaunch_specific_optuna_load.sh