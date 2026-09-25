# AVEC TL - SOURCE , INTI
## Python

Setup
``` shell
module load python
cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
```

Voir les différents arguments disponible ([search_source_pretrain.py](../CustOmics/search_source_pretrain.py), [eval_source_pretrain.py](../CustOmics/eval_source_pretrain.py))
``` shell
python search_source_pretrain.py -h
```

Exemple espace latent supervisé + donnée clinique 
``` shell
python search_source_pretrain.py \
    --cancer KIRP        \
    --inner_splits   5 \
    --n_trials_per_worker 10 \
    --validation loss --cv_strategy loco \
    --n_folds_loco 5 --domain_adv --name_suffix lossdann_    
```


Exemple simulation donnée manquante + modality dropout
``` shell
python search_source_pretrain.py \
    --cancer KIRP        \
    --inner_splits   5 \
    --n_trials_per_worker 10 \
    --validation loss --cv_strategy loco \
    --n_folds_loco 5 --name_suffix missing_  \
    --modality_dropout --missing_rate 0.7 --missing_strategy impute  
```

## Sbatch

``` shell
cd /env/cnrgh/proj/math_stats/scratch/hlegrand
```

Fichier [tl_source_optu_multi.sh](../scripts/avec/tl_source_optu_multi.sh) à modifier pour changer les paramètres de lancement (cancer, nombre d'inner folds, d'outer folds, paramètres de CustCox à regarder avec -h, nombre de trial optuna par worker, nombre de worker optuna ... )
1 cancer
``` shell
sbatch scripts/source/tl_source_optu_multi.sh KIRP
```
tous les cancers
``` shell
sbatch scripts/source/tl_source_run_all_cancer.sh
```