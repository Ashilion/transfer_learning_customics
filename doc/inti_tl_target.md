# AVEC TL - TARGET , INTI
## Python

Setup
``` shell
module load python
cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
```

Voir les différents arguments disponible ([search_target_finetune.py](../CustOmics/search_target_finetune.py),  [eval_target_finetune.py](../CustOmics/eval_target_finetune.py))
``` shell
python search_target_finetune.py -h
```

Exemple espace latent supervisé + donnée clinique 
``` shell
python search_target_finetune.py \
    --cancer KIRP  --outer_fold  0 --inner_splits 5 \
    --n_trials_per_worker 10  --ridge \
    --use_saved_folds --pretrain_ckpt loss_pretrained_model.pt \
    --best_params_in loss_best_params_source.json \
    --add_clinical --supervised --name_suffix supclinridge_ 
```


Exemple simulation donnée manquante + modality dropout
``` shell
python search_target_finetune.py \
    --cancer KIRP  --outer_fold  0 --inner_splits 5 \
    --n_trials_per_worker 10  --use_saved_folds \
    --pretrain_ckpt lossmiss50_pretrained_model.pt \
    --best_params_in lossmiss50_best_params_source.json \
    --ridge --name_suffix tlimp50_30_\
    --modality_dropout --missing_rate 0.3 --missing_strategy impute
```


## Sbatch

``` shell
cd /env/cnrgh/proj/math_stats/scratch/hlegrand
```

Fichier [search_custcox_sans_tl.sh](../scripts/sans_tl/search_custcox_sans_tl.sh) à modifier pour changer les paramètres de lancement (cancer, nombre d'inner folds, d'outer folds, paramètres de CustCox à regarder avec -h, nombre de trial optuna par worker, nombre de worker optuna ... )
``` shell
sbatch scripts/target/tl_target_optu_multi_outer.sh
```

Pour relancer seulement [eval_target_finetune.py](../CustOmics/eval_target_finetune.py) (par exemple si l'ensemble des trials n'ont pas été terminé mais qu'on veut quand même évaluer les meilleurs hyperparamètres avec les trials qui ont été fait)

``` shell
sbatch scripts/target/tl_target_eval_all.sh
```