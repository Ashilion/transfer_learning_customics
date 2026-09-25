# SANS TL, INTI
## Python

Setup
``` shell
module load python
cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
```

Voir les différents arguments disponible ([search_ncv.py](../CustOmics/search_ncv.py), [eval_ncv.py](../CustOmics/eval_ncv.py))
``` shell
python search_ncv.py -h
```

Exemple espace latent supervisé + donnée clinique 
``` shell
python search_ncv.py --ridge --cancer KIRP --outer_fold 0 --add_clinical --supervised --name_suffix supclinridge_ --n_trials_per_worker 10 
```

Cette commande lance seulement un worker sur un outer_fold pour un certain nombre d'essais. Pour lancer sur tous les outer folds et avoir plusieurs worker en parallèle il faut utiliser le script sbatch (voir après)

Exemple simulation donnée manquante + modality dropout
``` shell
python search_ncv.py --ridge --cancer KIRP --outer_fold 0 --modality_dropout --missing_rate 0.7 --missing_strategy impute --name_suffix imp70_ --n_trials_per_worker 10 
```


## Sbatch

``` shell
cd /env/cnrgh/proj/math_stats/scratch/hlegrand
```

Fichier [search_custcox_sans_tl.sh](../scripts/sans_tl/search_custcox_sans_tl.sh) à modifier pour changer les paramètres de lancement (cancer, nombre d'inner folds, d'outer folds, paramètres de CustCox à regarder avec -h, nombre de trial optuna par worker, nombre de worker optuna ...)
``` shell
sbatch scripts/sans_tl/search_custcox_sans_tl.sh
```


Pour relancer seulement [eval_ncv.py](../CustOmics/eval_ncv.py) (par exemple si l'ensemble des trials n'ont pas été terminé mais qu'on veut quand même évaluer les meilleurs hyperparamètres avec les trials qui ont été fait)

```shell
sbatch scripts/sans_tl/eval_all_custcox_sans_tl.sh
```