# TL / CustCox


## Introduction

## File Structure
| Version | Optuna - Recherche hyperparamètres | Evaluation Meilleurs HP
| --- | --- | --- |
| Sans TL          | [search_ncv.py](../CustOmics/search_ncv.py)             | [eval_ncv.py](../CustOmics/eval_ncv.py) |
| Avec TL - Source | [search_source_pretrain.py](../CustOmics/search_source_pretrain.py) | [eval_source_pretrain.py](../CustOmics/eval_source_pretrain.py)|
| Avec TL - Target | [search_target_finetune.py](../CustOmics/search_target_finetune.py) | [eval_target_finetune.py](../CustOmics/eval_target_finetune.py)|

## Lancer dans sshell (1 outer fold)
Setup
``` shell
module load python
cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics
source ../venv_sksurv_dev/bin/activate
```

Lancer un worker optuna sans TL
``` shell
python search_ncv.py --cancer KIRP --outer_fold 0 --supervised --ridge --add_clinical --name_suffix supclinridge_ 
```

Voir les différents arguments disponible
``` shell
python search_ncv.py -h
```

Exemple simulation donnée manquante + modality dropout
``` shell
python search_ncv.py --ridge --cancer KIRP --outer_fold 0 --modality_dropout --missing_rate 0.7 --missing_strategy impute --name_suffix imp70_ 
```

## Lancer sur INTI
``` shell
cd /env/cnrgh/proj/math_stats/scratch/hlegrand
```
---
### script SANS TL 
``` shell
sbatch scripts/sans_tl/search_custcox_sans_tl.sh
```
---
### script AVEC TL SOURCE

1 cancer
``` shell
sbatch scripts/source/tl_source_optu_multi.sh KIRP
```
tous les cancers
``` shell
sbatch scripts/source/tl_source_load_all.sh
```

---
### script AVEC TL TARGET
``` shell
sbatch scripts/target/tl_target_optu_multi_outer.sh
```


---
### Regrouper les csv
Les scripts d'évaluation écrivent dans des fichiers unique spécifique au fold -> il faut regrouper après pour avoir tous les résultats dans un même fichier

``` shell
# Usage: sbatch scripts/combine_results_outer.sh <cancer> <method> <name_suffix>
sbatch scripts/combine_results_outer.sh KIRP custcox supclinridge_
```

---
### Résultats
Pas de scripts slurm spécifique, lancés directement depuis un sshell
-> results_analysis/
-> CustOmics/analysis

## Lancer sur TGCC
script SANS TL 

script AVEC TL SOURCE

script AVEC TL TARGET

scripts données manquantes