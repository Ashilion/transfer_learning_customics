# TGCC

## Rsync (depuis inti)
Inti vers TGCC (code source)
``` shell
rsync -avz --chmod=Dg+s --chown=:fg0062 CustOmics/  topaze:/ccc/work/cont007/fg0062/legrandh/CustOmics --exclude="figures" --exclude="splits" --exclude="toy_data" --exclude="__pycache__" --exclude="optuna_journal"
#autres folders moins fréquents
rsync -avz --chmod=Dg+s --chown=:fg0062 utils/ topaze:/ccc/work/cont007/fg0062/legrandh/utils/
rsync -avz --chmod=Dg+s --chown=:fg0062 requirements_minimal.txt  topaze:/ccc/work/cont007/fg0062/legrandh/requirements.txt
rsync -avz --chmod=Dg+s --chown=:fg0062 wheels/  topaze:/ccc/work/cont007/fg0062/legrandh/wheels/
```

TGCC vers inti (resultats)
``` shell
rsync -avz --chmod=Dg+s --chown=:fg0062 topaze:/ccc/work/cont007/fg0062/legrandh/CustOmics/optuna_journal/ results_tgcc/optuna_journal/ --exclude="folds"
```


## Sans TL

Setup
``` shell
cd /ccc/work/cont007/fg0062/legrandh
```

Les commandes suivantes génèrent le fichier de tache (eg tasks.dag), pour modifier ce qui est généré dans le fichier de tache voir scripts_ccc/generate_task_allc_allf.sh . Pour modifier les paramètres CCCMSUB, voir scripts_ccc/all_cancer_all_folds.sh     
``` shell
./run_scripts/allc_allf.sh --supervised --add_clinical --name_suffix supclinridge_
./run_scripts/allc_allf.sh --add_clinical --name_suffix clinridge_
./run_scripts/allc_allf.sh  --supervised --name_suffix supridge_
./run_scripts/allc_allf.sh  --name_suffix ridge_
```


```shell
./run_scripts/eval_allc_allf.sh --supervised --add_clinical --name_suffix supclinridge_
./run_scripts/eval_allc_allf.sh --add_clinical --name_suffix clinridge_
./run_scripts/eval_allc_allf.sh --supervised --name_suffix supridge_
./run_scripts/eval_allc_allf.sh --name_suffix ridge_
```


## Avec TL - Source

Utilise partition A100 et gpu

```shell
ccc_msub scripts_ccc/tl_source_optu.sh
```

## Avec TL - Target

```shell
#search
./scripts_ccc/generate_task_tl_list_cancer.sh
ccc_msub scripts_ccc/tl_list_cancer.sh

#eval
./scripts_ccc/generate_task_tl_eval_all.sh
ccc_msub scripts_ccc/tl_load_all.sh
```
