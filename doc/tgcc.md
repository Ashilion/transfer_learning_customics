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

## Avec TL - Target Argument --multiproc

Sur Inti, les différents processus (ie worker optuna) sont crées à l'aide de srun qui permet de créer realiser creer un nombre de processus égal #SBATCH --ntasks-per-node=10    
```shell
srun python search_ncv.py ... 
```

### Pourquoi créer les processus depuis Python sur le TGCC ?

Sur le TGCC, les workers ne sont pas lancés séparément par le gestionnaire de jobs : c'est le script Python qui les crée lui-même (option `--multiproc`). Principalement pour 2 raisons :

**1. Garantir que tous les processus tournent sur le même nœud**

L'optimisation Optuna utilise un `JournalFileStorage` : l'état de l'étude est stocké dans un fichier journal que tous les workers lisent et écrivent. Ce mécanisme repose sur un fichier local partagé et suppose donc que tous les processus s'exécutent sur la même machine. En les créant depuis un unique processus Python, on s'assure qu'ils restent sur le nœud attribué au job.

> Cette contrainte disparaîtrait avec un `RDBStorage` (base de données, par ex. PostgreSQL ou MySQL mais pas SQLite seul présent sur le TGCC), qui permet à des workers répartis sur plusieurs nœuds de partager la même étude. Voir la documentation Optuna.

**2. Partager la mémoire entre les processus**

Les données TCGA (cliniques et omiques) sont chargées **une seule fois**, dans le processus principal, **avant** la création des workers. Sous Linux, les processus enfants créés par `fork` partagent alors les pages mémoire du parent tant qu'ils ne les modifient pas (mécanisme de *copy-on-write*). Comme les workers se contentent de lire ces données sans jamais les modifier, elles ne sont pas dupliquées en mémoire :

- selon la taille du cancer étudié, un processus peut consommer plus de 2 Go de RAM ;
- sur le TGCC, la RAM est allouée proportionnellement au nombre de cœurs demandés : il faut réserver **2 cœurs physiques** pour disposer d'assez de mémoire pour un seul processus ;
- sans partage mémoire, ces 2 cœurs ne feraient donc tourner qu'**un seul** worker ;
- grâce au partage, les données ne sont présentes qu'une fois en mémoire, et plusieurs workers peuvent tenir dans la RAM de ces 2 cœurs.

De plus, l'hyperthreading est activé sur le TGCC : chaque cœur physique expose 2 cœurs logiques, soit **2 cœurs physiques = 4 cœurs logiques**. On peut donc lancer `--multiproc 4` pour créer 4 workers, un par cœur logique, sans réserver davantage de ressources.

---

### ⚠️ Attention : affinité CPU et librairies Python

**Le problème**

L'*affinité CPU* d'un processus est la liste des cœurs sur lesquels le système d'exploitation a le droit de l'exécuter. Un processus enfant hérite de l'affinité de son parent au moment de sa création.

Or certaines librairies (PyTorch, NumPy, via leurs backends de calcul parallèle) modifient l'affinité du processus au moment de leur import et la restreignent à **un seul cœur**. Conséquence : tous les workers créés après l'import de PyTorch héritaient de cette affinité réduite et s'exécutaient **tous sur le même cœur**, les autres cœurs réservés restant inutilisés. Le parallélisme était donc purement apparent.

La solution (implémentée dans [CustOmics/pipeline_utils/parallel.py](../CustOmics/pipeline_utils/parallel.py))

1. **Avant tout import** de PyTorch, NumPy ou d'une librairie qui les charge, on enregistre l'affinité d'origine du processus, c'est-à-dire la liste complète des cœurs logiques alloués au job (déterminée par `cpu_per_tasks` dans le script de soumission TGCC).
2. **Après la création** de chaque worker, on redéfinit explicitement son affinité à partir de cette liste sauvegardée : les cœurs disponibles sont répartis entre les processus demandés (`--multiproc`), soit environ `nombre de cœurs logiques disponibles / nombre de processus` cœurs par worker.

Exemple : avec `cpu_per_tasks = 2` (2 cœurs physiques, donc 4 cœurs logiques) et `--multiproc 4`, chaque worker se voit attribuer son propre cœur logique au lieu de partager tous le même.