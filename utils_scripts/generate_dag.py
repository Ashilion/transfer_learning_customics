#!/usr/bin/env python3
"""
Génère tasks.dag pour pegasus-mpi-cluster (PMC), au format
    TASK <id> <commande> <args...>
    EDGE <parent> <enfant>

Combinatoire (identique au générateur Pegasus.api précédent) :
    - 3 modèles pré-entraînés (checkpoint miss 30 / 50 / 70)
    - 3 missing_rate de fine-tuning (30 / 50 / 70)
    - 2 stratégies (drop / impute)
    - par config : 20 folds (search -> eval) + 1 job d'agrégation

Usage :
    python3 generate_tasks_dag.py > tasks.dag
    # puis, sur le cluster :
    sbatch run_pmc.sbatch
"""

CUSTOMICS_DIR = "/env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics"
VENV_PYTHON = "/env/cnrgh/proj/math_stats/scratch/hlegrand/venv_sksurv_dev/bin/python3"

CANCER = "KIRP"
INNER_SPLITS = 5
N_TRIALS_PER_WORKER = 60
N_OUTER_FOLDS = 20

PRETRAIN_MISS_RATES = [30, 50, 70]   # modèle pré-entraîné (checkpoint)
FINETUNE_MISS_RATES = [30, 50, 70]   # missing_rate au fine-tuning
STRATEGIES = ["drop", "impute"]


def fmt_args(*args):
    return " ".join(str(a) for a in args)


def main():
    tasks = []  # lignes "TASK ..."
    edges = []  # lignes "EDGE ..."

    for pm in PRETRAIN_MISS_RATES:
        ckpt = f"{CUSTOMICS_DIR}/{CANCER}_lossmiss{pm}_pretrained_model.pt"
        params_src = f"{CUSTOMICS_DIR}/{CANCER}_lossmiss{pm}_best_params_source.json"

        for fm in FINETUNE_MISS_RATES:
            for strategy in STRATEGIES:
                cfg_id = f"pm{pm}_fm{fm}_{strategy}"
                name_suffix = f"tlimp{fm}_{strategy}_"
                eval_ids = []

                common = [
                    "--cancer", CANCER,
                    "--use_saved_folds",
                    "--pretrain_ckpt", ckpt,
                    "--best_params_in", params_src,
                    "--ridge",
                    "--name_suffix", name_suffix,
                    "--modality_dropout",
                    "--missing_rate", f"{fm / 100:.2f}",
                    "--missing_strategy", strategy,
                ]

                for fold in range(N_OUTER_FOLDS):
                    search_id = f"search_{cfg_id}_f{fold}"
                    eval_id = f"eval_{cfg_id}_f{fold}"

                    search_cmd = fmt_args(
                        VENV_PYTHON, f"{CUSTOMICS_DIR}/search_target_finetune.py",
                        *common,
                        "--outer_fold", fold,
                        "--inner_splits", INNER_SPLITS,
                        "--n_trials_per_worker", N_TRIALS_PER_WORKER,
                    )
                    eval_cmd = fmt_args(
                        VENV_PYTHON, f"{CUSTOMICS_DIR}/eval_target_finetune.py",
                        *common,
                        "--outer_fold", fold,
                    )

                    tasks.append(f"TASK {search_id} {search_cmd}")
                    tasks.append(f"TASK {eval_id} {eval_cmd}")
                    edges.append(f"EDGE {search_id} {eval_id}")

                    eval_ids.append(eval_id)

                # Job d'agrégation des 20 folds de cette config
                aggregate_id = f"aggregate_{cfg_id}"
                aggregate_cmd = fmt_args(
                    VENV_PYTHON, f"{CUSTOMICS_DIR}/aggregate_target_finetune.py",
                    "--cancer", CANCER,
                    "--name_suffix", name_suffix,
                    "--outer_splits", N_OUTER_FOLDS,
                )
                tasks.append(f"TASK {aggregate_id} {aggregate_cmd}")
                for eval_id in eval_ids:
                    edges.append(f"EDGE {eval_id} {aggregate_id}")

    print("# Tâches pegasus-mpi-cluster")
    print(f"# {len(tasks)} TASK, générées pour {CANCER}")
    for line in tasks:
        print(line)
    print()
    print("# Flèches de dépendance")
    for line in edges:
        print(line)


if __name__ == "__main__":
    main()