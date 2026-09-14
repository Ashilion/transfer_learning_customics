"""
Analyse et visualisation des résultats Optuna (CustOMICS + CoxNet),
agrégées sur tous les cancers -- version "transfer learning loss source",
où il n'y a plus qu'UNE SEULE étude (outer fold) par cancer.

Fichiers journal attendus :
    journal_tl_loss_source_BRCA.log
    journal_tl_loss_source_COAD.log
    ...

Contrairement au script original (nested CV, plusieurs outer folds par
cancer), ici on a un seul point de données par cancer : les boxplots par
fold (agrégés sur les folds d'un même cancer) n'ont donc plus de sens et
ont été retirés. Les figures "all cancers" (une valeur par cancer) sont
tracées en barplots plutôt qu'en boxplots.

"""

import argparse
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.importance import get_param_importances

import re
import glob
import os

from analyze_optuna_figs import ordered_categories_by_layers_and_size

# ======= Argument Parsing ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Optuna (1 seul outer fold par cancer) pour CustOMICS + CoxNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="tl_loss_source_",
        help="Préfixe présent dans les noms de fichiers journal (journal_{name_suffix}{cancer}.log)."
    )
    parser.add_argument(
        "--skip_importance",
        action="store_true",
        default=False,
        help="Skip computing param/duration importances (peut être lent)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures",
        help="Directory where figures are saved."
    )
    parser.add_argument(
        "--journal_dir",
        type=str,
        default="../results_tgcc/optuna_journal",
        help="Répertoire contenant les fichiers journal_*.log."
    )
    parser.add_argument(
        "--omics_order",
        type=str,
        default="rna,mirna,cnv,mutation",
        help=(
            "Ordre des omics correspondant à hidden_dim_0, hidden_dim_1, ... "
            "(doit matcher list(omics_df.keys()) / cfg['sources'] du script d'entraînement)."
        ))
    return parser.parse_args()


# ======= Hidden dim <-> omics naming =========================================================

def hidden_dim_col_to_omics_name(col, omics_order):
    """
    Traduit un nom de colonne 'hidden_dim_i' vers le nom de l'omic
    correspondant, en se basant sur l'ordre des sources utilisé lors de
    l'entraînement (cfg["sources"] = list(omics_df.keys())).
    Retourne le nom de colonne original si l'index est hors de omics_order
    (par sécurité, pour ne pas planter si la liste ne matche pas).
    """
    try:
        idx = int(col.rsplit("_", 1)[-1])
        return omics_order[idx]
    except (ValueError, IndexError):
        return col


def rename_hidden_dims_to_omics(d, omics_order):
    """
    Renomme les clés 'hidden_dim_i' d'un dict (best_params ou importances)
    en noms d'omics ('rna', 'mirna', ...) selon omics_order. Les autres
    clés (lr, beta, best_alpha, ...) sont laissées inchangées.
    """
    renamed = {}
    for key, value in d.items():
        if key.startswith("hidden_dim"):
            new_key = hidden_dim_col_to_omics_name(key, omics_order)
        else:
            new_key = key
        renamed[new_key] = value
    return renamed


# ======= Study loading (1 seul outer fold par cancer) ========================================

def load_single_study(cancer_name, name_suffix, omics_order, journal_dir, do_importance=True):
    """
    Charge study Optuna disponible pour un cancer donné (plus de
    boucle sur les folds), et récupère :
      - la study elle-même
      - les best_params (+ best_alpha), avec les hidden_dim_i déjà renommés
        en noms d'omics
      - les importances de paramètres / durée (si do_importance=True),
        également renommées
    """
    study = optuna.load_study(
        study_name=f"source_loss_{cancer_name}",
        storage=JournalStorage(
            JournalFileBackend(
                file_path=os.path.join(journal_dir, f"journal_{name_suffix}{cancer_name}.log")
            )
        ),
    )

    params_importance = {}
    duration_importance = {}
    if do_importance:
        params_importance = rename_hidden_dims_to_omics(get_param_importances(study), omics_order)
        duration_importance = rename_hidden_dims_to_omics(
            get_param_importances(study, target=lambda t: t.duration.total_seconds()),
            omics_order,
        )

    best_params = study.best_trial.params
    best_params = rename_hidden_dims_to_omics(best_params, omics_order)

    return study, best_params, params_importance, duration_importance


# ======= Hidden dim (omics) ordering & helpers ================================================

def as_tuple(value):
    """
    Force une valeur en tuple d'ints, quelle que soit sa forme d'origine :
      - vrai tuple/list Python : (1024, 256) -> (1024, 256)
      - string représentant un tuple : "(1024, 256)" -> (1024, 256)
      - scalaire isolé : 256 -> (256,)
    """
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value),)


# ======= Main =================================================================================

def main():
    args = parse_args()

    name_suffix = args.name_suffix
    do_importance = not args.skip_importance
    out_dir = args.out_dir
    journal_dir = args.journal_dir
    omics_order = [s.strip() for s in args.omics_order.split(",") if s.strip()]

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Name suffix       : {name_suffix}")
    print(f"  Importances       : {do_importance}")
    print(f"  Omics order       : {omics_order}")
    print(f"  Journal dir       : {journal_dir}")
    print(f"{'='*60}\n")

    # journal_tl_loss_source_BRCA.log -> capture "BRCA" (pas de _foldN ici)
    pattern = re.compile(
        rf"journal_{re.escape(name_suffix)}([A-Za-z0-9]+)\.log$"
    )

    cancers = sorted({
        pattern.match(os.path.basename(f)).group(1)
        for f in glob.glob(os.path.join(journal_dir, "*.log"))
        if pattern.match(os.path.basename(f))
    })

    print(f"Cancers trouvés ({len(cancers)}) : {cancers}")

    all_importances = []
    all_best_params = []
    all_hidden_dims = []

    for cancer in cancers:
        print(cancer)

        study, best_params, params_importance, duration_importance = load_single_study(
            cancer,
            name_suffix,
            omics_order,
            journal_dir,
            do_importance=do_importance,
        )

        for param, value in params_importance.items():
            all_importances.append({
                "Cancer": cancer,
                "Parameter": param,
                "Importance": value,
            })

        row = best_params.copy()
        row["Cancer"] = cancer
        all_best_params.append(row)

        for col in best_params:
            if col in omics_order:
                architecture = as_tuple(best_params[col])
                all_hidden_dims.append({
                    "Cancer": cancer,
                    "Omics": col,
                    "ArchitectureTuple": architecture,
                    "Architecture": "×".join(map(str, architecture)),
                })

    importances_df = pd.DataFrame(all_importances)
    best_params_df = pd.DataFrame(all_best_params)
    hidden_df = pd.DataFrame(all_hidden_dims)

    # ---- Importances (1 valeur par cancer/paramètre -> barplot, pas boxplot) ----
    if not importances_df.empty:
        g = sns.catplot(
            data=importances_df,
            x="Parameter",
            y="Importance",
            col="Cancer",
            col_wrap=6,
            kind="bar",
            sharey=True,
            height=3,
        )

        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=60)

        g.figure.suptitle(
            f"Hyperparameter importances across {len(cancers)} cancers (1 fold each)",
            fontsize=16,
            y=1.02,
        )

        g.tight_layout()

        out_path = f"{out_dir}/{name_suffix}allcancer_param_importances_barplot.png"
        g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(g.figure)
        print(f"Sauvegardé : {out_path}")

    # ---- Meilleurs hyperparamètres (1 valeur par cancer -> barplot) ----
    cols = [
        c for c in best_params_df.columns
        if c not in omics_order and c not in ["Cancer"]
    ]

    if cols:
        fig, axes = plt.subplots(len(cols), 1, figsize=(14, 4 * len(cols)))
        axes = np.atleast_1d(axes)

        for ax, col in zip(axes, cols):
            sns.barplot(
                data=best_params_df,
                x="Cancer",
                y=col,
                ax=ax,
                color="#2a78d6",
            )
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=90)

            if col in ["lr", "beta", "delta_min"]:
                ax.set_yscale("log")

        fig.tight_layout()
        out_path = f"{out_dir}/{name_suffix}allcancer_best_params_barplot.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Sauvegardé : {out_path}")

    # ---- Heatmaps / barplots d'architecture (hidden_dim) : inchangés ----
    if not hidden_df.empty:
        heat = (
            hidden_df
            .groupby(["Cancer", "Omics", "Architecture"])
            .size()
            .reset_index(name="Count")
        )

        for omic in hidden_df["Omics"].unique():
            tmp = heat[heat["Omics"] == omic]

            mat = tmp.pivot(
                index="Cancer",
                columns="Architecture",
                values="Count"
            ).fillna(0)

            architecture_order_tuple = ordered_categories_by_layers_and_size(
                hidden_df.loc[hidden_df["Omics"] == omic, "ArchitectureTuple"]
            )

            architecture_order = [
                "x".join(map(str, arch))
                for arch in architecture_order_tuple
            ]

            mat = mat.reindex(columns=architecture_order, fill_value=0)

            fig, ax = plt.subplots(figsize=(12, 6))

            sns.heatmap(
                mat,
                cmap="Blues",
                annot=True,
                fmt=".0f",
                ax=ax,
            )

            ax.set_title(omic)
            fig.tight_layout()

            out_path = f"{out_dir}/{name_suffix}hidden_dim_heatmap_{omic}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Sauvegardé : {out_path}")

        for omic in hidden_df["Omics"].unique():
            tmp = heat[heat["Omics"] == omic].copy()

            architecture_order_tuple = ordered_categories_by_layers_and_size(
                hidden_df.loc[hidden_df["Omics"] == omic, "ArchitectureTuple"]
            )
            architecture_order = [
                "x".join(map(str, arch))
                for arch in architecture_order_tuple
            ]

            tmp["Architecture"] = pd.Categorical(
                tmp["Architecture"],
                categories=architecture_order,
                ordered=True,
            )

            fig, ax = plt.subplots(figsize=(14, 6))

            sns.barplot(
                data=tmp,
                x="Cancer",
                y="Count",
                hue="Architecture",
                hue_order=architecture_order,
                ax=ax,
            )

            ax.set_title(omic)
            ax.set_xlabel("Cancer")
            ax.set_ylabel("Count")

            ax.legend(
                title="Architecture",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )

            fig.tight_layout()

            out_path = f"{out_dir}/{name_suffix}hidden_dim_barplot_{omic}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Sauvegardé : {out_path}")


if __name__ == "__main__":
    main()