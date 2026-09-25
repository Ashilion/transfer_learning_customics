"""
Analyse et visualisation des résultats Optuna (nested CV, CustOMICS + CoxNet).

Ce script charge les studies Optuna sauvegardées pour chaque fold externe,
puis génère plusieurs figures :
  - grilles 2x2 des plots Optuna standards (optimization history, param
    importances, duration importances, timeline) pour les premiers folds
  - boxplots des importances de paramètres / durée à travers les folds
  - boxplots des meilleurs hyperparamètres (hors hidden_dim) à travers les folds
  - barplots de la distribution des hidden_dim_i.

"""

import argparse
import math
from io import BytesIO
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.importance import get_param_importances

from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_timeline,
)
import sys
sys.path.append("..")
# ======= Argument Parsing ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested CV with CustOMICS + CoxNet + Optuna for survival prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default="COAD",
        help="Cancer type name (e.g. COAD, BRCA, LUAD)."
    )
    parser.add_argument(
        "--outer_splits",
        type=int,
        default=50,
        help="Number of outer CV folds."
    )
    parser.add_argument(
        "--n_folds_to_plot",
        type=int,
        default=4,
        help="Number of outer folds to display in the 2x2 grids of Optuna plots."
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="String to add to the name of the results files (and journal files)."
    )
    parser.add_argument(
        "--skip_importance",
        action="store_true",
        default=False,
        help="Skip computing param/duration importances (can be slow for many folds)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures",
        help="Directory where figures are saved."
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


# ======= Study loading =======================================================================

def load_studies(cancer_name, n_outer, name_suffix, do_importance=True):
    """
    Charge les studies Optuna de chaque fold externe et récupère :
      - la liste des studies
      - les best_params (+ best_alpha) de chaque fold
      - les importances de paramètres / durée (si do_importance=True)
    """
    studies = []
    params_importances = []
    durations_importances = []
    best_params_tab = []

    for outer_fold in range(n_outer):
        study = optuna.load_study(
            study_name=f"journal_storage_multiprocess_{name_suffix}fold{outer_fold}",
            # study_name=f"journal_storage_multiprocess_{name_suffix}{cancer_name}_fold{outer_fold}",
            storage=JournalStorage(
                JournalFileBackend(
                    file_path=f"./optuna_journal/journal_{name_suffix}fold{outer_fold}.log"
                    #file_path=f"./optuna_journal/journal_{name_suffix}{cancer_name}_fold{outer_fold}.log"
                )
            ),
        )
        studies.append(study)

        if do_importance:
            params_importances.append(get_param_importances(study))
            durations_importances.append(
                get_param_importances(study, target=lambda t: t.duration.total_seconds())
            )

        best_params = study.best_trial.params
        best_params["best_alpha"] = study.best_trial.user_attrs["best_alpha"]
        best_params_tab.append(best_params)

    return studies, best_params_tab, params_importances, durations_importances


# ======= Per-fold Optuna plots ================================================================

def get_default_plot_funcs():
    """Liste des (nom, fonction) de plots Optuna à assembler en grille par fold."""
    return [
        ("Optimization history", lambda s: plot_optimization_history(s)),
        ("Param importances", lambda s: plot_param_importances(s)),
        ("Duration importances", lambda s: plot_param_importances(
            s,
            target=lambda t: t.duration.total_seconds(),
            target_name="duration",
        )),
        ("Timeline", lambda s: plot_timeline(s)),
    ]


def plot_folds_grid(studies, plot_funcs, n_folds_to_plot, cancer_name, name_suffix, out_dir="figures"):
    """Pour chaque type de plot Optuna, assemble une grille 2x2 des n premiers folds."""
    for func_name, fn in plot_funcs:
        images = []
        for outer_fold in range(n_folds_to_plot):
            tmp_ax = fn(studies[outer_fold])
            tmp_ax.set_title(f"Fold {outer_fold}", fontsize=10)
            tmp_fig = tmp_ax.get_figure()
            tmp_fig.tight_layout()

            buf = BytesIO()
            tmp_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            images.append(mpimg.imread(buf))
            plt.close(tmp_fig)

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        for ax, img in zip(axes.flat, images):
            ax.imshow(img)
            ax.axis("off")

        fig.suptitle(func_name.replace("_", " ").title(), fontsize=14)
        fig.tight_layout()

        out_path = f"{out_dir}/{name_suffix}{cancer_name}_{func_name}_all_folds.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Sauvegardé : {out_path}")


# ======= Importances boxplots ================================================================

def plot_importances_boxplot(importances_list, value_name, title, filename, cancer_name, name_suffix, out_dir="figures"):
    """Boxplot générique pour une liste de dicts d'importances (un dict par fold)."""
    all_importances = pd.DataFrame(importances_list).fillna(0)
    df_long = all_importances.melt(var_name="Paramètre", value_name=value_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_long, x="Paramètre", y=value_name, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out_path = f"{out_dir}/{name_suffix}{cancer_name}_{filename}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Best params boxplots ================================================================

def plot_best_params_boxplots(best_params_tab, cancer_name, name_suffix, out_dir="figures"):
    """Boxplots (un par hyperparamètre, hors hidden_dim) des meilleurs paramètres par fold externe."""
    all_params = pd.DataFrame(best_params_tab).fillna(0)
    hidden_dim_cols = [c for c in all_params.columns if c.startswith("hidden_dim")]
    all_params = all_params.drop(columns=hidden_dim_cols)

    n_params = len(all_params.columns)
    ncols = 3
    nrows = math.ceil(n_params / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, all_params.columns):
        sns.boxplot(y=all_params[col], ax=ax)
        ax.set_title(col)
        ax.set_ylabel("Value")
        if col in ("lr", "delta_min", "beta"):
            ax.set_yscale("log")

    for ax in axes[n_params:]:
        fig.delaxes(ax)

    fig.suptitle("Best params across outer folds", fontsize=16)
    fig.tight_layout()

    out_path = f"{out_dir}/{name_suffix}{cancer_name}_best_params_boxplots_separate.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Hidden dim ordering & barplots ======================================================
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
 
 
def tuple_n_layers(t):
    """Nombre de couches dans l'architecture, ex: (512, 256, 128) -> 3."""
    return len(t)
 
 
def tuple_param_count(t):
    """
    Proxy du nombre de paramètres : produit des éléments du tuple.
    Ex: (512, 256, 128) -> 512*256*128 = 16 777 216.
    """
    n = 1
    for v in t:
        n *= v
    return n
 
 
def ordered_categories_by_layers_and_size(values):
    """
    Détermine l'ordre des catégories (tuples) d'une colonne hidden_dim_i,
    trié par :
      1. le nombre de couches du tuple (ex: (512,256) a 2 couches,
         (512,256,128) en a 3)
      2. le produit des éléments du tuple, comme proxy du nombre de
         paramètres, en cas d'égalité (ou pour ordonner au sein d'un
         même nombre de couches)
    """
    uniques = pd.unique(values)
    tuples = [as_tuple(v) for v in uniques]
    ordered = sorted(
        zip(uniques, tuples),
        key=lambda pair: (tuple_n_layers(pair[1]), tuple_param_count(pair[1]))
    )
    return [orig for orig, _ in ordered]


def plot_hidden_dim_barplots(best_params_tab, cancer_name, name_suffix, omics_order, out_dir="figures"):
    """
    Barplots de la distribution des valeurs de chaque hidden_dim_i,
    avec les catégories ordonnées par (nb de layers, taille) plutôt que
    par un simple tri numérique.
    """
    best_params_df = pd.DataFrame(best_params_tab)
    hidden_dim_cols = [c for c in best_params_df.columns if c.startswith("hidden_dim")]

    if not hidden_dim_cols:
        return

    n_hd = len(hidden_dim_cols)
    ncols_hd = min(n_hd, 2)
    nrows_hd = math.ceil(n_hd / ncols_hd)

    fig_hd, axes_hd = plt.subplots(nrows_hd, ncols_hd, figsize=(5 * ncols_hd, 4 * nrows_hd))
    axes_hd = np.array(axes_hd).flatten()

    for ax, col in zip(axes_hd, hidden_dim_cols):
        raw_counts = best_params_df[col].value_counts()
        ordered_values = ordered_categories_by_layers_and_size(best_params_df[col])
        counts = raw_counts.reindex(ordered_values).fillna(0)

        labels = ["×".join(str(x) for x in as_tuple(v)) for v in counts.index]
        omics_name = hidden_dim_col_to_omics_name(col, omics_order)
        ax.bar(labels, counts.values, color="#2a78d6", width=0.5)
        ax.set_title(omics_name)
        ax.set_xlabel("Architecture")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=60)
        

    for ax in axes_hd[n_hd:]:
        fig_hd.delaxes(ax)

    fig_hd.suptitle("Hidden dim — distribution par catégorie", fontsize=14)
    fig_hd.tight_layout()

    out_path = f"{out_dir}/{name_suffix}{cancer_name}_hidden_dim_barplots.png"
    fig_hd.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig_hd)
    print(f"Sauvegardé : {out_path}")


# ======= Main =================================================================================

def main():
    args = parse_args()

    cancer_name = args.cancer
    n_outer = args.outer_splits
    name_suffix = args.name_suffix
    n_folds_to_plot = args.n_folds_to_plot
    do_importance = not args.skip_importance
    out_dir = args.out_dir
    omics_order = [s.strip() for s in args.omics_order.split(",") if s.strip()]
    print(f"\n{'='*60}")
    print(f"  Cancer            : {cancer_name}")
    print(f"  Outer folds       : {n_outer}")
    print(f"  Folds affichés    : {n_folds_to_plot}")
    print(f"  Importances       : {do_importance}")
    print(f"{'='*60}\n")

    studies, best_params_tab, params_importances, durations_importances = load_studies(
        cancer_name, n_outer, name_suffix, do_importance=do_importance
    )


    if do_importance:
        plot_funcs = get_default_plot_funcs()
        plot_folds_grid(studies, plot_funcs, n_folds_to_plot, cancer_name, name_suffix, out_dir=out_dir)

        plot_importances_boxplot(
            params_importances,
            value_name="Importance",
            title="Param importances across outer folds",
            filename="param_importances_boxplot",
            cancer_name=cancer_name,
            name_suffix=name_suffix,
            out_dir=out_dir,
        )
        plot_importances_boxplot(
            durations_importances,
            value_name="Duration Importance",
            title="Param duration importances across outer folds",
            filename="duration_importances_boxplot",
            cancer_name=cancer_name,
            name_suffix=name_suffix,
            out_dir=out_dir,
        )

    plot_best_params_boxplots(best_params_tab, cancer_name, name_suffix, out_dir=out_dir)
    plot_hidden_dim_barplots(best_params_tab, cancer_name, name_suffix, omics_order, out_dir=out_dir)


if __name__ == "__main__":
    main()