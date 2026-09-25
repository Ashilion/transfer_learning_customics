"""
Analyse et visualisation des résultats Optuna (nested CV, CustOMICS + CoxNet),
agrégées sur tous les cancers.

Ce script charge les studies Optuna sauvegardées pour chaque fold externe et
pour chaque cancer, puis génère plusieurs figures agrégées :
  - boxplots des importances de paramètres, par cancer
  - boxplots des meilleurs hyperparamètres, par cancer
  - heatmaps / barplots de la distribution des architectures (hidden_dim_i),
    par omic et par cancer.

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
import re
import glob
import os

from analyse_optuna_figs import ordered_categories_by_layers_and_size
sys.path.append("..")
# ======= Argument Parsing ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested CV with CustOMICS + CoxNet + Optuna for survival prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--outer_splits",
        type=int,
        default=50,
        help="Number of outer CV folds."
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


# ======= Study loading =======================================================================

def load_studies(cancer_name, n_outer, name_suffix, omics_order, do_importance=True):
    """
    Charge les studies Optuna de chaque fold externe et récupère :
      - la liste des studies
      - les best_params (+ best_alpha) de chaque fold, avec les hidden_dim_i
        déjà renommés en noms d'omics
      - les importances de paramètres / durée (si do_importance=True),
        également renommées
    """
    studies = []
    params_importances = []
    durations_importances = []
    best_params_tab = []

    for outer_fold in range(n_outer):
        study = optuna.load_study(
            # study_name=f"journal_storage_multiprocess_{name_suffix}fold{outer_fold}",
            # study_name=f"journal_storage_multiprocess_{name_suffix}{cancer_name}_fold{outer_fold}",
            study_name=f"ft_{name_suffix}{cancer_name}_fold{outer_fold}",

            storage=JournalStorage(
                JournalFileBackend(
                    # file_path=f"./optuna_journal/journal_{name_suffix}fold{outer_fold}.log"
                    file_path=f"../results_tgcc/optuna_journal/journal_{name_suffix}ft_{cancer_name}_fold{outer_fold}.log"
                )
            ),
        )
        studies.append(study)

        if do_importance:
            params_importances.append(
                rename_hidden_dims_to_omics(get_param_importances(study), omics_order)
            )
            durations_importances.append(
                rename_hidden_dims_to_omics(
                    get_param_importances(study, target=lambda t: t.duration.total_seconds()),
                    omics_order,
                )
            )

        best_params = study.best_trial.params
        best_params["best_alpha"] = study.best_trial.user_attrs["best_alpha"]
        best_params = rename_hidden_dims_to_omics(best_params, omics_order)
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

def plot_best_params_boxplots(best_params_tab, cancer_name, name_suffix, omics_order, out_dir="figures"):
    """Boxplots (un par hyperparamètre, hors omics/hidden_dim) des meilleurs paramètres par fold externe."""
    all_params = pd.DataFrame(best_params_tab).fillna(0)
    hidden_dim_cols = [c for c in all_params.columns if c in omics_order]
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


# ======= Hidden dim (omics) ordering & barplots ==============================================

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


def plot_hidden_dim_barplots(best_params_tab, cancer_name, name_suffix, omics_order, out_dir="figures"):
    """
    Barplots de la distribution des valeurs de chaque colonne d'omic
    (rna, mirna, cnv, mutation, ...), avec les catégories ordonnées par
    (nb de layers, taille) plutôt que par un simple tri numérique.
    """
    best_params_df = pd.DataFrame(best_params_tab)
    hidden_dim_cols = [c for c in omics_order if c in best_params_df.columns]

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

        labels = ["x".join(str(x) for x in as_tuple(v)) for v in counts.index]
        ax.bar(labels, counts.values, color="#2a78d6", width=0.5)
        ax.set_title(col)
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

    n_outer = args.outer_splits
    name_suffix = args.name_suffix
    do_importance = not args.skip_importance
    out_dir = args.out_dir
    omics_order = [s.strip() for s in args.omics_order.split(",") if s.strip()]
    print(f"\n{'='*60}")
    print(f"  Outer folds       : {n_outer}")
    print(f"  Importances       : {do_importance}")
    print(f"  Omics order       : {omics_order}")
    print(f"{'='*60}\n")

    pattern = re.compile(
        rf"journal_{re.escape(name_suffix)}([A-Za-z0-9]+)_fold\d+\.log"
    )

    journal_dir = "../results_tgcc/optuna_journal"

    cancers = sorted({
        pattern.match(os.path.basename(f)).group(1)
        for f in glob.glob(os.path.join(journal_dir, "*.log"))
        if pattern.match(os.path.basename(f))
    })

    print(cancers)

    all_importances = []
    all_best_params = []
    all_hidden_dims = []

    for cancer in cancers:
        print(cancer)

        studies, best_params_tab, params_importances, durations_importances = load_studies(
            cancer,
            n_outer,
            name_suffix,
            omics_order,
            do_importance=True
        )

        name_suffix = f"ft_{name_suffix}"
        for fold, imp in enumerate(params_importances):

            for param, value in imp.items():

                all_importances.append({
                    "Cancer": cancer,
                    "Fold": fold,
                    "Parameter": param,
                    "Importance": value
                })

        for fold, params in enumerate(best_params_tab):

            row = params.copy()
            row["Cancer"] = cancer
            row["Fold"] = fold

            all_best_params.append(row)

            for col in params:

                if col in omics_order:

                    architecture = as_tuple(params[col])

                    all_hidden_dims.append({
                        "Cancer": cancer,
                        "Omics": col,
                        "ArchitectureTuple": architecture,
                        "Architecture": "x".join(map(str, architecture))
                    })

    importances_df = pd.DataFrame(all_importances)
    best_params_df = pd.DataFrame(all_best_params)
    hidden_df = pd.DataFrame(all_hidden_dims)

    #importance
    g = sns.catplot(
        data=importances_df,
        x="Parameter",
        y="Importance",
        col="Cancer",
        col_wrap=6,
        kind="box",
        sharey=True,
        height=3
    )

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=60)

    g.figure.suptitle(
        "Hyperparameter importances across the 18 cancers",
        fontsize=16,
        y=1.02
    )

    g.tight_layout()

    out_path = f"{out_dir}/{name_suffix}allcancer_param_importances_boxplot.png"
    g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)

    #meilleurs hyperparamètres
    cols = [
        c for c in best_params_df.columns
        if c not in omics_order
        and c not in ["Cancer","Fold"]
    ]

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=(14,4*len(cols))
    )

    for ax, col in zip(axes, cols):

        sns.boxplot(
            data=best_params_df,
            x="Cancer",
            y=col,
            ax=ax
        )

        ax.set_title(col)
        ax.tick_params(axis='x', rotation=90)

        if col in ["lr","beta","delta_min"]:
            ax.set_yscale("log")

    fig.tight_layout()
    out_path = f"{out_dir}/{name_suffix}allcancer_best_params.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    #heat
    heat = (
        hidden_df
        .groupby(["Cancer","Omics","Architecture"])
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
            tmp["ArchitectureTuple"]
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
            ax=ax
        )

        ax.set_title(omic)

        fig.tight_layout()

        out_path = f"{out_dir}/{name_suffix}hidden_dim_heatmap_{omic}.png"

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    architecture_order_tuple = ordered_categories_by_layers_and_size(
        tmp["ArchitectureTuple"]
    )

    architecture_order = [
        "x".join(map(str, arch))
        for arch in architecture_order_tuple
    ]
    for omic in hidden_df["Omics"].unique():

        tmp = heat[heat["Omics"] == omic].copy()

        tmp["Architecture"] = pd.Categorical(
            tmp["Architecture"],
            categories=architecture_order,
            ordered=True
        )

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.barplot(
            data=tmp,
            x="Cancer",
            y="Count",
            hue="Architecture",
            hue_order=architecture_order,
            ax=ax
        )

        ax.set_title(omic)
        ax.set_xlabel("Cancer")
        ax.set_ylabel("Count")

        ax.legend(
            title="Architecture",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        fig.tight_layout()

        out_path = (
            f"{out_dir}/{name_suffix}"
            f"hidden_dim_barplot_{omic}.png"
        )

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
if __name__ == "__main__":
    main()