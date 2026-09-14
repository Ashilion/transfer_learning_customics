"""
Analyse et visualisation des résultats Optuna (nested CV, CustOMICS + CoxNet)
pour UN SEUL cancer (ex : KIRP), à partir du cache CSV déjà produit par le
script d'analyse multi-cancers (cache_importances.csv, cache_best_params.csv,
cache_hidden_dims.csv dans --cache_dir).

Ce script ne relit JAMAIS les journaux Optuna : il repart uniquement du
cache CSV. S'il est absent, le script s'arrête avec un message d'erreur.

Contrairement au script multi-cancers, ici il n'y a plus besoin de facetter
par "Cancer" (un seul cancer), donc les figures sont réorganisées en
orientation horizontale (boxplots horizontaux, facettes par méthode
côte à côte) pour un rendu plus compact et lisible.

Figures produites (dans --out_dir) :
  - best_params_{cancer}_sans_tl.png / _avec_tl.png :
      pour chaque hyperparamètre, un boxplot horizontal (Méthode en
      ordonnée, valeur en abscisse).
  - param_importances_{cancer}_sans_tl.png / _avec_tl.png :
      importances des hyperparamètres, boxplots horizontaux
      (Paramètre en ordonnée, Importance en abscisse), une colonne par méthode.
  - hidden_dim_heatmap_{cancer}_{sans_tl|avec_tl}_{omic}.png :
      heatmap horizontale (Méthode en ordonnée, Architecture en abscisse).
"""

import argparse
import ast
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ======= Constantes (miroir du script multi-cancers) =========================================

METHODS = ["ridge", "clinridge", "supridge", "supclinridge"]

METHOD_LABELS = {
    "ridge": "CustCox",
    "supridge": "CustCox sup",
    "clinridge": "CustCox clin",
    "supclinridge": "CustCox sup clin",
}

EXCLUDED_ARCHITECTURES = {
    "mutation": {(512, 256, 128)},
}

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
ANNOT_FONTSIZE = 14
LEGEND_FONTSIZE = 16

IMPORTANCES_CSV_NAME = "cache_importances.csv"
BEST_PARAMS_CSV_NAME = "cache_best_params.csv"
HIDDEN_DIMS_CSV_NAME = "cache_hidden_dims.csv"


# ======= Argument Parsing ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Optuna pour un seul cancer, figures horizontales, depuis le cache CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cancer", type=str, default="KIRP", help="Nom du cancer à analyser.")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Méthodes à comparer.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="figures",
        help="Dossier contenant les CSV de cache produits par le script multi-cancers "
             "(cache_importances.csv, cache_best_params.csv, cache_hidden_dims.csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Dossier de sortie des figures. Par défaut : <cache_dir>/<cancer>.",
    )
    parser.add_argument(
        "--omics_order",
        type=str,
        default="rna,mirna,cnv,mutation",
        help="Ordre des omics, utilisé pour exclure les colonnes omics des hyperparamètres.",
    )
    parser.add_argument("--skip_importance", action="store_true", default=False,
                         help="Ne pas tracer les importances de paramètres.")
    return parser.parse_args()


# ======= Helpers ==============================================================================

def ordered_categories_by_layers_and_size(architectures):
    """Trie les architectures (tuples) par nb de couches puis par taille totale."""
    uniq = sorted(set(architectures), key=lambda a: (len(a), sum(a), a))
    return uniq


def as_tuple(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value),)


def load_cache(cache_dir, cancer):
    importances_csv = os.path.join(cache_dir, IMPORTANCES_CSV_NAME)
    best_params_csv = os.path.join(cache_dir, BEST_PARAMS_CSV_NAME)
    hidden_csv = os.path.join(cache_dir, HIDDEN_DIMS_CSV_NAME)

    missing = [p for p in (importances_csv, best_params_csv, hidden_csv) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Cache CSV introuvable : " + ", ".join(missing) +
            "\nCe script nécessite le cache produit par le script multi-cancers "
            "(exécuté au moins une fois sans --use_cache pour le générer)."
        )

    importances_df = pd.read_csv(importances_csv)
    best_params_df = pd.read_csv(best_params_csv)
    hidden_df = pd.read_csv(hidden_csv)
    if not hidden_df.empty and "ArchitectureTuple" in hidden_df.columns:
        hidden_df["ArchitectureTuple"] = hidden_df["ArchitectureTuple"].apply(ast.literal_eval)

    for df, name in ((importances_df, "importances"), (best_params_df, "best_params"), (hidden_df, "hidden_dims")):
        if "Cancer" not in df.columns:
            raise ValueError(f"Colonne 'Cancer' absente du cache {name}.")

    importances_df = importances_df[importances_df["Cancer"] == cancer].copy()
    best_params_df = best_params_df[best_params_df["Cancer"] == cancer].copy()
    hidden_df = hidden_df[hidden_df["Cancer"] == cancer].copy()

    if best_params_df.empty:
        raise ValueError(f"Aucune donnée en cache pour le cancer '{cancer}'.")

    return importances_df, best_params_df, hidden_df


# ======= Plot : meilleurs hyperparamètres, boxplots verticaux alignés horizontalement ========

def plot_best_params_horizontal(best_params_df, tl_label, methods, omics_order, out_dir, cancer,
                                 max_cols_per_row=3):
    cols = [
        c for c in best_params_df.columns
        if c not in omics_order and c not in ("Cancer", "Fold", "Method", "TL")
    ]
    usable_cols = []
    for c in cols:
        if best_params_df[c].notna().sum() == 0:
            print(f"  [warn] colonne '{c}' entièrement vide pour {tl_label}, ignorée")
            continue
        usable_cols.append(c)
    cols = usable_cols

    if not cols:
        print(f"  [warn] aucun hyperparamètre à tracer pour {tl_label}")
        return

    # Boxplots verticaux classiques (Méthode en abscisse, valeur en ordonnée),
    # disposés côte à côte : une ligne de subplots si peu de paramètres,
    # sinon on enchaîne les lignes en gardant max_cols_per_row colonnes.
    n_cols_grid = min(max_cols_per_row, len(cols))
    n_rows_grid = math.ceil(len(cols) / n_cols_grid)

    fig, axes = plt.subplots(
        n_rows_grid, n_cols_grid, figsize=(4.2 * n_cols_grid, 4.5 * n_rows_grid), squeeze=False
    )
    axes = axes.flatten()

    present_methods = [m for m in methods if m in best_params_df["Method"].unique()]

    for ax, col in zip(axes, cols):
        data = best_params_df.dropna(subset=[col])
        sns.boxplot(
            data=data,
            x="Method",
            y=col,
            order=present_methods,
            width=0.6,
            ax=ax,
        )
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in present_methods], rotation=30, ha="right")
        ax.set_title(col, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        if col in ("lr", "beta", "delta_min"):
            ax.set_yscale("log")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Meilleurs hyperparamètres — {cancer} — {tl_label}", fontsize=TITLE_FONTSIZE + 2, y=1.02)
    fig.tight_layout()

    suffix = "avec_tl" if tl_label == "Avec TL" else "sans_tl"
    out_path = os.path.join(out_dir, f"best_params_{cancer}_{suffix}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Plot : importances des hyperparamètres, boxplots verticaux, colonnes = méthodes ====

def plot_param_importances_horizontal(importances_df, tl_label, methods, out_dir, cancer):
    subset = importances_df[importances_df["TL"] == tl_label]
    if subset.empty:
        print(f"  [warn] pas de données d'importance pour {tl_label}")
        return

    present_methods = [m for m in methods if m in subset["Method"].unique()]

    g = sns.catplot(
        data=subset,
        x="Parameter", y="Importance",
        col="Method",
        col_order=present_methods,
        kind="box",
        sharex=True,
        sharey=True,
        height=4.5,
        aspect=0.8,
    )
    for ax, method in zip(g.axes.flat, present_methods):
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=60)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    g.set_axis_labels("", "Importance", fontsize=LABEL_FONTSIZE)

    g.figure.suptitle(f"Importances des hyperparamètres — {cancer} — {tl_label}", fontsize=TITLE_FONTSIZE, y=1.03)
    g.figure.subplots_adjust(top=0.85, wspace=0.1)

    suffix = "avec_tl" if tl_label == "Avec TL" else "sans_tl"
    out_path = os.path.join(out_dir, f"param_importances_{cancer}_{suffix}.png")
    g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Sauvegardé : {out_path}")


# ======= Plot : heatmap horizontale des architectures (hidden_dim), méthodes en ordonnée ====

def plot_hidden_dim_heatmap_horizontal(hidden_df, omic, tl_label, methods, out_dir, cancer):
    sub = hidden_df[(hidden_df["TL"] == tl_label) & (hidden_df["Omics"] == omic)]
    if sub.empty:
        return

    heat = sub.groupby(["Method", "Architecture"]).size().reset_index(name="Count")
    present_methods = [m for m in methods if m in heat["Method"].unique()]

    mat = heat.pivot(index="Method", columns="Architecture", values="Count").fillna(0)
    mat = mat.reindex(index=present_methods)

    arch_order_tuple = ordered_categories_by_layers_and_size(
        sub["ArchitectureTuple"]
    )
    arch_order = ["x".join(map(str, a)) for a in arch_order_tuple]
    mat = mat.reindex(columns=arch_order, fill_value=0)
    mat.index = [METHOD_LABELS.get(m, m) for m in mat.index]

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * mat.shape[1] + 4), max(3, 0.8 * mat.shape[0] + 1.5)))
    sns.heatmap(
        mat, cmap="Blues", annot=True, fmt=".0f", ax=ax,
        annot_kws={"fontsize": ANNOT_FONTSIZE},
        cbar_kws={"label": "Count"},
    )
    ax.set_title(f"{omic} — {cancer} — {tl_label}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Architecture", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Méthode", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=45)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, rotation=0)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label("Count", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()

    suffix = "avec_tl" if tl_label == "Avec TL" else "sans_tl"
    out_path = os.path.join(out_dir, f"hidden_dim_heatmap_{cancer}_{suffix}_{omic}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Main =================================================================================

def main():
    args = parse_args()

    cancer = args.cancer
    methods = args.methods
    omics_order = [s.strip() for s in args.omics_order.split(",") if s.strip()]
    out_dir = args.out_dir or os.path.join(args.cache_dir, cancer)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Cancer            : {cancer}")
    print(f"  Méthodes          : {methods}")
    print(f"  Cache dir         : {args.cache_dir}")
    print(f"  Out dir           : {out_dir}")
    print(f"{'='*60}\n")

    importances_df, best_params_df, hidden_df = load_cache(args.cache_dir, cancer)

    # ---- 1) Meilleurs hyperparamètres, boxplots horizontaux, un graphique par condition TL
    for tl_label in ("Sans TL", "Avec TL"):
        subset = best_params_df[best_params_df["TL"] == tl_label]
        if subset.empty:
            print(f"  [warn] pas de données pour {tl_label}, best_params ignoré")
            continue
        plot_best_params_horizontal(subset, tl_label, methods, omics_order, out_dir, cancer)

    # ---- 2) Importances des hyperparamètres, horizontal, une colonne par méthode
    if not args.skip_importance and not importances_df.empty:
        for tl_label in ("Sans TL", "Avec TL"):
            plot_param_importances_horizontal(importances_df, tl_label, methods, out_dir, cancer)

    # ---- 3) Heatmaps horizontales des architectures (hidden_dim), par condition TL / omic
    if not hidden_df.empty:
        for tl_label in ("Sans TL", "Avec TL"):
            sub_tl = hidden_df[hidden_df["TL"] == tl_label]
            if sub_tl.empty:
                continue
            for omic in sub_tl["Omics"].unique():
                plot_hidden_dim_heatmap_horizontal(hidden_df, omic, tl_label, methods, out_dir, cancer)


if __name__ == "__main__":
    main()