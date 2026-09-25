"""
Heatmap comparant l'effet du transfer learning (TL) pour plusieurs méthodes
(ridge, clinridge, supridge, supclinridge), sur les cancers pour lesquels
TOUTES les combinaisons méthode x {avec TL, sans TL} sont disponibles.

Convention de nommage attendue dans le dossier de résultats :
  - sans TL : ncv_custcox_optuna_paral_{method}_{CANCER}.csv
  - avec TL : ncv_finetune_optuna_paral_{method}_{CANCER}.csv

Pour chaque (méthode, cancer), on merge les deux CSV sur la colonne 'fold',
puis on calcule diff = valeur(avec TL) - valeur(sans TL), moyennée sur les
folds. Un heatmap est produit par métrique (C-index, IBS), avec les cancers
en ordonnée et les méthodes en abscisse.

Exemple d'utilisation (CLI) :
    python heatmap_tl_diff.py \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

import argparse
import glob
import os
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_COLUMNS = ["cindex_default", "graf"]
METRIC_LABELS = {"cindex_default": "C-index", "graf": "IBS"}

METHODS = ["ridge", "clinridge", "supridge", "supclinridge"]

# Noms d'affichage des méthodes (utilisés uniquement pour les labels/plots,
# la découverte des fichiers continue de se baser sur les clés de METHODS).
METHOD_LABELS = {
    "ridge": "CustCox",
    "supridge": "CustCox\nsup",
    "clinridge": "CustCox\nclin",
    "supclinridge": "CustCox\nsup clin",
}

NO_TL_PREFIX = "ncv_custcox_optuna_paral_"
TL_PREFIX = "ncv_finetune_optuna_paral_"


# --------------------------------------------------------------------------- #
# Découverte des cancers disponibles pour toutes les méthodes / conditions TL
# --------------------------------------------------------------------------- #

def _cancers_for_prefix(results_dir: str, prefix: str, method: str) -> set:
    """Cancers disponibles pour un prefixe (no-TL ou TL) et une méthode donnée."""
    full_prefix = f"{prefix}{method}_"
    pattern = os.path.join(results_dir, f"{full_prefix}*.csv")
    files = glob.glob(pattern)
    return {
        os.path.basename(f).replace(full_prefix, "").replace(".csv", "")
        for f in files
    }


def discover_common_cancers(results_dir: str, methods: List[str] = METHODS) -> List[str]:
    """
    Cancers pour lesquels TOUTES les méthodes ont à la fois un fichier
    'sans TL' et un fichier 'avec TL'. Intersection sur méthodes x conditions.
    """
    common: Optional[set] = None
    for method in methods:
        no_tl_cancers = _cancers_for_prefix(results_dir, NO_TL_PREFIX, method)
        tl_cancers = _cancers_for_prefix(results_dir, TL_PREFIX, method)
        method_cancers = no_tl_cancers & tl_cancers
        common = method_cancers if common is None else common & method_cancers
    return sorted(common) if common else []


def get_cancer_sizes(cancer_names: List[str], clinical_dir: str) -> Dict[str, int]:
    """Nombre d'individus par cancer, lu depuis les pickles cliniques (tri optionnel)."""
    sizes = {}
    for cancer_name in cancer_names:
        path = os.path.join(clinical_dir, f"{cancer_name}_clinical.pickle")
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            sizes[cancer_name] = len(df)
        except FileNotFoundError:
            sizes[cancer_name] = 0
    return sizes


# --------------------------------------------------------------------------- #
# Calcul des diffs avec TL - sans TL
# --------------------------------------------------------------------------- #

def load_metric_diff(results_dir: str, method: str, cancer: str) -> Dict[str, float]:
    """Charge les 2 CSV (sans TL / avec TL) pour un (méthode, cancer), merge sur
    'fold' et retourne la diff moyenne (avec TL - sans TL) pour chaque métrique.
    """
    no_tl_path = os.path.join(results_dir, f"{NO_TL_PREFIX}{method}_{cancer}.csv")
    tl_path = os.path.join(results_dir, f"{TL_PREFIX}{method}_{cancer}.csv")

    df_no_tl = pd.read_csv(no_tl_path)[["fold"] + METRIC_COLUMNS]
    df_tl = pd.read_csv(tl_path)[["fold"] + METRIC_COLUMNS]

    merged = pd.merge(df_no_tl, df_tl, on="fold", suffixes=("_no_tl", "_tl"))

    diffs = {}
    for col in METRIC_COLUMNS:
        diffs[col] = (merged[f"{col}_tl"] - merged[f"{col}_no_tl"]).mean()
    return diffs


def build_diff_table(
    results_dir: str, cancers: List[str], methods: List[str] = METHODS
) -> Dict[str, pd.DataFrame]:
    """
    Construit, pour chaque métrique, un dataframe (cancers en index, méthodes
    en colonnes) contenant la diff moyenne avec TL - sans TL.
    Retourne {metric_label: DataFrame}.
    """
    tables = {label: pd.DataFrame(index=cancers, columns=methods, dtype=float)
              for label in METRIC_LABELS.values()}

    for method in methods:
        for cancer in cancers:
            try:
                diffs = load_metric_diff(results_dir, method, cancer)
            except FileNotFoundError as e:
                print(f"  [warn] fichier manquant pour method={method} cancer={cancer}: {e}")
                continue
            for col, label in METRIC_LABELS.items():
                tables[label].loc[cancer, method] = diffs[col]

    return tables


def rename_method_columns(
    tables: Dict[str, pd.DataFrame], method_labels: Dict[str, str] = METHOD_LABELS
) -> Dict[str, pd.DataFrame]:
    """Renomme les colonnes (méthodes) des tables pour l'affichage, en gardant
    les clés internes intactes pour la découverte des fichiers en amont.
    Les méthodes absentes de `method_labels` gardent leur nom d'origine.
    """
    return {
        label: table.rename(columns=lambda m: method_labels.get(m, m))
        for label, table in tables.items()
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

# def plot_heatmaps(tables: Dict[str, pd.DataFrame], output_dir: str) -> None:
#     """Un heatmap par métrique : cancers en ordonnée, méthodes en abscisse."""
#     os.makedirs(output_dir, exist_ok=True)
#     for label, table in tables.items():
#         if table.empty:
#             continue
#         vmax = table.abs().max().max()
#         vmax = vmax if pd.notna(vmax) and vmax > 0 else 1.0

#         fig, ax = plt.subplots(figsize=(0.9 * len(table.columns) + 2, 0.4 * len(table.index) + 2))
#         sns.heatmap(
#             table.astype(float),
#             annot=True,
#             fmt=".3f",
#             cmap="RdBu_r" if label!="IBS" else "RdBu",
#             center=0,
#             vmin=-vmax,
#             vmax=vmax,
#             linewidths=0.5,
#             linecolor="white",
#             cbar_kws={"label": f"Diff {label} (avec TL - sans TL)"},
#             ax=ax,
#         )
#         ax.set_title(f"Effet du transfer learning - {label}")
#         ax.set_xlabel("Méthode")
#         ax.set_ylabel("Cancer")

#         # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#         plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
#         plt.setp(ax.get_yticklabels(), rotation=0)
        
#         plt.tight_layout()
#         fname = f"heatmap_tl_diff_{label.replace('-', '_').lower()}.png"
#         fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
#         plt.show()

def plot_heatmaps(tables: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Deux heatmaps (C-index et IBS) côte à côte avec un axe Y partagé."""
    
    os.makedirs(output_dir, exist_ok=True)

    labels = list(tables.keys())
    n_plots = len(labels)

    # Même échelle Y, une colonne par métrique
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(0.9 * len(next(iter(tables.values())).columns) * n_plots + 2,
                 0.4 * len(next(iter(tables.values())).index) + 2),
        sharey=True,
    )

    # Si une seule métrique
    if n_plots == 1:
        axes = [axes]

    for i, (ax, (label, table)) in enumerate(zip(axes, tables.items())):

        vmax = table.abs().max().max()
        vmax = vmax if pd.notna(vmax) and vmax > 0 else 1.0

        sns.heatmap(
            table.astype(float),
            annot=True,
            fmt=".3f",
            cmap="RdBu_r" if label != "IBS" else "RdBu",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": f"Diff {label} (avec TL - sans TL)"},
            ax=ax,
        )

        ax.set_title(label)
        ax.set_xlabel("Méthode")

        # Labels X horizontaux sur 2 lignes si nécessaire
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Afficher les cancers seulement sur le premier graphique
        if i == 0:
            ax.set_ylabel("Cancer")
            plt.setp(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle("Effet du transfer learning", fontsize=14)

    plt.tight_layout()

    fname = "heatmap_tl_diff_combined.png"
    fig.savefig(
        os.path.join(output_dir, fname),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def compare_tl_heatmap(
    results_dir: str,
    output_dir: str,
    methods: List[str] = METHODS,
    cancers: Optional[List[str]] = None,
    clinical_dir: Optional[str] = None,
    method_labels: Dict[str, str] = METHOD_LABELS,
) -> Dict[str, pd.DataFrame]:
    if cancers is None:
        cancers = discover_common_cancers(results_dir, methods)

    if not cancers:
        print("Aucun cancer avec toutes les méthodes/conditions TL disponibles.")
        return {}

    if clinical_dir:
        sizes = get_cancer_sizes(cancers, clinical_dir)
        cancers = sorted(sizes, key=sizes.get, reverse=True)

    print(f"Cancers retenus (toutes méthodes x TL présentes): {cancers}")

    tables = build_diff_table(results_dir, cancers, methods)
    tables = rename_method_columns(tables, method_labels)

    for label, table in tables.items():
        print(f"\n--- {label} ---")
        print(table)
        table.to_csv(os.path.join(output_dir, f"heatmap_tl_diff_{label.replace('-', '_').lower()}.csv"))

    plot_heatmaps(tables, output_dir)
    return tables


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Heatmap de la différence (avec TL - sans TL) par méthode et par "
            "cancer, pour les cancers où toutes les méthodes sont disponibles."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Dossier contenant les CSV ncv_custcox_...csv et ncv_finetune_...csv",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Dossier où enregistrer les heatmaps et les tableaux CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help=f"Liste des méthodes à comparer (default: {METHODS})",
    )
    parser.add_argument(
        "--cancers",
        nargs="+",
        default=None,
        help="Liste explicite de cancers. Si absent, découverte automatique (intersection).",
    )
    parser.add_argument(
        "--clinical-dir",
        default=None,
        help="Dossier contenant les {cancer}_clinical.pickle, pour trier les cancers par taille (optionnel).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_tl_heatmap(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        methods=args.methods,
        cancers=args.cancers,
        clinical_dir=args.clinical_dir,
    )