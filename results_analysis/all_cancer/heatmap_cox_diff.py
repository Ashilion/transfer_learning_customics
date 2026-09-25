"""
Heatmap comparant les 8 méthodes (ridge, clinridge, supridge, supclinridge,
chacune avec et sans transfer learning - TL) au modèle de Cox seul (baseline),
sur les cancers pour lesquels TOUTES les combinaisons sont disponibles.

Convention de nommage attendue dans le dossier de résultats :
  - Cox seul (baseline)  : outer_cv_results_vvh_{cox_suffix}{CANCER}.csv
  - méthode sans TL       : ncv_custcox_optuna_paral_{method}_{CANCER}.csv
  - méthode avec TL       : ncv_finetune_optuna_paral_{method}_{CANCER}.csv

Pour chaque (méthode, condition TL, cancer), on merge le CSV de la méthode
avec le CSV du Cox seul sur la colonne 'fold', puis on calcule
diff = valeur(méthode) - valeur(Cox seul), moyennée sur les folds.
Un heatmap est produit par métrique (C-index, IBS), avec les cancers en
ordonnée et les 8 méthodes (méthode x TL) en abscisse.

Exemple d'utilisation (CLI) :
    python heatmap_cox_diff.py \
        --results-dir /path/to/results \
        --output-dir /path/to/figures \
        --cox-suffix ""
"""

import argparse
import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_COLUMNS = ["cindex_default", "graf"]
METRIC_LABELS = {"cindex_default": "C-index", "graf": "IBS"}

# BASE_METHODS = ["ridge", "clinridge", "supridge", "supclinridge"]

# # Noms d'affichage des méthodes (utilisés uniquement pour les labels/plots,
# # la découverte des fichiers continue de se baser sur les clés de BASE_METHODS).
# METHOD_LABELS = {
#     "ridge": "CC",
#     "supridge": "CC sup",
#     "clinridge": "CC clin",
#     "supclinridge": "CC sup clin",
# }

BASE_METHODS = ["ridge", "clinridge"]

# Noms d'affichage des méthodes (utilisés uniquement pour les labels/plots,
# la découverte des fichiers continue de se baser sur les clés de BASE_METHODS).
METHOD_LABELS = {
    "ridge": "CustCox",
    "clinridge": "CustCox clin"
}


NO_TL_PREFIX = "ncv_custcox_optuna_paral_"
TL_PREFIX = "ncv_finetune_optuna_paral_"
COX_PREFIX = "outer_cv_results_vvh_"


def build_method_conditions(
    base_methods: List[str] = BASE_METHODS,
    method_labels: Dict[str, str] = METHOD_LABELS,
) -> List[Tuple[str, str, bool]]:
    """
    Retourne une liste de tuples (label_colonne, method, is_tl) définissant
    les 8 conditions comparées au Cox seul.

    Le label de colonne utilise le nom d'affichage (method_labels) et est
    écrit sur deux lignes ("<méthode>\n<sans/avec> TL") pour permettre des
    ticks horizontaux qui ne débordent pas.
    """
    conditions = []
    for method in base_methods:
        display = method_labels.get(method, method)
        conditions.append((f"{display}\nsans TL", method, False))
        conditions.append((f"{display}\navec TL", method, True))
    return conditions


def _method_path(results_dir: str, method: str, is_tl: bool, cancer: str) -> str:
    prefix = TL_PREFIX if is_tl else NO_TL_PREFIX
    return os.path.join(results_dir, f"{prefix}{method}_{cancer}.csv")


def _cox_path(results_dir: str, cox_suffix: str, cancer: str) -> str:
    return os.path.join(results_dir, f"{COX_PREFIX}{cox_suffix}{cancer}.csv")


# --------------------------------------------------------------------------- #
# Découverte des cancers disponibles pour Cox seul + toutes les méthodes
# --------------------------------------------------------------------------- #

def _cancers_for_prefix(results_dir: str, prefix: str) -> set:
    """Cancers disponibles pour un prefixe de fichier donné."""
    pattern = os.path.join(results_dir, f"{prefix}*.csv")
    files = glob.glob(pattern)
    return {
        os.path.basename(f).replace(prefix, "").replace(".csv", "")
        for f in files
    }


def discover_common_cancers(
    results_dir: str,
    cox_suffix: str = "",
    base_methods: List[str] = BASE_METHODS,
) -> List[str]:
    """
    Cancers pour lesquels le Cox seul ET toutes les méthodes (avec et sans TL)
    ont un fichier de résultats disponible.
    """
    common = _cancers_for_prefix(results_dir, f"{COX_PREFIX}{cox_suffix}")
    for method in base_methods:
        no_tl_cancers = _cancers_for_prefix(results_dir, f"{NO_TL_PREFIX}{method}_")
        tl_cancers = _cancers_for_prefix(results_dir, f"{TL_PREFIX}{method}_")
        common = common & no_tl_cancers & tl_cancers
    return sorted(common)


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
# Calcul des diffs méthode - Cox seul
# --------------------------------------------------------------------------- #

def load_metric_diff(
    results_dir: str, method: str, is_tl: bool, cancer: str, cox_suffix: str
) -> Dict[str, float]:
    """Charge le CSV de la méthode (avec/sans TL) et le CSV Cox seul, merge sur
    'fold' et retourne la diff moyenne (méthode - Cox seul) pour chaque métrique.
    """
    method_path = _method_path(results_dir, method, is_tl, cancer)
    cox_path = _cox_path(results_dir, cox_suffix, cancer)

    df_method = pd.read_csv(method_path)[["fold"] + METRIC_COLUMNS]
    df_cox = pd.read_csv(cox_path)[["fold"] + METRIC_COLUMNS]

    merged = pd.merge(df_method, df_cox, on="fold", suffixes=("_method", "_cox"))

    diffs = {}
    for col in METRIC_COLUMNS:
        diffs[col] = (merged[f"{col}_method"] - merged[f"{col}_cox"]).mean()
    return diffs


def build_diff_table(
    results_dir: str,
    cancers: List[str],
    cox_suffix: str,
    base_methods: List[str] = BASE_METHODS,
    method_labels: Dict[str, str] = METHOD_LABELS,
) -> Dict[str, pd.DataFrame]:
    """
    Construit, pour chaque métrique, un dataframe (cancers en index, les 8
    méthodes en colonnes) contenant la diff moyenne méthode - Cox seul.
    Retourne {metric_label: DataFrame}.
    """
    conditions = build_method_conditions(base_methods, method_labels)
    columns = [label for label, _, _ in conditions]

    tables = {
        label: pd.DataFrame(index=cancers, columns=columns, dtype=float)
        for label in METRIC_LABELS.values()
    }

    for col_label, method, is_tl in conditions:
        for cancer in cancers:
            try:
                diffs = load_metric_diff(results_dir, method, is_tl, cancer, cox_suffix)
            except FileNotFoundError as e:
                print(f"  [warn] fichier manquant pour {col_label!r} cancer={cancer}: {e}")
                continue
            for metric_col, metric_label in METRIC_LABELS.items():
                tables[metric_label].loc[cancer, col_label] = diffs[metric_col]

    return tables


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_heatmaps(tables: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Un heatmap par métrique : cancers en ordonnée, 8 méthodes en abscisse.

    Les labels de colonnes sont sur deux lignes (méthode / avec-sans TL) et
    affichés horizontalement (rotation=0) pour rester lisibles sans être
    penchés ni tronqués.
    """
    os.makedirs(output_dir, exist_ok=True)
    for label, table in tables.items():
        if table.empty:
            continue
        vmax = table.abs().max().max()
        vmax = vmax if pd.notna(vmax) and vmax > 0 else 1.0

        # Largeur par colonne un peu plus généreuse car les labels sont
        # horizontaux (donc plus larges que penchés à 45°).
        fig, ax = plt.subplots(figsize=(1.3 * len(table.columns) + 2, 0.4 * len(table.index) + 2))
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
            cbar_kws={"label": f"Diff {label} (méthode - Cox seul)"},
            ax=ax,
        )
        ax.set_title(f"Effet des méthodes vs Cox seul - {label}")
        ax.set_xlabel("Méthode")
        ax.set_ylabel("Cancer")
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        plt.tight_layout()
        fname = f"heatmap_cox_diff_{label.replace('-', '_').lower()}.png"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.show()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def compare_cox_heatmap(
    results_dir: str,
    output_dir: str,
    base_methods: List[str] = BASE_METHODS,
    cancers: Optional[List[str]] = None,
    clinical_dir: Optional[str] = None,
    cox_suffix: str = "",
    method_labels: Dict[str, str] = METHOD_LABELS,
) -> Dict[str, pd.DataFrame]:
    if cancers is None:
        cancers = discover_common_cancers(results_dir, cox_suffix, base_methods)

    if not cancers:
        print("Aucun cancer avec Cox seul + les 8 méthodes disponibles.")
        return {}

    if clinical_dir:
        sizes = get_cancer_sizes(cancers, clinical_dir)
        cancers = sorted(sizes, key=sizes.get, reverse=True)

    print(f"Cancers retenus (Cox seul + 8 méthodes présentes): {cancers}")

    tables = build_diff_table(results_dir, cancers, cox_suffix, base_methods, method_labels)

    for label, table in tables.items():
        print(f"\n--- {label} ---")
        print(table)
        table.to_csv(os.path.join(output_dir, f"heatmap_cox_diff_{label.replace('-', '_').lower()}.csv"))

    plot_heatmaps(tables, output_dir)
    return tables


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Heatmap de la différence (méthode - Cox seul) pour les 8 méthodes "
            "(4 méthodes x avec/sans TL), pour les cancers où Cox seul et toutes "
            "les méthodes sont disponibles."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Dossier contenant les CSV ncv_custcox_...csv, ncv_finetune_...csv et outer_cv_results_vvh_...csv",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Dossier où enregistrer les heatmaps et les tableaux CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=BASE_METHODS,
        help=f"Liste des méthodes de base à comparer (default: {BASE_METHODS})",
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
    parser.add_argument(
        "--cox-suffix",
        default="",
        help=(
            "Suffixe inséré entre 'outer_cv_results_vvh_' et le nom du cancer "
            "dans le nom du fichier Cox seul, ex: outer_cv_results_vvh_{cox-suffix}{cancer}.csv "
            "(default: vide)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_cox_heatmap(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        base_methods=args.methods,
        cancers=args.cancers,
        clinical_dir=args.clinical_dir,
        cox_suffix=args.cox_suffix,
    )