"""
compare_multiple_1_cancer.py

Compare plusieurs fichiers de résultats de survie (C-index, IBS) pour COAD :
diffs par rapport à un fichier de référence, courbes par fold ou boxplots.

Usage:
    python results_analysis/compare_multiple_1_cancer.py \
    --files results/outer_cv_results_vvh_ridge_KIRP.csv  \
            results/ncv_custcox_optuna_paral_ridge_KIRP.csv \
            results/ncv_custcox_optuna_paral_clinridge_KIRP.csv \
            results/ncv_custcox_optuna_paral_supridge_KIRP.csv \
            results/ncv_custcox_optuna_paral_supclinridge_KIRP.csv \
            results/ncv_finetune_optuna_paral_ridge_KIRP.csv \
            results/ncv_finetune_optuna_paral_clinridge_KIRP.csv \
            results/ncv_finetune_optuna_paral_supridge_KIRP.csv \
            results/ncv_finetune_optuna_paral_supclinridge_KIRP.csv \
    --labels "Cox" "CustCox" "CustCox clin" "CustCox sup" "CustCox clin sup" \
            "TL CustCox" "TL CustCox clin" "TL CustCox sup" "TL CustCox clin sup" \
    --graph "KIRP_allridge" --boxplot 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from functools import reduce


def load_file(path, type_):
    """
    Charge un fichier de résultats et retourne un DataFrame normalisé
    avec les colonnes: fold, cindex_default, graf.
    """
    df = pd.read_csv(path)

    if type_ == "ref":
        df = df[df["learner_id"] == "cox_ref"]
        df = df[df["task_id"] == "COAD"]
        df["fold"] = df["iteration"] - 1  # indice en R commence a 1
    else:
        # fichier "new": on suppose une colonne fold, sinon on retombe sur iteration
        if "fold" not in df.columns and "iteration" in df.columns:
            df["fold"] = df["iteration"]

    df = df[["fold", "cindex_default", "graf"]].reset_index(drop=True)
    return df


def compare_results(paths, types, labels, graph_name, boxplot=False):
    assert len(paths) == len(types), "Il faut autant de --types que de --files"
    assert len(paths) == len(labels), "Il faut autant de --labels que de --files"

    # Chargement de chaque fichier, avec colonnes suffixées par le label
    dfs = []
    for path, type_, label in zip(paths, types, labels):
        df = load_file(path, type_)
        print(f"\n--- {label} ({path}) ---")
        print(df.head())
        df = df.rename(columns={
            "cindex_default": f"cindex_default__{label}",
            "graf": f"graf__{label}",
        })
        dfs.append(df)

    # Merge successif de tous les fichiers sur 'fold'
    merged = reduce(lambda left, right: pd.merge(left, right, on="fold", how="inner"), dfs)

    # Différences par rapport au premier fichier (référence par défaut)
    ref_label = labels[0]
    print(f"\n=== Différences par rapport à '{ref_label}' ===")
    for label in labels[1:]:
        cindex_diff = merged[f"cindex_default__{label}"] - merged[f"cindex_default__{ref_label}"]
        graf_diff = merged[f"graf__{label}"] - merged[f"graf__{ref_label}"]
        print(f"{label} vs {ref_label} -> "
              f"C-index diff (mean): {cindex_diff.mean():.4f}, "
              f"Graf diff (mean): {graf_diff.mean():.4f}")

    if not boxplot:
        # Plot C-index
        plt.figure()
        for label in labels:
            plt.plot(merged["fold"], merged[f"cindex_default__{label}"], label=label)
        plt.legend()
        plt.title("C-index comparison")
        plt.savefig(f"{graph_name}_cindex.png", bbox_inches="tight")
        plt.show()

        # Plot IBS
        plt.figure()
        for label in labels:
            plt.plot(merged["fold"], merged[f"graf__{label}"], label=label)
        plt.legend()
        plt.title("IBS comparison")
        plt.savefig(f"{graph_name}_ibs.png", bbox_inches="tight")
        plt.show()

    else:
        # Reshape en format long pour boxplot
        cindex_parts = []
        graf_parts = []
        for label in labels:
            cindex_parts.append(pd.DataFrame({
                "Value": merged[f"cindex_default__{label}"],
                "Metric": "C-index",
                "Source": label
            }))
            graf_parts.append(pd.DataFrame({
                "Value": merged[f"graf__{label}"],
                "Metric": "IBS",
                "Source": label
            }))

        plot_data = pd.concat(cindex_parts + graf_parts, ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, metric in zip(axes, ["C-index", "IBS"]):
            subset = plot_data[plot_data["Metric"] == metric]
            sns.boxplot(x="Source", y="Value", data=subset, palette="Set2", ax=ax, showfliers=False)
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        plt.tight_layout()
        plt.savefig(f"{graph_name}_boxplot.png", bbox_inches="tight")
        plt.show()

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare plusieurs fichiers de résultats de survie.")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Liste des chemins des fichiers à comparer")
    parser.add_argument("--types", nargs="+", default=None, choices=["ref", "new"],
                        help="Type de chaque fichier (ref ou new), dans le même ordre que --files. "
                             "Par défaut: 'new' pour tous les fichiers.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels optionnels pour la légende (même ordre que --files). "
                             "Par défaut: nom du fichier sans extension.")
    parser.add_argument("--graph", required=True, help="Préfixe du nom des graphiques sauvegardés")
    parser.add_argument(
        "--boxplot",
        action="store_true",
        default=False,
        help="Utiliser des boxplots au lieu de courbes"
    )
    args = parser.parse_args()

    if args.types is None:
        types = ["new"] * len(args.files)
    else:
        if len(args.files) != len(args.types):
            parser.error("--files et --types doivent avoir le même nombre d'éléments")
        types = args.types

    if args.labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.files]
    else:
        if len(args.labels) != len(args.files):
            parser.error("--labels doit avoir le même nombre d'éléments que --files")
        labels = args.labels

    merged = compare_results(args.files, types, labels, args.graph, boxplot=args.boxplot)