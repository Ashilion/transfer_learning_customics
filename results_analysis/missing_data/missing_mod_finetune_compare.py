"""
Compare, pour du fine-tuning (Transfer Learning), les stratégies Drop vs
Impute à différents taux d'entraînement (XX) et d'évaluation (YY), avec
Cox comme référence. Recherche automatiquement les fichiers per-fold
(motif tl(drop|imp)XX_YY_*.csv) dans un dossier, affiche des statistiques,
et génère des boxplots (par XX, puis regroupés par facette).

Usage:
    python missing_mod_finetune_compare.py --results-dir results --cox cox_ref.csv --graph out/kirp
    python missing_mod_finetune_compare.py --results-dir results --cancer-type KIRP --cox cox_ref.csv --graph out/kirp

    python results_analysis/missing_data/missing_mod_finetune_compare.py --cox results/outer_cv_results_vvh_ridge_KIRP.csv --graph "figures/KIRP/missing_data/missing_finetune_results"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Couleur dédiée à Cox
COX_COLOR = "#55A868"

# Regex de secours pour retrouver type/XX/YY dans le nom de fichier
# ex: ncv_finetune_optuna_paral_tldrop30_30_KIRP.csv -> type=drop, train_rate(XX)=30, rate(YY)=30
#     ncv_finetune_optuna_paral_tlimp30_50_KIRP.csv  -> type=impute, train_rate(XX)=30, rate(YY)=50
# XX = taux de données manquantes utilisé pendant le fine-tuning
# YY = taux de données manquantes utilisé pour l'évaluation (fold)
FILENAME_PATTERN = re.compile(r"tl(drop|imp)(\d+)_(\d+)_")

# Regex utilisée pour repérer les fichiers de résultats dans le dossier
# results/. Matche n'importe quel .csv contenant tldropXX_YY ou tlimpXX_YY.
RESULTS_FILENAME_REGEX = re.compile(r"tl(?:drop|imp)\d*_\d+_.*\.csv$")


# ---------------------------------------------------------------------------
# Recherche des fichiers de résultats dans un dossier
# ---------------------------------------------------------------------------

def find_finetune_files(results_dir, cancer_type=None):

    if not os.path.isdir(results_dir):
        raise ValueError(f"Le dossier '{results_dir}' n'existe pas.")

    matches = []

    for filename in sorted(os.listdir(results_dir)):

        if not RESULTS_FILENAME_REGEX.search(filename):
            continue

        if cancer_type is not None and cancer_type not in filename:
            continue

        matches.append(os.path.join(results_dir, filename))

    if not matches:
        raise ValueError(
            f"Aucun fichier trouvé dans '{results_dir}' correspondant au "
            f"motif tl(drop|imp)XX_YY_....csv"
            + (f" pour '{cancer_type}'." if cancer_type else ".")
        )

    return matches


# ---------------------------------------------------------------------------
# Chargement d'un fichier "per-fold" (fold, cindex_default, graf, ...)
# ---------------------------------------------------------------------------

def load_finetune_file(path):
    """
    Charge un fichier de résultats fine-tuning au format :

        fold,cindex_default,graf,...,missing_rate,missing_strategy

    Une ligne par fold. Le type (drop/impute) et le taux de données
    manquantes utilisé pour l'évaluation (YY) sont lus dans les colonnes
    missing_strategy / missing_rate quand elles existent, sinon ils sont
    déduits du nom du fichier. Le taux utilisé pendant le fine-tuning (XX)
    n'est disponible que dans le nom du fichier.

    Retourne un DataFrame avec :
        method, fold, cindex, ibs, type, rate, train_rate
    """

    df = pd.read_csv(path)

    if "fold" not in df.columns:
        if "iteration" in df.columns:
            df["fold"] = df["iteration"] - 1
        else:
            raise ValueError(
                f"{path} : colonne 'fold' (ou 'iteration') manquante."
            )

    if "cindex_default" not in df.columns:
        raise ValueError(f"{path} : colonne 'cindex_default' manquante.")

    if "graf" not in df.columns:
        raise ValueError(f"{path} : colonne 'graf' manquante.")

    # ------------------------------------------------------------------
    # Type (drop / impute) et taux d'évaluation (YY)
    # ------------------------------------------------------------------

    method_type = None
    rate = None

    if "missing_strategy" in df.columns and df["missing_strategy"].notna().any():
        raw_type = str(df["missing_strategy"].dropna().iloc[0]).strip().lower()
        if raw_type in ("drop", "impute"):
            method_type = raw_type
        elif raw_type in ("imp",):
            method_type = "impute"

    if "missing_rate" in df.columns and df["missing_rate"].notna().any():
        raw_rate = float(df["missing_rate"].dropna().iloc[0])
        # missing_rate peut être exprimé en fraction (0.5) ou en % (50)
        rate = round(raw_rate * 100) if raw_rate <= 1 else round(raw_rate)

    # ------------------------------------------------------------------
    # Taux d'entraînement / fine-tuning (XX) : uniquement dans le nom de
    # fichier, et repli sur le nom de fichier si type/YY manquants.
    # ------------------------------------------------------------------

    m = FILENAME_PATTERN.search(os.path.basename(path))

    if m is None and (method_type is None or rate is None):
        raise ValueError(
            f"{path} : impossible de déterminer le type/taux "
            f"(ni dans les colonnes, ni dans le nom du fichier)."
        )

    train_rate = None

    if m is not None:
        file_type, file_xx, file_yy = m.group(1), m.group(2), m.group(3)

        train_rate = int(file_xx)

        if method_type is None:
            method_type = "drop" if file_type == "drop" else "impute"
        if rate is None:
            rate = int(file_yy)

    if train_rate is None:
        raise ValueError(
            f"{path} : impossible de déterminer le taux de fine-tuning (XX) "
            f"depuis le nom du fichier."
        )

    result = pd.DataFrame({
        "fold": df["fold"].astype(int),
        "cindex": df["cindex_default"].astype(float),
        "ibs": df["graf"].astype(float),
    })

    result["type"] = method_type
    result["rate"] = rate
    result["train_rate"] = train_rate
    result["method"] = f"{method_type}_{rate}"

    return result


def load_all_finetune_files(paths):
    """
    Charge et concatène plusieurs fichiers per-fold (un par
    combinaison XX/type/YY, ex: drop30_30, drop30_50, ..., impute70_70).
    """

    frames = []

    for path in paths:
        frames.append(load_finetune_file(path))

    df = pd.concat(frames, ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Chargement du fichier Cox (référence)
# ---------------------------------------------------------------------------

def load_cox_file(path):

    cox_df = pd.read_csv(path)

    if "fold" not in cox_df.columns:
        if "iteration" in cox_df.columns:
            cox_df["fold"] = cox_df["iteration"] - 1
        else:
            raise ValueError(
                "Le fichier Cox doit contenir 'fold' ou 'iteration'."
            )

    if "cindex_default" not in cox_df.columns:
        raise ValueError(
            "Le fichier Cox doit contenir la colonne 'cindex_default'."
        )

    if "graf" not in cox_df.columns:
        raise ValueError(
            "Le fichier Cox doit contenir la colonne 'graf'."
        )

    cox_df = (
        cox_df[["fold", "cindex_default", "graf"]]
        .drop_duplicates("fold")
        .sort_values("fold")
    )

    cox_cindex = cox_df.set_index("fold")["cindex_default"]
    cox_ibs = cox_df.set_index("fold")["graf"]

    return cox_cindex, cox_ibs


# ---------------------------------------------------------------------------
# Affichage des statistiques
# ---------------------------------------------------------------------------

def print_comparison(df, cox_cindex, cox_ibs):

    print("\n" + "=" * 80)
    print("COMPARAISON DROP / IMPUTE (fine-tuning)")
    print("=" * 80)

    print("\nRéférence Cox :")
    print(f"  C-index moyen : {cox_cindex.mean():.4f}")
    print(f"  IBS moyen     : {cox_ibs.mean():.4f}")

    for train_rate in sorted(df["train_rate"].unique()):

        print("\n" + "#" * 80)
        print(f"Taux de fine-tuning (XX) : {train_rate}%")
        print("#" * 80)

        tr_subset = df[df["train_rate"] == train_rate]

        for rate in sorted(tr_subset["rate"].unique()):

            print("\n" + "-" * 80)
            print(f"Taux de données manquantes (YY) : {rate}%")
            print("-" * 80)

            subset = tr_subset[tr_subset["rate"] == rate]

            for method_type in ["drop", "impute"]:

                data = subset[subset["type"] == method_type]

                if data.empty:
                    print(f"\n{method_type.upper()} : MISSING")
                    continue

                cox_cindex_aligned = cox_cindex.reindex(data["fold"])
                cox_ibs_aligned = cox_ibs.reindex(data["fold"])

                cindex_diff = data["cindex"].values - cox_cindex_aligned.values
                ibs_diff = data["ibs"].values - cox_ibs_aligned.values

                print(f"\n{method_type.upper()}")
                print(f"  C-index moyen : {data['cindex'].mean():.4f}")
                print(f"  C-index vs Cox : {np.nanmean(cindex_diff):+.4f}")
                print(f"  IBS moyen : {data['ibs'].mean():.4f}")
                print(f"  IBS vs Cox : {np.nanmean(ibs_diff):+.4f}")
                print(f"  Nombre de folds : {len(data)}")


# ---------------------------------------------------------------------------
# Graphiques : courbes par fold
# ---------------------------------------------------------------------------

def plot_results(df, cox_cindex, cox_ibs, graph_name):

    rates = sorted(df["rate"].unique())
    all_folds = sorted(
        set(df["fold"].unique()) | set(cox_cindex.index)
    )

    method_colors = {
        "drop": "#4C72B0",
        "impute": "#DD8452",
        "Cox": COX_COLOR,
    }

    for metric, cox_series, ylabel, suffix in [
        ("cindex", cox_cindex, "C-index", "cindex_folds"),
        ("ibs", cox_ibs, "IBS", "ibs_folds"),
    ]:

        plt.figure(figsize=(12, 6))

        plt.plot(
            cox_series.index,
            cox_series.values,
            color=COX_COLOR,
            linewidth=2.5,
            marker="o",
            label="Cox",
        )

        for rate in rates:
            for method_type in ["drop", "impute"]:

                subset = df[(df["rate"] == rate) & (df["type"] == method_type)]

                if subset.empty:
                    continue

                subset = subset.sort_values("fold")

                plt.plot(
                    subset["fold"],
                    subset[metric],
                    marker="o",
                    linestyle="-",
                    color=method_colors[method_type],
                    alpha=0.8,
                    label=f"{method_type.capitalize()} {rate}%",
                )

        plt.axhline(
            cox_series.mean(),
            color=COX_COLOR,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

        plt.title(f"{ylabel} : Drop vs Impute vs Cox")
        plt.xlabel("Fold")
        plt.ylabel(ylabel)
        plt.xticks(all_folds)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

        plt.tight_layout()

        plt.savefig(f"{graph_name}_{suffix}.png", dpi=300, bbox_inches="tight")

        plt.show()


# ---------------------------------------------------------------------------
# Graphiques : boxplots côte à côte par taux (YY), pour un train_rate (XX) donné
# ---------------------------------------------------------------------------

def plot_boxplots(df, cox_cindex, cox_ibs, graph_name, train_rate=None):
    """
    Génère les boxplots Drop vs Impute par taux YY.

    Si `train_rate` (XX) est fourni, le df est supposé déjà filtré sur ce
    XX : le titre et le nom de fichier de sortie l'indiquent.
    """

    rates = sorted(df["rate"].unique())

    hue_order = ["Drop", "Impute"]
    palette = {"Drop": "#4C72B0", "Impute": "#DD8452"}

    title_suffix = f" (fine-tuning {train_rate}%)" if train_rate is not None else ""
    file_suffix = f"_train{train_rate}" if train_rate is not None else ""

    for metric, cox_series, ylabel, suffix in [
        ("cindex", cox_cindex, "C-index", "boxplot_cindex"),
        ("ibs", cox_ibs, "IBS", "boxplot_ibs"),
    ]:

        rows = []

        for _, value in cox_series.items():
            rows.append({"rate": "Cox", "method": "Cox", "value": value})

        for _, row in df.iterrows():
            rows.append({
                "rate": f"{row['rate']}%",
                "method": row["type"].capitalize(),
                "value": row[metric],
            })

        plot_df = pd.DataFrame(rows)

        order = ["Cox"] + [f"{r}%" for r in rates]

        fig, ax = plt.subplots(figsize=(2 * len(order) + 3, 6))

        sns.boxplot(
            data=plot_df[plot_df["method"] == "Cox"],
            x="rate",
            y="value",
            order=["Cox"],
            color=COX_COLOR,
            ax=ax,
            showfliers=False,
        )

        sns.boxplot(
            data=plot_df[plot_df["method"] != "Cox"],
            x="rate",
            y="value",
            hue="method",
            order=[f"{r}%" for r in rates],
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            showfliers=False,
            dodge=True,
        )

        cox_median = cox_series.median()

        ax.axhline(
            cox_median,
            color=COX_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="Cox median",
        )

        ax.set_title(f"{ylabel} : Drop vs Impute par taux{title_suffix}")
        ax.set_xlabel("Taux de données manquantes (test)")
        ax.set_ylabel(ylabel)

        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()

        plt.savefig(f"{graph_name}{file_suffix}_{suffix}.png", dpi=300, bbox_inches="tight")

        plt.show()


def plot_boxplots_by_train_rate(df, cox_cindex, cox_ibs, graph_name):
    """
    Génère un jeu de boxplots séparé pour chaque valeur de XX
    (taux de fine-tuning) : un fichier par XX et par métrique.
    """

    for train_rate in sorted(df["train_rate"].unique()):
        subset = df[df["train_rate"] == train_rate]
        plot_boxplots(subset, cox_cindex, cox_ibs, graph_name, train_rate=train_rate)


# ---------------------------------------------------------------------------
# Graphique combiné : toutes les valeurs de XX regroupées dans une seule
# figure, mais sous forme de facettes (donc pas dans les mêmes boxplots)
# ---------------------------------------------------------------------------

def plot_boxplots_grouped(df, cox_cindex, cox_ibs, graph_name):
    """
    Une seule figure par métrique, avec une facette (sous-plot) par valeur
    de XX. Les boxplots Drop/Impute restent groupés par XX et ne sont pas
    mélangés entre eux.
    """

    train_rates = sorted(df["train_rate"].unique())
    rates = sorted(df["rate"].unique())

    hue_order = ["Drop", "Impute"]
    palette = {"Drop": "#4C72B0", "Impute": "#DD8452"}

    for metric, cox_series, ylabel, suffix in [
        ("cindex", cox_cindex, "C-index", "grouped_boxplot_cindex"),
        ("ibs", cox_ibs, "IBS", "grouped_boxplot_ibs"),
    ]:

        fig, axes = plt.subplots(
            1, len(train_rates),
            figsize=(4.5 * len(train_rates) + 2, 6),
            sharey=True,
        )

        if len(train_rates) == 1:
            axes = [axes]

        cox_median = cox_series.median()

        handles, labels = None, None

        for ax, train_rate in zip(axes, train_rates):

            subset = df[df["train_rate"] == train_rate]

            rows = []
            for _, row in subset.iterrows():
                rows.append({
                    "rate": f"{row['rate']}%",
                    "method": row["type"].capitalize(),
                    "value": row[metric],
                })

            plot_df = pd.DataFrame(rows)

            sns.boxplot(
                data=plot_df,
                x="rate",
                y="value",
                hue="method",
                order=[f"{r}%" for r in rates],
                hue_order=hue_order,
                palette=palette,
                ax=ax,
                showfliers=False,
                dodge=True,
            )

            ax.axhline(cox_median, color=COX_COLOR, linestyle="--", linewidth=1.5)

            ax.set_title(f"Fine-tuning {train_rate}%")
            ax.set_xlabel("Taux de données manquantes (test)")

            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

            if ax.legend_ is not None:
                ax.legend_.remove()

        axes[0].set_ylabel(ylabel)
        for ax in axes[1:]:
            ax.set_ylabel("")

        cox_handle = plt.Line2D(
            [0], [0], color=COX_COLOR, linestyle="--", linewidth=1.5,
        )
        handles = list(handles) + [cox_handle]
        labels = list(labels) + ["Cox (médiane)"]

        fig.suptitle(f"{ylabel} : Drop vs Impute par taux de fine-tuning (XX)")
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.92), loc="upper left")

        plt.tight_layout()

        plt.savefig(f"{graph_name}_{suffix}.png", dpi=300, bbox_inches="tight")

        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare Drop et Impute (30/50/70%) pour le fine-tuning, "
            "avec Cox comme référence, à partir de fichiers per-fold."
        )
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help=(
            "Dossier dans lequel chercher les fichiers de résultats "
            "(par défaut : 'results/'). Les fichiers sont repérés via "
            "le motif tl(drop|imp)XX_YY_....csv"
        ),
    )

    parser.add_argument(
        "--cancer-type",
        default=None,
        help=(
            "Filtre optionnel sur le nom du fichier (ex: 'KIRP') pour ne "
            "garder que les fichiers de ce type de cancer."
        ),
    )

    parser.add_argument(
        "--cox",
        required=True,
        help="CSV contenant les résultats du modèle Cox (référence).",
    )

    parser.add_argument(
        "--graph",
        required=True,
        help="Préfixe pour les graphiques générés.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Recherche puis chargement des résultats Drop / Impute
    # -----------------------------------------------------------------------

    result_files = find_finetune_files(args.results_dir, args.cancer_type)

    print("Fichiers trouvés :")
    for f in result_files:
        print(f"  - {f}")

    df = load_all_finetune_files(result_files)

    print("\nRésultats chargés :")
    print(df)

    # -----------------------------------------------------------------------
    # Chargement Cox
    # -----------------------------------------------------------------------

    cox_cindex, cox_ibs = load_cox_file(args.cox)

    # -----------------------------------------------------------------------
    # Comparaison numérique
    # -----------------------------------------------------------------------

    print_comparison(df, cox_cindex, cox_ibs)

    # -----------------------------------------------------------------------
    # Graphiques
    # -----------------------------------------------------------------------

    # plot_results(df, cox_cindex, cox_ibs, args.graph)

    # 1 jeu de boxplots par valeur de XX (train_rate) : 3 x 2 métriques = 6 plots
    plot_boxplots_by_train_rate(df, cox_cindex, cox_ibs, args.graph)

    # 1 plot combiné par métrique, avec une facette par XX (boxplots séparés)
    plot_boxplots_grouped(df, cox_cindex, cox_ibs, args.graph)