"""
Compare des stratégies de gestion des données manquantes (Drop vs Impute,
à différents taux) à une référence Cox, sur C-index et IBS. Affiche des
statistiques par taux, des courbes par fold et des boxplots.

Deux modes de chargement des résultats Drop/Impute :

1) Mode "fichiers individuels" (NOUVEAU DÉFAUT) :
   Un fichier par méthode et par taux, au format long
   (fold, cindex_default, graf, ...), comme le fichier Cox.
   Nommage attendu :
       results/ncv_custcox_optuna_paral_drop{rate}_{cancer}.csv
       results/ncv_custcox_optuna_paral_imp{rate}_{cancer}.csv

   Usage:
       python missing_mod_compare.py --cox cox_ref.csv --graph out/kirp
       python missing_mod_compare.py --results-dir results --rates 30 50 70 \
           --cancer KIRP --cox results/outer_cv_results_vvh_ridge_KIRP.csv \
           --graph "figures/KIRP/missing_data/missing_results"

2) Mode "pivot" (ancien comportement, conservé pour compatibilité) :
   Un seul CSV pivot (method, fold0, ..., fold9), cellules "cindex/ibs".
   Activé dès que --results est fourni.

       python missing_mod_compare.py --results drop_impute.csv \
           --cox cox_ref.csv --graph out/kirp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import argparse
import os
import re


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Couleur dédiée à Cox
COX_COLOR = "#55A868"

# Taux par défaut et préfixes de fichiers pour le mode "fichiers individuels"
DEFAULT_RATES = [30, 50, 70]

# type interne -> préfixe utilisé dans le nom de fichier
TYPE_TO_FILE_PREFIX = {
    "drop": "drop",
    "impute": "imp",
}

DEFAULT_PATTERN = "ncv_custcox_optuna_paral_{prefix}{rate}_{cancer}.csv"


# ---------------------------------------------------------------------------
# Chargement du fichier pivot (ancien format)
# ---------------------------------------------------------------------------

def load_results(path):
    """
    Charge un fichier CSV pivot de la forme :

        method,fold0,fold1,...,fold9
        impute_50,MISSING,...
        drop_30,0.8929/0.1514,...

    Chaque valeur est :
        C-index / IBS

    Retourne un DataFrame long avec :
        method
        fold
        cindex
        ibs
    """

    df = pd.read_csv(path)

    fold_cols = [c for c in df.columns if re.match(r"fold\d+", str(c))]

    rows = []

    for _, row in df.iterrows():

        method = row["method"]

        for fold_col in fold_cols:

            value = row[fold_col]

            if pd.isna(value) or str(value).upper() == "MISSING":
                continue

            value = str(value).strip()

            # Sécurité si valeur mal formée
            if "/" not in value:
                continue

            try:
                cindex, ibs = value.split("/", 1)

                cindex = float(cindex)
                ibs = float(ibs)

            except (ValueError, TypeError):
                continue

            fold = int(re.search(r"\d+", fold_col).group())

            rows.append({
                "method": method,
                "fold": fold,
                "cindex": cindex,
                "ibs": ibs
            })

    result = pd.DataFrame(rows)

    return result


# ---------------------------------------------------------------------------
# Chargement d'un fichier individuel au format "long" (fold, cindex_default, graf)
# ---------------------------------------------------------------------------

def _load_long_format_file(filepath):
    """
    Charge un fichier au format long (celui produit par les runs de CV),
    contenant au minimum les colonnes 'fold' (ou 'iteration') et
    'cindex_default', 'graf'.

    Retourne un DataFrame avec les colonnes : fold, cindex, ibs
    """

    file_df = pd.read_csv(filepath)

    if "fold" not in file_df.columns:
        if "iteration" in file_df.columns:
            file_df["fold"] = file_df["iteration"] - 1
        else:
            raise ValueError(
                f"{filepath} doit contenir la colonne 'fold' ou 'iteration'."
            )

    if "cindex_default" not in file_df.columns:
        raise ValueError(
            f"{filepath} doit contenir la colonne 'cindex_default'."
        )

    if "graf" not in file_df.columns:
        raise ValueError(
            f"{filepath} doit contenir la colonne 'graf'."
        )

    file_df = (
        file_df[["fold", "cindex_default", "graf"]]
        .drop_duplicates("fold")
        .sort_values("fold")
        .rename(columns={"cindex_default": "cindex", "graf": "ibs"})
    )

    return file_df


# ---------------------------------------------------------------------------
# Chargement des fichiers individuels par méthode / taux (NOUVEAU DÉFAUT)
# ---------------------------------------------------------------------------

def load_individual_results(
    results_dir,
    rates=DEFAULT_RATES,
    cancer="KIRP",
    pattern=DEFAULT_PATTERN
):
    """
    Charge les résultats Drop/Impute à partir de fichiers individuels,
    un fichier par méthode (drop/impute) et par taux de données manquantes.

    Nom de fichier attendu (paramétrable via `pattern`) :
        ncv_custcox_optuna_paral_drop{rate}_{cancer}.csv
        ncv_custcox_optuna_paral_imp{rate}_{cancer}.csv

    Un fichier manquant est simplement signalé et ignoré (pas d'erreur
    bloquante), pour permettre des comparaisons partielles.

    Retourne un DataFrame long avec :
        method   (ex: "drop_30", "impute_30")
        fold
        cindex
        ibs
    """

    rows = []

    for rate in rates:

        for method_type, prefix in TYPE_TO_FILE_PREFIX.items():

            filename = pattern.format(
                prefix=prefix,
                rate=rate,
                cancer=cancer
            )

            filepath = os.path.join(results_dir, filename)

            if not os.path.exists(filepath):
                print(f"[Info] Fichier manquant, ignoré : {filepath}")
                continue

            file_df = _load_long_format_file(filepath)

            method_name = f"{method_type}_{rate}"

            for _, row in file_df.iterrows():
                rows.append({
                    "method": method_name,
                    "fold": int(row["fold"]),
                    "cindex": row["cindex"],
                    "ibs": row["ibs"]
                })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "Aucun fichier de résultats trouvé dans "
            f"'{results_dir}' pour les taux {list(rates)}. "
            "Vérifie --results-dir, --rates, --cancer et --pattern, "
            "ou utilise --results pour le mode pivot."
        )

    return result


# ---------------------------------------------------------------------------
# Identification Drop / Impute / taux
# ---------------------------------------------------------------------------

def parse_method(method):
    """
    Transforme :

        drop_30   -> type='drop',   rate=30
        impute_30 -> type='impute', rate=30

    """

    m = re.match(r"^(drop|impute)_(\d+)$", method)

    if m is None:
        return None, None

    method_type = m.group(1)
    rate = int(m.group(2))

    return method_type, rate


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def prepare_data(df):

    df = df.copy()

    parsed = df["method"].apply(parse_method)

    df["type"] = parsed.apply(lambda x: x[0])
    df["rate"] = parsed.apply(lambda x: x[1])

    # On enlève les lignes qui ne correspondent pas à drop_xx / impute_xx
    df = df.dropna(subset=["type", "rate"])

    df["rate"] = df["rate"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Comparaison avec Cox
# ---------------------------------------------------------------------------

def compare_to_cox(df, cox_cindex, cox_ibs):
    """
    Compare chaque résultat Drop/Impute au modèle de Cox.

    cox_cindex et cox_ibs doivent contenir les résultats Cox
    par fold.
    """

    cox = pd.DataFrame({
        "fold": cox_cindex.index,
        "cox_cindex": cox_cindex.values,
        "cox_ibs": cox_ibs.values
    })

    merged = df.merge(cox, on="fold", how="left")

    merged["cindex_diff_cox"] = (
        merged["cindex"] - merged["cox_cindex"]
    )

    merged["ibs_diff_cox"] = (
        merged["ibs"] - merged["cox_ibs"]
    )

    return merged


# ---------------------------------------------------------------------------
# Affichage des statistiques
# ---------------------------------------------------------------------------

def print_comparison(df, cox_cindex, cox_ibs):

    print("\n" + "=" * 80)
    print("COMPARAISON DROP / IMPUTE")
    print("=" * 80)

    print("\nRéférence Cox :")

    print(
        f"  C-index moyen : {cox_cindex.mean():.4f}"
    )

    print(
        f"  IBS moyen     : {cox_ibs.mean():.4f}"
    )

    for rate in sorted(df["rate"].unique()):

        print("\n" + "-" * 80)
        print(f"Taux de données manquantes : {rate}%")
        print("-" * 80)

        subset = df[df["rate"] == rate]

        for method_type in ["drop", "impute"]:

            data = subset[subset["type"] == method_type]

            if data.empty:
                print(f"\n{method_type.upper()} : MISSING")
                continue

            cox_cindex_aligned = cox_cindex.reindex(data["fold"])
            cox_ibs_aligned = cox_ibs.reindex(data["fold"])

            cindex_diff = (
                data["cindex"].values -
                cox_cindex_aligned.values
            )

            ibs_diff = (
                data["ibs"].values -
                cox_ibs_aligned.values
            )

            print(f"\n{method_type.upper()}")

            print(
                f"  C-index moyen : {data['cindex'].mean():.4f}"
            )

            print(
                f"  C-index vs Cox : {np.nanmean(cindex_diff):+.4f}"
            )

            print(
                f"  IBS moyen : {data['ibs'].mean():.4f}"
            )

            print(
                f"  IBS vs Cox : {np.nanmean(ibs_diff):+.4f}"
            )

            print(
                f"  Nombre de folds : {len(data)}"
            )


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_results(
    df,
    cox_cindex,
    cox_ibs,
    graph_name
):

    rates = sorted(df["rate"].unique())

    # -----------------------------------------------------------------------
    # Ajout de Cox au dataframe pour les graphiques
    # -----------------------------------------------------------------------

    cox_df = pd.DataFrame({
        "fold": cox_cindex.index,
        "method": "Cox",
        "type": "Cox",
        "rate": 0,
        "cindex": cox_cindex.values,
        "ibs": cox_ibs.values
    })

    plot_df = pd.concat(
        [df, cox_df],
        ignore_index=True
    )

    # -----------------------------------------------------------------------
    # Couleurs
    # -----------------------------------------------------------------------

    method_colors = {
        "drop": "#4C72B0",
        "impute": "#DD8452",
        "Cox": COX_COLOR
    }

    # -----------------------------------------------------------------------
    # C-index par fold
    # -----------------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    # Cox
    plt.plot(
        cox_cindex.index,
        cox_cindex.values,
        color=COX_COLOR,
        linewidth=2.5,
        marker="o",
        label="Cox"
    )

    # Drop / Impute
    for rate in rates:

        for method_type in ["drop", "impute"]:

            subset = df[
                (df["rate"] == rate) &
                (df["type"] == method_type)
            ]

            if subset.empty:
                continue

            plt.plot(
                subset["fold"],
                subset["cindex"],
                marker="o",
                linestyle="-",
                color=method_colors[method_type],
                alpha=0.8,
                label=f"{method_type.capitalize()} {rate}%"
            )

    plt.axhline(
        cox_cindex.mean(),
        color=COX_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8
    )

    plt.title("C-index : Drop vs Impute vs Cox")
    plt.xlabel("Fold")
    plt.ylabel("C-index")
    plt.xticks(range(10))
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        f"{graph_name}_cindex_folds.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # -----------------------------------------------------------------------
    # IBS par fold
    # -----------------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    # Cox
    plt.plot(
        cox_ibs.index,
        cox_ibs.values,
        color=COX_COLOR,
        linewidth=2.5,
        marker="o",
        label="Cox"
    )

    # Drop / Impute
    for rate in rates:

        for method_type in ["drop", "impute"]:

            subset = df[
                (df["rate"] == rate) &
                (df["type"] == method_type)
            ]

            if subset.empty:
                continue

            plt.plot(
                subset["fold"],
                subset["ibs"],
                marker="o",
                linestyle="-",
                color=method_colors[method_type],
                alpha=0.8,
                label=f"{method_type.capitalize()} {rate}%"
            )

    plt.axhline(
        cox_ibs.mean(),
        color=COX_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8
    )

    plt.title("IBS : Drop vs Impute vs Cox")
    plt.xlabel("Fold")
    plt.ylabel("IBS")
    plt.xticks(range(10))
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        f"{graph_name}_ibs_folds.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ---------------------------------------------------------------------------
# Boxplots côte à côte par taux
# ---------------------------------------------------------------------------

def plot_boxplots(
    df,
    cox_cindex,
    cox_ibs,
    graph_name
):

    rates = sorted(df["rate"].unique())

    # -----------------------------------------------------------------------
    # Préparation C-index
    # -----------------------------------------------------------------------

    rows = []

    # Cox
    for fold, value in cox_cindex.items():
        rows.append({
            "rate": "Cox",
            "method": "Cox",
            "value": value
        })

    # Drop / Impute
    for _, row in df.iterrows():
        rows.append({
            "rate": f"{row['rate']}%",
            "method": row["type"].capitalize(),
            "value": row["cindex"]
        })

    cindex_plot = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Préparation IBS
    # -----------------------------------------------------------------------

    rows = []

    # Cox
    for fold, value in cox_ibs.items():
        rows.append({
            "rate": "Cox",
            "method": "Cox",
            "value": value
        })

    # Drop / Impute
    for _, row in df.iterrows():
        rows.append({
            "rate": f"{row['rate']}%",
            "method": row["type"].capitalize(),
            "value": row["ibs"]
        })

    ibs_plot = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Ordre des catégories
    # -----------------------------------------------------------------------

    order = ["Cox"] + [f"{r}%" for r in rates]

    hue_order = ["Drop", "Impute"]

    palette = {
        "Drop": "#4C72B0",
        "Impute": "#DD8452"
    }

    # -----------------------------------------------------------------------
    # Figure C-index
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(2 * len(order) + 3, 6)
    )

    # Cox séparément
    sns.boxplot(
        data=cindex_plot[
            cindex_plot["method"] == "Cox"
        ],
        x="rate",
        y="value",
        order=["Cox"],
        color=COX_COLOR,
        ax=ax,
        showfliers=False
    )

    # Drop / Impute
    sns.boxplot(
        data=cindex_plot[
            cindex_plot["method"] != "Cox"
        ],
        x="rate",
        y="value",
        hue="method",
        order=[f"{r}%" for r in rates],
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        showfliers=False,
        dodge=True
    )

    # Médiane Cox
    cox_median = cox_cindex.median()

    ax.axhline(
        cox_median,
        color=COX_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Cox median"
    )

    ax.set_title("C-index : Drop vs Impute par taux")
    ax.set_xlabel("Taux de données manquantes")
    ax.set_ylabel("C-index")

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        f"{graph_name}_boxplot_cindex.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # -----------------------------------------------------------------------
    # Figure IBS
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(2 * len(order) + 3, 6)
    )

    # Cox
    sns.boxplot(
        data=ibs_plot[
            ibs_plot["method"] == "Cox"
        ],
        x="rate",
        y="value",
        order=["Cox"],
        color=COX_COLOR,
        ax=ax,
        showfliers=False
    )

    # Drop / Impute
    sns.boxplot(
        data=ibs_plot[
            ibs_plot["method"] != "Cox"
        ],
        x="rate",
        y="value",
        hue="method",
        order=[f"{r}%" for r in rates],
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        showfliers=False,
        dodge=True
    )

    # Médiane Cox
    cox_median = cox_ibs.median()

    ax.axhline(
        cox_median,
        color=COX_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Cox median"
    )

    ax.set_title("IBS : Drop vs Impute par taux")
    ax.set_xlabel("Taux de données manquantes")
    ax.set_ylabel("IBS")

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        f"{graph_name}_boxplot_ibs.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare Drop et Impute pour différents taux de données "
            "manquantes, avec Cox comme référence."
        )
    )

    # -- Mode "fichiers individuels" (nouveau défaut) ------------------------

    parser.add_argument(
        "--results-dir",
        default="results",
        help=(
            "Dossier contenant les fichiers individuels par méthode/taux "
            "(défaut : 'results'). Ignoré si --results est fourni."
        )
    )

    parser.add_argument(
        "--rates",
        type=int,
        nargs="+",
        default=DEFAULT_RATES,
        help=(
            "Taux de données manquantes à charger en mode fichiers "
            f"individuels (défaut : {DEFAULT_RATES})."
        )
    )

    parser.add_argument(
        "--cancer",
        default="KIRP",
        help="Code du cancer utilisé dans le nom des fichiers (défaut : KIRP)."
    )

    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=(
            "Patron de nom de fichier pour le mode fichiers individuels, "
            "avec les placeholders {prefix}, {rate}, {cancer}. "
            f"Défaut : '{DEFAULT_PATTERN}'."
        )
    )

    # -- Mode pivot (ancien comportement, optionnel) -------------------------

    parser.add_argument(
        "--results",
        default=None,
        help=(
            "CSV pivot contenant method, fold0, ..., fold9 "
            "(cellules 'C-index/IBS'). Si fourni, active l'ancien mode "
            "pivot à la place du mode fichiers individuels."
        )
    )

    # -- Commun ---------------------------------------------------------------

    parser.add_argument(
        "--cox",
        required=True,
        help=(
            "CSV contenant les résultats du modèle Cox."
        )
    )

    parser.add_argument(
        "--graph",
        required=True,
        help="Préfixe pour les graphiques générés."
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Chargement résultats Drop / Impute
    # -----------------------------------------------------------------------

    if args.results:
        # Ancien mode : un seul CSV pivot
        print(f"\nMode pivot : chargement de {args.results}")
        df = load_results(args.results)
    else:
        # Nouveau mode par défaut : fichiers individuels par méthode/taux
        print(
            "\nMode fichiers individuels : chargement depuis "
            f"'{args.results_dir}' pour les taux {args.rates} "
            f"(cancer={args.cancer})"
        )
        df = load_individual_results(
            results_dir=args.results_dir,
            rates=args.rates,
            cancer=args.cancer,
            pattern=args.pattern
        )

    df = prepare_data(df)

    print("\nRésultats chargés :")
    print(df)

    # -----------------------------------------------------------------------
    # Chargement Cox
    # -----------------------------------------------------------------------

    cox_df = pd.read_csv(args.cox)

    # Adapter ici si le fichier Cox possède des noms différents
    #
    # Le fichier doit contenir :
    # fold, cindex_default, graf
    #
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
        cox_df[
            ["fold", "cindex_default", "graf"]
        ]
        .drop_duplicates("fold")
        .sort_values("fold")
    )

    cox_cindex = cox_df.set_index("fold")["cindex_default"]
    cox_ibs = cox_df.set_index("fold")["graf"]

    # -----------------------------------------------------------------------
    # Comparaison numérique
    # -----------------------------------------------------------------------

    print_comparison(
        df,
        cox_cindex,
        cox_ibs
    )

    # -----------------------------------------------------------------------
    # Graphiques
    # -----------------------------------------------------------------------

    plot_results(
        df,
        cox_cindex,
        cox_ibs,
        args.graph
    )

    plot_boxplots(
        df,
        cox_cindex,
        cox_ibs,
        args.graph
    )