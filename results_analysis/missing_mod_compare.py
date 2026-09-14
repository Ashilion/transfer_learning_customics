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


# ---------------------------------------------------------------------------
# Chargement du fichier
# ---------------------------------------------------------------------------

def load_results(path):
    """
    Charge un fichier CSV de la forme :

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

    parser.add_argument(
        "--results",
        required=True,
        help=(
            "CSV contenant method, fold0, ..., fold9. "
            "Les cellules sont sous la forme C-index/IBS."
        )
    )

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

    df = load_results(args.results)

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