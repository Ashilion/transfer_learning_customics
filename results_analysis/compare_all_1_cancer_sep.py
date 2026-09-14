import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import os
import re
import seaborn as sns
from functools import reduce

# Couleur dediee a Cox, distincte de la palette Standard/TL
COX_COLOR = "#55A868"


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_file(path, type_, cancer=None):
    """
    Charge un fichier de resultats et retourne un DataFrame normalise
    avec les colonnes: fold, cindex_default, graf.
    """
    df = pd.read_csv(path)

    if type_ == "ref":
        df = df[df["learner_id"] == "cox_ref"]
        if cancer is not None and "task_id" in df.columns:
            df = df[df["task_id"] == cancer]
        df["fold"] = df["iteration"] - 1  # indice en R commence a 1
    else:
        # fichier "new": on suppose une colonne fold, sinon on retombe sur iteration
        if "fold" not in df.columns and "iteration" in df.columns:
            df["fold"] = df["iteration"]

    df = df[["fold", "cindex_default", "graf"]].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Generation automatique des fichiers/labels pour un cancer donne
# ---------------------------------------------------------------------------

# variantes de methode: (suffixe dans le nom de fichier, nom affiche)
METHOD_VARIANTS = [
    ("", "CustCox"),
    ("clin", "CustCox clin"),
    ("sup", "CustCox sup"),
    ("supclin", "CustCox clin sup"),
]


def auto_generate(cancer, suffix, results_dir, include_cox=True):
    """
    Reconstruit la liste --files/--labels/--types pour un cancer donne,
    en suivant la convention de nommage:
      - Cox (baseline)         : outer_cv_results_vvh_{suffix}_{cancer}.csv
      - CustCox (sans TL)      : ncv_custcox_optuna_paral_{prefix}{suffix}_{cancer}.csv
      - Transfer Learning CustCox (finetune)  : ncv_finetune_optuna_paral_{prefix}{suffix}_{cancer}.csv
    """
    files, labels, types = [], [], []

    if include_cox:
        files.append(os.path.join(results_dir, f"outer_cv_results_vvh_{suffix}_{cancer}.csv"))
        labels.append("Reference Model")
        types.append("new")

    # methodes sans TL
    for prefix, name in METHOD_VARIANTS:
        files.append(os.path.join(results_dir, f"ncv_custcox_optuna_paral_{prefix}{suffix}_{cancer}.csv"))
        labels.append(name)
        types.append("new")

    # methodes avec Transfer Learning (finetune)
    for prefix, name in METHOD_VARIANTS:
        files.append(os.path.join(results_dir, f"ncv_finetune_optuna_paral_{prefix}{suffix}_{cancer}.csv"))
        labels.append(f"Transfer Learning {name}")
        types.append("new")

    return files, labels, types


# ---------------------------------------------------------------------------
# Detection methode / Transfer Learning a partir du label
# ---------------------------------------------------------------------------

def parse_label(label):
    """
    Analyse le label affiche et retourne:
      - la famille de methode
      - le groupe: "Standard", "Transfer Learning" ou "Cox"

    Exemples:
      'Transfer Learning CustCox clin' -> ('CustCox clin', 'Transfer Learning')
      'CustCox clin'                   -> ('CustCox clin', 'Standard')
      'Reference Model'            -> ('Cox', 'Cox')
    """
    label = label.strip()

    if label == "Reference Model":
        return "Cox", "Cox"

    m = re.match(r"^\s*Transfer Learning\s+(.*)$", label)
    if m:
        return m.group(1).strip(), "Transfer Learning"

    return label, "Standard"


def wrap_method_label(name):
    """
    Coupe le nom de methode sur 2 lignes pour un affichage horizontal:
    'CustCox clin sup' -> 'CustCox\nclin sup'
    'Cox'               -> 'Cox'
    """
    parts = name.split(" ", 1)
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}\n{parts[1]}"


# ---------------------------------------------------------------------------
# Comparaison + graphiques
# ---------------------------------------------------------------------------

def compare_results(paths, types, labels, graph_name, cancer=None, boxplot=False):
    assert len(paths) == len(types), "Il faut autant de --types que de --files"
    assert len(paths) == len(labels), "Il faut autant de --labels que de --files"

    dfs = []
    for path, type_, label in zip(paths, types, labels):
        df = load_file(path, type_, cancer=cancer)
        print(f"\n--- {label} ({path}) ---")
        print(df.head())
        df = df.rename(columns={
            "cindex_default": f"cindex_default__{label}",
            "graf": f"graf__{label}",
        })
        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="fold", how="inner"), dfs)

    # Differences par rapport au premier fichier (reference par defaut)
    ref_label = labels[0]
    print(f"\n=== Differences par rapport a '{ref_label}' ===")
    for label in labels[1:]:
        cindex_diff = merged[f"cindex_default__{label}"] - merged[f"cindex_default__{ref_label}"]
        graf_diff = merged[f"graf__{label}"] - merged[f"graf__{ref_label}"]
        print(f"{label} vs {ref_label} -> "
              f"C-index diff (mean): {cindex_diff.mean():.4f}, "
              f"Graf diff (mean): {graf_diff.mean():.4f}")

    # Differences TL vs sans-TL, methode par methode (si applicable)
    parsed = {label: parse_label(label) for label in labels}
    families_order = []
    for label in labels:
        fam, _ = parsed[label]
        if fam not in families_order:
            families_order.append(fam)

    print("\n=== Gain apporté par le Transfer Learning (Transfer Learning vs Standard) ===")
    for fam in families_order:
        std_label = next((l for l in labels if parsed[l] == (fam, "Standard")), None)
        tl_label = next((l for l in labels if parsed[l] == (fam, "Transfer Learning")), None)
        if std_label is not None and tl_label is not None:
            cindex_gain = merged[f"cindex_default__{tl_label}"] - merged[f"cindex_default__{std_label}"]
            graf_gain = merged[f"graf__{tl_label}"] - merged[f"graf__{std_label}"]
            print(f"{fam}: Transfer Learning - Standard -> "
                  f"C-index gain (mean): {cindex_gain.mean():.4f}, "
                  f"Graf gain (mean): {graf_gain.mean():.4f}")

    if not boxplot:
        # Courbes par fold, couleur = methode, style de trait = Transfer Learning/Standard
        non_cox_families = [f for f in families_order if f != "Cox"]
        palette = sns.color_palette("tab10", n_colors=len(non_cox_families))
        color_map = dict(zip(non_cox_families, palette))
        if "Cox" in families_order:
            color_map["Cox"] = COX_COLOR

        for metric, col_prefix, title, suffix_name in [
            ("C-index", "cindex_default", "C-index comparison", "cindex"),
            ("IBS", "graf", "IBS comparison", "ibs"),
        ]:
            plt.figure()
            for label in labels:
                fam, tl_status = parsed[label]
                linestyle = "--" if tl_status == "Transfer Learning" else "-"
                plt.plot(
                    merged["fold"],
                    merged[f"{col_prefix}__{label}"],
                    label=label,
                    color=color_map[fam],
                    linestyle=linestyle,
                )
            plt.legend(fontsize=8)
            plt.title(title)
            plt.xlabel("fold")
            plt.ylabel(metric)
            plt.savefig(f"{graph_name}_{suffix_name}.png", bbox_inches="tight")
            plt.show()

    else:
        # Reshape en format long pour boxplot, avec colonnes Methode / Groupe
        # Groupe vaut "Reference Model" pour la baseline (couleur/box dediees), sinon
        # "Standard" ou "Transfer Learning" selon la methode.
        rows = []
        for label in labels:
            fam, tl_status = parsed[label]
            group = "Reference Model" if fam == "Cox" else tl_status
            for metric, col_prefix, metric_name in [
                ("cindex", "cindex_default", "C-index"),
                ("graf", "graf", "IBS"),
            ]:
                for value in merged[f"{col_prefix}__{label}"]:
                    rows.append({
                        "Value": value,
                        "Metric": metric_name,
                        "Method": fam,
                        "Group": group,
                    })
        plot_data = pd.DataFrame(rows)

        group_palette = {"Reference Model": COX_COLOR, "Standard": "#4C72B0", "Transfer Learning": "#DD8452"}
        has_cox = "Cox" in families_order

        fig, axes = plt.subplots(1, 2, figsize=(1.8 * len(families_order) + 3, 5))
        for ax, metric in zip(axes, ["C-index", "IBS"]):
            subset = plot_data[plot_data["Metric"] == metric]
            sns.boxplot(
                x="Method", y="Value", hue="Group",
                data=subset, order=families_order,
                hue_order=["Reference Model", "Standard", "Transfer Learning"],
                palette=group_palette, ax=ax, showfliers=False, dodge=True,
            )
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.set_xticklabels([wrap_method_label("Reference Model" if f == "Cox" else f) for f in families_order],
                                rotation=0, ha="center", fontsize=8)

            handles, legend_labels = ax.get_legend_handles_labels()

            # Ligne pointillee horizontale = médiane du Reference Model, pour comparer
            # visuellement toutes les methodes au Reference Model.
            if has_cox:
                cox_median = subset.loc[subset["Method"] == "Cox", "Value"].median()
                if pd.notna(cox_median):
                    ax.axhline(cox_median, color=COX_COLOR, linestyle="--",
                               linewidth=1.3, alpha=0.9, zorder=10)
                    handles.append(Line2D([0], [0], color=COX_COLOR, linestyle="--",
                                           linewidth=1.3))
                    legend_labels.append("Reference Model (médiane)")

            ax.legend(handles=handles, labels=legend_labels, title="", fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{graph_name}_boxplot.png", bbox_inches="tight")
        plt.show()

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare plusieurs fichiers de resultats de survie, "
                     "avec regroupement Transfer Learning vs Standard cote a cote par methode."
    )
    parser.add_argument("--cancer", default=None,
                        help="Code du cancer (ex: KIRP). Si fourni sans --files, "
                             "la liste des fichiers est generee automatiquement.")
    parser.add_argument("--suffix", default="ridge",
                        help="Suffixe de modele utilise dans les noms de fichiers "
                             "(ridge, lasso, ...). Par defaut: ridge.")
    parser.add_argument("--results-dir", default="results",
                        help="Dossier contenant les fichiers de resultats. Par defaut: results.")
    parser.add_argument("--no-cox", action="store_true", default=False,
                        help="N'inclut pas la baseline Cox dans la generation automatique.")

    parser.add_argument("--files", nargs="+", default=None,
                        help="Liste des chemins des fichiers a comparer "
                             "(remplace la generation automatique via --cancer).")
    parser.add_argument("--types", nargs="+", default=None, choices=["ref", "new"],
                        help="Type de chaque fichier (ref ou new), meme ordre que --files. "
                             "Par defaut: 'new' pour tous les fichiers.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels optionnels pour la legende (meme ordre que --files). "
                             "Convention: prefixer par 'Transfer Learning ' pour les methodes avec Transfer "
                             "Learning, ex: 'Transfer Learning CustCox clin'.")
    parser.add_argument("--graph", required=True, help="Prefixe du nom des graphiques sauvegardes")
    parser.add_argument("--boxplot", action="store_true", default=False,
                        help="Utiliser des boxplots (methode en x, Transfer Learning/Standard cote a cote) "
                             "au lieu de courbes.")
    args = parser.parse_args()

    if args.files is None:
        if args.cancer is None:
            parser.error("Fournir soit --files, soit --cancer pour la generation automatique.")
        files, labels, types = auto_generate(
            cancer=args.cancer,
            suffix=args.suffix,
            results_dir=args.results_dir,
            include_cox=not args.no_cox,
        )
        # Overrides eventuels par l'utilisateur
        if args.labels is not None:
            if len(args.labels) != len(files):
                parser.error("--labels doit avoir le meme nombre d'elements que les fichiers generes "
                              f"({len(files)}).")
            labels = args.labels
        if args.types is not None:
            if len(args.types) != len(files):
                parser.error("--types doit avoir le meme nombre d'elements que les fichiers generes "
                              f"({len(files)}).")
            types = args.types
    else:
        files = args.files

        if args.types is None:
            types = ["new"] * len(files)
        else:
            if len(files) != len(args.types):
                parser.error("--files et --types doivent avoir le meme nombre d'elements")
            types = args.types

        if args.labels is None:
            labels = [os.path.splitext(os.path.basename(p))[0] for p in files]
        else:
            if len(args.labels) != len(files):
                parser.error("--labels doit avoir le meme nombre d'elements que --files")
            labels = args.labels

    merged = compare_results(
        files, types, labels, args.graph,
        cancer=args.cancer, boxplot=args.boxplot,
    )