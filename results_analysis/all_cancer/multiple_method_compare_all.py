"""
Compare des résultats de modèles de survie (C-index, IBS) entre un nombre
quelconque de sources (ex: référence R, implémentation Python, variante ridge,
etc.) pour plusieurs types de cancer.

Chaque source est lue avec l'un des deux "readers" suivants :
  - "ref"  : un unique gros CSV contenant tous les cancers, filtré par
             `learner_id` puis par `task_id` (c'est le format des résultats
             de référence R).
  - "file" : un CSV par cancer, nommé "{file_basename}{cancer_name}.csv"
             dans un dossier de résultats (format des résultats Python).

Exemple d'utilisation (CLI) :
    python multiple_method_compare_all.py \
        --source name=vincent_r type=ref path=/path/ref_results.csv learner_id=glmnet_ref \
        --source name=python type=file dir=/path/results basename=outer_cv_results_vvh_ \
        --source name=python_ridge type=file dir=/path/results basename=outer_cv_results_vvh_ridge_ \
        --baseline vincent_r \
        --plot boxplot mad
"""

import argparse
import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METRIC_COLUMNS = ["cindex_default", "graf"]
METRIC_LABELS = {"cindex_default": "C-index", "graf": "IBS"}


# --------------------------------------------------------------------------- #
# Structures de données
# --------------------------------------------------------------------------- #

@dataclass
class Source:
    """Description d'une source de résultats à comparer."""
    name: str
    reader: str  # "ref" ou "file"
    # reader == "ref"
    path: Optional[str] = None
    learner_id: Optional[str] = None
    # reader == "file"
    results_dir: Optional[str] = None
    file_basename: Optional[str] = None

    def __post_init__(self):
        if self.reader not in ("ref", "file"):
            raise ValueError(f"Reader inconnu '{self.reader}' pour la source '{self.name}'")
        if self.reader == "ref" and not self.path:
            raise ValueError(f"Source '{self.name}': reader=ref nécessite 'path'")
        if self.reader == "file" and not (self.results_dir and self.file_basename):
            raise ValueError(f"Source '{self.name}': reader=file nécessite 'dir' et 'basename'")


def parse_source_spec(tokens: List[str]) -> Source:
    """Transforme une liste de tokens 'clé=valeur' (après --source) en Source."""
    kv = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Token --source invalide '{tok}', attendu clé=valeur")
        k, v = tok.split("=", 1)
        kv[k.strip()] = v.strip()

    if "name" not in kv or "type" not in kv:
        raise ValueError(f"--source nécessite au moins 'name' et 'type', reçu: {kv}")

    return Source(
        name=kv["name"],
        reader=kv["type"],
        path=kv.get("path"),
        learner_id=kv.get("learner_id"),
        results_dir=kv.get("dir"),
        file_basename=kv.get("basename"),
    )


# --------------------------------------------------------------------------- #
# Découverte des cancers disponibles / tailles
# --------------------------------------------------------------------------- #

def discover_cancer_names(source: Source) -> List[str]:
    """
    Liste les cancers disponibles pour une source de type 'file'.
 
    Les fichiers contenant "ridge" dans leur nom sont exclus, sauf si "ridge"
    fait partie du `file_basename` lui-même (auquel cas on cherche
    explicitement des résultats ridge).
    """
    if source.reader != "file":
        raise ValueError("La découverte des cancers ne fonctionne que pour reader='file'")
    pattern = os.path.join(source.results_dir, f"{source.file_basename}*.csv")
    files = sorted(glob.glob(pattern))
    if "ridge" not in source.file_basename:
        files = [f for f in files if "ridge" not in os.path.basename(f)]
    return [
        os.path.basename(f).replace(source.file_basename, "").replace(".csv", "")
        for f in files
    ]


def get_cancer_sizes(cancer_names: List[str], clinical_dir: str) -> Dict[str, int]:
    """Nombre d'individus par cancer, lu depuis les pickles cliniques."""
    sizes = {}
    for cancer_name in cancer_names:
        path = os.path.join(clinical_dir, f"{cancer_name}_clinical.pickle")
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            sizes[cancer_name] = len(df)
        except FileNotFoundError:
            print(f"  [warn] fichier clinique introuvable pour '{cancer_name}', taille inconnue")
            sizes[cancer_name] = 0
    return sizes


# --------------------------------------------------------------------------- #
# Chargement d'une source
# --------------------------------------------------------------------------- #

def load_ref_source(source: Source, cancer_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Charge une source de type 'ref' : un gros CSV, filtré par cancer/task_id."""
    df_all = pd.read_csv(source.path)
    if source.learner_id:
        df_all = df_all[df_all["learner_id"] == source.learner_id]

    data = {}
    for cancer_name in cancer_names:
        sub = df_all[df_all["task_id"] == cancer_name][
            ["iteration"] + METRIC_COLUMNS
        ].rename(columns={"iteration": "fold"})
        sub = sub.copy()
        sub["fold"] = sub["fold"] - 1  # les indices R commencent à 1
        data[cancer_name] = sub
    return data


def load_file_source(source: Source, cancer_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Charge une source de type 'file' : un CSV par cancer."""
    data = {}
    for cancer_name in cancer_names:
        path = os.path.join(source.results_dir, f"{source.file_basename}{cancer_name}.csv")
        try:
            data[cancer_name] = pd.read_csv(path)[["fold"] + METRIC_COLUMNS]
        except FileNotFoundError:
            print(f"  [warn] pas de fichier pour la source '{source.name}' / cancer '{cancer_name}'")
    return data


def load_source(source: Source, cancer_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Dispatch vers le bon loader selon source.reader."""
    if source.reader == "ref":
        return load_ref_source(source, cancer_names)
    return load_file_source(source, cancer_names)


def load_all_sources(
    sources: List[Source], cancer_names: List[str]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Charge toutes les sources. Retourne {nom_source: {cancer: df}}."""
    return {source.name: load_source(source, cancer_names) for source in sources}


# --------------------------------------------------------------------------- #
# Construction du dataframe long (pour les boxplots)
# --------------------------------------------------------------------------- #

def build_long_dataframe(
    all_data: Dict[str, Dict[str, pd.DataFrame]], cancer_names: List[str]
) -> pd.DataFrame:
    """
    Transforme {source: {cancer: df}} en un dataframe long avec les colonnes
    Value, Metric, Source, Cancer, prêt pour les boxplots seaborn.
    Aucun merge/alignement entre sources n'est nécessaire ici : chaque source
    apporte ses propres valeurs par fold, on compare des distributions.
    """
    rows = []
    for source_name, per_cancer in all_data.items():
        for cancer_name in cancer_names:
            df = per_cancer.get(cancer_name)
            if df is None or df.empty:
                continue
            for col in METRIC_COLUMNS:
                rows.append(pd.DataFrame({
                    "Value": df[col],
                    "Metric": METRIC_LABELS[col],
                    "Source": source_name,
                    "Cancer": cancer_name,
                }))
    if not rows:
        return pd.DataFrame(columns=["Value", "Metric", "Source", "Cancer"])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Diffs par rapport à une source de référence (nécessite un merge par fold)
# --------------------------------------------------------------------------- #

def compute_diffs_vs_baseline(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    cancer_names: List[str],
    baseline: str,
) -> pd.DataFrame:
    """
    Pour chaque source non-baseline, merge sur 'fold' avec la source baseline,
    par cancer, et calcule diff moyenne / std / MAD par métrique.
    Retourne un dataframe: cancer, source, metric, diff_mean, diff_std, diff_mad
    """
    if baseline not in all_data:
        raise ValueError(f"Source baseline '{baseline}' introuvable parmi les sources chargées")

    records = []
    baseline_data = all_data[baseline]

    for source_name, per_cancer in all_data.items():
        if source_name == baseline:
            continue
        for cancer_name in cancer_names:
            ref_df = baseline_data.get(cancer_name)
            new_df = per_cancer.get(cancer_name)
            if ref_df is None or new_df is None or ref_df.empty or new_df.empty:
                continue
            merged = pd.merge(ref_df, new_df, on="fold", suffixes=("_ref", "_new"))
            if merged.empty:
                continue
            for col in METRIC_COLUMNS:
                diff = merged[f"{col}_new"] - merged[f"{col}_ref"]
                records.append({
                    "cancer": cancer_name,
                    "source": source_name,
                    "metric": METRIC_LABELS[col],
                    "diff_mean": diff.mean(),
                    "diff_std": diff.std(),
                    "diff_mad": diff.abs().mean(),
                })
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_boxplots(long_df: pd.DataFrame, cancer_order: List[str], results_dir: str) -> None:
    """Un boxplot par métrique, cancers en x, une boîte par source (hue)."""
    for metric in long_df["Metric"].unique():
        subset = long_df[long_df["Metric"] == metric]
        order = [c for c in cancer_order if c in subset["Cancer"].unique()]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            x="Cancer", y="Value", hue="Source", data=subset,
            palette="Set2", ax=ax, showfliers=False, order=order,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fname = f"all_cancer_boxplot_{metric.replace('-', '_').lower()}_clin.png"
        plt.savefig(os.path.join(results_dir, fname), bbox_inches="tight")
        plt.show()


def _grouped_bar(ax, cancers, groups: Dict[str, List[float]],
                  errs: Optional[Dict[str, List[float]]], title: str, ylabel: str) -> None:
    """Bar chart groupé: un groupe de barres par source, une barre par cancer dans le groupe."""
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    x = np.arange(len(cancers))
    for i, (name, values) in enumerate(groups.items()):
        offset = (i - (n_groups - 1) / 2) * width
        yerr = errs[name] if errs else None
        ax.bar(x + offset, values, width=width, yerr=yerr, capsize=3, label=name)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cancers, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_meandiff(diffs_df: pd.DataFrame, cancer_order: List[str], baseline: str, results_dir: str) -> None:
    """Diff moyenne (+/- std) par cancer, un panneau par métrique, barres groupées par source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["C-index", "IBS"]):
        subset = diffs_df[diffs_df["metric"] == metric]
        cancers = [c for c in cancer_order if c in subset["cancer"].unique()]
        groups, errs = {}, {}
        for source_name in subset["source"].unique():
            src_sub = subset[subset["source"] == source_name].set_index("cancer")
            groups[source_name] = [src_sub.loc[c, "diff_mean"] if c in src_sub.index else np.nan for c in cancers]
            errs[source_name] = [src_sub.loc[c, "diff_std"] if c in src_sub.index else 0 for c in cancers]
        _grouped_bar(ax, cancers, groups, errs, f"Diff moyenne {metric} (vs {baseline})", f"diff {metric}")
    plt.suptitle("Différence moyenne par cancer", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_all_cancers.png"), bbox_inches="tight")
    plt.show()


def plot_mad(diffs_df: pd.DataFrame, cancer_order: List[str], baseline: str, results_dir: str) -> None:
    """MAD par cancer, un panneau par métrique, barres groupées par source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["C-index", "IBS"]):
        subset = diffs_df[diffs_df["metric"] == metric]
        cancers = [c for c in cancer_order if c in subset["cancer"].unique()]
        groups = {}
        for source_name in subset["source"].unique():
            src_sub = subset[subset["source"] == source_name].set_index("cancer")
            groups[source_name] = [src_sub.loc[c, "diff_mad"] if c in src_sub.index else np.nan for c in cancers]
        _grouped_bar(ax, cancers, groups, None, f"MAD {metric} (vs {baseline})", f"MAD {metric}")
    plt.suptitle("Mean Absolute Difference par cancer", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mad_all_cancers.png"), bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def compare_all_cancers(
    sources: List[Source],
    results_dir: str,
    cancer_names: Optional[List[str]] = None,
    clinical_dir: str = "/env/cnrgh/proj/math_stats/scratch/hlegrand/data/clinical",
    plot_modes: Optional[List[str]] = None,
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare un nombre quelconque de sources de résultats à travers les cancers.

    Parameters
    ----------
    sources : list of Source
        Sources à comparer (2 ou plus). Chacune a son propre reader ("ref" ou "file").
    results_dir : str
        Dossier où enregistrer les figures produites.
    cancer_names : list of str, optionnel
        Si absent, découvert depuis la première source de type 'file'.
    clinical_dir : str
        Dossier contenant "{cancer}_clinical.pickle", utilisé pour trier les cancers par taille.
    plot_modes : list of str
        Un ou plusieurs parmi "boxplot", "meandiff", "mad".
    baseline : str, optionnel
        Nom de la source utilisée comme référence pour les plots "meandiff"/"mad".
        Obligatoire si ces modes sont demandés.
    """
    plot_modes = plot_modes or ["boxplot"]

    if cancer_names is None:
        file_sources = [s for s in sources if s.reader == "file"]
        if not file_sources:
            raise ValueError("Aucun cancer_names fourni et aucune source 'file' pour les découvrir")
        cancer_names = discover_cancer_names(file_sources[0])

    if not cancer_names:
        print("Aucun fichier de résultats trouvé.")
        return pd.DataFrame()

    print(f"Résultats trouvés pour: {cancer_names}")

    cancer_sizes = get_cancer_sizes(cancer_names, clinical_dir)
    cancer_order = sorted(cancer_sizes, key=cancer_sizes.get, reverse=True)
    print(cancer_order)
    print({c: cancer_sizes[c] for c in cancer_order})

    all_data = load_all_sources(sources, cancer_names)

    needs_baseline = "meandiff" in plot_modes or "mad" in plot_modes
    diffs_df = pd.DataFrame()
    if needs_baseline:
        if not baseline:
            raise ValueError("Les modes 'meandiff'/'mad' nécessitent --baseline")
        diffs_df = compute_diffs_vs_baseline(all_data, cancer_names, baseline)

    if "boxplot" in plot_modes:
        long_df = build_long_dataframe(all_data, cancer_names)
        plot_boxplots(long_df, cancer_order, results_dir)

    if "meandiff" in plot_modes and not diffs_df.empty:
        plot_meandiff(diffs_df, cancer_order, baseline, results_dir)

    if "mad" in plot_modes and not diffs_df.empty:
        plot_mad(diffs_df, cancer_order, baseline, results_dir)

    if not diffs_df.empty:
        summary = diffs_df.pivot_table(
            index=["cancer", "source"], columns="metric",
            values=["diff_mean", "diff_mad"]
        )
        print(summary)
        return diffs_df

    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare des résultats de modèles de survie entre un nombre quelconque de sources et de cancers."
    )
    parser.add_argument(
        "--source",
        action="append",
        nargs="+",
        metavar="KEY=VALUE",
        required=True,
        help=(
            "Définit une source de résultats. Répéter ce flag une fois par source. "
            "Clés: name (requis), type=ref|file (requis), "
            "pour type=ref: path=<csv>, learner_id=<id>; "
            "pour type=file: dir=<results_dir>, basename=<prefixe>. "
            "Exemple: --source name=vincent_r type=ref path=ref.csv learner_id=glmnet_ref"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="/env/cnrgh/proj/math_stats/scratch/hlegrand/results",
        help="Dossier où enregistrer les figures (default: %(default)s)",
    )
    parser.add_argument(
        "--clinical-dir",
        default="/env/cnrgh/proj/math_stats/scratch/hlegrand/data/clinical",
        help="Dossier contenant les {cancer}_clinical.pickle (default: %(default)s)",
    )
    parser.add_argument(
        "--cancers",
        nargs="+",
        default=None,
        help="Liste explicite de cancers. Si absent, découverte depuis la première source 'file'.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Nom de la source utilisée comme référence pour les plots 'meandiff'/'mad'.",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=["boxplot", "meandiff", "mad"],
        default=["boxplot"],
        metavar="MODE",
        help="Mode(s) de plot: boxplot, meandiff, mad (default: boxplot)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parsed_sources = [parse_source_spec(tokens) for tokens in args.source]
    compare_all_cancers(
        sources=parsed_sources,
        results_dir=args.results_dir,
        cancer_names=args.cancers,
        clinical_dir="data/clinical",
        plot_modes=args.plot,
        baseline=args.baseline,
    )