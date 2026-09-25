"""
Compte, pour chaque cancer, le nombre de lignes (folds) où `n_nonzero_coefs`
vaut 0 (i.e. le modèle n'a sélectionné aucune variable), et affiche un
barplot du nombre de "beta nuls" par cancer.

Une seule source en entrée, lue avec l'un des deux "readers" suivants
(repris du script multiple_method_compare_all.py) :
  - "ref"  : un unique gros CSV contenant tous les cancers, filtré par
             `learner_id` (optionnel) puis par `task_id`.
  - "file" : un CSV par cancer, nommé "{file_basename}{cancer_name}.csv"
             dans un dossier de résultats.

Exemples d'utilisation (CLI) :
    python barplot_nzero_coefs.py \
        --type ref --path /path/ref_results.csv --learner-id glmnet_ref

    python barplot_nzero_coefs.py \
        --type file --dir /path/results --basename outer_cv_results_vvh_
"""

import argparse
import glob
import os
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COL_NZERO = "n_nonzero_coefs"


# --------------------------------------------------------------------------- #
# Découverte des cancers disponibles (reader "file")
# --------------------------------------------------------------------------- #

def discover_cancer_names(results_dir: str, file_basename: str) -> List[str]:
    """Liste les cancers disponibles à partir des fichiers présents dans results_dir."""
    pattern = os.path.join(results_dir, f"{file_basename}*.csv")
    files = sorted(glob.glob(pattern))
    if "ridge" not in file_basename:
        files = [f for f in files if "ridge" not in os.path.basename(f)]
    return [
        os.path.basename(f).replace(file_basename, "").replace(".csv", "")
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
# Chargement des données
# --------------------------------------------------------------------------- #

def load_ref_source(
    path: str, cancer_names: List[str], learner_id: Optional[str]
) -> Dict[str, pd.DataFrame]:
    """Charge un gros CSV unique, filtré par learner_id (optionnel) puis par task_id."""
    df_all = pd.read_csv(path)
    if learner_id:
        df_all = df_all[df_all["learner_id"] == learner_id]

    data = {}
    for cancer_name in cancer_names:
        sub = df_all[df_all["task_id"] == cancer_name][[COL_NZERO]]
        data[cancer_name] = sub
    return data


def load_file_source(
    results_dir: str, file_basename: str, cancer_names: List[str]
) -> Dict[str, pd.DataFrame]:
    """Charge un CSV par cancer."""
    data = {}
    for cancer_name in cancer_names:
        path = os.path.join(results_dir, f"{file_basename}{cancer_name}.csv")
        try:
            data[cancer_name] = pd.read_csv(path)[[COL_NZERO]]
        except FileNotFoundError:
            print(f"  [warn] pas de fichier pour le cancer '{cancer_name}'")
        except KeyError:
            print(f"  [warn] colonne '{COL_NZERO}' absente pour le cancer '{cancer_name}'")
    return data


# --------------------------------------------------------------------------- #
# Comptage des beta nuls
# --------------------------------------------------------------------------- #

def count_zero_nzero_coefs(
    all_data: Dict[str, pd.DataFrame], cancer_names: List[str]
) -> pd.DataFrame:
    """
    Pour chaque cancer, compte le nombre de lignes où n_nonzero_coefs == 0,
    ainsi que le nombre total de lignes (folds).
    Retourne un dataframe: cancer, n_zero, n_total.
    """
    records = []
    for cancer_name in cancer_names:
        df = all_data.get(cancer_name)
        if df is None or df.empty:
            continue
        n_zero = int((df[COL_NZERO] == 0).sum())
        n_total = len(df)
        records.append({"cancer": cancer_name, "n_zero": n_zero, "n_total": n_total})
    return pd.DataFrame.from_records(records)


def count_below_tresh(
    all_data: Dict[str, pd.DataFrame], cancer_names: List[str], tresh:int
) -> pd.DataFrame:

    records = []
    for cancer_name in cancer_names:
        df = all_data.get(cancer_name)
        if df is None or df.empty:
            continue
        n_zero = int((df[COL_NZERO] < tresh).sum())
        n_total = len(df)
        records.append({"cancer": cancer_name, "n_zero": n_zero, "n_total": n_total})
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_zero_barplot(counts_df: pd.DataFrame, cancer_order: List[str], results_dir: str) -> None:
    """Barplot du nombre de beta nuls (n_nonzero_coefs == 0) par cancer."""
    order = [c for c in cancer_order if c in counts_df["cancer"].unique()]
    subset = counts_df.set_index("cancer").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="cancer", y="n_zero", data=subset, color="steelblue", ax=ax)

    
    for i, row in subset.iterrows():
        ax.text(
            i, row["n_zero"] + 0.05 * max(subset["n_zero"].max(), 1),
            f"{row['n_zero']}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title(f"Nombre de folds avec {COL_NZERO} = 0, par cancer")
    ax.set_xlabel("")
    ax.set_ylabel("Nombre de folds tous les beta nuls")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "barplot_nzero_coefs.png"), bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def compare_zero_coefs(
    reader: str,
    results_dir: str,
    path: Optional[str] = None,
    learner_id: Optional[str] = None,
    source_dir: Optional[str] = None,
    file_basename: Optional[str] = None,
    cancer_names: Optional[List[str]] = None,
    clinical_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compte le nombre de beta nuls (n_nonzero_coefs == 0) par cancer pour une
    source unique, et produit un barplot.

    Parameters
    ----------
    reader : "ref" ou "file"
    results_dir : dossier où enregistrer la figure
    path : chemin du CSV (reader="ref")
    learner_id : filtre optionnel sur learner_id (reader="ref")
    source_dir, file_basename : dossier et préfixe des fichiers (reader="file")
    cancer_names : liste explicite de cancers ; si absente et reader="file",
        découverte automatiquement depuis les fichiers présents.
    clinical_dir : si fourni, utilisé pour trier les cancers par taille
        (sinon tri alphabétique).
    """
    if reader not in ("ref", "file"):
        raise ValueError(f"Reader inconnu '{reader}'")
    if reader == "ref" and not path:
        raise ValueError("reader='ref' nécessite --path")
    if reader == "file" and not (source_dir and file_basename):
        raise ValueError("reader='file' nécessite --dir et --basename")

    if cancer_names is None:
        if reader != "file":
            raise ValueError("--cancers est requis quand reader='ref'")
        cancer_names = discover_cancer_names(source_dir, file_basename)

    if not cancer_names:
        print("Aucun fichier de résultats trouvé.")
        return pd.DataFrame()

    print(f"Cancers trouvés: {cancer_names}")

    if clinical_dir:
        cancer_sizes = get_cancer_sizes(cancer_names, clinical_dir)
        cancer_order = sorted(cancer_sizes, key=cancer_sizes.get, reverse=True)
    else:
        cancer_order = sorted(cancer_names)

    if reader == "ref":
        all_data = load_ref_source(path, cancer_names, learner_id)
    else:
        all_data = load_file_source(source_dir, file_basename, cancer_names)

    counts_df = count_zero_nzero_coefs(all_data, cancer_names)
    # counts_df = count_below_tresh(all_data, cancer_names, 3)
    if counts_df.empty:
        print("Aucune donnée exploitable.")
        return counts_df

    print(counts_df.set_index("cancer").loc[[c for c in cancer_order if c in counts_df["cancer"].values]])

    plot_zero_barplot(counts_df, cancer_order, results_dir)
    return counts_df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Barplot du nombre de beta nuls (n_nonzero_coefs == 0) par cancer, pour une source de résultats."
    )
    parser.add_argument("--type", choices=["ref", "file"], default="file", help="Type de source: ref ou file")
    parser.add_argument("--path", default=None, help="Chemin du CSV unique (reader=ref)")
    parser.add_argument("--learner-id", default=None, help="Filtre optionnel sur learner_id (reader=ref)")
    parser.add_argument("--dir", default=None, help="Dossier des résultats (reader=file)")
    parser.add_argument("--basename", default=None, help="Préfixe des fichiers, ex: outer_cv_results_vvh_ (reader=file)")
    parser.add_argument("--cancers", nargs="+", default=None, help="Liste explicite de cancers")
    parser.add_argument(
        "--clinical-dir", default="/env/cnrgh/proj/math_stats/scratch/hlegrand/data/clinical",
        help="Dossier contenant les {cancer}_clinical.pickle, pour trier par taille (optionnel)",
    )
    parser.add_argument(
        "--results-dir", default=".",
        help="Dossier où enregistrer la figure (default: dossier courant)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_zero_coefs(
        reader=args.type,
        results_dir=args.results_dir,
        path=args.path,
        learner_id=args.learner_id,
        source_dir=args.dir,
        file_basename=args.basename,
        cancer_names=args.cancers,
        clinical_dir=args.clinical_dir,
    )