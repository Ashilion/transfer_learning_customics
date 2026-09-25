"""
Analyse et visualisation des résultats Optuna (nested CV, CustOMICS + CoxNet),
agrégées sur tous les cancers, pour les 4 méthodes (ridge, clinridge,
supridge, supclinridge), avec comparaison avec TL / sans TL.

Seuls les cancers pour lesquels les 4 méthodes ET les 2 conditions
(avec/sans TL) sont disponibles (fichiers CSV de résultats présents) sont
conservés, comme dans heatmap_tl_diff.py.

Convention de nommage des journaux Optuna (clarifiée avec l'utilisateur) :
  - sans TL : journal_{method}_{cancer}_fold{fold}.log
              study_name = journal_storage_multiprocess_{method}_{cancer}_fold{fold}
  - avec TL : journal_{method}_ft_{cancer}_fold{fold}.log
              study_name = ft_{method}_{cancer}_fold{fold}

Figures produites :
  - best_params_comparison_sans_tl.png / _avec_tl.png :
      pour chaque hyperparamètre, un boxplot avec les cancers en abscisse
      et une couleur par méthode (côte à côte), un graphique par condition TL.
  - param_importances_sans_tl.png / _avec_tl.png :
      importances des hyperparamètres, une ligne par méthode, une colonne par cancer.
  - hidden_dim_heatmap_{method}_{sans_tl|avec_tl}_{omic}.png :
      distribution des architectures par cancer, méthode et condition TL.

Cache CSV :
  Les données agrégées (importances, best_params, hidden_dims) sont sauvegardées
  en CSV dans out_dir après calcul. Utiliser --use_cache pour les recharger
  directement sans relire les journaux Optuna (utile pour ne retoucher que les
  graphiques).
"""

import argparse
import math
import os
import glob
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.importance import get_param_importances

from analyse_optuna_figs import ordered_categories_by_layers_and_size
sys.path.append("..")
# ======= Constantes ==========================================================================

METHODS = ["ridge", "clinridge", "supridge", "supclinridge"]

METHOD_LABELS = {
    "ridge": "CC",
    "supridge": "CC sup",
    "clinridge": "CC clin",
    "supclinridge": "CC sup clin",
}
# Préfixes des CSV de résultats finaux, utilisés uniquement pour déterminer
# quels cancers ont bien les 4 méthodes x {avec TL, sans TL} (cf. heatmap_tl_diff.py)
NO_TL_CSV_PREFIX = "ncv_custcox_optuna_paral_"
TL_CSV_PREFIX = "ncv_finetune_optuna_paral_"

# Architectures à exclure de la heatmap hidden_dim, par omic
# (clé = nom d'omic, valeur = set de tuples d'architecture à exclure)
EXCLUDED_ARCHITECTURES = {
    "mutation": {(512, 256, 128)},
}

# Tailles de police pour les figures hidden_dim heatmap / param_importances
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
ANNOT_FONTSIZE = 16
LEGEND_FONTSIZE = 18

# Noms des fichiers de cache CSV
IMPORTANCES_CSV_NAME = "cache_importances.csv"
BEST_PARAMS_CSV_NAME = "cache_best_params.csv"
HIDDEN_DIMS_CSV_NAME = "cache_hidden_dims.csv"


# ======= Argument Parsing ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Optuna multi-méthodes (CustOMICS + CoxNet), avec/sans TL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outer_splits", type=int, default=50, help="Nombre de folds externes.")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Méthodes à comparer.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Dossier contenant les CSV ncv_custcox_...csv / ncv_finetune_...csv "
             "(utilisé uniquement pour déterminer les cancers communs aux 4 méthodes).",
    )
    parser.add_argument(
        "--journal_dir",
        type=str,
        default="./optuna_journal",
        help="Dossier contenant les journaux Optuna (.log).",
    )
    parser.add_argument("--skip_importance", action="store_true", default=False,
                         help="Ne pas calculer les importances de paramètres (plus rapide).")
    parser.add_argument("--out_dir", type=str, default="figures", help="Dossier de sortie des figures.")
    parser.add_argument(
        "--omics_order",
        type=str,
        default="rna,mirna,cnv,mutation",
        help="Ordre des omics correspondant à hidden_dim_0, hidden_dim_1, ...",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=False,
        help="Charger les données agrégées depuis les CSV de cache (dans out_dir) "
             "au lieu de relire les journaux Optuna. Pratique pour ne retoucher "
             "que les graphiques sans tout recalculer. Si les CSV de cache sont "
             "absents, le script retombe automatiquement sur le calcul complet.",
    )
    return parser.parse_args()


# ======= Hidden dim <-> omics naming =========================================================

def hidden_dim_col_to_omics_name(col, omics_order):
    try:
        idx = int(col.rsplit("_", 1)[-1])
        return omics_order[idx]
    except (ValueError, IndexError):
        return col


def rename_hidden_dims_to_omics(d, omics_order):
    renamed = {}
    for key, value in d.items():
        new_key = hidden_dim_col_to_omics_name(key, omics_order) if key.startswith("hidden_dim") else key
        renamed[new_key] = value
    return renamed


def as_tuple(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value),)


# ======= Découverte des cancers communs aux 4 méthodes (avec + sans TL) =====================
# Reprend exactement la logique de heatmap_tl_diff.py : discover_common_cancers

def _cancers_for_prefix(results_dir, prefix, method):
    full_prefix = f"{prefix}{method}_"
    pattern = os.path.join(results_dir, f"{full_prefix}*.csv")
    files = glob.glob(pattern)
    return {
        os.path.basename(f).replace(full_prefix, "").replace(".csv", "")
        for f in files
    }


def discover_common_cancers(results_dir, methods):
    common = None
    for method in methods:
        no_tl_cancers = _cancers_for_prefix(results_dir, NO_TL_CSV_PREFIX, method)
        tl_cancers = _cancers_for_prefix(results_dir, TL_CSV_PREFIX, method)
        method_cancers = no_tl_cancers & tl_cancers
        common = method_cancers if common is None else common & method_cancers
    return sorted(common) if common else []


# ======= Chargement des studies Optuna (multi-méthodes, avec/sans TL) =======================

def load_studies_method(cancer_name, method, tl, n_outer, journal_dir, omics_order, do_importance=True):
    """
    Charge les studies Optuna d'un (cancer, méthode, condition TL) sur tous
    les folds externes disponibles. Les folds dont le journal est manquant
    sont ignorés (avec un warning), plutôt que de faire planter le script.
    """
    studies = []
    params_importances = []
    durations_importances = []
    best_params_tab = []

    for outer_fold in range(n_outer):
        if tl:
            study_name = f"ft_{method}_{cancer_name}_fold{outer_fold}"
            file_path = os.path.join(journal_dir, f"journal_{method}_ft_{cancer_name}_fold{outer_fold}.log")
        else:
            study_name = f"journal_storage_multiprocess_{method}_{cancer_name}_fold{outer_fold}"
            file_path = os.path.join(journal_dir, f"journal_{method}_{cancer_name}_fold{outer_fold}.log")

        if not os.path.exists(file_path):
            print(f"  [warn] journal manquant, fold ignoré : {file_path}")
            continue

        study = optuna.load_study(
            study_name=study_name,
            storage=JournalStorage(JournalFileBackend(file_path=file_path)),
        )
        studies.append(study)

        if do_importance:
            try:
                params_importances.append(
                    rename_hidden_dims_to_omics(get_param_importances(study), omics_order)
                )
                durations_importances.append(
                    rename_hidden_dims_to_omics(
                        get_param_importances(study, target=lambda t: t.duration.total_seconds()),
                        omics_order,
                    )
                )
            except Exception as e:
                print(f"  [warn] importances impossibles pour {study_name}: {e}")

        # Ignore les studies sans aucun trial COMPLETE
        try:
            best_trial = study.best_trial
        except ValueError:
            print(f"  [warn] aucun trial complété, fold ignoré : {study_name}")
            continue
        best_params = dict(best_trial.params)
        best_params["best_alpha"] = best_trial.user_attrs.get("best_alpha")
        best_params = rename_hidden_dims_to_omics(best_params, omics_order)
        best_params["Fold"] = outer_fold
        best_params_tab.append(best_params)

    return studies, best_params_tab, params_importances, durations_importances


# ======= Plot : meilleurs hyperparamètres, méthodes côte à côte =============================
def plot_best_params_comparison(best_params_df, tl_label, methods, omics_order, out_dir):
    """
    Un subplot par hyperparamètre (hors omics/Cancer/Fold/Method/TL), avec les
    cancers en abscisse et une couleur par méthode (boxplots côte à côte).
    Disposition en grille à 2 colonnes, boxplots deux fois plus fins.
    """
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

    n_cols_grid = 2
    n_rows_grid = math.ceil(len(cols) / n_cols_grid)

    fig, axes = plt.subplots(
        n_rows_grid, n_cols_grid, figsize=(10 * n_cols_grid, 4 * n_rows_grid)  # 10 au lieu de 16
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        data = best_params_df.dropna(subset=[col])
        sns.boxplot(
            data=data,
            x="Cancer",
            y=col,
            hue="Method",
            hue_order=methods,
            width=0.8,
            gap=0.1,
            ax=ax,
        )
        ax.set_title(col, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Cancer", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(col, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        if col in ("lr", "beta", "delta_min"):
            ax.set_yscale("log")

        handles, labels = ax.get_legend_handles_labels()
        new_labels = [METHOD_LABELS.get(lbl, lbl) for lbl in labels]
        ax.legend(
            handles,
            new_labels,
            title="Méthode",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_FONTSIZE,
        )

    # Cache les axes vides si nombre impair de colonnes
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Meilleurs hyperparamètres par cancer et méthode — {tl_label}", fontsize=TITLE_FONTSIZE + 2, y=1.01)
    fig.tight_layout()

    suffix = "avec_tl" if tl_label == "Avec TL" else "sans_tl"
    out_path = os.path.join(out_dir, f"best_params_comparison_{suffix}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")

# ======= Plot : importances des hyperparamètres, une ligne par méthode ======================
def plot_param_importances_by_method(importances_df, tl_label, methods, out_dir, aspect=1.0):
    """
    Grille complète (row=Method, col=Cancer) sans col_wrap : chaque ligne
    correspond à une méthode (4 lignes), chaque colonne à un cancer.
    Pas de cellule vide (contrairement à un col_wrap non multiple du nb de
    cancers), et margin_titles=True évite de répéter les titres sur chaque
    sous-graphe, ce qui réduit l'espace blanc.
    """
    subset = importances_df[importances_df["TL"] == tl_label]
    if subset.empty:
        return

    n_cancers = subset["Cancer"].nunique()

    g = sns.catplot(
        data=subset,
        x="Parameter", y="Importance",
        row="Method", col="Cancer",
        row_order=methods,
        kind="box",
        sharey=True,
        height=2.6,
        aspect=aspect,
        margin_titles=True,
    )
    g.set_titles(row_template="{row_name}", col_template="{col_name}", size=LABEL_FONTSIZE)
    g.set_axis_labels("", "Importance", fontsize=LABEL_FONTSIZE)

    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=60)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    g.figure.suptitle(f"Importances des hyperparamètres — {tl_label}", fontsize=TITLE_FONTSIZE, y=1.0)
    g.figure.subplots_adjust(top=0.93, wspace=0.05, hspace=0.15)

    suffix = "avec_tl" if tl_label == "Avec TL" else "sans_tl"
    out_path = os.path.join(out_dir, f"param_importances_{suffix}.png")
    g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Sauvegardé : {out_path}")


# ======= Plot : heatmap des architectures (hidden_dim) =======================================
def plot_hidden_dim_heatmap(mat, omic, method, tl_label, out_dir):
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * mat.shape[1] + 4), max(6, 0.6 * mat.shape[0] + 2)))
    sns.heatmap(
        mat, cmap="Blues", annot=True, fmt=".0f", ax=ax,
        annot_kws={"fontsize": ANNOT_FONTSIZE},
        cbar_kws={"label": "Count"},
    )
    ax.set_title(f"{omic} — {method} — {tl_label}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Architecture", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Cancer", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=45)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, rotation=0)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label("Count", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()

    suffix = "avec_tl" if tl_label == "Avec TL" else "sans_tl"
    out_path = os.path.join(out_dir, f"hidden_dim_heatmap_{method}_{suffix}_{omic}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Main =================================================================================

def main():
    args = parse_args()

    n_outer = args.outer_splits
    methods = args.methods
    do_importance = not args.skip_importance
    out_dir = args.out_dir
    journal_dir = args.journal_dir
    omics_order = [s.strip() for s in args.omics_order.split(",") if s.strip()]
    os.makedirs(out_dir, exist_ok=True)

    importances_csv = os.path.join(out_dir, IMPORTANCES_CSV_NAME)
    best_params_csv = os.path.join(out_dir, BEST_PARAMS_CSV_NAME)
    hidden_csv = os.path.join(out_dir, HIDDEN_DIMS_CSV_NAME)

    print(f"\n{'='*60}")
    print(f"  Méthodes          : {methods}")
    print(f"  Outer folds       : {n_outer}")
    print(f"  Importances       : {do_importance}")
    print(f"  Omics order       : {omics_order}")
    print(f"  Use cache         : {args.use_cache}")
    print(f"{'='*60}\n")

    cache_available = all(os.path.exists(p) for p in (importances_csv, best_params_csv, hidden_csv))

    if args.use_cache and cache_available:
        print("Chargement des données agrégées depuis le cache CSV (pas de relecture des journaux Optuna)...\n")
        importances_df = pd.read_csv(importances_csv)
        best_params_df = pd.read_csv(best_params_csv)
        hidden_df = pd.read_csv(hidden_csv)
        if not hidden_df.empty and "ArchitectureTuple" in hidden_df.columns:
            hidden_df["ArchitectureTuple"] = hidden_df["ArchitectureTuple"].apply(ast.literal_eval)
    else:
        if args.use_cache and not cache_available:
            print("[warn] --use_cache demandé mais cache CSV introuvable, recalcul complet depuis les journaux Optuna.\n")

        common_cancers = discover_common_cancers(args.results_dir, methods)
        print(f"Cancers avec les {len(methods)} méthodes (avec + sans TL) : {common_cancers}\n")

        if not common_cancers:
            print("Aucun cancer commun trouvé, arrêt.")
            return

        all_importances = []
        all_best_params = []
        all_hidden_dims = []

        for method in methods:
            for tl in (False, True):
                tl_label = "Avec TL" if tl else "Sans TL"
                for cancer in common_cancers:
                    print(f"[{method} | {tl_label}] {cancer}")

                    _, best_params_tab, params_importances, _ = load_studies_method(
                        cancer, method, tl, n_outer, journal_dir, omics_order, do_importance=do_importance
                    )

                    if not best_params_tab:
                        print(f"  [warn] aucune donnée chargée pour {method}/{cancer}/{tl_label}, ignoré")
                        continue

                    for fold_idx, imp in enumerate(params_importances):
                        for param, value in imp.items():
                            all_importances.append({
                                "Cancer": cancer, "Method": method, "TL": tl_label,
                                "Fold": fold_idx, "Parameter": param, "Importance": value,
                            })

                    for params in best_params_tab:
                        row = params.copy()
                        row["Cancer"] = cancer
                        row["Method"] = method
                        row["TL"] = tl_label
                        all_best_params.append(row)

                        for col in params:
                            if col in omics_order:
                                architecture = as_tuple(params[col])
                                excluded = EXCLUDED_ARCHITECTURES.get(col, set())
                                if architecture in excluded:
                                    continue
                                all_hidden_dims.append({
                                    "Cancer": cancer, "Method": method, "TL": tl_label,
                                    "Omics": col, "ArchitectureTuple": architecture,
                                    "Architecture": "x".join(map(str, architecture)),
                                })

        importances_df = pd.DataFrame(all_importances)
        best_params_df = pd.DataFrame(all_best_params)
        hidden_df = pd.DataFrame(all_hidden_dims)

        importances_df.to_csv(importances_csv, index=False)
        best_params_df.to_csv(best_params_csv, index=False)
        hidden_df.to_csv(hidden_csv, index=False)
        print(f"\nCache CSV sauvegardé : {importances_csv}, {best_params_csv}, {hidden_csv}\n")

    # ---- 1) Meilleurs hyperparamètres : méthodes côte à côte, un graphique par condition TL
    for tl_label in ("Sans TL", "Avec TL"):
        subset = best_params_df[best_params_df["TL"] == tl_label]
        if subset.empty:
            print(f"  [warn] pas de données pour {tl_label}, best_params ignoré")
            continue
        plot_best_params_comparison(subset, tl_label, methods, omics_order, out_dir)

    # ---- 2) Importances des hyperparamètres, une ligne par méthode, une colonne par cancer
    if do_importance and not importances_df.empty:
        plot_param_importances_by_method(importances_df, "Sans TL", methods, out_dir, aspect=1.0)
        plot_param_importances_by_method(importances_df, "Avec TL", methods, out_dir, aspect=0.6)

    # ---- 3) Heatmaps des architectures (hidden_dim), par méthode / condition TL / omic
    if not hidden_df.empty:
        heat = (
            hidden_df
            .groupby(["Cancer", "Method", "TL", "Omics", "Architecture"])
            .size()
            .reset_index(name="Count")
        )

        for method in methods:
            for tl_label in ("Sans TL", "Avec TL"):
                sub_hidden = hidden_df[(hidden_df["Method"] == method) & (hidden_df["TL"] == tl_label)]
                if sub_hidden.empty:
                    continue

                for omic in sub_hidden["Omics"].unique():
                    tmp = heat[
                        (heat["Method"] == method) & (heat["TL"] == tl_label) & (heat["Omics"] == omic)
                    ]
                    mat = tmp.pivot(index="Cancer", columns="Architecture", values="Count").fillna(0)

                    arch_order_tuple = ordered_categories_by_layers_and_size(
                        sub_hidden.loc[sub_hidden["Omics"] == omic, "ArchitectureTuple"]
                    )
                    arch_order = ["x".join(map(str, a)) for a in arch_order_tuple]
                    mat = mat.reindex(columns=arch_order, fill_value=0)

                    plot_hidden_dim_heatmap(mat, omic, method, tl_label, out_dir)


if __name__ == "__main__":
    main()