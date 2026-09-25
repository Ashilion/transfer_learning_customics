"""
Analyse et visualisation des résultats Optuna (nested CV, CustOMICS + CoxNet),
pour UN SEUL cancer et UNE SEULE méthode, en faisant varier le name_suffix
de fine-tuning (Drop vs Impute, à différents taux d'entraînement XX et
d'évaluation YY).

Ce script est un dérivé de analyze_optuna_multi_methods.py (script 1),
adapté pour explorer non pas {méthode x cancer x avec/sans TL} mais
{type (drop/impute) x XX (taux d'entraînement) x YY (taux d'évaluation)},
à cancer et méthode fixés.

Convention de nommage des journaux Optuna (confirmée par l'utilisateur à
partir d'un exemple réel) :

  - fichier     : journal_{name_suffix}ft_{cancer}_fold{fold}.log
                  ex : journal_tldrop30_30_ft_KIRP_fold14.log
  - study_name  : ft_{name_suffix}{cancer}_fold{fold}
                  ex : ft_tldrop30_30_KIRP_fold14

  name_suffix est de la forme "tl(drop|imp)XX_YY_" où :
    - XX = taux de données manquantes utilisé pendant le fine-tuning
    - YY = taux de données manquantes utilisé pour l'évaluation (fold)

  Remarque : la méthode n'apparaît PAS dans le nom des journaux (à la
  différence du script 1). --method est donc conservé uniquement comme
  étiquette pour les sorties (CSV / titres de figures) ; il n'intervient
  pas dans la construction des chemins de fichiers ni des study_name. Si
  journal_dir contient plusieurs méthodes mélangées sous ce même schéma
  de nommage, il faut les séparer soi-même (sous-dossiers par méthode,
  par exemple) avant d'utiliser ce script.

Figures produites (dans out_dir) :
  - best_params_comparison_train{XX}.png :
      pour chaque hyperparamètre, un boxplot avec les taux d'évaluation
      (YY) en abscisse et une couleur par type (Drop/Impute), un
      graphique par taux d'entraînement (XX).
  - param_importances_train{XX}.png :
      importances des hyperparamètres, une ligne par type (Drop/Impute),
      une colonne par taux d'évaluation (YY), un graphique par XX.
  - hidden_dim_heatmap_{type}_train{XX}_{omic}.png :
      distribution des architectures par taux d'évaluation (YY), pour un
      type et un XX donnés.

Cache CSV :
  Les données agrégées (importances, best_params, hidden_dims) sont
  sauvegardées en CSV dans out_dir après calcul. Utiliser --use_cache
  pour les recharger directement sans relire les journaux Optuna (utile
  pour ne retoucher que les graphiques).
"""

import argparse
import math
import os
import glob
import re
import ast
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.importance import get_param_importances

sys.path.append("..")
from analyse_optuna_figs import ordered_categories_by_layers_and_size

# ======= Constantes ==========================================================================

TYPE_LABELS = {"drop": "Drop", "impute": "Impute"}
TYPE_PALETTE = {"Drop": "#4C72B0", "Impute": "#DD8452"}

# Regex de découverte des name_suffix dans les noms de fichiers journaux
# Matche "tldrop30_50_" ou "tlimp50_30_" (avec le "_" final inclus dans le match)
NAME_SUFFIX_PATTERN = re.compile(r"tl(drop|imp)(\d+)_(\d+)_")

# Formats de nommage des journaux / study_name (cf. docstring en tête de fichier)
# NB : la méthode n'apparaît pas dans ce schéma de nommage.
NAME_SUFFIX_FILE_FORMAT = "journal_{name_suffix}ft_{cancer}_fold{fold}.log"
NAME_SUFFIX_STUDY_FORMAT = "ft_{name_suffix}{cancer}_fold{fold}"

# Architectures à exclure de la heatmap hidden_dim, par omic
EXCLUDED_ARCHITECTURES = {
    "mutation": {(512, 256, 128)},
}

# Tailles de police pour les figures
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
ANNOT_FONTSIZE = 16
LEGEND_FONTSIZE = 18

# Noms des fichiers de cache CSV
IMPORTANCES_CSV_NAME = "cache_importances_missingmod.csv"
BEST_PARAMS_CSV_NAME = "cache_best_params_missingmod.csv"
HIDDEN_DIMS_CSV_NAME = "cache_hidden_dims_missingmod.csv"


# ======= Argument Parsing ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse Optuna fine-tuning (Drop vs Impute, XX/YY variables), "
                    "pour un seul cancer et une seule méthode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cancer_type", type=str, required=True, help="Nom du cancer (ex: KIRP).")
    parser.add_argument("--method", type=str, required=True,
                         help="Méthode (ex: ridge, clinridge, supridge, supclinridge).")
    parser.add_argument("--outer_splits", type=int, default=50, help="Nombre de folds externes.")
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
             "au lieu de relire les journaux Optuna.",
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


# ======= Découverte des name_suffix disponibles pour {method, cancer} =======================

def discover_configs(journal_dir, cancer):
    """
    Scanne journal_dir à la recherche des journaux correspondant à
    {cancer} (schéma journal_{name_suffix}ft_{cancer}_fold{fold}.log), et
    en extrait tous les name_suffix (tl(drop|imp)XX_YY_) disponibles, avec
    leur type / train_rate (XX) / test_rate (YY).

    Retourne un dict {name_suffix: {"type": ..., "train_rate": ..., "test_rate": ...}}
    trié par (type, train_rate, test_rate).
    """
    pattern = os.path.join(journal_dir, f"journal_tl*_ft_{cancer}_fold*.log")
    files = glob.glob(pattern)
    print("pattern", pattern)
    configs = {}
    for f in files:
        base = os.path.basename(f)
        m = NAME_SUFFIX_PATTERN.search(base)
        if m is None:
            continue
        name_suffix = m.group(0)  # ex: "tlimp50_30_"
        raw_type, xx, yy = m.groups()
        configs[name_suffix] = {
            "type": "impute" if raw_type == "imp" else "drop",
            "train_rate": int(xx),
            "test_rate": int(yy),
        }

    return dict(
        sorted(configs.items(), key=lambda kv: (kv[1]["type"], kv[1]["train_rate"], kv[1]["test_rate"]))
    )


# ======= Chargement des studies Optuna pour une config (type, XX, YY) donnée ================

def load_studies_config(cancer, name_suffix, n_outer, journal_dir, omics_order, do_importance=True):
    """
    Charge les studies Optuna d'une config (name_suffix) sur tous les folds
    externes disponibles. Les folds dont le journal est manquant sont
    ignorés (avec un warning), plutôt que de faire planter le script.
    """
    studies = []
    params_importances = []
    durations_importances = []
    best_params_tab = []

    for outer_fold in range(n_outer):
        study_name = NAME_SUFFIX_STUDY_FORMAT.format(name_suffix=name_suffix,
                                                       cancer=cancer, fold=outer_fold)
        file_name = NAME_SUFFIX_FILE_FORMAT.format(name_suffix=name_suffix,
                                                     cancer=cancer, fold=outer_fold)
        file_path = os.path.join(journal_dir, file_name)

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


# ======= Plot : meilleurs hyperparamètres, un graphique par XX ==============================

def plot_best_params_comparison(best_params_df, train_rate, omics_order, out_dir):
    """
    Un subplot par hyperparamètre (hors omics/Cancer/Fold/TestRate/Type/...),
    avec les taux d'évaluation (YY) en abscisse et une couleur par type
    (Drop/Impute). Disposition en grille à 2 colonnes.
    """
    meta_cols = ("Cancer", "Method", "Fold", "Type", "TrainRate", "TestRate", "ConfigLabel")
    cols = [c for c in best_params_df.columns if c not in omics_order and c not in meta_cols]

    usable_cols = []
    for c in cols:
        if best_params_df[c].notna().sum() == 0:
            print(f"  [warn] colonne '{c}' entièrement vide pour XX={train_rate}, ignorée")
            continue
        usable_cols.append(c)
    cols = usable_cols

    if not cols:
        print(f"  [warn] aucun hyperparamètre à tracer pour XX={train_rate}")
        return

    n_cols_grid = 2
    n_rows_grid = math.ceil(len(cols) / n_cols_grid)

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(10 * n_cols_grid, 4 * n_rows_grid))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        data = best_params_df.dropna(subset=[col])
        sns.boxplot(
            data=data,
            x="TestRate",
            y=col,
            hue="Type",
            hue_order=["Drop", "Impute"],
            palette=TYPE_PALETTE,
            width=0.8,
            gap=0.1,
            ax=ax,
        )
        ax.set_title(col, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Taux de données manquantes (test, YY)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(col, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        if col in ("lr", "beta", "delta_min"):
            ax.set_yscale("log")

        ax.legend(
            title="Type",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_FONTSIZE,
        )

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Meilleurs hyperparamètres par taux d'évaluation — fine-tuning {train_rate}%",
        fontsize=TITLE_FONTSIZE + 2, y=1.01,
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"best_params_comparison_train{train_rate}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Plot : importances des hyperparamètres, une ligne par type =========================

def plot_param_importances_by_train_rate(importances_df, train_rate, out_dir, aspect=1.0):
    """
    Grille complète (row=Type, col=TestRate) sans col_wrap, pour un XX donné.
    """
    subset = importances_df[importances_df["TrainRate"] == train_rate]
    if subset.empty:
        return

    g = sns.catplot(
        data=subset,
        x="Parameter", y="Importance",
        row="Type", col="TestRate",
        row_order=["Drop", "Impute"],
        kind="box",
        sharey=True,
        height=2.6,
        aspect=aspect,
        margin_titles=True,
    )
    g.set_titles(row_template="{row_name}", col_template="YY={col_name}%", size=LABEL_FONTSIZE)
    g.set_axis_labels("", "Importance", fontsize=LABEL_FONTSIZE)

    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=60)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    g.figure.suptitle(
        f"Importances des hyperparamètres — fine-tuning {train_rate}%",
        fontsize=TITLE_FONTSIZE, y=1.0,
    )
    g.figure.subplots_adjust(top=0.90, wspace=0.05, hspace=0.15)

    out_path = os.path.join(out_dir, f"param_importances_train{train_rate}.png")
    g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Sauvegardé : {out_path}")


# ======= Plot : heatmap des architectures (hidden_dim) =======================================

def plot_hidden_dim_heatmap(mat, omic, type_label, train_rate, out_dir):
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * mat.shape[1] + 4), max(6, 0.6 * mat.shape[0] + 2)))
    sns.heatmap(
        mat, cmap="Blues", annot=True, fmt=".0f", ax=ax,
        annot_kws={"fontsize": ANNOT_FONTSIZE},
        cbar_kws={"label": "Count"},
    )
    ax.set_title(f"{omic} — {type_label} — fine-tuning {train_rate}%", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Architecture", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Taux de données manquantes (test, YY)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=45)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, rotation=0)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label("Count", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()

    out_path = os.path.join(
        out_dir, f"hidden_dim_heatmap_{type_label.lower()}_train{train_rate}_{omic}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ======= Main =================================================================================

def main():
    args = parse_args()

    n_outer = args.outer_splits
    cancer = args.cancer_type
    method = args.method
    do_importance = not args.skip_importance
    out_dir = args.out_dir
    journal_dir = args.journal_dir
    omics_order = [s.strip() for s in args.omics_order.split(",") if s.strip()]
    os.makedirs(out_dir, exist_ok=True)

    importances_csv = os.path.join(out_dir, IMPORTANCES_CSV_NAME)
    best_params_csv = os.path.join(out_dir, BEST_PARAMS_CSV_NAME)
    hidden_csv = os.path.join(out_dir, HIDDEN_DIMS_CSV_NAME)

    print(f"\n{'='*60}")
    print(f"  Cancer            : {cancer}")
    print(f"  Méthode           : {method}")
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

        configs = discover_configs(journal_dir, cancer)
        print(f"Configs (name_suffix) trouvées pour {cancer} : {list(configs.keys())}\n")

        if not configs:
            print("Aucune config (name_suffix) trouvée, arrêt.")
            return

        all_importances = []
        all_best_params = []
        all_hidden_dims = []

        for name_suffix, meta in configs.items():
            type_label = TYPE_LABELS[meta["type"]]
            train_rate = meta["train_rate"]
            test_rate = meta["test_rate"]
            config_label = f"{type_label} {train_rate}/{test_rate}"

            print(f"[{config_label}] name_suffix={name_suffix}")

            _, best_params_tab, params_importances, _ = load_studies_config(
                cancer, name_suffix, n_outer, journal_dir, omics_order, do_importance=do_importance
            )

            if not best_params_tab:
                print(f"  [warn] aucune donnée chargée pour {config_label}, ignoré")
                continue

            for fold_idx, imp in enumerate(params_importances):
                for param, value in imp.items():
                    all_importances.append({
                        "Cancer": cancer, "Method": method, "Type": type_label,
                        "TrainRate": train_rate, "TestRate": test_rate,
                        "ConfigLabel": config_label, "Fold": fold_idx,
                        "Parameter": param, "Importance": value,
                    })

            for params in best_params_tab:
                row = params.copy()
                row["Cancer"] = cancer
                row["Method"] = method
                row["Type"] = type_label
                row["TrainRate"] = train_rate
                row["TestRate"] = test_rate
                row["ConfigLabel"] = config_label
                all_best_params.append(row)

                for col in params:
                    if col in omics_order:
                        architecture = as_tuple(params[col])
                        excluded = EXCLUDED_ARCHITECTURES.get(col, set())
                        if architecture in excluded:
                            continue
                        all_hidden_dims.append({
                            "Cancer": cancer, "Method": method, "Type": type_label,
                            "TrainRate": train_rate, "TestRate": test_rate,
                            "ConfigLabel": config_label,
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

    if best_params_df.empty:
        print("Aucune donnée disponible, arrêt avant les graphiques.")
        return

    train_rates = sorted(best_params_df["TrainRate"].unique())

    # ---- 1) Meilleurs hyperparamètres : un graphique par XX (taux d'entraînement)
    for train_rate in train_rates:
        subset = best_params_df[best_params_df["TrainRate"] == train_rate]
        if subset.empty:
            continue
        plot_best_params_comparison(subset, train_rate, omics_order, out_dir)

    # ---- 2) Importances des hyperparamètres, une ligne par type, une colonne par YY, une figure par XX
    if do_importance and not importances_df.empty:
        for train_rate in train_rates:
            plot_param_importances_by_train_rate(importances_df, train_rate, out_dir, aspect=0.8)

    # ---- 3) Heatmaps des architectures (hidden_dim), par type / XX / omic
    if not hidden_df.empty:
        heat = (
            hidden_df
            .groupby(["Type", "TrainRate", "TestRate", "Omics", "Architecture"])
            .size()
            .reset_index(name="Count")
        )

        for type_label in ("Drop", "Impute"):
            for train_rate in train_rates:
                sub_hidden = hidden_df[
                    (hidden_df["Type"] == type_label) & (hidden_df["TrainRate"] == train_rate)
                ]
                if sub_hidden.empty:
                    continue

                for omic in sub_hidden["Omics"].unique():
                    tmp = heat[
                        (heat["Type"] == type_label)
                        & (heat["TrainRate"] == train_rate)
                        & (heat["Omics"] == omic)
                    ]
                    mat = tmp.pivot(index="TestRate", columns="Architecture", values="Count").fillna(0)

                    arch_order_tuple = ordered_categories_by_layers_and_size(
                        sub_hidden.loc[sub_hidden["Omics"] == omic, "ArchitectureTuple"]
                    )
                    arch_order = ["x".join(map(str, a)) for a in arch_order_tuple]
                    mat = mat.reindex(columns=arch_order, fill_value=0)

                    plot_hidden_dim_heatmap(mat, omic, type_label, train_rate, out_dir)


if __name__ == "__main__":
    main()