"""
Compare les résultats de validation croisée (outer CV) de modèles de survie
obtenus en Python avec des résultats de référence obtenus en R (pipeline de
Vincent, `glmnet_ref`), pour plusieurs types de cancer à la fois.

Le script :
1. Charge les résultats de référence (R) soit depuis un unique fichier
   contenant tous les cancers (`type_ref="ref"`, colonne `learner_id` filtrée
   sur `glmnet_ref`), soit depuis un fichier par cancer (`type_ref="file"`).
2. Recherche dans `results_dir` tous les fichiers Python au format
   `{file_basename}{cancer}.csv` (ex: `outer_cv_results_vvh_BRCA.csv`).
3. Fusionne (merge) les résultats R et Python fold par fold, pour chaque
   cancer trouvé.
4. Calcule, par cancer, les différences moyennes (et écarts-types / MAD)
   entre R et Python pour deux métriques : le C-index (`cindex_default`) et
   l'IBS (`graf`).
5. Génère des visualisations selon les modes demandés :
   - "boxplot"  : boxplots comparant R vs Python par cancer, pour chaque métrique
   - "meandiff" : barplots de la différence moyenne (R - Python) par cancer
   - "mad"      : barplots de la Mean Absolute Difference par cancer
   Les figures sont sauvegardées dans `results_dir` (fichiers .png) et
   affichées à l'écran.
6. Retourne (et affiche) un tableau récapitulatif (`summary`) avec, pour
   chaque cancer, les différences moyennes (et MAD si demandé).

Le script utilise aussi des fichiers clinique (`{cancer}_clinical.pickle`)
pour trier les cancers du plus grand au plus petit effectif (utile pour
l'ordre d'affichage des graphiques).

Utilisation en ligne de commande
---------------------------------
    python compare_all_clinical.py \\
        --results-dir /chemin/vers/results \\
        --path-ref /chemin/vers/results/ref_results.csv \\
        --type-ref ref \\
        --plot boxplot mad \\
        --filename outer_cv_results_vvh_

Exemple minimal (valeurs par défaut) :
    python compare_all_clinical.py

Exemple avec plusieurs modes de plot et un autre dossier de résultats :
    python compare_all_clinical.py --results-dir ./mes_resultats --plot boxplot meandiff mad

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import pickle
import argparse


def compare_all_cancers(path_ref, results_dir, type_ref="ref", plot_modes=None, file_basename="outer_cv_results_vvh_"):
    if plot_modes is None:
        plot_modes = ["boxplot"]

    ridge = False
    if type_ref == "ref":
        df_ref_all = pd.read_csv(path_ref)
        df_ref_all = df_ref_all[df_ref_all["learner_id"] == "glmnet_ref"]

    ridge_str = "ridge_" if ridge else ""
    print(f"{file_basename}{ridge_str}")
    new_files = glob.glob(os.path.join(results_dir, f"{file_basename}{ridge_str}*.csv"))
    if not ridge:
        new_files = list(filter(lambda x: "ridge" not in x, new_files))
    cancer_names = [
        os.path.basename(f).replace(f"{file_basename}{ridge_str}", "").replace(".csv", "")
        for f in new_files
    ]

    if not cancer_names:
        print("No result files found.")
        return

    print(f"Found results for: {cancer_names}")

    # Nombre d'individus par cancer
    cancer_sizes = {}
    wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
    for cancer_name in cancer_names:
        path = os.path.join(wkd_path, f"data/clinical/{cancer_name}_clinical.pickle")
        with open(path, "rb") as f:
            df = pickle.load(f)
        cancer_sizes[cancer_name] = len(df)

    # Cancers triés du plus grand au plus petit
    cancer_order = sorted(cancer_sizes, key=cancer_sizes.get, reverse=True)
    print(cancer_order)
    print({c: cancer_sizes[c] for c in cancer_order})

    cindex_diffs = {}
    cindex_stds = {}
    cindex_mads = {}

    graf_diffs = {}
    graf_stds = {}
    graf_mads = {}

    all_cancer = pd.DataFrame()

    for cancer_name, new_file in zip(cancer_names, new_files):
        try:
            if type_ref == "ref":
                ref_df = df_ref_all[df_ref_all["task_id"] == cancer_name][
                    ["iteration", "cindex_default", "graf"]
                ].rename(columns={"iteration": "fold"})
            else:
                ref_df = pd.read_csv(path_ref)[["fold", "cindex_default", "graf"]]

            new_df = pd.read_csv(new_file)[["fold", "cindex_default", "graf"]]
            ref_df["fold"] = ref_df["fold"] - 1  # indice en R commence à 1
            print(ref_df["fold"].unique())
            print(new_df["fold"].unique())
            merged = pd.merge(ref_df, new_df, on="fold", suffixes=("_ref", "_new"))
            print(f"{cancer_name}: ref={len(ref_df)}, new={len(new_df)}, merged={len(merged)}")

            cindex_diff = merged["cindex_default_new"] - merged["cindex_default_ref"]
            graf_diff = merged["graf_new"] - merged["graf_ref"]

            cindex_diffs[cancer_name] = cindex_diff.mean()
            cindex_stds[cancer_name] = cindex_diff.std()
            cindex_mads[cancer_name] = cindex_diff.abs().mean()

            graf_diffs[cancer_name] = graf_diff.mean()
            graf_stds[cancer_name] = graf_diff.std()
            graf_mads[cancer_name] = graf_diff.abs().mean()

            cindex_data = pd.DataFrame({
                "Value": pd.concat(
                    [merged["cindex_default_ref"], merged["cindex_default_new"]],
                    ignore_index=True
                ),
                "Metric": "C-index",
                "Source": ["vincent_r"] * len(merged) + ["python"] * len(merged),
                "Cancer": cancer_name,
            })
            graf_data = pd.DataFrame({
                "Value": pd.concat(
                    [merged["graf_ref"], merged["graf_new"]],
                    ignore_index=True
                ),
                "Metric": "IBS",
                "Source": ["vincent_r"] * len(merged) + ["python"] * len(merged),
                "Cancer": cancer_name,
            })
            plot_data = pd.concat([cindex_data, graf_data], ignore_index=True)
            all_cancer = pd.concat([all_cancer, plot_data], ignore_index=True)

        except Exception as e:
            print(f"  Skipping {cancer_name}: {e}")

    if not cindex_diffs:
        print("No valid comparisons could be made")
        return

    cancers = sorted(cindex_diffs.keys())
    x = np.arange(len(cancers))

    if "boxplot" in plot_modes:
        cancer_order_plot = [c for c in cancer_order if c in all_cancer["Cancer"].unique()]
        all_cancer["Cancer"] = pd.Categorical(
            all_cancer["Cancer"], categories=cancer_order_plot, ordered=True
        )
        print(all_cancer.head())
        print(all_cancer.groupby(["Cancer", "Metric", "Source"]).size())

        for metric in ["C-index", "IBS"]:
            subset = all_cancer[all_cancer["Metric"] == metric]
            print(f"\n=== {metric} ===")
            print(
                subset.groupby(["Cancer", "Source"])["Value"]
                .describe()[["min", "25%", "50%", "75%", "max"]]
            )

        for metric in ["C-index", "IBS"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            subset = all_cancer[all_cancer["Metric"] == metric]
            sns.boxplot(
                x="Cancer",
                y="Value",
                hue="Source",
                data=subset,
                palette="Set2",
                ax=ax,
                showfliers=False,
                order=cancer_order_plot,
            )
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.savefig(
                os.path.join(results_dir, f"all_cancer_boxplot_{metric.replace('-', '_').lower()}.png"),
                bbox_inches="tight",
            )
            plt.show()

    if "meandiff" in plot_modes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors_cindex = ["blue" if v >= 0 else "red" for v in [cindex_diffs[c] for c in cancers]]
        axes[0].bar(
            x,
            [cindex_diffs[c] for c in cancers],
            yerr=[cindex_stds[c] for c in cancers],
            capsize=5,
            color=colors_cindex,
        )
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cancers, rotation=45, ha="right")
        axes[0].set_title("Mean C-index difference (R vs Python)")
        axes[0].set_ylabel("diff C-index")

        colors_graf = ["blue" if v <= 0 else "red" for v in [graf_diffs[c] for c in cancers]]
        axes[1].bar(
            x,
            [graf_diffs[c] for c in cancers],
            yerr=[graf_stds[c] for c in cancers],
            capsize=5,
            color=colors_graf,
        )
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cancers, rotation=45, ha="right")
        axes[1].set_title("Mean IBS difference (R vs Python)")
        axes[1].set_ylabel("diff IBS")

        plt.suptitle("Mean difference per cancer (R/Python)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"comparison_all_cancers_{ridge_str}.png"),
            bbox_inches="tight",
        )
        plt.show()

    if "mad" in plot_modes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(x, [cindex_mads[c] for c in cancers], color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cancers, rotation=45, ha="right")
        axes[0].set_title("Mean Absolute Difference — C-index (R vs Python)")
        axes[0].set_ylabel("MAD C-index")

        axes[1].bar(x, [graf_mads[c] for c in cancers], color="darkorange")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cancers, rotation=45, ha="right")
        axes[1].set_title("Mean Absolute Difference — IBS (R vs Python)")
        axes[1].set_ylabel("MAD IBS")

        plt.suptitle("Mean Absolute Difference per cancer (R/Python)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, f"mad_all_cancers_{ridge_str}.png"),
            bbox_inches="tight",
        )
        plt.show()

    # Summary table
    summary = pd.DataFrame({
        "cancer": cancers,
        "cindex_diff_mean": [cindex_diffs[c] for c in cancers],
        "graf_diff_mean": [graf_diffs[c] for c in cancers],
        **(
            {
                "cindex_mad": [cindex_mads[c] for c in cancers],
                "graf_mad": [graf_mads[c] for c in cancers],
            }
            if "mad" in plot_modes
            else {}
        ),
    })
    print(summary.to_string(index=False))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare survival model results (R vs Python) across cancer types."
    )
    parser.add_argument(
        "--results-dir",
        default="/env/cnrgh/proj/math_stats/scratch/hlegrand/results",
        help="Directory containing the outer_cv_results_vvh_*.csv files (default: %(default)s)",
    )
    parser.add_argument(
        "--path-ref",
        default="/env/cnrgh/proj/math_stats/scratch/hlegrand/results/ref_results.csv",
        help="Path to the reference CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--type-ref",
        choices=["ref", "file"],
        default="ref",
        help="Reference type: 'ref' uses glmnet_ref results from vincent "
             "'file' uses a standalone per-cancer CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=["boxplot", "meandiff", "mad"],
        default=["boxplot"],
        metavar="MODE",
        help="Plot mode(s) to display. Choose one or more from: boxplot, meandiff, mad "
             "(default: boxplot). Example: --plot boxplot mad",
    )
    parser.add_argument(
        "--filename",
        default="outer_cv_results_vvh_",
        help="names of the {filename}*.csv files (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = compare_all_cancers(
        path_ref=args.path_ref,
        results_dir=args.results_dir,
        type_ref=args.type_ref,
        plot_modes=args.plot,
        file_basename=args.filename
    )