import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from scipy import stats

def compare_results(path_ref, path_new, graph_name, type_ref="ref", boxplot=False, paired_ttest=False):
    
    # Noms courts pour la légende
    label_ref = os.path.splitext(os.path.basename(path_ref))[0]
    label_new = os.path.splitext(os.path.basename(path_new))[0]

    # Lecture référence
    if type_ref == "ref":
        df = pd.read_csv(path_ref)
        ref_df = df[df["learner_id"] == "cox_ref"]
        ref_df = ref_df[ref_df["task_id"] == "COAD"]
        ref_df = ref_df[['nr', 'task_id', 'iteration', 'cindex_default', 'graf', 'time_train', 'time_both']]
    else:
        ref_df = pd.read_csv(path_ref)

    df = pd.read_csv(path_new)

    print(ref_df.head())
    print(df.head())
    ref_df['fold'] = ref_df['fold'] - 1  # indice en R commence a 1
    ref_df = ref_df.rename(columns={"iteration": "fold"})

    # Merge et analyse
    merged = pd.merge(ref_df, df, on="fold", suffixes=("_ref", "_new"))

    merged["cindex_diff"] = merged["cindex_default_new"] - merged["cindex_default_ref"]
    merged["graf_diff"]   = merged["graf_new"] - merged["graf_ref"]

    print(merged[["fold", "cindex_default_ref", "cindex_default_new", "cindex_diff",
                  "graf_ref", "graf_new", "graf_diff"]])
    print("C-index difference (mean):", merged["cindex_diff"].mean())
    print("Graf difference (mean):",    merged["graf_diff"].mean())

    # Paired t-test entre ref et new
    if paired_ttest:
        run_paired_ttest(merged, label_ref, label_new)

    # Plot C-index
    if not boxplot:
        plt.figure()
        plt.plot(merged["fold"], merged["cindex_default_ref"], label=label_ref)
        plt.plot(merged["fold"], merged["cindex_default_new"], label=label_new)
        plt.legend()
        plt.title("C-index comparison")
        plt.savefig(f"{graph_name}_cindex.png", bbox_inches="tight")
        plt.show()

        # Plot IBS
        plt.figure()
        plt.plot(merged["fold"], merged["graf_ref"], label=label_ref)
        plt.plot(merged["fold"], merged["graf_new"], label=label_new)
        plt.legend()
        plt.title("IBS comparison")
        plt.savefig(f"{graph_name}_ibs.png", bbox_inches="tight")
        plt.show()
    
    else:
        # Reshape merged data for boxplot: ref vs new
        cindex_data = pd.DataFrame({
            'Value': pd.concat([merged['cindex_default_ref'], merged['cindex_default_new']]),
            'Metric': 'C-index',
            'Source': [label_ref] * len(merged) + [label_new] * len(merged)
        })
        graf_data = pd.DataFrame({
            'Value': pd.concat([merged['graf_ref'], merged['graf_new']]),
            'Metric': 'IBS',
            'Source': [label_ref] * len(merged) + [label_new] * len(merged)
        })
        plot_data = pd.concat([cindex_data, graf_data], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, metric in zip(axes, ['C-index', 'IBS']):
            subset = plot_data[plot_data['Metric'] == metric]
            sns.boxplot(x='Source', y='Value', data=subset, palette='Set2', ax=ax)
            ax.set_title(f"{metric} ")
            ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(f"{graph_name}_boxplot.png", bbox_inches="tight")
        plt.show()

    return merged


def run_paired_ttest(merged, label_ref, label_new, alpha=0.05):
    """
    Effectue un test t apparié (paired t-test) entre les valeurs ref et new
    pour le C-index et le score de Graf (IBS), et affiche les résultats.
    """
    print("\n" + "=" * 60)
    print("PAIRED T-TEST RESULTS")
    print("=" * 60)

    for metric_name, col_ref, col_new in [
        ("C-index", "cindex_default_ref", "cindex_default_new"),
        ("IBS (Graf)", "graf_ref", "graf_new"),
    ]:
        vals_ref = merged[col_ref].values
        vals_new = merged[col_new].values

        t_stat, p_value = stats.ttest_rel(vals_new, vals_ref)

        diff = vals_new - vals_ref
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)
        n = len(diff)
        sem_diff = std_diff / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(1 - alpha, df=n - 1, loc=mean_diff, scale=sem_diff)

        significant = "Oui" if p_value < alpha else "Non"

        print(f"\n--- {metric_name} ({label_new} vs {label_ref}) ---")
        print(f"n paires            : {n}")
        print(f"Différence moyenne  : {mean_diff:.6f}")
        print(f"Écart-type diff.    : {std_diff:.6f}")
        # print(f"IC {int((1-alpha)*100)}% de la diff.   : [{ci_low:.6f}, {ci_high:.6f}]")
        print(f"t-statistic         : {t_stat:.4f}")
        print(f"p-value             : {p_value:.6f}")
        print(f"Significatif (a={alpha}) : {significant}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare deux fichiers de résultats de survie.")
    parser.add_argument("--ref", required=True, help="Chemin vers le fichier de référence")
    parser.add_argument("--new", required=True, help="Chemin vers le nouveau fichier de résultats")
    parser.add_argument("--graph", required=True, help="Préfixe du nom des graphiques sauvegardés")
    parser.add_argument("--type-ref", default="new", choices=["ref", "new"],
                        help="Type du fichier de référence (default: new)")
    parser.add_argument(
        "--boxplot",
        action="store_true",
        default=False,
        help="Use boxplots instead of line charts"
    )
    parser.add_argument(
        "--ttest",
        action="store_true",
        default=False,
        help="Effectuer un paired t-test entre les résultats ref et new"
    )
    args = parser.parse_args()

    merged = compare_results(args.ref, args.new, args.graph, type_ref=args.type_ref,
                              boxplot=args.boxplot, paired_ttest=args.ttest)