"""
Paired t-test (C-index, IBS) entre modèles CustCox avec/sans Transfer
Learning (finetune), pour un cancer et plusieurs suffixes de modèle
(ridge, clinridge, supridge, supclinridge). Option correction FDR
(Benjamini-Hochberg) et forest plot récapitulatif.

Usage:
    python ttest_all_method_1_cancer.py --cancer KIRP --data-dir results --summary
    python ttest_all_method_1_cancer.py --cancer KIRP --data-dir results --fdr --plot --fig-out out/kirp_forest.png

"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import argparse
import os

SUFFIXES = ["ridge", "clinridge", "supclinridge", "supridge"]

METHOD_LABELS = {
    "ridge": "CustCox",
    "supridge": "CustCox sup",
    "clinridge": "CustCox clin",
    "supclinridge": "CustCox sup clin",
}

def load_results(path):
    df = pd.read_csv(path)
    # normalise le nom de colonne fold/iteration selon le format du fichier
    if "fold" not in df.columns and "iteration" in df.columns:
        df = df.rename(columns={"iteration": "fold"})
    if "fold" in df.columns and df["fold"].min() == 1:
        # certains fichiers (type "ref") indexent les folds a partir de 1
        pass  # decommenter la ligne suivante si besoin de recaler sur 0
        # df["fold"] = df["fold"] - 1
    return df


def paired_ttest_metric(vals_finetune, vals_sans, alpha=0.05):
    t_stat, p_value = stats.ttest_rel(vals_finetune, vals_sans)
    diff = vals_finetune - vals_sans
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    n = len(diff)
    sem_diff = std_diff / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(1 - alpha, df=n - 1, loc=mean_diff, scale=sem_diff)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def compare_suffix(data_dir, cancer, suffix,
                    finetune_prefix="ncv_finetune_optuna_paral",
                    sans_prefix="ncv_custcox_optuna_paral",
                    alpha=0.05):
    path_finetune = os.path.join(data_dir, f"{finetune_prefix}_{suffix}_{cancer}.csv")
    path_sans = os.path.join(data_dir, f"{sans_prefix}_{suffix}_{cancer}.csv")

    if not os.path.exists(path_finetune) or not os.path.exists(path_sans):
        print(f"[{suffix}] fichier manquant -> skip")
        print(f"    attendu: {path_finetune}")
        print(f"    attendu: {path_sans}")
        return None

    df_finetune = load_results(path_finetune)
    df_sans = load_results(path_sans)

    merged = pd.merge(df_sans, df_finetune, on="fold", suffixes=("_sans", "_finetune"))

    if merged.empty:
        print(f"[{suffix}] merge vide (folds non alignes) -> skip")
        return None

    results = {}
    for metric_name, col_base in [("C-index", "cindex_default"), ("IBS (Graf)", "graf")]:
        col_sans = f"{col_base}_sans"
        col_finetune = f"{col_base}_finetune"
        if col_sans not in merged.columns or col_finetune not in merged.columns:
            print(f"[{suffix}] colonne '{col_base}' introuvable -> skip metrique")
            continue
        res = paired_ttest_metric(merged[col_finetune].values, merged[col_sans].values, alpha)
        results[metric_name] = res

    return results


def print_results(cancer, suffix, results):
    print("\n" + "=" * 70)
    print(f"CANCER: {cancer} | SUFFIX: {suffix} (finetune vs sans)")
    print("=" * 70)
    for metric_name, res in results.items():
        print(f"\n--- {metric_name} ---")
        print(f"n paires            : {res['n']}")
        print(f"Différence moyenne  : {res['mean_diff']:.6f}")
        print(f"Écart-type diff.    : {res['std_diff']:.6f}")
        print(f"IC 95%              : [{res['ci_low']:.6f}, {res['ci_high']:.6f}]")
        print(f"t-statistic         : {res['t_stat']:.4f}")
        print(f"p-value             : {res['p_value']:.6f}")
        print(f"Significatif        : {'Oui' if res['significant'] else 'Non'}")


def plot_forest(summary_df, cancer, fig_out, use_fdr=False):
    """Forest plot : différence moyenne (finetune - sans) avec IC 95%, un point par suffixe."""
    metrics = list(summary_df["metric"].unique())

    sig_col = "significant_adj" if use_fdr and "significant_adj" in summary_df.columns else "significant"
    pval_col = "p_value_adj" if use_fdr and "p_value_adj" in summary_df.columns else "p_value"
    pval_label = "q" if use_fdr and "p_value_adj" in summary_df.columns else "p"

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 0.75 * summary_df["suffix"].nunique() + 1.5))
    if len(metrics) == 1:
        axes = [axes]

    color_sig = "#2E7D32"       # vert : significatif
    color_nonsig = "#B0B0B0"    # gris : non significatif

    for ax, metric in zip(axes, metrics):
        sub = summary_df[summary_df["metric"] == metric].copy()
        
        sub = sub.sort_values("suffix_label", ascending=False).reset_index(drop=True)
        y_pos = np.arange(len(sub))

        # bandes horizontales alternées, très légères, à la place du quadrillage
        for i in y_pos:
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#F5F5F5", zorder=0)

        colors = [color_sig if s else color_nonsig for s in sub[sig_col]]

        for i, row in sub.iterrows():
            err_low = row["mean_diff"] - row["ci_low"]
            err_high = row["ci_high"] - row["mean_diff"]
            ax.errorbar(
                row["mean_diff"], i, xerr=[[err_low], [err_high]],
                fmt="none", ecolor=colors[i], elinewidth=2.4, capsize=5, zorder=2,
            )
        ax.scatter(sub["mean_diff"], y_pos, color=colors, s=100, zorder=3, edgecolor="white", linewidth=0.9)

        ax.axvline(0, color="#B71C1C", linestyle="--", linewidth=1.2, zorder=1)

        for i, row in sub.iterrows():
            ax.annotate(
                f"{pval_label}={row[pval_col]:.3f}", (row["mean_diff"], i),
                xytext=(0, 12), textcoords="offset points",
                va="bottom", ha="center", fontsize=9, color="#444444",
            )

        ax.set_yticks(y_pos)
        
        ax.set_yticklabels(sub["suffix_label"], fontsize=11)
        ax.set_xlabel("Différence moyenne  (avec - sans TL)", fontsize=10)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(-0.5, len(sub) - 0.5)
        ax.margins(x=0.3, y=0.08)
        ax.grid(False)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(axis="both", length=0)

    sig_label = "Significatif (q<0.05, BH)" if use_fdr else "Significatif (p<0.05)"
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_sig, markersize=9, label=sig_label),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_nonsig, markersize=9, label="Non significatif"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(f"Avec/Sans Transfer Learning — {cancer}", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure sauvegardée: {fig_out}")


def apply_fdr_correction(summary_df, alpha=0.05, scope="all"):
    """Corrige les p-values avec Benjamini-Hochberg (scipy.stats.false_discovery_control).

    scope="all"    -> correction sur l'ensemble des tests (tous suffixes x métriques)
    scope="metric" -> correction séparée par métrique (C-index d'un côté, IBS de l'autre)
    """
    summary_df = summary_df.copy()

    if scope == "metric":
        summary_df["p_value_adj"] = np.nan
        for metric in summary_df["metric"].unique():
            mask = summary_df["metric"] == metric
            summary_df.loc[mask, "p_value_adj"] = false_discovery_control(
                summary_df.loc[mask, "p_value"].values, method="bh"
            )
    else:
        summary_df["p_value_adj"] = false_discovery_control(
            summary_df["p_value"].values, method="bh"
        )

    summary_df["significant_adj"] = summary_df["p_value_adj"] < alpha
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Paired t-test finetune vs sans pour un cancer donné, sur plusieurs suffixes de modèle."
    )
    parser.add_argument("--cancer", required=True, help="Nom du cancer (ex: KIRP)")
    parser.add_argument("--data-dir", default=".", help="Répertoire contenant les CSV")
    parser.add_argument("--suffixes", nargs="+", default=SUFFIXES,
                         help="Liste des suffixes à tester (défaut: ridge clinridge supclinridge supridge)")
    parser.add_argument("--finetune-prefix", default="ncv_finetune_optuna_paral")
    parser.add_argument("--sans-prefix", default="ncv_custcox_optuna_paral")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--summary", action="store_true", help="Afficher un tableau récapitulatif à la fin")
    parser.add_argument("--csv-out", default=None, help="Chemin pour sauvegarder le récapitulatif en CSV")
    parser.add_argument("--plot", action="store_true", help="Générer un forest plot du récapitulatif")
    parser.add_argument("--fig-out", default=None, help="Chemin de sauvegarde de la figure (défaut: ttest_{cancer}.png)")
    parser.add_argument("--fdr", action="store_true", help="Corriger les p-values avec Benjamini-Hochberg")
    parser.add_argument("--fdr-scope", choices=["all", "metric"], default="all",
                         help="Portée de la correction FDR: 'all' (les 8 tests ensemble) ou 'metric' (séparément par métrique)")
    args = parser.parse_args()

    all_results = {}
    for suffix in args.suffixes:
        results = compare_suffix(
            args.data_dir, args.cancer, suffix,
            finetune_prefix=args.finetune_prefix,
            sans_prefix=args.sans_prefix,
            alpha=args.alpha,
        )
        if results:
            all_results[suffix] = results
            print_results(args.cancer, suffix, results)

    if all_results:
        rows = []
        for suffix, results in all_results.items():
            for metric_name, res in results.items():
                rows.append({
                    "cancer": args.cancer,
                    "suffix": suffix,
                    "suffix_label": METHOD_LABELS.get(suffix, suffix),
                    "metric": metric_name,
                    "n": res["n"],
                    "mean_diff": res["mean_diff"],
                    "std_diff": res["std_diff"],
                    "t_stat": res["t_stat"],
                    "p_value": res["p_value"],
                    "significant": res["significant"],
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                })
        summary_df = pd.DataFrame(rows)

        if args.fdr:
            summary_df = apply_fdr_correction(summary_df, alpha=args.alpha, scope=args.fdr_scope)

        if args.summary:
            print("\n" + "=" * 70)
            print("RÉCAPITULATIF")
            print("=" * 70)
            print(summary_df.to_string(index=False))

        if args.csv_out:
            summary_df.to_csv(args.csv_out, index=False)
            print(f"\nRécapitulatif sauvegardé dans: {args.csv_out}")

        if args.plot:
            fig_out = args.fig_out or f"ttest_{args.cancer}.png"
            plot_forest(summary_df, args.cancer, fig_out, use_fdr=args.fdr)


if __name__ == "__main__":
    main()