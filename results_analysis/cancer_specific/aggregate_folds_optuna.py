"""
Agrège les fichiers de résultats par fold (nested CV) d'une méthode donnée
(custcox, finetune, pclsurv, cma) en un seul CSV, avec stats C-index/IBS.

Usage:
    python aggregate_folds_optuna.py --cancer COAD --method cma --outer_splits 50
    python aggregate_folds_optuna.py --cancer BRCA --method custcox --name_suffix v2_
"""

import argparse
import pandas as pd
import glob
import sys

# Un pattern d'entrée + un chemin de sortie par méthode.
METHOD_CONFIG = {
    "custcox": {
        "pattern": "results/folds/ncv_custcox_optuna_{name_suffix}{cancer}_fold*.csv",
        "out":     "results/ncv_custcox_optuna_paral_{name_suffix}{cancer}.csv",
    },
    "finetune": {
        "pattern": "CustOmics/tl_ckpt/ncv_finetune_optuna_{name_suffix}{cancer}_fold*.csv",
        "out":     "results/ncv_finetune_optuna_paral_{name_suffix}{cancer}.csv",
    },
    "pclsurv": {
        "pattern": "results/folds/ncv_pclsurv_{name_suffix}{cancer}_fold*.csv",
        "out":     "results/ncv_pclsurv_{name_suffix}{cancer}.csv",
    },
    "cma": {
        "pattern": "results/folds/ncv_cma_{name_suffix}{cancer}_fold*_cls_token.csv",
        "out":     "results/ncv_cma_{name_suffix}{cancer}_cls_token.csv",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cancer", type=str, default="COAD")
    p.add_argument("--method", type=str, default="cma", choices=sorted(METHOD_CONFIG.keys()))
    p.add_argument("--outer_splits", type=int, default=50)
    p.add_argument("--name_suffix", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    name_suffix = args.name_suffix
    cfg = METHOD_CONFIG[args.method]

    pattern = cfg["pattern"].format(name_suffix=name_suffix, cancer=args.cancer)
    out = cfg["out"].format(name_suffix=name_suffix, cancer=args.cancer)

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching: {pattern}")
        sys.exit(1)

    missing = [
        i for i in range(args.outer_splits)
        if f"fold{i}.csv" not in " ".join(files)
    ]
    if missing:
        print(f"WARNING: missing folds {missing} — aggregating available folds only.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("fold").reset_index(drop=True)

    print(f"\n=== Résultats nested CV ({args.method}) ===")
    print(df.to_string(index=False))
    df = df.rename(columns={"cindex": "cindex_default", "ibs": "graf"})
    print(f"\nC-index : {df['cindex_default'].mean():.4f} ± {df['cindex_default'].std():.4f}")
    print(f"IBS     : {df['graf'].mean():.4f} ± {df['graf'].std():.4f}")

    df.to_csv(out, index=False)
    print(f"\nAggregated results saved to {out}")


if __name__ == "__main__":
    main()