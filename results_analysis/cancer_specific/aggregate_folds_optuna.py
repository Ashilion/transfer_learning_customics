"""
Agrège les fichiers de résultats par fold (nested CV) d'une méthode donnée
(custcox, finetune, pclsurv, cma) en un seul CSV, avec stats C-index/IBS.

Les dossiers d'entrée / de sortie ont des valeurs par défaut par méthode,
surchargeables via --input_dir / --output_dir.

Usage:
    python aggregate_folds_optuna.py --cancer COAD --method cma --outer_splits 50
    python aggregate_folds_optuna.py --cancer BRCA --method custcox --name_suffix v2_
    python aggregate_folds_optuna.py --cancer COAD --method finetune \
        --input_dir results/folds_tl --output_dir results/tl
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Par méthode : dossier + nom de fichier (pattern) en entrée, dossier + nom en sortie.
METHOD_CONFIG = {
    "custcox": {
        "in_dir":  "results/folds",
        "in_file": "ncv_custcox_optuna_{name_suffix}{cancer}_fold*.csv",
        "out_dir":  "results",
        "out_file": "ncv_custcox_optuna_paral_{name_suffix}{cancer}.csv",
    },
    "finetune": {
        "in_dir":  "CustOmics/tl_ckpt",
        "in_file": "ncv_finetune_optuna_{name_suffix}{cancer}_fold*.csv",
        "out_dir":  "results",
        "out_file": "ncv_finetune_optuna_paral_{name_suffix}{cancer}.csv",
    },
    "pclsurv": {
        "in_dir":  "results/folds",
        "in_file": "ncv_pclsurv_{name_suffix}{cancer}_fold*.csv",
        "out_dir":  "results",
        "out_file": "ncv_pclsurv_{name_suffix}{cancer}.csv",
    },
    "cma": {
        "in_dir":  "results/folds",
        "in_file": "ncv_cma_{name_suffix}{cancer}_fold*_cls_token.csv",
        "out_dir":  "results",
        "out_file": "ncv_cma_{name_suffix}{cancer}_cls_token.csv",
    },
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cancer", type=str, default="COAD")
    p.add_argument("--method", type=str, default="cma", choices=sorted(METHOD_CONFIG.keys()))
    p.add_argument("--outer_splits", type=int, default=50)
    p.add_argument("--name_suffix", type=str, default="")
    p.add_argument("--input_dir", type=str, default=None,
        help="Dossier où lire les CSV par fold (défaut : dossier propre à la méthode).")
    p.add_argument("--output_dir", type=str, default=None,
        help="Dossier où écrire le CSV agrégé (défaut : dossier propre à la méthode).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = METHOD_CONFIG[args.method]
    fmt = dict(name_suffix=args.name_suffix, cancer=args.cancer)

    in_dir = args.input_dir if args.input_dir is not None else cfg["in_dir"]
    out_dir = args.output_dir if args.output_dir is not None else cfg["out_dir"]

    pattern = os.path.join(in_dir, cfg["in_file"].format(**fmt))
    out = os.path.join(out_dir, cfg["out_file"].format(**fmt))

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching: {pattern}")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("fold").reset_index(drop=True)

    missing = sorted(set(range(args.outer_splits)) - set(df["fold"].astype(int)))
    if missing:
        print(f"WARNING: missing folds {missing} — aggregating available folds only.")

    print(f"\n=== Résultats nested CV ({args.method}) ===")
    print(df.to_string(index=False))
    df = df.rename(columns={"cindex": "cindex_default", "ibs": "graf"})
    print(f"\nC-index : {df['cindex_default'].mean():.4f} ± {df['cindex_default'].std():.4f}")
    print(f"IBS     : {df['graf'].mean():.4f} ± {df['graf'].std():.4f}")

    os.makedirs(out_dir or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nAggregated results saved to {out}")


if __name__ == "__main__":
    main()