import argparse
import glob
import os
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cancer", type=str, default="KIRP")
    p.add_argument("--outer_splits", type=int, default=20)
    p.add_argument("--name_suffix", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    name_suffix = args.name_suffix
    pattern = f"results/tl_folds/ncv_finetune_optuna_{name_suffix}{args.cancer}_fold*.csv"

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

    print("\n=== Résultats nested CV — fine-tuning target ===")
    print(df.to_string(index=False))
    df = df.rename(columns={"cindex": "cindex_default", "ibs": "graf"})
    print(f"\nC-index : {df['cindex_default'].mean():.4f} ± {df['cindex_default'].std():.4f}")
    print(f"IBS     : {df['graf'].mean():.4f} ± {df['graf'].std():.4f}")

    out = f"results/ncv_finetune_optuna_paral_{name_suffix}{args.cancer}.csv"

    try:
        df.to_csv(out, index=False)
    except Exception as e:
        print(f"\nERROR: failed to save aggregated results to {out}: {e}")
        sys.exit(1)

    print(f"\nAggregated results saved to {out}")

    # Agrégation réussie -> on peut supprimer les fichiers de fold individuels
    n_deleted, n_failed = 0, 0
    for f in files:
        try:
            os.remove(f)
            n_deleted += 1
        except OSError as e:
            n_failed += 1
            print(f"WARNING: could not delete {f}: {e}")

    print(f"Deleted {n_deleted}/{len(files)} individual fold files"
          + (f" ({n_failed} failures)" if n_failed else ""))


if __name__ == "__main__":
    main()