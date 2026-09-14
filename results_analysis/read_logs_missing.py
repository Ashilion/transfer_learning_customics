#!/usr/bin/env python3
"""
Parse CustOmics missing-data optuna log files (.out) to recover results
that weren't saved properly to CSV.

For each log file, extracts (per outer-fold result block found):
    - filename, job_id, fold (from filename)
    - cancer, outer_folds, inner_folds, optuna_trials, ridge, nb_features
    - missing_rate, missing_strategy
    - study_name (encodes method, e.g. journal_storage_multiprocess_imp30KIRP_fold0)
    - method (derived from study_name, e.g. "imp" -> impute)
    - optuna_best_alpha
    - c_index, ibs
    - coefs_nonzero, coefs_total

Usage:
    python parse_logs.py logs/missing_data/*.out -o results_from_logs.csv
"""
import re
import sys
import csv
import argparse
from pathlib import Path

FILENAME_RE = re.compile(r"opt_fold_(?P<fold>\d+)_(?P<jobid>\d+)\.out$")

JOB_RE = re.compile(r"Job\s*:\s*(\d+)")
CANCER_RE = re.compile(r"Cancer\s*:\s*(\S+)")
OUTER_FOLDS_RE = re.compile(r"Outer folds\s*:\s*(\d+)")
INNER_FOLDS_RE = re.compile(r"Inner folds\s*:\s*(\d+)")
TRIALS_RE = re.compile(r"Optuna trials\s*:\s*(\d+)")
RIDGE_RE = re.compile(r"Ridge\s*:\s*(\S+)")
NBFEAT_RE = re.compile(r"Nb features\s*:\s*(\d+)")

MISSING_RATE_RE = re.compile(r"Missing rate\s*:\s*([\d.]+)")
MISSING_STRAT_RE = re.compile(r"Missing strategy:\s*(\S+)")

STUDY_NAME_RE = re.compile(r"study created in Journal with name:\s*(\S+)")
BEST_ALPHA_RE = re.compile(r"Optuna best alpha\s*:\s*([\d.eE+-]+)")

# Final metrics block, may appear once (or more, if multiple folds per file)
RESULT_RE = re.compile(
    r"C-index\s*:\s*([\d.]+)\s*\|\s*IBS\s*:\s*([\d.]+)\s*\n"
    r"\s*Coefs non-nuls\s*:\s*(\d+)\s*/\s*(\d+)\s*\(nuls\s*:\s*(\d+)\)"
)

OUTER_FOLD_HEADER_RE = re.compile(r"OUTER FOLD\s*(\d+)")


def derive_method(study_name):
    """Try to derive a human-readable method label from the optuna study name.
    e.g. journal_storage_multiprocess_imp30KIRP_fold0 -> method=impute, rate=30
    """
    if not study_name:
        return None
    m = re.search(r"multiprocess_([a-zA-Z]+)(\d+)", study_name)
    if m:
        tag, rate = m.group(1), m.group(2)
        tag_map = {"imp": "impute", "drop": "drop", "mean": "mean"}
        return f"{tag_map.get(tag, tag)}_{rate}"
    return study_name


def parse_file(path: Path):
    text = path.read_text(errors="replace")

    fname_m = FILENAME_RE.search(path.name)
    fold_from_name = fname_m.group("fold") if fname_m else None
    jobid_from_name = fname_m.group("jobid") if fname_m else None

    job = JOB_RE.search(text)
    cancer = CANCER_RE.search(text)
    outer_folds = OUTER_FOLDS_RE.search(text)
    inner_folds = INNER_FOLDS_RE.search(text)
    trials = TRIALS_RE.search(text)
    ridge = RIDGE_RE.search(text)
    nbfeat = NBFEAT_RE.search(text)
    missing_rate = MISSING_RATE_RE.search(text)
    missing_strat = MISSING_STRAT_RE.search(text)
    study_name = STUDY_NAME_RE.search(text)

    # There can be several "OUTER FOLD n" headers (once per multiproc task
    # printing its own banner), but usually a single final result block per file.
    outer_fold_headers = OUTER_FOLD_HEADER_RE.findall(text)
    outer_fold_seen = outer_fold_headers[0] if outer_fold_headers else None

    best_alpha_matches = BEST_ALPHA_RE.findall(text)
    best_alpha = best_alpha_matches[-1] if best_alpha_matches else None

    results = RESULT_RE.findall(text)

    rows = []
    if not results:
        rows.append({
            "file": path.name,
            "job_id": jobid_from_name or (job.group(1) if job else None),
            "fold": fold_from_name or outer_fold_seen,
            "cancer": cancer.group(1) if cancer else None,
            "missing_rate": missing_rate.group(1) if missing_rate else None,
            "missing_strategy": missing_strat.group(1) if missing_strat else None,
            "study_name": study_name.group(1) if study_name else None,
            "method": derive_method(study_name.group(1)) if study_name else None,
            "optuna_best_alpha": best_alpha,
            "outer_folds": outer_folds.group(1) if outer_folds else None,
            "inner_folds": inner_folds.group(1) if inner_folds else None,
            "optuna_trials": trials.group(1) if trials else None,
            "ridge": ridge.group(1) if ridge else None,
            "nb_features": nbfeat.group(1) if nbfeat else None,
            "c_index": None,
            "ibs": None,
            "coefs_nonzero": None,
            "coefs_total": None,
            "coefs_null": None,
            "PARSE_WARNING": "no C-index/IBS block found",
        })
        return rows

    for c_index, ibs, nz, tot, nul in results:
        rows.append({
            "file": path.name,
            "job_id": jobid_from_name or (job.group(1) if job else None),
            "fold": fold_from_name or outer_fold_seen,
            "cancer": cancer.group(1) if cancer else None,
            "missing_rate": missing_rate.group(1) if missing_rate else None,
            "missing_strategy": missing_strat.group(1) if missing_strat else None,
            "study_name": study_name.group(1) if study_name else None,
            "method": derive_method(study_name.group(1)) if study_name else None,
            "optuna_best_alpha": best_alpha,
            "outer_folds": outer_folds.group(1) if outer_folds else None,
            "inner_folds": inner_folds.group(1) if inner_folds else None,
            "optuna_trials": trials.group(1) if trials else None,
            "ridge": ridge.group(1) if ridge else None,
            "nb_features": nbfeat.group(1) if nbfeat else None,
            "c_index": c_index,
            "ibs": ibs,
            "coefs_nonzero": nz,
            "coefs_total": tot,
            "coefs_null": nul,
            "PARSE_WARNING": "",
        })
    return rows


def build_pivot(rows):
    """method x fold -> 'c_index/ibs' pivot, to eyeball completeness at a glance."""
    methods = sorted({r["method"] or "UNKNOWN" for r in rows})
    folds = sorted({r["fold"] or "?" for r in rows}, key=lambda x: (len(x), x))
    grid = {(r["method"] or "UNKNOWN", r["fold"] or "?"): r for r in rows}

    header = ["method"] + [f"fold{f}" for f in folds]
    lines = [",".join(header)]
    for m in methods:
        row = [m]
        for f in folds:
            r = grid.get((m, f))
            if r and r["c_index"]:
                row.append(f"{r['c_index']}/{r['ibs']}")
            else:
                row.append("MISSING")
        lines.append(",".join(row))
    return "\n".join(lines), methods, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfiles", nargs="+", help=".out log files to parse (glob ok)")
    ap.add_argument("-o", "--output", default="results_from_logs.csv")
    ap.add_argument("--pivot", default="results_pivot.csv",
                     help="method x fold summary grid, 'MISSING' flags gaps")
    ap.add_argument("--expect-files", type=int, default=None,
                     help="expected total file count (e.g. 60), warn if off")
    args = ap.parse_args()

    all_rows = []
    n_files = 0
    for f in args.logfiles:
        p = Path(f)
        if not p.exists():
            print(f"!! missing file: {f}", file=sys.stderr)
            continue
        n_files += 1
        rows = parse_file(p)
        all_rows.extend(rows)
        for r in rows:
            flag = f"  [{r['PARSE_WARNING']}]" if r['PARSE_WARNING'] else ""
            print(f"{p.name}: fold={r['fold']} method={r['method']} "
                  f"c_index={r['c_index']} ibs={r['ibs']} "
                  f"coefs={r['coefs_nonzero']}/{r['coefs_total']}{flag}")

    fieldnames = [
        "file", "job_id", "fold", "cancer", "missing_rate", "missing_strategy",
        "study_name", "method", "optuna_best_alpha", "outer_folds", "inner_folds",
        "optuna_trials", "ridge", "nb_features", "c_index", "ibs",
        "coefs_nonzero", "coefs_total", "coefs_null", "PARSE_WARNING",
    ]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    pivot_text, methods, folds = build_pivot(all_rows)
    with open(args.pivot, "w") as fh:
        fh.write(pivot_text + "\n")

    n_ok = sum(1 for r in all_rows if r["c_index"])
    n_warn = sum(1 for r in all_rows if r["PARSE_WARNING"])
    print(f"\nParsed {n_files} file(s) -> {len(all_rows)} result row(s) "
          f"({n_ok} OK, {n_warn} with warnings)")
    print(f"Methods found ({len(methods)}): {methods}")
    print(f"Folds found ({len(folds)}): {folds}")
    if args.expect_files is not None and n_files != args.expect_files:
        print(f"!! WARNING: expected {args.expect_files} files, found {n_files}",
              file=sys.stderr)
    n_missing = pivot_text.count("MISSING")
    if n_missing:
        print(f"!! WARNING: {n_missing} method x fold cell(s) missing "
              f"a result -- see {args.pivot}", file=sys.stderr)
    print(f"\nWrote {len(all_rows)} row(s) to {args.output}")
    print(f"Wrote pivot summary to {args.pivot}")


if __name__ == "__main__":
    main()