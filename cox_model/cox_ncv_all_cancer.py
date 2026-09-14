import pandas as pd
import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import warnings
import time
import sys

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
sys.path.append(wkd_path)
from utils.vvh_cv import vvh_cv
import utils.folds_utils as fold_utils
import argparse

# ======= Argument Parsing ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested CV Coxnet on all cancer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="vvh",
        help="Validation function (vvh or ibs)"
    )
    # parser.add_argument(
    #     "--use_saved_folds",
    #     action="store_true",
    #     default=False,
    #     help="Use pre-saved outer fold splits from splits.json instead of generating new ones."
    # )
    parser.add_argument(
        "--ridge",
        action="store_true",
        default=False,
        help="Use ridge (L2) CoxPHSurvivalAnalysis function instead of Elastic Net CoxnetSurvivalAnalysis"
    )
    return parser.parse_args()

# ======= Main ===================================================================

def main(): 
    args = parse_args()

    validation_function = args.validation  # "vvh" or "ibs"
    ridge = args.ridge

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)

    os.makedirs(os.path.join(wkd_path, "results"), exist_ok=True)

    # Discover all available cancer pickles
    pickle_files = glob.glob(os.path.join(wkd_path, "data/clinical", "*_clinical.pickle"))
    cancer_names = [os.path.basename(f).replace("_clinical.pickle", "") for f in pickle_files]

    if not cancer_names:
        print("No clinical pickle files found. Run prepare_all_cancers.py first.")
        sys.exit(1)

    print(f"Found {len(cancer_names)} cancer(s): {cancer_names}\n")

    for cancer_name in cancer_names:
        print(f"\n{'='*50}")
        print(f"CANCER: {cancer_name}")
        print(f"{'='*50}")

        if not ridge:
            results_path = os.path.join(wkd_path, f"results/outer_cv_results_{validation_function}_{cancer_name}.csv")
        else:
            results_path = os.path.join(wkd_path, f"results/outer_cv_results_{validation_function}_ridge_{cancer_name}.csv")

        if os.path.exists(results_path):
            print(f"  Results already exist at {results_path}, skipping.")
            continue

        try:
            path = os.path.join(wkd_path, f"data/clinical/{cancer_name}_clinical.pickle")
            with open(path, "rb") as f:
                df = pickle.load(f)

            Xt = df[list(set(df.columns) - set(["time", "bcr_patient_barcode", "status"]))]
            y = np.array(
                [(bool(i), j) for i, j in zip(df["status"], df["time"])],
                dtype=[("status", "bool_"), ("time", "<f4")]
            )

            train_folds, test_folds = fold_utils.get_folds(cancer_name)

            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

            outer_results = []
            l1_ratio = 0.01

            for outer_fold, (train_idx, test_idx) in enumerate(zip(train_folds, test_folds)):
                print(f"  OUTER FOLD {outer_fold}")

                X_train_outer = Xt.iloc[train_idx, :]
                X_test_outer = Xt.iloc[test_idx, :]

                y_train_outer = y[train_idx] 
                # get list of alphas in the outer fold to have the same alphas for the inner folds
                coxnet_pipe = make_pipeline(
                    StandardScaler(),
                    CoxnetSurvivalAnalysis(n_alphas=100, l1_ratio=l1_ratio, alpha_min_ratio=0.001, max_iter=100, fit_baseline_model=True)
                )
                coxnet_pipe.fit(X_train_outer, y[train_idx])
                estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

                alpha_scores = {alpha: [] for alpha in estimated_alphas}

                tot_time_train = 0
                tot_time_validate = 0

                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y[train_idx]["status"]):
                    X_train_inner = X_train_outer.iloc[inner_train_idx, :]
                    X_val_inner = X_train_outer.iloc[inner_val_idx, :]

                    y_train_inner = y_train_outer[inner_train_idx]
                    y_val_inner = y_train_outer[inner_val_idx]

                    start_train = time.time()
                    if not ridge :
                        coxnet_pipe = make_pipeline(
                            StandardScaler(),
                            CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=estimated_alphas, fit_baseline_model=True)
                        )
                        coxnet_pipe.fit(X_train_inner, y_train_inner)
                        model = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]
                        scaler = coxnet_pipe.named_steps["standardscaler"]
                        X_val_scaled  = scaler.transform(X_val_inner)
                    tot_time_train += time.time() - start_train

                    start_validate = time.time()
                    for i, alpha in enumerate(estimated_alphas):
                        if ridge:
                            coxnet_pipe = make_pipeline(
                                StandardScaler(),
                                CoxPHSurvivalAnalysis(alpha=alpha)
                            )
                            coxnet_pipe.fit(X_train_inner, y_train_inner)
                            model = coxnet_pipe.named_steps["coxphsurvivalanalysis"]
                            scaler = coxnet_pipe.named_steps["standardscaler"]
                            X_val_scaled  = scaler.transform(X_val_inner)
                        if validation_function == "ibs":
                            survs = model.predict_survival_function(X_val_scaled, alpha = alpha)
                            times = np.sort(np.unique(y_val_inner["time"]))
                            upper = min(
                                np.max( y_train_inner["time"]),
                                np.max(y_val_inner["time"])
                            )

                            times = times[times < upper]
                            preds = np.vstack([fn(times) for fn in survs])
                            score = integrated_brier_score(y_val_inner,y_val_inner,preds,times)
                        elif validation_function == "vvh":
                            if not ridge:
                                coefs = model.coef_[:, i]
                            else:
                                coefs = model.coef_
                            nonzero_mask = coefs != 0
                            if(nonzero_mask.sum()<3):
                                score = np.inf
                            else:
                                score = vvh_cv(model, alpha, X_train_inner,y_train_inner, X_val_scaled, y_val_inner, ridge=ridge)
                        alpha_scores[alpha].append(score)
                    tot_time_validate += time.time() - start_validate

                mean_alpha_scores = {alpha: np.mean(scores) for alpha, scores in alpha_scores.items()}
                best_alpha = min(mean_alpha_scores, key=mean_alpha_scores.get)

                # Retrain on full outer train with best alpha
                if not ridge:
                    coxnet_pipe = make_pipeline(
                        StandardScaler(),
                        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_alpha], fit_baseline_model=True)
                    )
                    coxnet_pipe.fit(X_train_outer, y[train_idx])
                    model = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]
                else:
                    coxnet_pipe = make_pipeline(
                        StandardScaler(),
                        CoxPHSurvivalAnalysis(alpha=best_alpha)
                    )
                    coxnet_pipe.fit(X_train_outer, y[train_idx])
                    model = coxnet_pipe.named_steps["coxphsurvivalanalysis"]
                scaler = coxnet_pipe.named_steps["standardscaler"]
                X_test_scaled = scaler.transform(X_test_outer)

                if not ridge:
                    survs = model.predict_survival_function(X_test_scaled, alpha=best_alpha)
                else:
                    survs = model.predict_survival_function(X_test_scaled)

                times = np.sort(np.unique(y[test_idx]["time"]))
                upper = min(np.max(y[train_idx]["time"]), np.max(y[test_idx]["time"]))
                times = times[times < upper]

                print(f"    time range train: [{min(y[train_idx]['time'])}, {max(y[train_idx]['time'])}]")

                preds = np.vstack([fn(times) for fn in survs])
                ibs_score = integrated_brier_score(y[test_idx], y[test_idx], preds, times)

                if not ridge:
                    preds_for_cindex = model.predict(X_test_scaled, alpha=best_alpha)
                else:
                    preds_for_cindex = model.predict(X_test_scaled)

                c_index = concordance_index_ipcw(y[test_idx], y[test_idx], preds_for_cindex)[0]

                step_values = {
                    "fold": outer_fold,
                    "cindex_default": c_index,
                    "graf": ibs_score,
                    "time_train": tot_time_train,
                    "time_eval": tot_time_validate,
                }
                print(f"    {step_values}")
                outer_results.append(step_values)

            results_df = pd.DataFrame(outer_results)
            results_df.to_csv(results_path, index=False)
            print(f"  Saved results to {results_path}")

        except Exception as e:
            print(f"  ERROR on {cancer_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll cancers processed.")

#===============================================================================================
if __name__ == "__main__":
    main()