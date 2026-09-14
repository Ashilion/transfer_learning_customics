import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
import optuna

from sklearn.model_selection import StratifiedKFold
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
import utils.folds_utils as fold_utils
from custcox_utils import (
    fit_feature_selector, apply_feature_selector,
    build_customics_model, build_survival_array, fit_coxnet,
    evaluate_survival,fit_transfer, get_finetune_architecture,
    freeze_and_reset_optimizer, unfreeze_and_reset_optimizer
)

from tl_custcox_source_eval_all import build_customics_model_plus

import os
from tl_custcox_target_multi_study_1_outer import load_cancer_data

# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Optuna evaluation for TARGET fine-tuning (one outer fold).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default="COAD",
        help="Target cancer type (e.g. COAD, BRCA, LUAD)."
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default="pretrained_model.pt",
        help="Filename (relative to --output_dir, prefixed by cancer) of the pre-trained checkpoint."
    )
    parser.add_argument(
        "--best_params_in",
        type=str,
        default="best_params_source.json",
        help="Filename (relative to --output_dir, prefixed by cancer) of the JSON with best pretrain hyperparameters."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tl_ckpt",
        help="Directory to save output results (and where checkpoint/config are looked up)."
    )
    parser.add_argument("--outer_splits", type=int, default=5,
        help="Number of outer CV folds.")
    parser.add_argument("--outer_fold", type=int, required=True,
        help="Index of the outer fold to evaluate.")
    parser.add_argument("--n_epochs_ft", type=int, default=600,
        help="Max fine-tuning epochs.")
    parser.add_argument("--l1_ratio", type=float, default=0.01,
        help="L1 ratio for CoxNet regularization.")
    parser.add_argument("--use_saved_folds", action="store_true", default=False,
        help="Use pre-saved outer fold splits from splits.json.")
    parser.add_argument(
        "--add_clinical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate clinical features to the latent representation before CoxNet."
    )
    parser.add_argument("--name_suffix", type=str, default="",
        help="Suffix used in journal/study name during training.")
    parser.add_argument("--limit_epochs", type=int, default=None,
        help="Hard-limit epochs (disables early stopping).")
    parser.add_argument("--supervised", action="store_true", default=False,
        help="Train in supervised mode.")
    parser.add_argument(
        "--ridge",
        action="store_true",
        default=False,
        help="Use ridge (L2) CoxPHSurvivalAnalysis instead of Elastic Net CoxnetSurvivalAnalysis"
    )
    return parser.parse_args()


# ===== Main ============================================================================================

def main():
    args = parse_args()

    TARGET_CANCER = args.cancer
    outer_fold    = args.outer_fold
    name_suffix   = args.name_suffix
    limit_epochs  = args.limit_epochs
    n_epochs_ft   = limit_epochs if limit_epochs else args.n_epochs_ft
    l1_ratio      = args.l1_ratio
    add_clinical_to_cox = args.add_clinical
    unsupervised   = not args.supervised

    print(f"\n{'='*60}")
    print(f"  [TARGET eval]  Cancer : {TARGET_CANCER}")
    print(f"  Outer fold     : {outer_fold} / {args.outer_splits}")
    print(f"  Study suffix   : '{name_suffix}'")
    print(f"  Max epochs FT  : {n_epochs_ft}")
    print(f"  Add clinical   : {add_clinical_to_cox}")
    print(f"  Limit Epochs   : {limit_epochs}")
    print(f"{'='*60}\n")

    study_name   = f"ft_{name_suffix}{TARGET_CANCER}_fold{outer_fold}"
    journal_path = f"optuna_journal/journal_{name_suffix}ft_{TARGET_CANCER}_fold{outer_fold}.log"

    study = optuna.load_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(file_path=journal_path)),
    )

    best_trial    = study.best_trial
    best_params   = best_trial.params
    best_alpha    = best_trial.user_attrs["best_alpha"]
    estimated_alphas = best_trial.user_attrs.get("estimated_alphas", None)
    best_score    = best_trial.value

    best_lr             = best_params["lr"]
    best_lr2            = best_params["lr2"]*10
    best_patience       = best_params["patience"]
    best_delta_min      = best_params["delta_min"]
    best_finetune_arch  = best_params.get("finetune_arch", "full_retrain")
    if unsupervised:
        lambda_surv = 5
    else:
        lambda_surv = best_params["lambda_surv"]

    print(f"Best trial #{best_trial.number}")
    print(f"  lr={best_lr:.2e}  patience={best_patience}  delta_min={best_delta_min:.2e} lr2 ={best_lr2:.2e}")
    print(f"  finetune_arch={best_finetune_arch}")
    print(f"  alpha={best_alpha:.6f}  score={best_score:.4f}")
    print(f" lambda surv {lambda_surv}")


    ckpt_filename    = f"{TARGET_CANCER}_{os.path.basename(args.pretrain_ckpt)}"
    config_filename  = f"{TARGET_CANCER}_{os.path.basename(args.best_params_in)}"
    pretrain_ckpt_path  = os.path.join(args.output_dir, ckpt_filename)
    best_params_in_path = os.path.join(args.output_dir, config_filename)

    with open(best_params_in_path, "r") as f:
        best_config = json.load(f)

    pretrain_params = best_config["best_params"]
    arch            = best_config["architecture"]

    hidden_dim     = arch["hidden_dim"]
    central_hidden = arch["central_hidden"]
    classifier_dim = arch["classifier_dim"]
    survival_dim   = arch["survival_dim"]
    num_classes    = arch["num_classes"]
    dropout        = arch["dropout"]

    # --- Fine-tune architecture variant selected by Optuna (mirrors the search script) ---
    expand_load = (best_finetune_arch != "full_retrain")
    arch_variant = get_finetune_architecture(
        {
            "hidden_dim":     hidden_dim,
            "central_hidden": central_hidden,
            "classifier_dim": classifier_dim,
            "survival_dim":   survival_dim,
            "dropout":        dropout,
            "num_classes":    num_classes,
        },
        best_finetune_arch,
    )

    selector_path = f"{pretrain_ckpt_path}.selector.pkl"
    with open(selector_path, "rb") as f:
        sel_outer = pickle.load(f)

    data = load_cancer_data(TARGET_CANCER)
    clinical_df = data["clinical"]

    omics_df = {
        "_rna":  data["_rna"],
        "mirna": data["mirna"],
        "cnv":   data["cnv"],
        "mutation": data["mutation"],
    }

    # --- Clinical data used only if --add_clinical is set (same convention as the search script) ---
    clinical_test = None
    offset = None
    if add_clinical_to_cox:
        clinical_path = f"../data/clinical/{TARGET_CANCER}_clinical.pickle"
        with open(clinical_path, "rb") as f:
            df_clin = pickle.load(f)
        clinical_test = df_clin[list(set(df_clin.columns) - {"time", "bcr_patient_barcode", "status"})]
        offset = clinical_df.index[0] - clinical_test.index[0]

    lt_samples = list(clinical_df.index)
    sources    = list(omics_df.keys())
    device     = torch.device("cpu")

    if args.use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(TARGET_CANCER, src="../data/splits.json")
        train_idx, test_idx = train_folds[outer_fold], test_folds[outer_fold]
    else:
        outer_cv = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
        splits   = list(outer_cv.split(lt_samples, clinical_df.loc[:, "status"]))
        train_idx, test_idx = splits[outer_fold]

    print(f"\n{'='*50}")
    print(f" OUTER FOLD {outer_fold}")
    print(f"{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer  = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw  = get_sub_omics_df(omics_df, samples_test_outer)

    omics_train_outer = apply_feature_selector(omics_train_outer_raw, sel_outer)
    omics_test_outer  = apply_feature_selector(omics_test_outer_raw,  sel_outer)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, "status", "time")
    y_test_outer  = build_survival_array(clinical_df, samples_test_outer,  "status", "time")

    print("\n=== Final fine-tuning on outer train ===")
    
    n_epochs_central = n_epochs_ft // 2
    n_epochs_all     = n_epochs_ft - n_epochs_central
    patience         = None if limit_epochs else best_patience

   
    final_model = build_customics_model_plus(
        omics_train_outer, sources, pretrain_params, device,
        arch_variant["hidden_dim"], arch_variant["central_hidden"], arch_variant["classifier_dim"],
        arch_variant["survival_dim"], arch_variant["dropout"], arch_variant["num_classes"],
        unsupervised, switch_epoch=0,
    )
    final_model = fit_transfer(
        final_model, pretrain_ckpt_path, device,
        omics_train_outer, clinical_df,
        "status", "status", "time",
        32, n_epochs_central, n_epochs_all, "survival",
        lr1=best_lr,lr2=best_lr2, patience=patience, min_delta=best_delta_min,
        expand_load=expand_load,track_loss_components=True
    )
    
    Z_train_outer = final_model.get_latent_representation(omics_train_outer)
    Z_test_outer  = final_model.get_latent_representation(omics_test_outer)

    
    if add_clinical_to_cox:
        train_adapted = [idx - offset for idx in samples_train_outer]
        clin_train = clinical_test.loc[train_adapted, :]
        X_train_cox = np.concatenate([clin_train, Z_train_outer], axis=1)

        test_adapted = [idx - offset for idx in samples_test_outer]
        clin_test = clinical_test.loc[test_adapted, :]
        X_test_cox = np.concatenate([clin_test, Z_test_outer], axis=1)
    else:
        X_train_cox = Z_train_outer
        X_test_cox  = Z_test_outer

    alpha_list = [best_alpha] if estimated_alphas is None else estimated_alphas
    coxnet, scaler = fit_coxnet(X_train_cox, y_train_outer, l1_ratio, alpha_list, ridge = args.ridge)
    X_test_scaled  = scaler.transform(X_test_cox)

    if args.ridge:
        model = coxnet[best_alpha]
        risk_scores = model.predict(X_test_scaled)
        survs = model.predict_survival_function(X_test_scaled)
    else:
        model = coxnet
        risk_scores = model.predict(X_test_scaled, alpha=best_alpha)
        survs = model.predict_survival_function(X_test_scaled, alpha=best_alpha)
    # --- Metrics ---
    c_index     = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]


    times  = np.sort(np.unique(y_test_outer["time"]))
    upper  = min(y_train_outer["time"].max(), y_test_outer["time"].max())
    times  = times[times < upper]
    preds  = np.vstack([fn(times) for fn in survs])
    ibs    = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs:.4f}")
    # --- Save result ---
    os.makedirs(args.output_dir, exist_ok=True)
    result = [{
        "fold":        outer_fold,
        "cindex":      c_index,
        "ibs":         ibs,
        "best_lr":     best_lr,
        "best_patience":  best_patience,
        "best_delta_min": best_delta_min,
        "best_finetune_arch": best_finetune_arch,
        "best_alpha":  best_alpha,
        "best_score":  best_score,
        "trial_number": best_trial.number,
        "add_clinical": add_clinical_to_cox,
        "limit_epochs": limit_epochs,
    }]
    out_path = f"{args.output_dir}/ncv_finetune_optuna_{name_suffix}{TARGET_CANCER}_fold{outer_fold}.csv"
    pd.DataFrame(result).to_csv(out_path, index=False)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    main()