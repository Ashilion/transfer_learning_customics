import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
import time
import optuna

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os

import sys
sys.path.append('..')
from utils.vvh_cv import vvh_cv
import utils.folds_utils as fold_utils

from custcox_utils import (
    fit_feature_selector, apply_feature_selector, build_customics_model,
    build_survival_array, fit_coxnet, evaluate_survival,fit_transfer, get_finetune_architecture,
    freeze_and_reset_optimizer, unfreeze_and_reset_optimizer
)

from tl_custcox_source_eval_all import build_customics_model_plus

# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested CV with pre-trained CustOMICS fine-tuning + CoxNet, Optuna HP search.",
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
        help="Path to the pre-trained model checkpoint."
    )
    parser.add_argument(
        "--best_params_in", 
        type=str, 
        default="best_params_source.json",
        help="Path to the JSON file with best hyperparameters from pre-training."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="tl_ckpt",
        help="Directory to save output results."
    )
    parser.add_argument("--outer_splits", 
        type=int, 
        default=5,
        help="Number of outer CV folds."
    )
    parser.add_argument(
        "--inner_splits", 
        type=int, 
        default=3,
        help="Number of inner CV folds."
    )
    parser.add_argument(
        "--n_epochs_ft", 
        type=int, 
        default=200,
        help="Max fine-tuning epochs (ceiling when early stopping is active)."
    )
    parser.add_argument(
        "--l1_ratio", 
        type=float, 
        default=0.01,
        help="L1 ratio for CoxNet regularization."
    )
    parser.add_argument(
        "--use_saved_folds", 
        action="store_true", 
        default=False,
        help="Use pre-saved outer fold splits from splits.json."
    )
    parser.add_argument(
        "--add_clinical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate clinical features to the latent representation before CoxNet."
    )
    parser.add_argument(
        "--limit_epochs",
        type=int,
        default=None,
        help="Limit the number of fine-tuning epochs (overrides --n_epochs_ft) and disables early stopping."
    )

    parser.add_argument(
        "--outer_fold", 
        type=int, 
        required=True,
        help="Index of the outer fold to process."
    )
    parser.add_argument(
        "--n_trials_per_worker", 
        type=int, 
        default=30,
        help="Number of Optuna trials."
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=None,
        help="Optuna timeout in seconds."
    )
    parser.add_argument(
        "--name_suffix", 
        type=str, 
        default="",
        help="String suffix added to study name and journal filename."
    )
    parser.add_argument(
        "--supervised", 
        action="store_true", 
        default=False,
        help="Train in supervised mode."
    )
    parser.add_argument(
        "--ridge",
        action="store_true",
        default=False,
        help="Use ridge (L2) CoxPHSurvivalAnalysis instead of Elastic Net CoxnetSurvivalAnalysis"
    )
    return parser.parse_args()


# =====================================================================================

def load_cancer_data(cancer_name, per_cancer_dir="../data/union_separated_cancer", prefix="dict_pancancer"):
    path = f"{per_cancer_dir}/{prefix}_{cancer_name}.pickle"
    with open(path, "rb") as f:
        return pickle.load(f)



# ===== Optuna objective ================================================================================

def make_objective(cfg):

    def objective(trial):
        lr        = trial.suggest_float("lr",        5e-4, 1e-2, log=True)
        lr2       = trial.suggest_float("lr2",        5e-5, 1e-3, log=True)
        patience  = trial.suggest_int("patience",    5, 25, step=5)
        delta_min = trial.suggest_float("delta_min", 1e-8, 1e-3, log=True)
        dropout  = trial.suggest_float("dropout", 0, 0.3)
        beta =    trial.suggest_float("beta", 1e-4, 1e4, log=True),
        # finetune_arch = trial.suggest_categorical(
        #     "finetune_arch",
        #     ["full_retrain", "widen_10", "widen_25", "add_layer16"],
        # )
        finetune_arch = "full_retrain"
        expand_load = (finetune_arch != "full_retrain")

        arch_variant = get_finetune_architecture(
            {
                "hidden_dim":     cfg["hidden_dim"],
                "central_hidden": cfg["central_hidden"],
                "classifier_dim": cfg["classifier_dim"],
                "survival_dim":   cfg["survival_dim"],
                "dropout":        dropout,
                "num_classes":    cfg["num_classes"],
            },
            finetune_arch,
        )

        if cfg["unsupervised"]:
            lambda_surv = 5
        else:
            lambda_surv = trial.suggest_float("lambda_surv", 1, 1e3)

        if cfg["limit_epochs"]:
            effective_patience = None
            n_epochs_ft_eff = cfg["limit_epochs"]
        else:
            effective_patience = patience
            n_epochs_ft_eff = cfg["n_epochs_ft"]

        n_epochs_central = n_epochs_ft_eff // 2
        n_epochs_all     = n_epochs_ft_eff - n_epochs_central

        ref_model = build_customics_model_plus(
            cfg["omics_train_outer"], cfg["sources"], cfg["best_params"], cfg["device"],
            arch_variant["hidden_dim"], arch_variant["central_hidden"], arch_variant["classifier_dim"],
            arch_variant["survival_dim"], arch_variant["dropout"], arch_variant["num_classes"],
            cfg["unsupervised"], switch_epoch=0, lambda_surv=lambda_surv
        )
        ref_model = fit_transfer(
            ref_model, cfg["ckpt_path"], cfg["device"],
            cfg["omics_train_outer"], cfg["clinical_df"],
            cfg["label"], cfg["event"], cfg["surv_time"],
            cfg["batch_size"], n_epochs_central, n_epochs_all, cfg["task"],
            lr1=lr, lr2=lr2, patience=effective_patience, min_delta=delta_min,
            expand_load=expand_load,
        )
        Z_outer_ref = ref_model.get_latent_representation(cfg["omics_train_outer"])

        if cfg["add_clinical_to_cox"]:
            samples_adapted = [idx - cfg["offset"] for idx in cfg["samples_train_outer"]]
            clin_outer = cfg["clinical_test"].loc[samples_adapted, :]
            X_for_alpha = np.concatenate([clin_outer, Z_outer_ref], axis=1)
        else:
            X_for_alpha = Z_outer_ref

        ref_cox, _ = fit_coxnet(X_for_alpha, cfg["y_train_outer"], cfg["l1_ratio"])
        estimated_alphas = ref_cox.alphas_
        print(f"  -> {len(estimated_alphas)} alpha candidates")

        # --- Inner CV ---
        alpha_scores = {a: [] for a in estimated_alphas}

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            cfg["inner_cv"].split(cfg["samples_train_outer"], cfg["y_train_outer"]["status"])
        ):
            samples_train_inner = [cfg["samples_train_outer"][i] for i in inner_train_idx]
            samples_val_inner   = [cfg["samples_train_outer"][i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
            omics_val_raw   = get_sub_omics_df(cfg["omics_df"], samples_val_inner)

            sel         = cfg["sel_outer"]
            omics_train = apply_feature_selector(omics_train_raw, sel)
            omics_val   = apply_feature_selector(omics_val_raw,   sel)

            model = build_customics_model_plus(
                omics_train, cfg["sources"], cfg["best_params"], cfg["device"],
                arch_variant["hidden_dim"], arch_variant["central_hidden"], arch_variant["classifier_dim"],
                arch_variant["survival_dim"], arch_variant["dropout"], arch_variant["num_classes"],
                cfg["unsupervised"], switch_epoch=0,
            )
            
            model = fit_transfer(
                model, cfg["ckpt_path"], cfg["device"],
                omics_train, cfg["clinical_df"],
                cfg["label"], cfg["event"], cfg["surv_time"],
                cfg["batch_size"], n_epochs_central, n_epochs_all, cfg["task"],
                lr1=lr, lr2=lr, patience=effective_patience, min_delta=delta_min,
                expand_load=expand_load
            )
            model.plot_loss_detailed(save_path="results/loss_detailed_B.png")
            Z_train = model.get_latent_representation(omics_train)
            Z_val   = model.get_latent_representation(omics_val)

            y_train_struct = build_survival_array(cfg["clinical_df"], samples_train_inner, cfg["event"], cfg["surv_time"])
            y_val_struct   = build_survival_array(cfg["clinical_df"], samples_val_inner,   cfg["event"], cfg["surv_time"])

            if cfg["add_clinical_to_cox"]:
                train_adapted = [idx - cfg["offset"] for idx in samples_train_inner]
                clin_train = cfg["clinical_test"].loc[train_adapted, :]
                X_cox = np.concatenate([clin_train, Z_train], axis=1)

                val_adapted = [idx - cfg["offset"] for idx in samples_val_inner]
                clin_val = cfg["clinical_test"].loc[val_adapted, :]
                X_val = np.concatenate([clin_val, Z_val], axis=1)
            else:
                X_cox = Z_train
                X_val = Z_val

            coxnet, scaler = fit_coxnet(X_cox, y_train_struct, cfg["l1_ratio"], estimated_alphas,ridge=cfg["ridge"])
            X_val_scaled   = scaler.transform(X_val)

            for alpha in estimated_alphas:
                score = evaluate_survival(
                    coxnet, alpha,
                    X_cox, y_train_struct,
                    X_val_scaled, y_val_struct,
                    cfg["validation_function"],
                    ridge=cfg["ridge"]
                )
                alpha_scores[alpha].append(score)

            print(f"    inner fold {inner_fold} done")

            if cfg["trial_pruning"]:
                intermediate_best_score = np.inf
                for alpha in estimated_alphas:
                    mean_score = np.mean(alpha_scores[alpha])
                    if mean_score < intermediate_best_score:
                        intermediate_best_score = mean_score
                trial.report(intermediate_best_score, inner_fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        best_score = np.inf
        best_alpha = estimated_alphas[0]
        for alpha in estimated_alphas:
            mean_score = np.mean(alpha_scores[alpha])
            if mean_score < best_score:
                best_score = mean_score
                best_alpha = alpha

        trial.set_user_attr("best_alpha",        float(best_alpha))
        trial.set_user_attr("estimated_alphas",  list(estimated_alphas))
        return best_score

    return objective


# ===== Main ============================================================================================

def main():
    args = parse_args()

    TARGET_CANCER  = args.cancer
    PRETRAIN_CKPT  = args.pretrain_ckpt
    BEST_PARAMS_IN = args.best_params_in
    OUTPUT_DIR     = args.output_dir
    outer_fold     = args.outer_fold
    name_suffix    = args.name_suffix
    l1_ratio       = args.l1_ratio
    n_epochs_ft    = args.n_epochs_ft
    add_clinical_to_cox = args.add_clinical
    limit_epochs   = args.limit_epochs
    unsupervised   = not args.supervised
    ridge          = args.ridge 
    ckpt_filename = f"{TARGET_CANCER}_{os.path.basename(args.pretrain_ckpt)}"
    config_filename = f"{TARGET_CANCER}_{os.path.basename(args.best_params_in)}"

    pretrain_ckpt_path = os.path.join(args.output_dir, ckpt_filename)
    best_params_in_path = os.path.join(args.output_dir, config_filename)
    selector_path = f"{pretrain_ckpt_path}.selector.pkl"
    with open(selector_path, "rb") as f:
        sel_outer = pickle.load(f)

    print(f"\n{'='*60}")
    print(f"  Cancer         : {TARGET_CANCER}")
    print(f"  Checkpoint     : {pretrain_ckpt_path}")
    print(f"  Outer fold     : {outer_fold} / {args.outer_splits}")
    print(f"  Inner folds    : {args.inner_splits}")
    print(f"  Max epochs FT  : {n_epochs_ft}")
    print(f"  Add clinical   : {add_clinical_to_cox}")
    print(f"  Limit Epochs   : {limit_epochs}")
    print(f"  Optuna trials  : {args.n_trials_per_worker}  |  timeout: {args.timeout}s")
    print(f"  Saved folds    : {args.use_saved_folds}")
    print(f"  Ridge          : {ridge}")
    print(f"{'='*60}\n")

    with open(best_params_in_path, "r") as f:
        best_config = json.load(f)

    best_params    = best_config["best_params"]
    # best_alpha_src = best_config["best_alpha"]
    arch           = best_config["architecture"]
    print(f"Loaded hyperparams: {best_params}")

    data = load_cancer_data(TARGET_CANCER)
    clinical_df = data["clinical"]

    omics_df = {
        "protein":  data["_rna"],
        "gene_exp": data["mirna"],
        "methyl":   data["cnv"],
        "mutation": data["mutation"],
    }

    # --- Donnees cliniques (utilisees si --add_clinical) ---
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

    hidden_dim     = arch["hidden_dim"]
    central_hidden = arch["central_hidden"]
    classifier_dim = arch["classifier_dim"]
    survival_dim   = arch["survival_dim"]
    num_classes    = arch["num_classes"]
    dropout        = arch["dropout"]

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

    cfg = dict(
        omics_df            = omics_df,
        omics_train_outer   = omics_train_outer,
        omics_test_outer    = omics_test_outer,
        clinical_df         = clinical_df,
        clinical_test       = clinical_test,
        offset              = offset,
        samples_train_outer = samples_train_outer,
        y_train_outer       = y_train_outer,
        sources             = sources,
        best_params         = best_params,
        ckpt_path           = pretrain_ckpt_path,
        device              = device,
        hidden_dim          = hidden_dim,
        central_hidden      = central_hidden,
        classifier_dim      = classifier_dim,
        survival_dim        = survival_dim,
        num_classes         = num_classes,
        dropout             = dropout,
        unsupervised        = unsupervised,
        batch_size          = 32,
        n_epochs_ft         = n_epochs_ft,
        label               = "status",
        event               = "status",
        surv_time           = "time",
        task                = "survival",
        nbFeatures          = 5000,
        validation_function = "vvh",
        l1_ratio            = l1_ratio,
        ridge               = ridge,  
        add_clinical_to_cox = add_clinical_to_cox,
        limit_epochs        = limit_epochs,
        inner_cv            = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=0),
        sel_outer           = sel_outer,
        trial_pruning       = True,
    )

    os.makedirs("optuna_journal", exist_ok=True)
    journal_path = f"optuna_journal/journal_{name_suffix}ft_{TARGET_CANCER}_fold{outer_fold}.log"
    study_name   = f"ft_{name_suffix}{TARGET_CANCER}_fold{outer_fold}"

    storage = JournalStorage(JournalFileBackend(file_path=journal_path))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2, interval_steps=1)
    )
    study.optimize(
        make_objective(cfg),
        n_trials=args.n_trials_per_worker,
        timeout=args.timeout,
        catch=(Exception,),
    )


if __name__ == "__main__":
    main()