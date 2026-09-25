"""Post-Optuna evaluation for TARGET fine-tuning (one outer fold).

Recharge l'étude Optuna produite par `search_target_finetune.py` pour un
outer fold donné, refait le fine-tuning final sur l'outer train complet,
puis calcule C-index / IBS sur l'outer test. Sauvegarde un CSV par fold.

$
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

sys.path.append('..')
import utils.folds_utils as fold_utils


from custcox_utils import (fit_feature_selector, apply_feature_selector, 
fit_coxnet, build_survival_array, fit_scalers, apply_scalers)

from src.tools.utils import get_sub_omics_df
from missing_data_load_all import simulate_missing_modalities, apply_missing_modalities

from pipeline_utils.checkpoints import TransferPaths
from pipeline_utils.cli import (
    base_parser, add_cancer_arg, add_outer_cv_args, add_cox_args, add_transfer_paths_args,
    add_missing_modality_args,
)
from pipeline_utils.cox_cv import maybe_concat_clinical
from pipeline_utils.data import load_cancer_data, load_clinical_test, build_omics_dict
from pipeline_utils.finetune import resolve_finetune_architecture, build_and_finetune
from pipeline_utils.optuna_utils import load_best_trial


def parse_args():
    parser = base_parser("Post-Optuna evaluation for TARGET fine-tuning (one outer fold).")
    add_cancer_arg(parser)
    add_transfer_paths_args(parser)
    add_outer_cv_args(parser, outer_fold_required=True)
    add_cox_args(parser)
    add_missing_modality_args(parser)
    parser.add_argument("--n_epochs_ft", type=int, default=600, help="Max fine-tuning epochs.")
    parser.add_argument("--add_clinical", action="store_true", default=False,
        help="Concatenate clinical features to the latent representation before CoxNet.")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix used during training.")
    parser.add_argument("--limit_epochs", type=int, default=None, help="Hard-limit epochs.")
    parser.add_argument("--supervised", action="store_true", default=False, help="Train in supervised mode.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam",
        help="Optimizer used for the final fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
        help="Weight decay / L2 reg of the model's weights for the final fine-tuning.")
    return parser.parse_args()


def main():
    args = parse_args()
    unsupervised = not args.supervised
    n_epochs_ft = args.limit_epochs if args.limit_epochs else args.n_epochs_ft

    print(f"\n{'='*60}")
    print(f"  [TARGET eval]  Cancer : {args.cancer}")
    print(f"  Outer fold     : {args.outer_fold} / {args.outer_splits}")
    print(f"  Study suffix   : '{args.name_suffix}'")
    print(f"  Max epochs FT  : {n_epochs_ft}")
    print(f"  Add clinical   : {args.add_clinical}")
    print(f"  Limit Epochs   : {args.limit_epochs}")
    print(f"  Modality dropout : {args.modality_dropout}  |  mode : {args.md_mode}")
    print(f"  Missing rate   : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"{'='*60}\n")

    study_name = f"ft_{args.name_suffix}{args.cancer}_fold{args.outer_fold}"
    journal_file = f"optuna_journal/journal_{args.name_suffix}ft_{args.cancer}_fold{args.outer_fold}.log"
    _, best_trial = load_best_trial(study_name, journal_file)

    best_params = best_trial.params
    best_alpha = best_trial.user_attrs["best_alpha"]
    estimated_alphas = best_trial.user_attrs.get("estimated_alphas", None)
    best_score = best_trial.value

    best_lr = best_params["lr"]
    best_lr2 = best_params["lr2"] * 10
    best_patience = best_params["patience"]
    best_delta_min = best_params["delta_min"]
    best_finetune_arch = best_params.get("finetune_arch", "full_retrain")
    lambda_surv = 5 if unsupervised else best_params["lambda_surv"]

    print(f"Best trial #{best_trial.number}")
    print(f"  lr={best_lr:.2e}  patience={best_patience}  delta_min={best_delta_min:.2e}  lr2={best_lr2:.2e}")
    print(f"  finetune_arch={best_finetune_arch}")
    print(f"  alpha={best_alpha:.6f}  score={best_score:.4f}")
    print(f"  lambda surv {lambda_surv}")

    paths = TransferPaths(args.output_dir, args.cancer, args.pretrain_ckpt, args.best_params_in)

    with open(paths.best_params, "r") as f:
        best_config = json.load(f)
    pretrain_params = best_config["best_params"]
    arch = best_config["architecture"]

    arch_variant, expand_load = resolve_finetune_architecture(arch, best_finetune_arch)

    with open(paths.selector, "rb") as f:
        sel_outer = pickle.load(f)

    data = load_cancer_data(args.cancer)
    clinical_df = data["clinical"]
    # NB : mêmes clés de sources que lors du pré-entraînement / de la
    # recherche fine-tuning — voir docstring du module.
    omics_df = build_omics_dict(data, naming="clinical_labels")

    clinical_test, offset = None, None
    if args.add_clinical:
        clinical_test = load_clinical_test(args.cancer)
        offset = clinical_df.index[0] - clinical_test.index[0]

    lt_samples = list(clinical_df.index)
    sources = list(omics_df.keys())
    device = torch.device("cpu")

    if args.use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(args.cancer, src="../data/splits.json")
        train_idx, test_idx = train_folds[args.outer_fold], test_folds[args.outer_fold]
    else:
        outer_cv = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
        splits = list(outer_cv.split(lt_samples, clinical_df.loc[:, "status"]))
        train_idx, test_idx = splits[args.outer_fold]

    print(f"\n{'='*50}\n OUTER FOLD {args.outer_fold}\n{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

    omics_train_outer = apply_feature_selector(omics_train_outer_raw, sel_outer)
    omics_test_outer = apply_feature_selector(omics_test_outer_raw, sel_outer)

    # ===== Simulation de modalités manquantes (avant le scaling), même
    # repetition_id que search_target_finetune.py pour ce fold =====
    repetition_id = args.outer_fold // args.outer_splits
    missing_assignment = None
    mask_train_outer = None
    if args.missing_rate > 0:
        missing_assignment = simulate_missing_modalities(
            sample_ids=lt_samples,
            sources=sources,
            missing_rate=args.missing_rate,
            repetition_seed=repetition_id,
        )
        omics_train_outer, mask_train_outer = apply_missing_modalities(omics_train_outer, missing_assignment)
        n_missing = sum(1 for v in missing_assignment.values() if v is not None)
        print(f"  -> {n_missing}/{len(lt_samples)} patients avec une modalité manquante simulée sur le TRAIN "
              f"(repetition {repetition_id}, test set laissé complet)")

    # ===== Scaling (fit sur train uniquement) =====
    scalers = fit_scalers(omics_train_outer)
    omics_train_outer = apply_scalers(omics_train_outer, scalers)
    omics_test_outer = apply_scalers(omics_test_outer, scalers)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, "status", "time")
    y_test_outer = build_survival_array(clinical_df, samples_test_outer, "status", "time")

    print("\n=== Final fine-tuning on outer train ===")
    patience = None if args.limit_epochs else best_patience

    modality_dropout_p = best_params.get("modality_dropout_p")

    final_model = build_and_finetune(
        omics_train_outer, sources, pretrain_params, device,
        arch_variant, unsupervised, paths.pretrain_ckpt,
        clinical_df, "status", "status", "time",
        32, n_epochs_ft, "survival",
        lr1=best_lr, lr2=best_lr2, patience=patience, min_delta=best_delta_min,
        expand_load=expand_load, lambda_surv=lambda_surv, track_loss_components=True,
        modality_dropout_p=modality_dropout_p, md_mode=args.md_mode,
        modality_mask_train=mask_train_outer, missing_strategy=args.missing_strategy,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
    )

    Z_train_outer = final_model.get_latent_representation(omics_train_outer)
    Z_test_outer = final_model.get_latent_representation(omics_test_outer)

    X_train_cox = maybe_concat_clinical(Z_train_outer, samples_train_outer, offset, clinical_test)
    X_test_cox = maybe_concat_clinical(Z_test_outer, samples_test_outer, offset, clinical_test)

    alpha_list = [best_alpha] if estimated_alphas is None else estimated_alphas
    coxnet, scaler = fit_coxnet(X_train_cox, y_train_outer, args.l1_ratio, alpha_list, ridge=args.ridge)
    X_test_scaled = scaler.transform(X_test_cox)

    if args.ridge:
        fitted = coxnet[best_alpha]
        risk_scores = fitted.predict(X_test_scaled)
        survs = fitted.predict_survival_function(X_test_scaled)
    else:
        risk_scores = coxnet.predict(X_test_scaled, alpha=best_alpha)
        survs = coxnet.predict_survival_function(X_test_scaled, alpha=best_alpha)

    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    times = np.sort(np.unique(y_test_outer["time"]))
    upper = min(y_train_outer["time"].max(), y_test_outer["time"].max())
    times = times[times < upper]
    preds = np.vstack([fn(times) for fn in survs])
    ibs = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    result = [{
        "fold": args.outer_fold,
        "cindex": c_index,
        "ibs": ibs,
        "best_lr": best_lr,
        "best_patience": best_patience,
        "best_delta_min": best_delta_min,
        "best_finetune_arch": best_finetune_arch,
        "best_alpha": best_alpha,
        "best_score": best_score,
        "trial_number": best_trial.number,
        "add_clinical": args.add_clinical,
        "limit_epochs": args.limit_epochs,
        "modality_dropout": args.modality_dropout,
        "modality_dropout_p": modality_dropout_p,
        "md_mode": args.md_mode,
        "missing_rate": args.missing_rate,
        "missing_strategy": args.missing_strategy,
    }]
    out_path = f"{args.output_dir}/ncv_finetune_optuna_{args.name_suffix}{args.cancer}_fold{args.outer_fold}.csv"
    pd.DataFrame(result).to_csv(out_path, index=False)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    main()