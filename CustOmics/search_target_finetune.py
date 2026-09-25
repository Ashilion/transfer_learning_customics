"""Nested CV with pre-trained CustOMICS fine-tuning + CoxNet, Optuna HP search.

Recherche Optuna pour UN outer fold donné (--outer_fold), en partant du
checkpoint pré-entraîné produit par `eval_source_pretrain.py`. Voir
`eval_target_finetune.py` pour le script pendant qui réentraîne le modèle
final et calcule les métriques.
"""
import sys

import numpy as np
import optuna
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.append('..')
import utils.folds_utils as fold_utils

from custcox_utils import (fit_feature_selector, apply_feature_selector, 
fit_coxnet, build_survival_array, fit_scalers, apply_scalers)
from src.tools.utils import get_sub_omics_df
from missing_data_load_all import simulate_missing_modalities, apply_missing_modalities

from pipeline_utils.checkpoints import TransferPaths
from pipeline_utils.cli import (
    base_parser, add_cancer_arg, add_outer_cv_args, add_inner_cv_args,
    add_optuna_args, add_cox_args, add_transfer_paths_args, add_missing_modality_args,
)
from pipeline_utils.cox_cv import estimate_alpha_grid, run_inner_cv_cox
from pipeline_utils.data import load_cancer_data, load_clinical_test, build_omics_dict
from pipeline_utils.finetune import resolve_finetune_architecture, build_and_finetune
from pipeline_utils.optuna_utils import get_or_create_study


def parse_args():
    parser = base_parser("Nested CV with pre-trained CustOMICS fine-tuning + CoxNet, Optuna HP search.")
    add_cancer_arg(parser)
    add_transfer_paths_args(parser)
    add_outer_cv_args(parser, outer_fold_required=True)
    add_inner_cv_args(parser)
    add_optuna_args(parser)
    add_cox_args(parser)
    add_missing_modality_args(parser)
    parser.add_argument("--n_epochs_ft", type=int, default=200,
        help="Max fine-tuning epochs (ceiling when early stopping is active).")
    parser.add_argument("--add_clinical", action="store_true", default=False,
        help="Concatenate clinical features to the latent representation before CoxNet.")
    parser.add_argument("--limit_epochs", type=int, default=None,
        help="Limit the number of fine-tuning epochs (overrides --n_epochs_ft) and disables early stopping.")
    parser.add_argument("--supervised", action="store_true", default=False, help="Train in supervised mode.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam",
        help="Optimizer used for fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
        help="Weight decay / L2 reg of the model's weights during fine-tuning.")
    return parser.parse_args()


# ===== Optuna objective =====================================================

def make_objective(cfg):

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
        lr2 = trial.suggest_float("lr2", 5e-5, 1e-3, log=True)
        patience = trial.suggest_int("patience", 5, 25, step=5)
        delta_min = trial.suggest_float("delta_min", 1e-8, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0, 0.3)

        finetune_arch = "full_retrain"  # trial.suggest_categorical(...) désactivé dans l'original
        arch_variant, expand_load = resolve_finetune_architecture(
            {**cfg["base_arch"], "dropout": dropout}, finetune_arch,
        )

        lambda_surv = 5 if cfg["unsupervised"] else trial.suggest_float("lambda_surv", 1, 1e3)

        if cfg["limit_epochs"]:
            effective_patience = None
            n_epochs_ft_eff = cfg["limit_epochs"]
        else:
            effective_patience = patience
            n_epochs_ft_eff = cfg["n_epochs_ft"]

        if cfg["modality_dropout"]:
            modality_dropout_p = trial.suggest_float("modality_dropout_p", 0.01, 0.5, log=True)
        else:
            modality_dropout_p = None

        ref_model = build_and_finetune(
            cfg["omics_train_outer"], cfg["sources"], cfg["best_params"], cfg["device"],
            arch_variant, cfg["unsupervised"], cfg["ckpt_path"],
            cfg["clinical_df"], cfg["label"], cfg["event"], cfg["surv_time"],
            cfg["batch_size"], n_epochs_ft_eff, cfg["task"],
            lr1=lr, lr2=lr2, patience=effective_patience, min_delta=delta_min,
            expand_load=expand_load, lambda_surv=lambda_surv,
            modality_dropout_p=modality_dropout_p, md_mode=cfg["md_mode"],
            modality_mask_train=cfg["mask_train_outer"], missing_strategy=cfg["missing_strategy"],
            optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
        )
        Z_outer_ref = ref_model.get_latent_representation(cfg["omics_train_outer"])

        estimated_alphas = estimate_alpha_grid(
            Z_outer_ref, cfg["samples_train_outer"], cfg["y_train_outer"], cfg["l1_ratio"],
            offset=cfg["offset"], clinical_test=cfg["clinical_test"],
        )
        print(f"  -> {len(estimated_alphas)} alpha candidates")

        def fit_and_embed(samples_train_inner, samples_val_inner):
            omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
            omics_val_raw = get_sub_omics_df(cfg["omics_df"], samples_val_inner)
            sel = cfg["sel_outer"]  # réutilise le sélecteur fitté au niveau outer, pas refit ici
            omics_train = apply_feature_selector(omics_train_raw, sel)
            omics_val = apply_feature_selector(omics_val_raw, sel)

            scalers_inner = fit_scalers(omics_train)
            omics_train = apply_scalers(omics_train, scalers_inner)
            omics_val = apply_scalers(omics_val, scalers_inner)

            mask_train_inner = None
            if cfg["missing_assignment"] is not None:
                omics_train, mask_train_inner = apply_missing_modalities(omics_train, cfg["missing_assignment"])
                omics_val, _ = apply_missing_modalities(omics_val, cfg["missing_assignment"])

            model = build_and_finetune(
                omics_train, cfg["sources"], cfg["best_params"], cfg["device"],
                arch_variant, cfg["unsupervised"], cfg["ckpt_path"],
                cfg["clinical_df"], cfg["label"], cfg["event"], cfg["surv_time"],
                cfg["batch_size"], n_epochs_ft_eff, cfg["task"],
                lr1=lr, lr2=lr2, patience=effective_patience, min_delta=delta_min,
                expand_load=expand_load, lambda_surv=lambda_surv,
                modality_dropout_p=modality_dropout_p, md_mode=cfg["md_mode"],
                modality_mask_train=mask_train_inner, missing_strategy=cfg["missing_strategy"],
                optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
            )
            Z_train = model.get_latent_representation(omics_train)
            Z_val = model.get_latent_representation(omics_val)
            return Z_train, Z_val

        best_alpha, best_score, _ = run_inner_cv_cox(
            cfg["inner_cv"], cfg["samples_train_outer"], cfg["y_train_outer"],
            estimated_alphas, cfg["l1_ratio"], cfg["validation_function"],
            fit_and_embed, cfg["clinical_df"], cfg["event"], cfg["surv_time"],
            offset=cfg["offset"], clinical_test=cfg["clinical_test"], ridge=cfg["ridge"],
            trial=trial, trial_pruning=cfg["trial_pruning"],
        )

        trial.set_user_attr("best_alpha", float(best_alpha))
        trial.set_user_attr("estimated_alphas", list(estimated_alphas))
        return best_score

    return objective


# ===== Main ==================================================================

def main():
    args = parse_args()
    unsupervised = not args.supervised
    paths = TransferPaths(args.output_dir, args.cancer, args.pretrain_ckpt, args.best_params_in)

    with open(paths.selector, "rb") as f:
        import pickle
        sel_outer = pickle.load(f)

    print(f"\n{'='*60}")
    print(f"  Cancer         : {args.cancer}")
    print(f"  Checkpoint     : {paths.pretrain_ckpt}")
    print(f"  Outer fold     : {args.outer_fold} / {args.outer_splits}")
    print(f"  Inner folds    : {args.inner_splits}")
    print(f"  Max epochs FT  : {args.n_epochs_ft}")
    print(f"  Add clinical   : {args.add_clinical}")
    print(f"  Limit Epochs   : {args.limit_epochs}")
    print(f"  Optuna trials  : {args.n_trials_per_worker}  |  timeout: {args.timeout}s")
    print(f"  Saved folds    : {args.use_saved_folds}")
    print(f"  Ridge          : {args.ridge}")
    print(f"  Modality dropout : {args.modality_dropout}  |  mode : {args.md_mode}")
    print(f"  Missing rate   : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"{'='*60}\n")

    import json
    with open(paths.best_params, "r") as f:
        best_config = json.load(f)
    best_params = best_config["best_params"]
    arch = best_config["architecture"]
    print(f"Loaded hyperparams: {best_params}")

    data = load_cancer_data(args.cancer)
    clinical_df = data["clinical"]
    # NB : mêmes clés de sources que lors du pré-entraînement (voir docstring
    # du module) — indispensable pour que fit_transfer recharge correctement
    # les poids par source depuis le checkpoint.
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
    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_train_outer = apply_feature_selector(omics_train_outer_raw, sel_outer)

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
        print(f"  -> {n_missing}/{len(lt_samples)} patients avec une modalité manquante simulée "
              f"(repetition {repetition_id})")

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, "status", "time")

    cfg = dict(
        omics_df=omics_df, omics_train_outer=omics_train_outer,
        clinical_df=clinical_df, clinical_test=clinical_test, offset=offset,
        samples_train_outer=samples_train_outer, y_train_outer=y_train_outer,
        sources=sources, best_params=best_params, ckpt_path=paths.pretrain_ckpt, device=device,
        base_arch=arch, unsupervised=unsupervised, batch_size=32, n_epochs_ft=args.n_epochs_ft,
        label="status", event="status", surv_time="time", task="survival",
        validation_function="vvh", l1_ratio=args.l1_ratio, ridge=args.ridge,
        add_clinical_to_cox=args.add_clinical, limit_epochs=args.limit_epochs,
        inner_cv=StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=0),
        sel_outer=sel_outer, trial_pruning=True,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        modality_dropout=args.modality_dropout, md_mode=args.md_mode,
        missing_assignment=missing_assignment, missing_strategy=args.missing_strategy,
        mask_train_outer=mask_train_outer,
    )

    study_name = f"ft_{args.name_suffix}{args.cancer}_fold{args.outer_fold}"
    journal_file = f"optuna_journal/journal_{args.name_suffix}ft_{args.cancer}_fold{args.outer_fold}.log"
    study = get_or_create_study(study_name, journal_file)
    study.optimize(make_objective(cfg), n_trials=args.n_trials_per_worker,
                    timeout=args.timeout, catch=(Exception,))


if __name__ == "__main__":
    main()