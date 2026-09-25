"""Nested CV with CustOMICS + CoxNet + Optuna for survival prediction.

Recherche Optuna pour UN outer fold donné (--outer_fold). Voir
`eval_ncv.py` pour le script pendant qui recharge l'étude et calcule les
métriques finales (C-index / IBS).

"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append('..')
import utils.folds_utils as fold_utils

from custcox_utils import (fit_feature_selector, apply_feature_selector, 
build_survival_array, fit_scalers, apply_scalers)
from src.tools.utils import get_sub_omics_df
from missing_data_load_all import simulate_missing_modalities, apply_missing_modalities

from pipeline_utils.cli import (
    base_parser, add_cancer_arg, add_outer_cv_args, add_inner_cv_args,
    add_optuna_args, add_training_args, add_cox_args, add_missing_modality_args,
)
from pipeline_utils.data import load_cancer_data, load_clinical_test, build_omics_dict
from pipeline_utils.hidden_dims_params import suggest_autoencoder_hidden_dims
from pipeline_utils.model import build_customics_model
from pipeline_utils.optuna_utils import get_or_create_study


def parse_args():
    parser = base_parser("Nested CV with CustOMICS + CoxNet + Optuna for survival prediction.")
    add_cancer_arg(parser)
    add_outer_cv_args(parser, outer_fold_required=True)
    add_inner_cv_args(parser)
    add_optuna_args(parser)
    add_training_args(parser)
    add_cox_args(parser)
    add_missing_modality_args(parser)
    parser.add_argument("--nb_features", type=int, default=5000,
        help="Number of maximum features per omics.")
    parser.add_argument("--linear_decoder", action="store_true", default=False,
        help="Remove activation layers from the central decoder to push the model "
             "to have a 'linear' latent representation.")
    parser.add_argument("--multiproc", type=int, default=1,
        help="Number of parallel worker processes for Optuna (each runs "
             "n_trials_per_worker trials, sharing the same JournalStorage file). "
             "1 = sequential, no multiprocessing.")
    return parser.parse_args()


# ===== Optuna objective =====================================================

def make_objective(cfg, args):

    def objective(trial):
        omics_train_outer = cfg["omics_train_outer"]
        samples_train_outer = cfg["samples_train_outer"]
        y_train_outer = cfg["y_train_outer"]

        autoencoder_hidden_dims = suggest_autoencoder_hidden_dims(
            trial, omics_train_outer, cfg["sources"]
        )

        params = {
            "latent_dim": 32,
            "rep_dim": 32,
            "central_hidden_dims": [64],
            "lambda_central": 1,
            "patience": trial.suggest_int("patience", 5, 50, step=5),
            "autoencoder_hidden_dims": autoencoder_hidden_dims,
            "dropout": trial.suggest_float("dropout", 0, 0.3),
            "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
            "beta": trial.suggest_float("beta", 1e-4, 1e4, log=True),
            "delta_min": trial.suggest_float("delta_min", 1e-8, 1e-3, log=True),
        }

        lambda_surv = 5 if cfg["unsupervised"] else trial.suggest_float("lambda_surv", 1, 1e3)

        if cfg["limit_epochs"]:
            effective_patience = None
            n_epochs_eff = cfg["limit_epochs"]
        else:
            effective_patience = params["patience"]
            n_epochs_eff = cfg["n_epochs"]

        if cfg["modality_dropout"]:
            modality_dropout_p = trial.suggest_float("modality_dropout_p", 0.01, 0.5, log=True)
        else:
            modality_dropout_p = None

        # ===== Estimation de la grille d'alphas sur l'outer train complet =====
        ref_model = build_customics_model(
            omics_train_outer, cfg["sources"], params, cfg["device"],
            cfg["classifier_dim"], cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
            cfg["unsupervised"], cfg["switch_epoch"],
            linear_decoder=cfg["linear_decoder"], lambda_surv=lambda_surv,
            modality_dropout_p=modality_dropout_p, md_mode=cfg["md_mode"],
            optimizer=args.optimizer, weight_decay=args.weight_decay,
        )
        ref_model.fit(
            omics_train=omics_train_outer, clinical_df=cfg["clinical_df"],
            label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
            omics_val=None, batch_size=cfg["batch_size"], n_epochs=n_epochs_eff,
            verbose=True, task=cfg["task"], patience=effective_patience,
            min_delta=params["delta_min"], early_stopping_on="train",
            modality_mask_train=cfg["mask_train_outer"],
            missing_strategy=cfg["missing_strategy"],
        )
        with torch.no_grad():
            Z_train_outer = ref_model.get_latent_representation(omics_train_outer)

        from pipeline_utils.cox_cv import estimate_alpha_grid, run_inner_cv_cox

        estimated_alphas = estimate_alpha_grid(
            Z_train_outer, samples_train_outer, y_train_outer, cfg["l1_ratio"],
            offset=cfg["offset"], clinical_test=cfg["clinical_test"], n_alphas=50,
        )
        print(f"  -> {len(estimated_alphas)} alpha candidates")

        # ===== CV interne =====
        def fit_and_embed(samples_train_inner, samples_val_inner):
            omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
            omics_val_raw = get_sub_omics_df(cfg["omics_df"], samples_val_inner)

            sel = fit_feature_selector(omics_train_raw, nbFeatures=cfg["nbFeatures"])
            omics_train = apply_feature_selector(omics_train_raw, sel)
            omics_val = apply_feature_selector(omics_val_raw, sel)

            scalers_inner = fit_scalers(omics_train)
            omics_train = apply_scalers(omics_train, scalers_inner)
            omics_val = apply_scalers(omics_val, scalers_inner)

            mask_train_inner = None
            mask_val_inner = None
            if cfg["missing_assignment"] is not None:
                omics_train, mask_train_inner = apply_missing_modalities(omics_train, cfg["missing_assignment"])
                omics_val, mask_val_inner = apply_missing_modalities(omics_val, cfg["missing_assignment"])

            model = build_customics_model(
                omics_train, cfg["sources"], params, cfg["device"],
                cfg["classifier_dim"], cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
                cfg["unsupervised"], cfg["switch_epoch"],
                linear_decoder=cfg["linear_decoder"], lambda_surv=lambda_surv,
                modality_dropout_p=modality_dropout_p, md_mode=cfg["md_mode"],
                optimizer=args.optimizer, weight_decay=args.weight_decay
            )
            model.fit(
                omics_train=omics_train, clinical_df=cfg["clinical_df"],
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=omics_val, batch_size=cfg["batch_size"], n_epochs=n_epochs_eff,
                verbose=False, task=cfg["task"], patience=effective_patience,
                min_delta=params["delta_min"], early_stopping_on="train",
                modality_mask_train=mask_train_inner, modality_mask_val=mask_val_inner,
                missing_strategy=cfg["missing_strategy"],
            )
            with torch.no_grad():
                Z_train = model.get_latent_representation(omics_train)
                Z_val = model.get_latent_representation(omics_val)
            return Z_train, Z_val

        best_alpha, best_score, alpha_scores = run_inner_cv_cox(
            cfg["inner_cv"], samples_train_outer, y_train_outer,
            estimated_alphas, cfg["l1_ratio"], cfg["validation_function"],
            fit_and_embed, cfg["clinical_df"], cfg["event"], cfg["surv_time"],
            offset=cfg["offset"], clinical_test=cfg["clinical_test"], ridge=cfg["ridge"],
            trial=trial, trial_pruning=cfg["trial_pruning"], gc_collect=True,
        )

        for alpha in estimated_alphas:
            print(f"Params {params} -> alpha {alpha:.6f} -> score {np.mean(alpha_scores[alpha]):.4f}")

        trial.set_user_attr("best_alpha", float(best_alpha))
        trial.set_user_attr("best_list_alpha", list(estimated_alphas))
        return best_score

    return objective


# ===== Worker (multiproc) ===================================================

_CFG = None
_ARGS = None
_STUDY_NAME = None
_JOURNAL_FILE = None
_N_TRIALS_PER_WORKER = None


def run_optimization(worker_id):
    print(f"[worker {worker_id}] starting in process {os.getpid()}")
    study = get_or_create_study(_STUDY_NAME, _JOURNAL_FILE)
    study.optimize(make_objective(_CFG, _ARGS), n_trials=_N_TRIALS_PER_WORKER, catch=(Exception,))
    print(f"[worker {worker_id}] done")


# ===== Main ==================================================================

def main():
    args = parse_args()
    unsupervised = not args.supervised
    multiproc = max(1, args.multiproc)

    print(f"\n{'='*60}")
    print(f"  Cancer       : {args.cancer}")
    print(f"  Outer fold   : {args.outer_fold} / {args.outer_splits}")
    print(f"  Inner folds  : {args.inner_splits}")
    print(f"  Add clinical : {args.add_clinical}")
    print(f"  Supervised   : {args.supervised}")
    print(f"  Saved folds  : {args.use_saved_folds}")
    print(f"  Optuna trials: {args.n_trials_per_worker}  |  timeout: {args.timeout}s")
    print(f"  Limit Epochs : {args.limit_epochs}")
    print(f"  Ridge        : {args.ridge}")
    print(f"  Nb features  : {args.nb_features}")
    print(f"  Linear Decoder: {args.linear_decoder}")
    print(f"  Multiproc    : {multiproc} worker(s)")
    print(f"  Modality dropout : {args.modality_dropout}  |  mode : {args.md_mode}")
    print(f"  Missing rate : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"{'='*60}\n")

    data = load_cancer_data(args.cancer)
    clinical_df = data["clinical"]
    omics_df = build_omics_dict(data, naming="raw", cast_float32=True)
    print(list(omics_df.keys()))

    clinical_test, offset = None, None
    if args.add_clinical:
        clinical_test = load_clinical_test(args.cancer)
        offset = clinical_df.index[0] - clinical_test.index[0]

    lt_samples = list(clinical_df.index)
    n_epochs = args.limit_epochs if args.limit_epochs else 400

    repetition_id = args.outer_fold // args.outer_splits
    missing_assignment = None
    if args.missing_rate > 0:
        missing_assignment = simulate_missing_modalities(
            sample_ids=lt_samples,
            sources=list(omics_df.keys()),
            missing_rate=args.missing_rate,
            repetition_seed=repetition_id,
        )
        n_missing = sum(1 for v in missing_assignment.values() if v is not None)
        print(f"  -> {n_missing}/{len(lt_samples)} patients avec une modalité manquante simulée "
              f"(repetition {repetition_id})")

    cfg = dict(
        omics_df=omics_df, clinical_df=clinical_df, clinical_test=clinical_test, offset=offset,
        sources=list(omics_df.keys()), device=torch.device("cpu"), unsupervised=unsupervised,
        num_classes=5, classifier_dim=[128, 64], survival_dim=[64, 32], dropout=0.2,
        batch_size=32, n_epochs=n_epochs, switch_epoch=n_epochs // 2,
        label="status", event="status", surv_time="time", task="survival",
        nbFeatures=args.nb_features, validation_function="vvh", l1_ratio=args.l1_ratio,
        ridge=args.ridge, add_clinical_to_cox=args.add_clinical,
        inner_cv=StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=0),
        limit_epochs=args.limit_epochs, linear_decoder=args.linear_decoder,
        trial_pruning=True,
        modality_dropout=args.modality_dropout,
        md_mode=args.md_mode,
        missing_assignment=missing_assignment,
        missing_strategy=args.missing_strategy,
    )

    if args.use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(args.cancer, src="../data/splits.json")
        train_idx, test_idx = train_folds[args.outer_fold], test_folds[args.outer_fold]
    else:
        outer_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
        splits = list(outer_cv.split(lt_samples))
        train_idx, test_idx = splits[args.outer_fold]

    print(f"\n{'='*50}\n OUTER FOLD {args.outer_fold}\n{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

    selector = fit_feature_selector(omics_train_outer_raw, nbFeatures=cfg["nbFeatures"])
    omics_train_outer = apply_feature_selector(omics_train_outer_raw, selector)
    omics_test_outer = apply_feature_selector(omics_test_outer_raw, selector)

    mask_train_outer = None
    mask_test_outer = None
    if cfg["missing_assignment"] is not None:
        omics_train_outer, mask_train_outer = apply_missing_modalities(omics_train_outer, cfg["missing_assignment"])
        omics_test_outer, mask_test_outer = apply_missing_modalities(omics_test_outer, cfg["missing_assignment"])

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, cfg["event"], cfg["surv_time"])

    cfg["samples_train_outer"] = samples_train_outer
    cfg["omics_train_outer"] = omics_train_outer
    cfg["y_train_outer"] = y_train_outer
    cfg["omics_test_outer"] = omics_test_outer
    cfg["mask_train_outer"] = mask_train_outer
    cfg["mask_test_outer"] = mask_test_outer

    study_name = f"journal_storage_multiprocess_{args.name_suffix}{args.cancer}_fold{args.outer_fold}"
    journal_file = f"optuna_journal/journal_{args.name_suffix}{args.cancer}_fold{args.outer_fold}.log"

    debut = time.time()

    if multiproc > 1:
        global _CFG, _ARGS, _STUDY_NAME, _JOURNAL_FILE, _N_TRIALS_PER_WORKER
        _CFG = cfg
        _ARGS = args
        _STUDY_NAME = study_name
        _JOURNAL_FILE = journal_file
        _N_TRIALS_PER_WORKER = args.n_trials_per_worker
        # Les workers héritent des globals via fork (copy-on-write) plutôt que
        # de les recevoir picklés à travers le Pool.
        with mp.Pool(processes=multiproc) as pool:
            pool.map(run_optimization, range(multiproc))
    else:
        study = get_or_create_study(study_name, journal_file)
        study.optimize(make_objective(cfg, args), n_trials=args.n_trials_per_worker,
                        timeout=args.timeout, catch=(Exception,))

    print(f"time study : {time.time() - debut}")


if __name__ == "__main__":
    main()