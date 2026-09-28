"""Post-Optuna evaluation of the nested CV pipeline (no transfer learning),
pour UN outer fold donné (--outer_fold)

Modification: possibilité d'utiliser des hyperparamètres prédéfinis (JSON)
au lieu de charger le meilleur essai Optuna, via --fixed_params_file.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

sys.path.append('..')
import utils.folds_utils as fold_utils

from custcox_utils import (
    fit_feature_selector, apply_feature_selector, build_survival_array, fit_coxnet,
    fit_scalers, apply_scalers,
)
from src.tools.utils import get_sub_omics_df
from pipeline_utils.missing_data import simulate_missing_modalities, apply_missing_modalities

from pipeline_utils.cli import (
    base_parser, add_cancer_arg, add_outer_cv_args, add_inner_cv_args,
    add_optuna_args, add_training_args, add_cox_args, add_missing_modality_args,
)
from pipeline_utils.cox_cv import maybe_concat_clinical
from pipeline_utils.data import load_cancer_data, load_clinical_test, build_omics_dict
from pipeline_utils.hidden_dims_params import reconstruct_autoencoder_hidden_dims
from pipeline_utils.model import build_customics_model
from pipeline_utils.optuna_utils import load_best_trial


def parse_args():
    parser = base_parser("Post-Optuna evaluation of the nested CV pipeline (no transfer learning).")
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
        help="Number of parallel worker processes used by Optuna during the search phase "
             "(kept here only for CLI compatibility with search_ncv.py; unused in this script).")
    parser.add_argument("--eval_patience", type=int, default=10,
        help="Patience used for the FINAL model, independent of the patience found by Optuna "
             "during the search (see module docstring, point 3).")
    parser.add_argument("--override_lambda_surv", type=float, default=None,
        help="Debug override: force lambda_surv to this value instead of the one found by "
             "Optuna (off by default, see module docstring, point 4).")
    parser.add_argument("--override_beta", type=float, default=None,
        help="Debug override: force beta to this value instead of the one found by Optuna "
             "(off by default, see module docstring, point 4).")
    parser.add_argument("--fixed_params_file", type=str, default=None,
        help="Chemin vers un fichier JSON contenant des hyperparametres predefinis, "
             "a utiliser a la place du meilleur essai Optuna. Format attendu : "
             "{\"params\": {...memes cles que best_trial.params...}, "
             "\"best_alpha\": <float>, \"best_list_alpha\": [<float>, ...] (optionnel)}. "
             "Si fourni, aucune etude Optuna n'est chargee.")
    return parser.parse_args()


def load_hyperparams(args):
    """Renvoie (best_params, best_alpha, best_list_alpha, source_label).

    Deux sources possibles :
      - fichier JSON fixe (--fixed_params_file), pour bypasser Optuna
      - meilleur essai de l'etude Optuna correspondant au cancer/fold
    """
    if args.fixed_params_file:
        with open(args.fixed_params_file, "r") as f:
            fixed = json.load(f)

        if "params" not in fixed:
            raise ValueError(
                f"'{args.fixed_params_file}' doit contenir une cle 'params' "
                "avec les hyperparametres (memes cles que best_trial.params)."
            )
        best_params = fixed["params"]
        best_alpha = fixed.get("best_alpha")
        best_list_alpha = fixed.get("best_list_alpha")

        if best_alpha is None:
            raise ValueError(
                f"'{args.fixed_params_file}' doit contenir 'best_alpha' "
                "(l'alpha du Cox net a utiliser pour predict/predict_survival_function)."
            )

        source_label = f"fixed params ({args.fixed_params_file})"
        return best_params, best_alpha, best_list_alpha, source_label

    study_name = f"journal_storage_multiprocess_{args.name_suffix}{args.cancer}_fold{args.outer_fold}"
    journal_file = f"optuna_journal/journal_{args.name_suffix}{args.cancer}_fold{args.outer_fold}.log"
    study, best_trial = load_best_trial(study_name, journal_file)

    best_params = best_trial.params
    best_alpha = best_trial.user_attrs["best_alpha"]
    best_list_alpha = best_trial.user_attrs["best_list_alpha"]
    source_label = f"optuna study ({study_name})"
    return best_params, best_alpha, best_list_alpha, source_label


def main():
    args = parse_args()
    unsupervised = not args.supervised

    print(f"\n{'='*60}")
    print(f"  Cancer       : {args.cancer}")
    print(f"  Outer fold   : {args.outer_fold} / {args.outer_splits}")
    print(f"  Inner folds  : {args.inner_splits}")
    print(f"  Add clinical : {args.add_clinical}")
    print(f"  Supervised   : {args.supervised}")
    print(f"  Saved folds  : {args.use_saved_folds}")
    print(f"  Ridge        : {args.ridge}")
    print(f"  Nb features  : {args.nb_features}")
    print(f"  Linear Decoder: {args.linear_decoder}")
    print(f"  Eval patience : {args.eval_patience}")
    print(f"  Modality dropout : {args.modality_dropout}  |  mode : {args.md_mode}")
    print(f"  Missing rate : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"  Hyperparams source : {'fixed file' if args.fixed_params_file else 'optuna'}")
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
    n_epochs = args.limit_epochs if args.limit_epochs else 1000

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
        clinical_df=clinical_df, clinical_test=clinical_test, offset=offset,
        sources=list(omics_df.keys()), device=torch.device("cpu"), unsupervised=unsupervised,
        num_classes=5, classifier_dim=[128, 64], survival_dim=[64, 32], dropout=0.2,
        batch_size=32, n_epochs=n_epochs, switch_epoch=n_epochs // 2,
        label="status", event="status", surv_time="time", task="survival",
        nbFeatures=args.nb_features, l1_ratio=args.l1_ratio, ridge=args.ridge,
        limit_epochs=args.limit_epochs, linear_decoder=args.linear_decoder,
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

    # ===== Simulation de modalités manquantes (avant le scaling) =====
    mask_train_outer = None
    if missing_assignment is not None:
        omics_train_outer, mask_train_outer = apply_missing_modalities(omics_train_outer, missing_assignment)

    # ===== Scaling (fit sur train uniquement) =====
    scalers = fit_scalers(omics_train_outer)
    omics_train_outer = apply_scalers(omics_train_outer, scalers)
    omics_test_outer = apply_scalers(omics_test_outer, scalers)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, cfg["event"], cfg["surv_time"])
    y_test_outer = build_survival_array(clinical_df, samples_test_outer, cfg["event"], cfg["surv_time"])

    best_params, best_alpha, best_list_alpha, hp_source = load_hyperparams(args)
    print(f"Hyperparams source : {hp_source}")
    print(f"Best params : {best_params}")
    print(f"Best alpha  : {best_alpha:.6f}")

    lambda_surv = best_params.get("lambda_surv", 5)
    if args.override_lambda_surv is not None:
        lambda_surv = args.override_lambda_surv

    autoencoder_hidden_dims = reconstruct_autoencoder_hidden_dims(best_params, cfg["sources"])
    full_best_params = {
        **best_params,
        "autoencoder_hidden_dims": autoencoder_hidden_dims,
        "latent_dim": 32,
        "rep_dim": 32,
        "central_hidden_dims": [64],
        "lambda_central": 1,
        "patience": args.eval_patience,
    }
    if args.override_beta is not None:
        full_best_params["beta"] = args.override_beta

    patience = None if cfg["limit_epochs"] else full_best_params["patience"]

    modality_dropout_p = best_params.get("modality_dropout_p")

    model = build_customics_model(
        omics_train_outer, cfg["sources"], full_best_params, cfg["device"],
        cfg["classifier_dim"], cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
        unsupervised, cfg["switch_epoch"], linear_decoder=cfg["linear_decoder"],
        lambda_surv=lambda_surv, modality_dropout_p=modality_dropout_p, md_mode=args.md_mode,
        optimizer=args.optimizer, weight_decay=args.weight_decay
    )
    model.fit(
        omics_train=omics_train_outer, clinical_df=clinical_df,
        label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
        omics_val=None, batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
        verbose=True, task=cfg["task"], patience=patience,
        min_delta=full_best_params["delta_min"], early_stopping_on="train",
        track_loss_components=True,
        modality_mask_train=mask_train_outer,
        missing_strategy=args.missing_strategy,
    )

    loss_plot_dir = "results"
    os.makedirs(loss_plot_dir, exist_ok=True)
    tag = f"{args.name_suffix}{args.cancer}_fold{args.outer_fold}"
    model.plot_loss_detailed(save_path=f"{loss_plot_dir}/loss_sans_tl_{tag}.png")
    model.plot_loss_detailed_stacked(save_path=f"{loss_plot_dir}/loss_sans_tl_stacked_{tag}.png")

    Z_train_outer = model.get_latent_representation(omics_train_outer)
    Z_test_outer = model.get_latent_representation(omics_test_outer)

    X_train_outer = maybe_concat_clinical(Z_train_outer, samples_train_outer, offset, clinical_test)
    X_test_outer = maybe_concat_clinical(Z_test_outer, samples_test_outer, offset, clinical_test)

    coxnet, scaler = fit_coxnet(X_train_outer, y_train_outer, cfg["l1_ratio"], best_list_alpha, ridge=cfg["ridge"])
    X_test_scaled = scaler.transform(X_test_outer)

    if cfg["ridge"]:
        fitted = coxnet[best_alpha]
        risk_scores = fitted.predict(X_test_scaled)
        survs = fitted.predict_survival_function(X_test_scaled)
        coefs = fitted.coef_
    else:
        risk_scores = coxnet.predict(X_test_scaled, alpha=best_alpha)
        survs = coxnet.predict_survival_function(X_test_scaled, alpha=best_alpha)
        alpha_idx = np.argmin(np.abs(coxnet.alphas_ - best_alpha))
        coefs = coxnet.coef_[:, alpha_idx]

    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    times = np.sort(np.unique(y_test_outer["time"]))
    upper = min(np.max(y_train_outer["time"]), np.max(y_test_outer["time"]))
    times = times[times < upper]
    preds = np.vstack([fn(times) for fn in survs])
    ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

    nonzero_mask = coefs != 0
    n_nonzero_coefs = int(nonzero_mask.sum())
    n_zero_coefs = int(len(coefs) - n_nonzero_coefs)
    print(f"  Coefs non-nuls : {n_nonzero_coefs} / {len(coefs)}  (nuls : {n_zero_coefs})")

    result = [{
        "fold": args.outer_fold,
        "cindex_default": c_index,
        "graf": ibs_score,
        "best_alpha": best_alpha,
        "n_nonzero_coefs": n_nonzero_coefs,
        "n_zero_coefs": n_zero_coefs,
        "modality_dropout": args.modality_dropout,
        "modality_dropout_p": modality_dropout_p,
        "md_mode": args.md_mode,
        "missing_rate": args.missing_rate,
        "missing_strategy": args.missing_strategy,
        "hp_source": hp_source,
        **{f"hp_{k}": v for k, v in best_params.items()},
    }]

    out_dir = "../results/folds"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/ncv_custcox_optuna_{args.name_suffix}{args.cancer}_fold{args.outer_fold}.csv"
    pd.DataFrame(result).to_csv(out_path, index=False)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()