"""Pan-cancer pre-training with CustOMICS + CoxNet — Optuna hyperparameter search.

Entraîne (et cherche les hyperparamètres d')un modèle CustOMICS sur toutes
les cancers SAUF --cancer (le "source" pré-entraînement du transfer
learning). Voir `eval_source_pretrain.py` pour le script pendant qui
réentraîne le modèle final et sauvegarde le checkpoint utilisé ensuite par
`search_target_finetune.py` / `eval_target_finetune.py`.
"""
import os
_INITIAL_AFFINITY = sorted(os.sched_getaffinity(0))  # avant que torch/OpenMP ne la restreigne

import sys
import time

import torch
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.append('..')

from custcox_utils import (
    fit_feature_selector, apply_feature_selector, build_survival_array,
    LeaveOneCancerOutCV, fit_scalers, apply_scalers,
)
from src.tools.utils import get_sub_omics_df
from pipeline_utils.missing_data import simulate_missing_modalities, apply_missing_modalities


from pipeline_utils.cli import (
    base_parser, add_cancer_arg, add_optuna_args, add_training_args, add_missing_modality_args,
    add_parallel_args,
)
from pipeline_utils.cox_cv import estimate_alpha_grid, run_inner_cv_cox, run_inner_cv_loss
from pipeline_utils.data import load_pancancer, get_source_data, load_clinical_test, build_omics_dict
from pipeline_utils.hidden_dims_params import suggest_autoencoder_hidden_dims
from pipeline_utils.model import build_customics_model
from pipeline_utils.parallel import run_study


def parse_args():
    parser = base_parser("Pan-cancer pre-training with CustOMICS + CoxNet — Optuna hyperparameter search.")
    add_cancer_arg(parser)
    add_optuna_args(parser)
    add_training_args(parser)
    add_missing_modality_args(parser)
    add_parallel_args(parser)
    parser.add_argument("--n_epochs", type=int, default=400,
        help="Max number of training epochs (ceiling when early stopping is active).")
    parser.add_argument("--final_epochs", type=int, default=400,
        help="Max number of training epochs for the final model (unused here, kept for parity "
             "with eval_source_pretrain.py's CLI).")
    parser.add_argument("--inner_splits", type=int, default=5, help="Number of inner CV folds.")
    parser.add_argument("--n_samples", type=int, default=-1,
        help="Number of source samples to use. Set to -1 to use all.")
    parser.add_argument("--l1_ratio", type=float, default=0.01, help="L1 ratio for CoxNet regularization.")
    parser.add_argument("--validation", type=str, default="loss", help="Validation function (vvh, ibs or loss).")
    parser.add_argument("--cv_strategy", type=str, default="stratified", choices=["stratified", "loco"],
        help="Inner CV strategy: 'stratified' (StratifiedKFold) or 'loco' (Leave-One-Cancer-Out).")
    parser.add_argument("--n_folds_loco", type=int, default=None,
        help="Max number of LOCO folds (cancers). None = all cancers. Ignored if --cv_strategy=stratified.")
    parser.add_argument("--domain_adv", action="store_true", default=False,
        help="Enable domain-adversarial training to make the latent space cancer-type invariant.")
    parser.add_argument("--lambda_domain", type=float, default=50,
        help="Weight for the domain-adversarial loss (GRL coefficient ceiling).")
    parser.add_argument("--domain_hidden_dim", type=int, nargs="+", default=[64],
        help="Hidden layer sizes for the domain classifier head.")
    parser.add_argument("--missing_repetition_seed", type=int, default=0,
        help="Repetition seed used for the missing-modality simulation (no outer-fold concept here, "
             "so this is a plain fixed seed rather than derived from a fold index).")
    return parser.parse_args()


# ===== Optuna objective =====================================================

def make_objective(cfg):

    def objective(trial):
        omics_source = cfg["omics_source"]
        clinical_df = cfg["clinical_df"]
        lt_samples = cfg["lt_samples"]
        sources = cfg["sources"]

        autoencoder_hidden_dims = suggest_autoencoder_hidden_dims(trial, omics_source, sources)

        params = {
            "latent_dim": 32, "rep_dim": 32, "central_hidden_dims": [64],
            "autoencoder_hidden_dims": autoencoder_hidden_dims,
            "dropout": trial.suggest_float("dropout", 0, 0.3),
            "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
            "beta": trial.suggest_float("beta", 1e-4, 1e4, log=True),
            "delta_min": trial.suggest_float("delta_min", 1e-8, 1e-3, log=True),
        }
        if cfg["modality_dropout"]:
            params["modality_dropout_p"] = trial.suggest_float("modality_dropout_p", 0.01, 0.5, log=True)
        else:
            params["modality_dropout_p"] = None

        patience = None if cfg["limit_epochs"] else 3
        n_epochs = cfg["n_epochs"]
        switch_epoch = cfg["switch_epoch"]

        def domain_params_for(dropout):
            if not cfg["domain_adv"]:
                return None
            return {
                "n_domains": cfg["n_domains"], "lambda": cfg["lambda_domain"],
                "hidden_layers": cfg["domain_hidden_dim"], "dropout": dropout,
            }

        y_all = build_survival_array(clinical_df, lt_samples, cfg["event"], cfg["surv_time"])

        # ===== Branche CoxNet (validation vvh / ibs) =====
        if cfg["validation_function"] != "loss":
            ref_model = build_customics_model(
                omics_source, sources, params, cfg["device"],
                cfg["classifier_dim"], cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
                cfg["unsupervised"], switch_epoch, domain_params=domain_params_for(params["dropout"]),
                modality_dropout_p=params["modality_dropout_p"], md_mode=cfg["md_mode"],
                optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
            )
            ref_model.fit(
                omics_train=omics_source, clinical_df=clinical_df,
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=None, batch_size=cfg["batch_size"], n_epochs=n_epochs,
                verbose=True, task=cfg["task"], patience=patience,
                min_delta=params["delta_min"], early_stopping_on="train",
                modality_mask_train=cfg["mask_source"],
                missing_strategy=cfg["missing_strategy"],
            )
            Z_all = ref_model.get_latent_representation(omics_source)

            estimated_alphas = estimate_alpha_grid(
                Z_all, lt_samples, y_all, cfg["l1_ratio"],
                offset=cfg["offset"], clinical_test=cfg["clinical_test"],
            )
            print(f"  -> {len(estimated_alphas)} alpha candidates")

            def fit_and_embed(samples_train_inner, samples_val_inner):
                omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
                omics_val_raw = get_sub_omics_df(cfg["omics_df"], samples_val_inner)
                sel = fit_feature_selector(omics_train_raw, nbFeatures=cfg["nbFeatures"])
                omics_train = apply_feature_selector(omics_train_raw, sel)
                omics_val = apply_feature_selector(omics_val_raw, sel)

                mask_train_inner = None
                mask_val_inner = None
                if cfg["missing_assignment"] is not None:
                    omics_train, mask_train_inner = apply_missing_modalities(omics_train, cfg["missing_assignment"])
                    omics_val, mask_val_inner = apply_missing_modalities(omics_val, cfg["missing_assignment"])

                model = build_customics_model(
                    omics_train, sources, params, cfg["device"],
                    cfg["classifier_dim"], cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
                    cfg["unsupervised"], switch_epoch, domain_params=domain_params_for(params["dropout"]),
                    modality_dropout_p=params["modality_dropout_p"], md_mode=cfg["md_mode"],
                    optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
                )
                model.fit(
                    omics_train=omics_train, clinical_df=clinical_df,
                    label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                    omics_val=omics_val, batch_size=cfg["batch_size"], n_epochs=n_epochs,
                    verbose=True, task=cfg["task"], patience=patience,
                    min_delta=params["delta_min"], early_stopping_on="train",
                    modality_mask_train=mask_train_inner, modality_mask_val=mask_val_inner,
                    missing_strategy=cfg["missing_strategy"],
                )
                Z_train = model.get_latent_representation(omics_train)
                Z_val = model.get_latent_representation(omics_val)
                return Z_train, Z_val

            best_alpha, best_score, _ = run_inner_cv_cox(
                cfg["inner_cv"], lt_samples, y_all, estimated_alphas, cfg["l1_ratio"],
                cfg["validation_function"], fit_and_embed, clinical_df, cfg["event"], cfg["surv_time"],
                offset=cfg["offset"], clinical_test=cfg["clinical_test"],
                trial=trial, trial_pruning=cfg["trial_pruning"],
            )
            trial.set_user_attr("best_alpha", float(best_alpha))
            trial.set_user_attr("estimated_alphas", list(estimated_alphas))
            return best_score

        # ===== Branche loss (pas de CoxNet) =====
        def fit_and_eval(samples_train_inner, samples_val_inner):
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
                omics_train, sources, params, cfg["device"],
                cfg["classifier_dim"], cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
                cfg["unsupervised"], switch_epoch, domain_params=domain_params_for(params["dropout"]),
                modality_dropout_p=params["modality_dropout_p"], md_mode=cfg["md_mode"],
                optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
            )
            model.fit(
                omics_train=omics_train, clinical_df=clinical_df,
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=omics_val, batch_size=cfg["batch_size"], n_epochs=n_epochs,
                verbose=True, task=cfg["task"], patience=patience,
                min_delta=params["delta_min"], early_stopping_on="train",
                modality_mask_train=mask_train_inner, modality_mask_val=mask_val_inner,
                missing_strategy=cfg["missing_strategy"],
            )
            return model.get_loss_eval(
                omics_val, clinical_df=clinical_df,
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
            )

        return run_inner_cv_loss(
            cfg["inner_cv"], lt_samples, y_all["status"], fit_and_eval,
            trial=trial, trial_pruning=cfg["trial_pruning"],
        )

    return objective


# ===== Main ==================================================================

def main():
    args = parse_args()
    unsupervised = not args.supervised
    multiproc = max(1, args.multiproc)
    n_epochs = args.limit_epochs if args.limit_epochs else args.n_epochs
    switch_epoch = n_epochs // 2

    print(f"\n{'='*60}")
    print(f"  Target cancer  : {args.cancer}")
    print(f"  Max epochs     : {n_epochs}")
    print(f"  Inner splits   : {args.inner_splits}")
    print(f"  Supervised     : {args.supervised}")
    print(f"  N samples      : {'all' if args.n_samples == -1 else args.n_samples}")
    print(f"  L1 ratio       : {args.l1_ratio}")
    print(f"  Optuna trials  : {args.n_trials_per_worker}  |  timeout: {args.timeout}s")
    print(f"  Limit epochs   : {args.limit_epochs}")
    print(f"  Validation     : {args.validation}")
    print(f"  Domain Adversarial: {args.domain_adv}")
    print(f"  Multiproc      : {multiproc} worker(s)  |  affinity : {args.affinity}")
    print(f"  Modality dropout : {args.modality_dropout}  |  mode : {args.md_mode}")
    print(f"  Missing rate   : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"{'='*60}\n")

    pancancer = load_pancancer()
    source_data = get_source_data(pancancer, args.cancer, n_samples=args.n_samples)
    clinical_df = source_data["clinical"]
    omics_df = build_omics_dict(source_data, naming="clinical_labels")

    lt_samples = list(clinical_df.index)
    print(f"Source samples : {len(lt_samples)}")

    clinical_test, offset = None, None
    if args.add_clinical:
        clinical_test = load_clinical_test(args.cancer)
        offset = clinical_df.index[0] - clinical_test.index[0]

    # En multiproc (fork), on reste sur CPU : un contexte CUDA ne survit pas au fork.
    if multiproc > 1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = list(omics_df.keys())

    print("=== Feature selection on source cancers ===")
    selector_source = fit_feature_selector(omics_df, nbFeatures=5000)
    omics_source = apply_feature_selector(omics_df, selector_source)

    missing_assignment = None
    mask_source = None
    if args.missing_rate > 0:
        missing_assignment = simulate_missing_modalities(
            sample_ids=lt_samples,
            sources=sources,
            missing_rate=args.missing_rate,
            repetition_seed=args.missing_repetition_seed,
        )
        omics_source, mask_source = apply_missing_modalities(omics_source, missing_assignment)
        n_missing = sum(1 for v in missing_assignment.values() if v is not None)
        print(f"  -> {n_missing}/{len(lt_samples)} patients avec une modalité manquante simulée "
              f"(seed {args.missing_repetition_seed})")

    if args.cv_strategy == "loco":
        inner_cv = LeaveOneCancerOutCV(
            clinical_df=clinical_df, cancer_col="cancer_type",
            n_folds=args.n_folds_loco, random_state=0,
        )
        print(f"  CV strategy    : LOCO  |  max folds: {args.n_folds_loco or 'all'}")
    else:
        inner_cv = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=0)
        print(f"  CV strategy    : Stratified  |  splits: {args.inner_splits}")

    n_domains = None
    if args.domain_adv:
        domain_label_encoder = LabelEncoder().fit(clinical_df["cancer_type"])
        clinical_df = clinical_df.copy()
        clinical_df["domain_label"] = domain_label_encoder.transform(clinical_df["cancer_type"])
        n_domains = len(domain_label_encoder.classes_)
        print(f"  N domains (cancer types in source) : {n_domains}")

    cfg = dict(
        omics_df=omics_df, omics_source=omics_source, clinical_df=clinical_df,
        lt_samples=lt_samples, sources=sources, device=device, unsupervised=unsupervised,
        num_classes=5, classifier_dim=[128, 64], survival_dim=[64, 32], dropout=0.2,
        batch_size=32, n_epochs=n_epochs, switch_epoch=switch_epoch,
        label="status", event="status", surv_time="time", task="survival",
        nbFeatures=5000, validation_function=args.validation, l1_ratio=args.l1_ratio,
        inner_cv=inner_cv, limit_epochs=args.limit_epochs, add_clinical_to_cox=args.add_clinical,
        clinical_test=clinical_test, offset=offset,
        domain_adv=args.domain_adv, lambda_domain=args.lambda_domain,
        domain_hidden_dim=args.domain_hidden_dim, n_domains=n_domains,
        trial_pruning=True,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        modality_dropout=args.modality_dropout, md_mode=args.md_mode,
        missing_assignment=missing_assignment, missing_strategy=args.missing_strategy,
        mask_source=mask_source,
    )

    study_name = f"source_{args.name_suffix}{args.cancer}"
    journal_file = f"optuna_journal/journal_tl_{args.name_suffix}source_{args.cancer}.log"

    debut = time.time()
    run_study(lambda: make_objective(cfg), study_name, journal_file,
              n_trials=args.n_trials_per_worker, timeout=args.timeout,
              multiproc=multiproc, affinity=args.affinity,
              initial_affinity=_INITIAL_AFFINITY)
    print(f"time study : {time.time() - debut}")


if __name__ == "__main__":
    main()