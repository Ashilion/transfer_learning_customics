import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
import time
import optuna

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

from custcox_utils import fit_feature_selector, apply_feature_selector, build_survival_array, fit_coxnet, evaluate_survival, LeaveOneCancerOutCV


# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pan-cancer pre-training with CustOMICS + CoxNet — Optuna hyperparameter search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cancer",
        type=str, 
        default="COAD",
        help="Target cancer type to exclude from source data (e.g. COAD, BRCA, LUAD)."
    )
    parser.add_argument(
        "--n_epochs", 
        type=int, 
        default=400,
        help="Max number of training epochs (used as ceiling when early stopping is active)."
    )
    parser.add_argument(
        "--final_epochs", 
        type=int, 
        default=400,
        help="Max number of training epochs for the final model."
    )
    parser.add_argument(
        "--inner_splits", 
        type=int, 
        default=5,
        help="Number of inner CV folds."
    )
    parser.add_argument(
        "--supervised", 
        action="store_true", 
        default=False,
        help="Train CustOMICS in supervised mode (survival loss included during training)."
        )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=-1,
        help="Number of source samples to use. Set to -1 to use all."
        )
    parser.add_argument(
        "--l1_ratio", 
        type=float, 
        default=0.01,
        help="L1 ratio for CoxNet regularization."
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
        "--limit_epochs", 
        type=int, 
        default=None,
        help="Hard-limit epochs (disables early stopping when set)."
        )
    parser.add_argument(
        "--add_clinical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate clinical features to the latent representation before CoxNet."
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="loss",
        help="Validation function (vvh, ibs or loss)"
    )
    parser.add_argument(
        "--cv_strategy",
        type=str,
        default="stratified",
        choices=["stratified", "loco"],
        help="Inner CV strategy: 'stratified' (StratifiedKFold) or 'loco' (Leave-One-Cancer-Out)."
    )
    parser.add_argument(
        "--n_folds_loco",
        type=int,
        default=None,
        help="Max number of LOCO folds (cancers). None = all cancers. Ignored if --cv_strategy=stratified."
    )
    parser.add_argument(
        "--domain_adv",
        action="store_true",
        default=False,
        help="Enable domain-adversarial training to make latent space cancer-type invariant."
        )
    parser.add_argument(
        "--lambda_domain",
        type=float,
        default=50,
        help="Weight for the domain-adversarial loss (GRL coefficient ceiling)... Pas optimisé dans les hyperparamètres pour l'instant"
    )
    parser.add_argument(
        "--domain_hidden_dim",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden layer sizes for the domain classifier head."
    )
    return parser.parse_args()


# ========================================================================

def build_customics_model_plus(omics_data, sources, params, device,
                                hidden_dim, central_hidden,
                                classifier_dim, survival_dim,
                                dropout, num_classes, unsupervised, switch_epoch, domain_params=None):

    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {
            'input_dim':  x_dim[i],
            'hidden_dim': params["autoencoder_hidden_dims"][i],
            'latent_dim': params['rep_dim'],
            'norm': True,
            'dropout': params["dropout"],
        }
        for i, src in enumerate(sources)
    }

    central_params = {
        'hidden_dim': params["central_hidden_dims"],
        'latent_dim': params['latent_dim'],
        'norm': True,
        'dropout': params["dropout"],
        'beta': params["beta"],
    }

    classif_params = {
        'n_class': num_classes,
        'lambda': 0,
        'hidden_layers': classifier_dim,
        'dropout': params["dropout"],
    }

    surv_params = {
        'lambda': 5,
        'dims': survival_dim,
        'activation': 'SELU',
        'l2_reg': 1e-2,
        'norm': True,
        'dropout': params["dropout"],
    }

    train_params = {'switch': switch_epoch, 'lr': params['lr']}

    model = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device,
        unsupervised=unsupervised,
        domain_params=domain_params
    ).to(device)

    return model


# ===== Optuna objective ================================================================================

def make_objective(cfg):

    def objective(trial):
        omics_source      = cfg["omics_source"]
        clinical_df       = cfg["clinical_df"]
        lt_samples        = cfg["lt_samples"]
        sources           = cfg["sources"]

        
        autoencoder_hidden_dims_possibilities = {
            "(1024, 512, 256, 128)": (1024, 512, 256, 128),
            "(1024, 256)":           (1024, 256),
            "(512, 128)":            (512, 128),
            "(1024, 256, 128)":      (1024, 256, 128),
            "(1024, 512, 128)":      (1024, 512, 128),
        }
        autoencoder_hidden_dims_possibilities_small = {
            "(512, 256, 128)": (512, 256, 128),
            "(256, 128)":      (256, 128),
            "(512, 256)":      (512, 256),
        }

        autoencoder_hidden_dims = []
        for i, src in enumerate(sources):
            if omics_source[src].shape[1] > 1200:
                possibilities = autoencoder_hidden_dims_possibilities
            else:
                possibilities = autoencoder_hidden_dims_possibilities_small
            key = trial.suggest_categorical(f"hidden_dim_{i}", list(possibilities.keys()))
            autoencoder_hidden_dims.append(possibilities[key])

        params = {
            "latent_dim":            32,
            "rep_dim":               32,
            "central_hidden_dims":   [64],
            "patience":              3,
            "autoencoder_hidden_dims": autoencoder_hidden_dims,
            "dropout":  trial.suggest_float("dropout", 0, 0.3),
            "lr":       trial.suggest_float("lr", 3e-5, 1e-3, log=True),
            "beta":     trial.suggest_float("beta", 1e-4, 1e4, log=True),
            "delta_min": trial.suggest_float("delta_min", 1e-8, 1e-3, log=True),
        }

        patience    = None if cfg["limit_epochs"] else params["patience"]
        n_epochs    = cfg["n_epochs"]
        switch_epoch = cfg["switch_epoch"]

        y_all = build_survival_array(clinical_df, lt_samples, cfg["event"], cfg["surv_time"])
        
        if cfg["validation_function"] != "loss":

            domain_params = None
            if cfg["domain_adv"]:
                domain_params = {
                    "n_domains": cfg["n_domains"],
                    "lambda": cfg["lambda_domain"],
                    "hidden_layers": cfg["domain_hidden_dim"],
                    "dropout": params["dropout"],
                }

            ref_model = build_customics_model_plus(
                omics_source, sources, params, cfg["device"],
                cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
                cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
                cfg["unsupervised"], switch_epoch, domain_params=domain_params
            )
            ref_model.fit(
                omics_train=omics_source, clinical_df=clinical_df,
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=None, batch_size=cfg["batch_size"], n_epochs=n_epochs,
                verbose=True, task=cfg["task"],
                patience=patience, min_delta=params["delta_min"],
                early_stopping_on="train",
            )
            Z_all = ref_model.get_latent_representation(omics_source)

            if cfg["add_clinical_to_cox"]:
                samples_adapted = [idx - cfg["offset"] for idx in lt_samples]
                clin_all = cfg["clinical_test"].loc[samples_adapted, :]
                X_for_alpha = np.concatenate([clin_all, Z_all], axis=1)
            else:
                X_for_alpha = Z_all

            ref_cox, _ = fit_coxnet(X_for_alpha, y_all, cfg["l1_ratio"])
            estimated_alphas = ref_cox.alphas_
            print(f"  -> {len(estimated_alphas)} alpha candidates")

            alpha_scores = {a: [] for a in estimated_alphas}
            best_score   = np.inf
            best_alpha   = estimated_alphas[0]

        loss_scores = []

        
        # --- Inner CV ---
        for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
            cfg["inner_cv"].split(lt_samples, y_all["status"])
        ):
            samples_train_inner = [lt_samples[i] for i in inner_train_idx]
            samples_val_inner   = [lt_samples[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
            omics_val_raw   = get_sub_omics_df(cfg["omics_df"], samples_val_inner)

            sel          = fit_feature_selector(omics_train_raw, nbFeatures=cfg["nbFeatures"])
            omics_train  = apply_feature_selector(omics_train_raw, sel)
            omics_val    = apply_feature_selector(omics_val_raw,   sel)

            domain_params = None
            if cfg["domain_adv"]:
                domain_params = {
                    "n_domains": cfg["n_domains"],
                    "lambda": cfg["lambda_domain"],
                    "hidden_layers": cfg["domain_hidden_dim"],
                    "dropout": params["dropout"],
                }

            model = build_customics_model_plus(
                omics_train, sources, params, cfg["device"],
                cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
                cfg["survival_dim"], cfg["dropout"], cfg["num_classes"],
                cfg["unsupervised"], switch_epoch, domain_params=domain_params
            )
            model.fit(
                omics_train=omics_train, clinical_df=clinical_df,
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=omics_val, batch_size=cfg["batch_size"], n_epochs=n_epochs,
                verbose=True, task=cfg["task"],
                patience=patience, min_delta=params["delta_min"],
                early_stopping_on="train",
            )

            Z_train = model.get_latent_representation(omics_train)
            Z_val   = model.get_latent_representation(omics_val)

            y_train_struct = build_survival_array(clinical_df, samples_train_inner, cfg["event"], cfg["surv_time"])
            y_val_struct   = build_survival_array(clinical_df, samples_val_inner,   cfg["event"], cfg["surv_time"])

            if cfg["validation_function"] != "loss":
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

                coxnet, scaler = fit_coxnet(X_cox, y_train_struct, cfg["l1_ratio"], estimated_alphas)
                X_val_scaled  = scaler.transform(X_val)

                for alpha in coxnet.alphas_:
                    score = evaluate_survival(
                        coxnet, alpha, X_cox, y_train_struct, X_val_scaled, y_val_struct,
                        cfg["validation_function"]
                    )
                    alpha_scores[alpha].append(score)
                if cfg["trial_pruning"]:
                    intermediate_best_score = np.inf
                    for alpha in estimated_alphas:
                        mean_score = np.mean(alpha_scores[alpha])
                        if mean_score < intermediate_best_score:
                            intermediate_best_score = mean_score
                    trial.report(intermediate_best_score, fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            else:
                score = model.get_loss_eval(omics_val, clinical_df=clinical_df,
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"])
                loss_scores.append(score)
                if cfg["trial_pruning"]:
                    intermediate_score = float(np.mean([loss.detach().cpu().numpy() for loss in loss_scores]))
                    trial.report(intermediate_score, fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            print(f"  fold {fold_idx} done")

        if cfg["validation_function"] != "loss":
            for alpha in estimated_alphas:
                mean_score = np.mean(alpha_scores[alpha])
                print(f"  alpha={alpha:.6f} -> score={mean_score:.4f}")
                if mean_score < best_score:
                    best_score = mean_score
                    best_alpha = alpha

            trial.set_user_attr("best_alpha", float(best_alpha))
            trial.set_user_attr("estimated_alphas", list(estimated_alphas))
            return best_score
        else:
            return float(np.mean([loss.detach().cpu().numpy() for loss in loss_scores]))

    return objective


# ===== Main ============================================================================================

def main():
    args = parse_args()

    TARGET_CANCER  = args.cancer
    OUTPUT_DIR     = "results"
    name_suffix    = args.name_suffix
    limit_epochs   = args.limit_epochs
    unsupervised   = not args.supervised
    n_samples      = args.n_samples
    l1_ratio       = args.l1_ratio
    add_clinical_to_cox = args.add_clinical
    validation_function = args.validation

    n_epochs     = limit_epochs if limit_epochs else args.n_epochs
    final_epochs = limit_epochs if limit_epochs else args.final_epochs
    switch_epoch       = n_epochs // 2
    final_switch_epoch = final_epochs // 2

    print(f"\n{'='*60}")
    print(f"  Target cancer  : {TARGET_CANCER}")
    print(f"  Max epochs (CV): {n_epochs}")
    print(f"  Max epochs (final): {final_epochs}")
    print(f"  Inner splits   : {args.inner_splits}")
    print(f"  Supervised     : {args.supervised}")
    print(f"  N samples      : {'all' if n_samples == -1 else n_samples}")
    print(f"  L1 ratio       : {l1_ratio}")
    print(f"  Optuna trials  : {args.n_trials_per_worker}  |  timeout: {args.timeout}s")
    print(f"  Limit epochs   : {limit_epochs}")
    print(f"  Validation     : {validation_function}")
    print(f"  Domain Adversarial: {args.domain_adv}")
    print(f"{'='*60}\n")

    # --- Load data ---
    path = "../data/dict_pancancer_union_mutation.pickle"
    with open(path, "rb") as f:
        pancancer = pickle.load(f)

    clinical_all   = pancancer["clinical"]
    source_mask    = clinical_all["cancer_type"] != TARGET_CANCER
    source_candidates = clinical_all[source_mask]

    if n_samples == -1:
        source_indices = source_candidates.index
    else:
        source_indices = source_candidates.sample(n_samples, random_state=42).index

    source_data = {name: df.loc[source_indices] for name, df in pancancer.items()}
    clinical_df = source_data["clinical"]

    path = f"../data/clinical/{TARGET_CANCER}_clinical.pickle"
    with open(path, "rb") as f:
        df = pickle.load(f)
    clinical_test = df[list(set(df.columns) - {"time", "bcr_patient_barcode", "status"})]

    omics_df = {
        "protein":  source_data["_rna"],
        "gene_exp": source_data["mirna"],
        "methyl":   source_data["cnv"],
        "mutation": source_data["mutation"],
    }

    lt_samples = list(clinical_df.index)
    print(f"Source samples : {len(lt_samples)}")

    offset = clinical_df.index[0] - clinical_test.index[0]

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = list(omics_df.keys())

    
    print("=== Feature selection on source cancers ===")
    selector_source = fit_feature_selector(omics_df, nbFeatures=5000)
    omics_source    = apply_feature_selector(omics_df, selector_source)

     # --- Inner CV splitter ---
    if args.cv_strategy == "loco":
        inner_cv = LeaveOneCancerOutCV(
            clinical_df=clinical_df,
            cancer_col="cancer_type",
            n_folds=args.n_folds_loco,
            random_state=0,
        )
        print(f"  CV strategy    : LOCO  |  max folds: {args.n_folds_loco or 'all'}")
    else:
        inner_cv = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=0)
        print(f"  CV strategy    : Stratified  |  splits: {args.inner_splits}")

    # --- domain adversarial preparation ---
    domain_label_encoder = LabelEncoder().fit(clinical_df["cancer_type"])
    clinical_df = clinical_df.copy()
    clinical_df["domain_label"] = domain_label_encoder.transform(clinical_df["cancer_type"])
    n_domains = len(domain_label_encoder.classes_)
    print(f"  N domains (cancer types in source) : {n_domains}")

    cfg = dict(
        omics_df            = omics_df,
        omics_source        = omics_source,
        clinical_df         = clinical_df,
        lt_samples          = lt_samples,
        sources             = sources,
        device              = device,
        unsupervised        = unsupervised,
        hidden_dim          = [512, 256],
        central_hidden      = [512, 256],
        num_classes         = 5,
        classifier_dim      = [128, 64],
        survival_dim        = [64, 32],
        dropout             = 0.2,
        batch_size          = 32,
        n_epochs            = n_epochs,
        switch_epoch        = switch_epoch,
        label               = "status",
        event               = "status",
        surv_time           = "time",
        task                = "survival",
        nbFeatures          = 5000,
        validation_function = validation_function,
        l1_ratio            = l1_ratio,
        inner_cv            = inner_cv,
        limit_epochs        = limit_epochs,
        add_clinical_to_cox = add_clinical_to_cox,
        clinical_test       = clinical_test,
        offset              = offset,
        domain_adv          = args.domain_adv,
        lambda_domain        = args.lambda_domain,
        domain_hidden_dim    = args.domain_hidden_dim,
        n_domains            = n_domains,
        domain_label_encoder = domain_label_encoder,
        trial_pruning      = True,
    )

   
    os.makedirs("optuna_journal", exist_ok=True)
    journal_path = f"optuna_journal/journal_tl_{name_suffix}source_{TARGET_CANCER}.log"
    study_name   = f"source_{name_suffix}{TARGET_CANCER}"

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