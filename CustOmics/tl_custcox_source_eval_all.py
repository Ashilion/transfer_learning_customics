import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
import optuna

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
from custcox_utils import (
    fit_feature_selector, apply_feature_selector,
    build_survival_array, fit_coxnet,
)
import os


# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Optuna evaluation for SOURCE pan-cancer pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cancer", type=str, default="COAD",
        help="Target cancer excluded from source data.")
    parser.add_argument("--n_samples", type=int, default=-1,
        help="Number of source samples used during training (-1 = all).")
    parser.add_argument("--name_suffix", type=str, default="",
        help="Suffix used in journal/study name during training.")
    parser.add_argument("--output_dir", type=str, default="tl_ckpt",
        help="Directory to write the result CSV and model checkpoint.")
    parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_model.pt",
        help="Path where the final model weights will be saved.")
    parser.add_argument("--best_params_out", type=str, default="best_params_source.json",
        help="Path where the best config JSON will be saved.")
    parser.add_argument("--n_epochs", type=int, default=1000,
        help="Max epochs for the final model (early stopping applies).")
    parser.add_argument("--supervised", action="store_true", default=False,
        help="Train in supervised mode.")
    parser.add_argument("--limit_epochs", type=int, default=None,
        help="Hard-limit epochs (disables early stopping).")

    # --- must match the flags used at training time ---
    parser.add_argument(
        "--domain_adv",
        action="store_true",
        default=False,
        help="Must match the --domain_adv setting used during Optuna training."
    )
    parser.add_argument(
        "--lambda_domain",
        type=float,
        default=1.0,
        help="Weight for the domain-adversarial loss (GRL coefficient ceiling)."
    )
    parser.add_argument(
        "--domain_hidden_dim",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden layer sizes for the domain classifier head."
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="loss",
        help="Validation function (vvh, ibs or loss)"
    )
    return parser.parse_args()


# =======================================================

def build_customics_model_plus(omics_data, sources, params, device,
                                hidden_dim, central_hidden,
                                classifier_dim, survival_dim,
                                dropout, num_classes, unsupervised, switch_epoch, 
                                domain_params=None, lambda_surv=5):

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
        'lambda': lambda_surv,
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


# ===== Main ============================================================================================

def main():
    args = parse_args()

    TARGET_CANCER  = args.cancer
    name_suffix    = args.name_suffix
    n_samples      = args.n_samples
    unsupervised   = not args.supervised
    limit_epochs   = args.limit_epochs
    n_epochs       = limit_epochs if limit_epochs else args.n_epochs
    switch_epoch   = n_epochs // 2

    print(f"\n{'='*60}")
    print(f"  [SOURCE eval]  Target cancer : {TARGET_CANCER}")
    print(f"  Study suffix   : '{name_suffix}'")
    print(f"  Max epochs     : {n_epochs}")
    print(f"  Domain Adversarial: {args.domain_adv}")
    print(f"{'='*60}\n")

    study_name   = f"source_{name_suffix}{TARGET_CANCER}"
    journal_path = f"optuna_journal/journal_tl_{name_suffix}source_{TARGET_CANCER}.log"

    study = optuna.load_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(file_path=journal_path)),
    )

    best_trial     = study.best_trial
    best_params_hp = best_trial.params
    best_score     = best_trial.value

    print(f"Best trial #{best_trial.number}")
    print(f"  params : {best_params_hp}")
    
    print(f"  score  : {best_score:.4f}")

    if args.validation != "loss":
        best_alpha     = best_trial.user_attrs["best_alpha"]
        estimated_alphas = best_trial.user_attrs.get("estimated_alphas", None)
        print(f"  alpha  : {best_alpha:.6f}")

    #Reconstruct full params dict
    autoencoder_hidden_dims_possibilities = {
        "(1024, 512, 256, 128)": (1024, 512, 256, 128),
        "(1024, 256)":           (1024, 256),
        "(512, 128)":            (512, 128),
        "(1024, 256, 128)":      (1024, 256, 128),
        "(1024, 512, 128)":      (1024, 512, 128),
        "(512, 256, 128)":       (512, 256, 128),
        "(256, 128)":            (256, 128),
        "(512, 256)":            (512, 256),
    }

    # --- Load data ---
    path = "../data/dict_pancancer_union_mutation.pickle"
    with open(path, "rb") as f:
        pancancer = pickle.load(f)

    clinical_all      = pancancer["clinical"]
    source_mask       = clinical_all["cancer_type"] != TARGET_CANCER
    source_candidates = clinical_all[source_mask]

    if n_samples == -1:
        source_indices = source_candidates.index
    else:
        source_indices = source_candidates.sample(n_samples, random_state=42).index

    source_data = {name: df.loc[source_indices] for name, df in pancancer.items()}
    clinical_df = source_data["clinical"]

    omics_df = {
        "protein":  source_data["_rna"],
        "gene_exp": source_data["mirna"],
        "methyl":   source_data["cnv"],
        "mutation": source_data["mutation"],
    }

    lt_samples = list(clinical_df.index)
    sources    = list(omics_df.keys())
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    num_classes    = 5
    classifier_dim = [128, 64]
    survival_dim   = [64, 32]
    dropout        = 0.2

    print(f"\nSource samples : {len(lt_samples)}")

    # --- domain adversarial preparation ---
    domain_params = None
    n_domains = None
    if args.domain_adv:
        domain_label_encoder = LabelEncoder().fit(clinical_df["cancer_type"])
        clinical_df = clinical_df.copy()
        clinical_df["domain_label"] = domain_label_encoder.transform(clinical_df["cancer_type"])
        n_domains = len(domain_label_encoder.classes_)
        print(f"  N domains (cancer types in source) : {n_domains}")

        domain_params = {
            "n_domains": n_domains,
            "lambda": args.lambda_domain,
            "hidden_layers": args.domain_hidden_dim,
            "dropout": dropout,
        }

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_filename = f"{TARGET_CANCER}_{os.path.basename(args.pretrain_ckpt)}"
    config_filename = f"{TARGET_CANCER}_{os.path.basename(args.best_params_out)}"

    pretrain_ckpt_path = os.path.join(args.output_dir, ckpt_filename)
    best_params_out_path = os.path.join(args.output_dir, config_filename)
    selector_path = f"{pretrain_ckpt_path}.selector.pkl"

    selector_source = fit_feature_selector(omics_df, nbFeatures=5000)
    omics_source    = apply_feature_selector(omics_df, selector_source)

    # save selector to reuse for target data
    with open(selector_path, "wb") as f:
        pickle.dump(selector_source, f)

    autoencoder_hidden_dims = []
    for i, src in enumerate(sources):
        key = best_params_hp[f"hidden_dim_{i}"]
        autoencoder_hidden_dims.append(autoencoder_hidden_dims_possibilities[key])

    best_params_full = {
        **best_params_hp,
        "autoencoder_hidden_dims": autoencoder_hidden_dims,
        "latent_dim":              32,
        "rep_dim":                 32,
        "central_hidden_dims":     [64],
        "patience":                best_params_hp.get("patience", 3),
    }

    print("\n=== Final training on all source data ===")
    patience = None if limit_epochs else best_params_full["patience"]

    final_model = build_customics_model_plus(
        omics_source, sources, best_params_full, device,
        best_params_full["autoencoder_hidden_dims"], best_params_full["central_hidden_dims"], classifier_dim, survival_dim,
        dropout, num_classes, unsupervised, switch_epoch,
        domain_params=domain_params,
    )
    final_model.fit(
        omics_train=omics_source, clinical_df=clinical_df,
        label="status", event="status", surv_time="time",
        omics_val=None, batch_size=32, n_epochs=n_epochs,
        verbose=True, task="survival",
        patience=patience, min_delta=best_params_full["delta_min"],
        early_stopping_on="train",
    )


    torch.save(final_model.state_dict(), pretrain_ckpt_path)
    print(f"Weights saved -> {pretrain_ckpt_path}")

    best_config = {
        "best_params":   best_params_full,
        "best_score":    float(best_score),
        "architecture": {
            "hidden_dim": best_params_full["autoencoder_hidden_dims"],
            "central_hidden": best_params_full["central_hidden_dims"],
            "classifier_dim": classifier_dim,
            "survival_dim":   survival_dim,
            "num_classes":    num_classes,
            "dropout":        dropout,
            "unsupervised":   unsupervised,
        },
        "domain_adversarial": {
            "enabled":          args.domain_adv,
            "lambda_domain":    args.lambda_domain,
            "domain_hidden_dim": args.domain_hidden_dim,
            "n_domains":        n_domains,
        },
        "target_cancer":      TARGET_CANCER,
        "nb_source_samples":  len(lt_samples),
        "optuna_study_name":  study_name,
        "optuna_journal":     journal_path,
        "optuna_trial":       best_trial.number,
    }

    if args.validation != "loss":
        best_config["best_alpha"] = float(best_alpha)

    with open(best_params_out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Config saved -> {best_params_out_path}")


if __name__ == "__main__":
    main()