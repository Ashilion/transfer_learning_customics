# -*- coding: utf-8 -*-
"""
Entraîne CustOMICS sur COAD avec les meilleurs hyperparamètres
du trial 0 (outer fold 0) récupérés depuis Optuna, puis sauvegarde les poids.

Usage :
    python analyse_latent_space_save_model.py
    python analyse_latent_space_save_model.py --outer_fold 0 --name_suffix "" 
"""

import argparse
import pickle
import os

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from sklearn.model_selection import KFold

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df
from custcox_utils import fit_feature_selector, apply_feature_selector


# ── Argument Parsing ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save CustOMICS weights trained with Optuna best params (COAD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cancer",       type=str, default="COAD")
    parser.add_argument("--outer_fold",   type=int, default=0)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--name_suffix",  type=str, default="")
    parser.add_argument("--n_epochs",     type=int, default=400,
                        help="Max epochs (early stopping désactivé si --no_early_stopping).")
    parser.add_argument("--no_early_stopping", action="store_true", default=False)
    parser.add_argument("--supervised",   action="store_true", default=False)
    parser.add_argument("--nbFeatures",   type=int, default=5000)
    parser.add_argument("--save_path",    type=str, default=None,
                        help="Chemin de sauvegarde du .pt. "
                             "Défaut : models/coad_fold{outer_fold}_best.pt")
    parser.add_argument("--journal_dir",  type=str, default="optuna_journal",
                        help="Dossier contenant les journaux Optuna.")
    parser.add_argument("--data_path",    type=str,
                        default="../data/dict_pancancer_union_mutation.pickle")
    parser.add_argument("--clinical_dir", type=str, default="../data/clinical")
    return parser.parse_args()


# ── Helpers ──────────────────────────────────

def load_pancancer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_cancer_data(pancancer, cancer_name):
    return {
        name: df[pancancer["clinical"]["cancer_type"] == cancer_name]
        for name, df in pancancer.items()
    }


HIDDEN_DIM_MAP = {
    "(1024, 512, 256, 128)": (1024, 512, 256, 128),
    "(1024, 256)":           (1024, 256),
    "(512, 128)":            (512, 128),
    "(1024, 256, 128)":      (1024, 256, 128),
    "(1024, 512, 128)":      (1024, 512, 128),
    "(512, 256, 128)":       (512, 256, 128),
    "(256, 128)":            (256, 128),
    "(512, 256)":            (512, 256),
}


def build_model(omics_data, sources, params, device, unsupervised, switch_epoch,
                num_classes=5, classifier_dim=None, survival_dim=None):
    if classifier_dim is None:
        classifier_dim = [128, 64]
    if survival_dim is None:
        survival_dim = [64, 32]

    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {
            "input_dim":  x_dim[i],
            "hidden_dim": params["autoencoder_hidden_dims"][i],
            "latent_dim": params["rep_dim"],
            "norm":       True,
            "dropout":    params["dropout"],
        }
        for i, src in enumerate(sources)
    }
    central_params = {
        "hidden_dim": params["central_hidden_dims"],
        "latent_dim": params["latent_dim"],
        "norm":       True,
        "dropout":    params["dropout"],
        "beta":       params["beta"],
    }
    classif_params = {
        "n_class":       num_classes,
        "lambda":        0,
        "hidden_layers": classifier_dim,
        "dropout":       params["dropout"],
    }
    surv_params = {
        "lambda":     5,
        "dims":       survival_dim,
        "activation": "SELU",
        "l2_reg":     1e-2,
        "norm":       True,
        "dropout":    params["dropout"],
    }
    train_params = {"switch": switch_epoch, "lr": params["lr"]}

    model = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device,
        unsupervised=unsupervised,
    ).to(device)

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cancer      = args.cancer
    outer_fold  = args.outer_fold
    suffix      = args.name_suffix
    device      = torch.device("cpu")
    unsupervised = not args.supervised

    # ── Données ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Cancer : {cancer}  |  outer fold : {outer_fold}")
    print(f"{'='*55}\n")

    pancancer   = load_pancancer(args.data_path)
    data        = get_cancer_data(pancancer, cancer)
    clinical_df = data["clinical"]

    omics_df = {
        "protein":  data["_rna"],
        "gene_exp": data["mirna"],
        "methyl":   data["cnv"],
        "mutation": data["mutation"],
    }
    sources    = list(omics_df.keys())
    lt_samples = list(clinical_df.index)

    # ── Split outer fold ──────────────────────────────────────────────────────
    outer_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
    splits   = list(outer_cv.split(lt_samples))
    train_idx, _ = splits[outer_fold]
    samples_train = [lt_samples[i] for i in train_idx]

    omics_train_raw = get_sub_omics_df(omics_df, samples_train)
    selector        = fit_feature_selector(omics_train_raw, nbFeatures=args.nbFeatures)
    omics_train     = apply_feature_selector(omics_train_raw, selector)

    # ── Charger l'étude Optuna ────────────────────────────────────────────────
    journal_path = os.path.join(
        args.journal_dir,
        f"journal_{suffix}{cancer}_fold{outer_fold}.log",
        # f"journal_{suffix}fold{outer_fold}.log",

    )
    study_name = f"journal_storage_multiprocess_{suffix}{cancer}_fold{outer_fold}"
    # study_name = f"journal_storage_multiprocess_{suffix}fold{outer_fold}"


    print(f"  Chargement de l'étude Optuna : {journal_path}")
    study = optuna.load_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(file_path=journal_path)),
    )

    # ── Récupérer le best trial ──────────────────────────────────────────────────
    best_trial = study.best_trial
    raw_params = best_trial.params
    best_alpha     = best_trial.user_attrs.get("best_alpha")
    best_list_alpha = best_trial.user_attrs.get("best_list_alpha")

    print(f"\n  Best Trial - params     : {raw_params}")
    print(f"  Best Trial - best_alpha : {best_alpha}")
    print(f"  Best Trial - value      : {best_trial.value}")

    # ── Reconstruire le dict de paramètres complet ────────────────────────────
    autoencoder_hidden_dims = [
        HIDDEN_DIM_MAP[raw_params[f"hidden_dim_{i}"]]
        for i in range(len(sources))
    ]

    params = {
        **raw_params,
        "autoencoder_hidden_dims": autoencoder_hidden_dims,
        "latent_dim":              32,
        "rep_dim":                 32,
        "central_hidden_dims":     [64],
        "patience":                5,
    }

    # ── Construire et entraîner le modèle ─────────────────────────────────────
    n_epochs     = args.n_epochs
    switch_epoch = n_epochs // 2
    patience     = None if args.no_early_stopping else params["patience"]

    model = build_model(
        omics_data   = omics_train,
        sources      = sources,
        params       = params,
        device       = device,
        unsupervised = unsupervised,
        switch_epoch = switch_epoch,
    )

    print(f"\n🚀  Entraînement  (n_epochs={n_epochs}, patience={patience}) …")
    model.fit(
        omics_train       = omics_train,
        clinical_df       = clinical_df,
        label             = "status",
        event             = "status",
        surv_time         = "time",
        omics_val         = None,
        batch_size        = 32,
        n_epochs          = n_epochs,
        verbose           = True,
        task              = "survival",
        patience          = patience,
        min_delta         = params.get("delta_min", 1e-3),
        early_stopping_on = "train",
    )

    # ── Sauvegarder ───────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    save_path = args.save_path or f"models/{cancer}_{suffix}fold{outer_fold}_best.pt"

    checkpoint = {
        "state_dict":   model.state_dict(),
        "params":       params,
        "sources":      sources,
        "cancer":       cancer,
        "outer_fold":   outer_fold,
        "trial_index":  0,
        "trial_value":  best_trial.value,
        "best_alpha":   best_alpha,
        "best_list_alpha": best_list_alpha,
        "final_epoch":  model.get_final_epoch(),
        "switch_epoch": model.get_switch_epoch(),
        "n_features":   {src: omics_train[src].shape[1] for src in sources},
    }

    torch.save(checkpoint, save_path)
    print(f"\n  Modèle sauvegardé → {save_path}")
    print(f"   Epoch finale     : {checkpoint['final_epoch']}")
    print(f"   Switch epoch     : {checkpoint['switch_epoch']}")
    print(f"   Latent dim       : {params['latent_dim']}")
    print(f"   Features/source  : {checkpoint['n_features']}")

    # ── Comment recharger ─────────────────────────────────────────────────────
    print(f"""
── Pour recharger ce modèle ──────────────────────────────────────────────────
  ckpt   = torch.load("{save_path}")
  model  = build_model(omics_train, sources, ckpt["params"], device,
                       unsupervised=True, switch_epoch=ckpt["switch_epoch"])
  model.load_state_dict(ckpt["state_dict"])
  model.eval_all()
  Z = model.get_latent_representation(omics_df)
──────────────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()