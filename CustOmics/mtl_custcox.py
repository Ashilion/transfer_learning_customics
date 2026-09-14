"""
Pan-cancer multitask survival (DeepSurv-style, pas de CoxPH) avec un tronc
partage (encodeurs par omique + VAE central) et UNE tete de survie MLP par
cancer (18 cancers).

Boucle : 50 outer folds. A chaque outer fold i, chaque cancer apporte son
fold pre-defini i (train_i / test_i) issu de splits.json ; les train_i de
tous les cancers sont poolees pour entrainer UN SEUL modele multitask ; les
test_i restent separes par cancer pour l'evaluation.

Pas de recherche d'hyperparametres : 1 seul split interne (train/val) pour
l'early stopping, avec des HP fixes (valeurs par defaut du script
tl_custcox_source_eval_all.py).

La prediction finale utilise directement la sortie du reseau (tete de
survie du cancer concerne), convertie en fonction de survie via un
estimateur de Breslow, pour calculer C-index (IPCW) et IBS -- pas de
CoxnetSurvivalAnalysis.
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
import utils.folds_utils as fold_utils

from custcox_utils import fit_feature_selector, apply_feature_selector, build_survival_array
from src.tools.utils import get_sub_omics_df
from src.network.customics_mtl import CustOMICSMultiTask
from mtl_deepsurv_utils import (
    TaskBalancedBatchSampler, breslow_baseline_cumhazard, predict_survival_function
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pan-cancer multitask DeepSurv (18 têtes) — 50 outer folds, HP fixes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path", type=str, default="../data/dict_pancancer_union_mutation.pickle")
    p.add_argument("--splits_path", type=str, default="../data/splits.json")
    p.add_argument("--output_dir", type=str, default="results_mtl")
    p.add_argument("--n_outer_folds", type=int, default=50)
    p.add_argument("--nbFeatures", type=int, default=5000)
    p.add_argument("--n_epochs", type=int, default=1000, help="epochs max (ceiling early stopping)")
    p.add_argument("--batch_size_per_task", type=int, default=6, help="echantillons par cancer et par batch")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.15, help="proportion de l'outer-train pour le split interne unique")
    p.add_argument("--start_fold", type=int, default=0)
    p.add_argument("--end_fold", type=int, default=None, help="exclu; defaut = n_outer_folds")
    p.add_argument("--outer_fold", type=int, required=True, help="index du SEUL outer fold a executer dans ce run ")
    return p.parse_args()


# ===== HP fixes (valeurs par defaut de tl_custcox_source_eval_all.py) ================

FIXED_HP = dict(
    latent_dim=64,
    rep_dim=64,
    central_hidden_dims=[128],
    hidden_dim=[512, 256],
    survival_dim=[32,16],
    dropout=0.2,
    lr=2e-4,
    beta=5,
)


def default_autoencoder_hidden_dims(omics_source, sources):
    """Choix fixe (pas de recherche Optuna) : gros dims si >1200 features, sinon plus petit."""
    dims = []
    for src in sources:
        if omics_source[src].shape[1] > 1200:
            dims.append((1024, 256))
        else:
            dims.append((512, 256))
    return dims


def build_model(omics_data, sources, cancer_ids, device):
    x_dim = [omics_data[src].shape[1] for src in sources]
    ae_hidden = default_autoencoder_hidden_dims(omics_data, sources)

    source_params = {
        src: {
            'input_dim': x_dim[i],
            'hidden_dim': ae_hidden[i],
            'latent_dim': FIXED_HP['rep_dim'],
            'norm': True,
            'dropout': FIXED_HP['dropout'],
        }
        for i, src in enumerate(sources)
    }
    central_params = {
        'hidden_dim': FIXED_HP['central_hidden_dims'],
        'latent_dim': FIXED_HP['latent_dim'],
        'norm': True,
        'dropout': FIXED_HP['dropout'],
        'beta': FIXED_HP['beta'],
    }
    surv_params = {
        'lambda': 100,
        'dims': FIXED_HP['survival_dim'],
        'activation': 'SELU',
        'l2_reg': 1e-2,
        'norm': True,
        'dropout': FIXED_HP['dropout'],
    }
    n_epochs = FIXED_HP.get('n_epochs')
    train_params = {'switch': None, 'lr': FIXED_HP['lr']}  # switch fixe plus bas

    model = CustOMICSMultiTask(
        source_params=source_params, central_params=central_params,
        surv_params=surv_params, train_params=train_params,
        device=device, cancer_types=cancer_ids,
    ).to(device)
    return model


def main():
    args = parse_args()
    outer_fold = args.outer_fold
    if not (0 <= outer_fold < args.n_outer_folds):
        raise ValueError(f"--outer_fold={outer_fold} doit etre dans [0, {args.n_outer_folds}).")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    with open(args.data_path, "rb") as f:
        pancancer = pickle.load(f)
    clinical_all = pancancer["clinical"].copy()
    cancer_types = sorted(clinical_all["cancer_type"].unique().tolist())
    print(f"{len(cancer_types)} types de cancer : {cancer_types}")
 
    # MultiOmicsDataset fait int(clinical_df.loc[sample, domain_col]) -> il faut
    # une colonne ENTIERE pour router chaque echantillon vers sa tete de survie.
    cancer_encoder = LabelEncoder().fit(cancer_types)
    clinical_all["cancer_id"] = cancer_encoder.transform(clinical_all["cancer_type"])
    cancer_ids_all = list(range(len(cancer_types)))  # == cancer_encoder.transform(cancer_types)
 
    omics_df = {
        "rna":  pancancer["_rna"],
        "mirna": pancancer["mirna"],
        "cnv":   pancancer["cnv"],
        "mutation": pancancer["mutation"],
    }
    sources = list(omics_df.keys())
 
    print(f"\n{'='*60}\nOUTER FOLD {outer_fold}/{args.n_outer_folds - 1}\n{'='*60}")
 
    all_results = []
 
    train_samples_pooled = []
    test_samples_by_cancer = {}
 
    for cancer in cancer_types:
        train_folds, test_folds = fold_utils.get_folds(cancer, src=args.splits_path)
        if outer_fold >= len(train_folds):
            raise ValueError(
                f"Le cancer {cancer} n'a que {len(train_folds)} folds pre-enregistres "
                f"(< {outer_fold + 1}). Verifie splits.json / --n_outer_folds."
            )
        tr_pos, te_pos = np.asarray(train_folds[outer_fold]), np.asarray(test_folds[outer_fold])
        cancer_sample_ids = clinical_all.index[clinical_all["cancer_type"] == cancer].to_numpy()
 
        max_pos = max(np.max(tr_pos), np.max(te_pos))
        if max_pos >= len(cancer_sample_ids):
            raise ValueError(
                f"[{cancer}] get_folds renvoie des positions jusqu'a {max_pos}, mais le "
                f"sous-dataframe clinique de ce cancer n'a que {len(cancer_sample_ids)} lignes. "
                f"L'ordre/la taille utilises pour construire splits.json ne correspond pas a "
                f"clinical_all -- verifie la source de splits.json pour ce cancer."
            )
 
        tr_idx = cancer_sample_ids[tr_pos]
        te_idx = cancer_sample_ids[te_pos]
 
        train_samples_pooled.extend(tr_idx.tolist())
        test_samples_by_cancer[cancer] = te_idx.tolist()
 
    clinical_train = clinical_all.loc[train_samples_pooled]
 
    # --- split interne UNIQUE (pas de k-fold, pas de tuning) ---
    inner_train_samples, inner_val_samples = train_test_split(
        train_samples_pooled, test_size=args.val_ratio, random_state=outer_fold,
        stratify=clinical_train["status"],
    )
 
    omics_inner_train_raw = get_sub_omics_df(omics_df, inner_train_samples)
    omics_inner_val_raw   = get_sub_omics_df(omics_df, inner_val_samples)
 
    selector = fit_feature_selector(omics_inner_train_raw, nbFeatures=args.nbFeatures)
    omics_inner_train = apply_feature_selector(omics_inner_train_raw, selector)
    omics_inner_val   = apply_feature_selector(omics_inner_val_raw, selector)
 
    n_epochs = args.n_epochs
    switch_epoch = n_epochs // 2
 
    model = build_model(omics_inner_train, sources, cancer_ids_all, device)
    model.switch_epoch = switch_epoch
 
    model.fit(
        omics_train=omics_inner_train, clinical_df=clinical_all,
        cancer_id_col="cancer_id", event="status", surv_time="time",
        omics_val=omics_inner_val, n_epochs=n_epochs, verbose=True,
        patience=args.patience, min_delta=args.min_delta, early_stopping_on="train",
        n_per_task_train=args.batch_size_per_task,
        n_per_task_val=max(2, args.batch_size_per_task // 2),
        sampler_seed=outer_fold,
    )
 
    out_path = os.path.join(args.output_dir, f"mtl_pancancer_deepsurv_fold{outer_fold}.csv")
 
    # --- evaluation par cancer, prediction directe (pas de CoxPH) ---
    for cancer in cancer_types:
        test_samples = test_samples_by_cancer[cancer]
        train_samples_this_cancer = [s for s in inner_train_samples
                                      if clinical_all.loc[s, "cancer_type"] == cancer]
        if len(train_samples_this_cancer) < 5 or len(test_samples) < 5:
            print(f"  [{cancer}] pas assez d'echantillons, fold ignore.")
            continue
 
        omics_train_c_raw = get_sub_omics_df(omics_df, train_samples_this_cancer)
        omics_test_c_raw  = get_sub_omics_df(omics_df, test_samples)
        omics_train_c = apply_feature_selector(omics_train_c_raw, selector)
        omics_test_c  = apply_feature_selector(omics_test_c_raw, selector)
 
        y_train_c = build_survival_array(clinical_all, train_samples_this_cancer, "status", "time")
        y_test_c  = build_survival_array(clinical_all, test_samples, "status", "time")
 
        cancer_id = int(cancer_encoder.transform([cancer])[0])
        risk_train = model.predict_risk(omics_train_c, cancer_id)
        risk_test  = model.predict_risk(omics_test_c, cancer_id)
 
        train_max_time = y_train_c["time"].max()
        test_mask = y_test_c["time"] < train_max_time
        if test_mask.sum() < 5:
            print(f"  [{cancer}] pas assez de test dans le support temporel du train, fold ignore.")
            continue
        y_test_eval    = y_test_c[test_mask]
        risk_test_eval = risk_test[test_mask]
 
        eval_times = np.sort(np.unique(y_test_eval["time"]))
        tau = eval_times.max()
 
        c_index = concordance_index_ipcw(y_train_c, y_test_eval, risk_test_eval, tau=tau)[0]
 
        event_times, cumhazard = breslow_baseline_cumhazard(
            risk_train, y_train_c["time"], y_train_c["status"])
 
        if len(eval_times) < 2 or len(event_times) == 0:
            ibs = np.nan
        else:
            preds = predict_survival_function(risk_test_eval, event_times, cumhazard, eval_times)
            ibs = integrated_brier_score(y_train_c, y_test_eval, preds, eval_times)
 
        print(f"  [{cancer}] C-index={c_index:.4f}  IBS={ibs:.4f}")
        all_results.append({
            "outer_fold": outer_fold,
            "cancer": cancer,
            "cindex": c_index,
            "ibs": ibs,
            "n_train": len(train_samples_this_cancer),
            "n_test": len(test_samples),
        })

        pd.DataFrame(all_results).to_csv(out_path, index=False)
 
    print(f"\nTermine. Resultats -> {out_path}")


if __name__ == "__main__":
    main()