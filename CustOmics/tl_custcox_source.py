import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
import time

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
from utils.vvh_cv import vvh_cv

from custcox_utils import fit_feature_selector, apply_feature_selector, build_customics_model, build_survival_array, fit_coxnet, evaluate_survival


# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pan-cancer pre-training with CustOMICS + CoxNet hyperparameter search.",
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
        default=4,
        help="Number of training epochs for inner CV models."
    )
    parser.add_argument(
        "--final_epochs",
        type=int,
        default=10,
        help="Number of training epochs for the final model."
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
        help="Number of source samples to use. Set to -1 to use all available samples."
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.01,
        help="L1 ratio for CoxNet regularization."
    )

    return parser.parse_args()

# =====================================================================================

def main():
    args = parse_args()

    TARGET_CANCER = args.cancer
    OUTPUT_DIR = "results"
    PRETRAIN_CKPT = "pretrained_pan_cancer.pt"
    BEST_PARAMS_OUT = "results/best_params_source.json"

    n_epochs = args.n_epochs
    final_epochs = args.final_epochs
    unsupervised = not args.supervised
    n_samples = args.n_samples
    l1_ratio = args.l1_ratio

    print(f"\n{'='*60}")
    print(f"  Target cancer  : {TARGET_CANCER}")
    print(f"  Epochs (CV)    : {n_epochs}")
    print(f"  Epochs (final) : {final_epochs}")
    print(f"  Inner splits   : {args.inner_splits}")
    print(f"  Supervised     : {args.supervised}")
    print(f"  N samples      : {'all' if n_samples == -1 else n_samples}")
    print(f"  L1 ratio       : {l1_ratio}")
    print(f"{'='*60}\n")

    path = "../data/dict_pancancer_union_mutation.pickle"
    with open(path, "rb") as f:
        pancancer = pickle.load(f)

    clinical_all = pancancer["clinical"]
    source_mask = clinical_all["cancer_type"] != TARGET_CANCER
    source_candidates = clinical_all[source_mask]

    if n_samples == -1:
        source_indices = source_candidates.index
    else:
        source_indices = source_candidates.sample(n_samples, random_state=42).index

    source_data = {}

    for name, df in pancancer.items():
        source_data[name] = df.loc[source_indices]

    clinical_df = source_data["clinical"]

    omics_df = {
        "protein":   source_data["_rna"],
        "gene_exp":  source_data["mirna"],
        "methyl":    source_data["cnv"],
        "mutation":  source_data["mutation"],
    }

    lt_samples = list(clinical_df.index)
    print("taille donnée :", len(lt_samples))

    device = torch.device("cpu")
    batch_size = 32

    label = "status"
    event = "status"
    surv_time = "time"
    task = "survival"

    sources = list(omics_df.keys())

    hidden_dim = [512, 256]
    central_hidden = [512, 256]
    num_classes = 5
    classifier_dim = [128, 64]
    survival_dim = [64, 32]
    dropout = 0.2
    switch_epoch = n_epochs // 2
    final_switch_epoch = final_epochs // 2

    param_grid = {
        "latent_dim": [64, 128],
        "rep_dim":    [64, 128],
        "lr":         [1e-3, 1e-4],
    }

    nbFeatures = 5000
    validation_function = "vvh"

    inner_cv = KFold(n_splits=args.inner_splits, shuffle=True, random_state=0)

    print("=== Feature selection sur les cancers sources ===")
    selector_source = fit_feature_selector(omics_df, nbFeatures=nbFeatures)
    omics_source = apply_feature_selector(omics_df, selector_source)


    print("=== Recherche des hyperparamètres (inner CV sur sources) ===")

    print("  -> Calcul de la grille d'alphas de référence...")
    ref_params = list(ParameterGrid(param_grid))[0]   # params quelconques pour l'init
    ref_model = build_customics_model(
        omics_source, sources, ref_params, device,
        hidden_dim, central_hidden, classifier_dim, survival_dim,
        dropout, num_classes, unsupervised,switch_epoch
    )
    ref_model.fit(
        omics_train=omics_source, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs,
        verbose=True, task=task,
    )
    y_all = build_survival_array(clinical_df, lt_samples, event, surv_time)
    Z_all = ref_model.get_latent_representation(omics_source)
    ref_cox, scaler = fit_coxnet(Z_all, y_all, l1_ratio)
    estimated_alphas = ref_cox.alphas_
    print(f"  -> {len(estimated_alphas)} alphas candidats")

    best_score = np.inf
    best_params = None
    best_alpha = None

    for params in ParameterGrid(param_grid):
        print(f"\n--- Params : {params} ---")
        alpha_scores = {alpha: [] for alpha in estimated_alphas}

        for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(lt_samples)):
            samples_train_inner = [lt_samples[i] for i in inner_train_idx]
            samples_val_inner = [lt_samples[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
            omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

            sel_inner = fit_feature_selector(omics_train_raw, nbFeatures=nbFeatures)
            omics_train = apply_feature_selector(omics_train_raw, sel_inner)
            omics_val = apply_feature_selector(omics_val_raw,   sel_inner)

            model = build_customics_model(
                omics_train, sources, params, device,
                hidden_dim, central_hidden, classifier_dim, survival_dim,
                dropout, num_classes, unsupervised,switch_epoch
            )
            model.fit(
                omics_train=omics_train, clinical_df=clinical_df,
                label=label, event=event, surv_time=surv_time,
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs,
                verbose=False, task=task,
            )

            Z_train = model.get_latent_representation(omics_train)
            Z_val = model.get_latent_representation(omics_val)

            y_train_struct = build_survival_array(clinical_df, samples_train_inner, event, surv_time)
            y_val_struct = build_survival_array(clinical_df, samples_val_inner, event, surv_time)

            coxnet, scaler = fit_coxnet(Z_train, y_train_struct, l1_ratio, estimated_alphas)

            for alpha in coxnet.alphas_:
                score = evaluate_survival(
                    coxnet, alpha,
                    Z_train, y_train_struct,
                    Z_val,   y_val_struct,
                    validation_function,
                )
                alpha_scores[alpha].append(score)

            print(f"  fold {fold_idx} done")

        for alpha in estimated_alphas:
            mean_score = np.mean(alpha_scores[alpha])
            print(f"  alpha={alpha:.6f} -> score={mean_score:.4f}")
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                best_alpha = alpha

    print(f"\n=== Meilleurs hyperparamètres : {best_params}  |  best_alpha : {best_alpha:.6f} ===")

    print("\n=== Entraînement final sur toutes les sources ===")
    final_model = build_customics_model(
        omics_source, sources, best_params, device,
        hidden_dim, central_hidden, classifier_dim, survival_dim,
        dropout, num_classes, unsupervised,final_switch_epoch
    )
    final_model.fit(
        omics_train=omics_source, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=final_epochs,
        verbose=False, task=task,
    )

    # 1) Poids du modèle
    torch.save(final_model.state_dict(), PRETRAIN_CKPT)
    print(f"Poids sauvegardés -> {PRETRAIN_CKPT}")

    # 2) Hyperparamètres + alpha + architecture
    best_config = {
        "best_params":     best_params,
        "best_alpha":      float(best_alpha),
        "best_score":      float(best_score),
        "architecture": {
            "hidden_dim":     hidden_dim,
            "central_hidden": central_hidden,
            "classifier_dim": classifier_dim,
            "survival_dim":   survival_dim,
            "num_classes":    num_classes,
            "dropout":        dropout,
            "unsupervised":   unsupervised,
        },
        "target_cancer":   TARGET_CANCER,
        "nb_source_samples": len(lt_samples),
    }

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(BEST_PARAMS_OUT, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Hyperparamètres sauvegardés -> {BEST_PARAMS_OUT}")


if __name__ == "__main__":
    main()
        