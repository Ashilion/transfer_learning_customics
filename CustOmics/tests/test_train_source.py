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


TARGET_CANCER = "COAD"          # cancer exclu des sources
OUTPUT_DIR = "results"
PRETRAIN_CKPT = "pretrained_pan_cancer.pt"
BEST_PARAMS_OUT = "results/best_params_source.json"

path = "../data/dict_pancancer_union_mutation.pickle"
with open(path, "rb") as f:
    pancancer = pickle.load(f)


clinical_all = pancancer["clinical"]
source_mask = clinical_all["cancer_type"] != TARGET_CANCER
source_indices = clinical_all[source_mask].sample(2000, random_state=42).index
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
# # Estimation utilisation mémoire
# for name, df in omics_df.items():
#     print(f"{name}: {df.memory_usage(deep=True).sum()/1e6:.1f} MB — shape {df.shape}")

lt_samples = list(clinical_df.index)
print("taille donnée :", len(lt_samples))


device = torch.device("cpu")
batch_size = 32
n_epochs = 4
switch_epoch = n_epochs //2

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
unsupervised = True
dropout = 0.2

param_grid = {
    "latent_dim": [64, 128],
    "rep_dim":    [64, 128],
    "lr":         [1e-3, 1e-4],
}

nbFeatures = 5000
validation_function = "vvh"

inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

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

        coxnet, scaler = fit_coxnet(Z_train, y_train_struct,l1_ratio, estimated_alphas)

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
final_epochs = 8 
final_switch = final_epochs //2
final_model = build_customics_model(
    omics_source, sources, best_params, device,
    hidden_dim, central_hidden, classifier_dim, survival_dim,
    dropout, num_classes, unsupervised, final_switch
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