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

TARGET_CANCER = "COAD"
PRETRAIN_CKPT = "pretrained_pan_cancer.pt"
BEST_PARAMS_IN = "results/best_params_source.json"
OUTPUT_DIR = "results"

with open(BEST_PARAMS_IN, "r") as f:
    best_config = json.load(f)

best_params = best_config["best_params"]
best_alpha_src = best_config["best_alpha"]
arch = best_config["architecture"]

print(f"Hyperparamètres chargés : {best_params}  |  alpha source : {best_alpha_src:.6f}")

path = "../data/dict_pancancer_union_mutation.pickle"
with open(path, "rb") as f:
    pancancer = pickle.load(f)

clinical_all = pancancer["clinical"]
target_mask = clinical_all["cancer_type"] == TARGET_CANCER

target_data = {name: df[target_mask] for name, df in pancancer.items()}
clinical_df = target_data["clinical"]

omics_df = {
    "protein":  target_data["_rna"],
    "gene_exp": target_data["mirna"],
    "methyl":   target_data["cnv"],
    "mutation": target_data["mutation"],
}

lt_samples = list(clinical_df.index)
sources = list(omics_df.keys())

device = torch.device("cpu")
batch_size = 32
n_epochs_ft = 10          

label = "status"
event = "status"
surv_time = "time"
task = "survival"

hidden_dim     = arch["hidden_dim"]
central_hidden = arch["central_hidden"]
classifier_dim = arch["classifier_dim"]
survival_dim   = arch["survival_dim"]
num_classes    = arch["num_classes"]
dropout        = arch["dropout"]
unsupervised   = arch["unsupervised"]

nbFeatures = 5000
validation_function = "vvh"

# Grille réduite
param_grid_ft = {
    "lr": [1e-3, 1e-4, 5e-5],
}

inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)

def build_model(omics_data, params):
    """Construit un CustOMICS avec les dims fixées par best_config."""
    x_dim = [omics_data[src].shape[1] for src in sources]
    source_params = {
        src: {
            "input_dim":  x_dim[i],
            "hidden_dim": hidden_dim,
            "latent_dim": best_params["rep_dim"],
            "norm":       True,
            "dropout":    dropout,
        }
        for i, src in enumerate(sources)
    }
    central_params = {
        "hidden_dim": central_hidden,
        "latent_dim": best_params["latent_dim"],
        "norm":       True,
        "dropout":    dropout,
        "beta":       1,
    }
    classif_params = {
        "n_class":       num_classes,
        "lambda":        0,
        "hidden_layers": classifier_dim,
        "dropout":       dropout,
    }
    surv_params = {
        "lambda":     5,
        "dims":       survival_dim,
        "activation": "SELU",
        "l2_reg":     1e-2,
        "norm":       True,
        "dropout":    dropout,
    }
    train_params = {"switch": 5, "lr": params["lr"]}
 
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
    
def load_pretrained(model, ckpt_path, strict=False):
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return model

def freeze_and_reset_optimizer(model, lr):
    model.freeze_autoencoders()
    model.update_optimizer(lr)
    return model

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(lt_samples)):
    print(f"\n{'='*50}")
    print(f" OUTER FOLD {outer_fold}")
    print(f"{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

    sel_outer  = fit_feature_selector(omics_train_outer_raw, nbFeatures)
    omics_train_outer = apply_feature_selector(omics_train_outer_raw, sel_outer)
    omics_test_outer = apply_feature_selector(omics_test_outer_raw,  sel_outer)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, event, surv_time)
    y_test_outer = build_survival_array(clinical_df, samples_test_outer,  event, surv_time)

    
    print("  -> Calcul de la grille d'alphas de référence (outer train)...")
    ref_model = build_model(omics_train_outer, {"lr": 1e-3})
    ref_model = load_pretrained(ref_model, PRETRAIN_CKPT, strict=False)
    ref_model = freeze_and_reset_optimizer(ref_model, lr=1e-3)
    #only train with phase 2 
    ref_model.switch_epoch = 0
    ref_model.fit(
        omics_train=omics_train_outer, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs_ft,
        verbose=False, task=task,
    )
    Z_outer_ref = ref_model.get_latent_representation(omics_train_outer)
    ref_cox, scaler = fit_coxnet(Z_outer_ref, y_train_outer, l1_ratio)
    estimated_alphas = ref_cox.alphas_
    print(f"  -> {len(estimated_alphas)} alphas candidats")

    
    best_score_inner = np.inf
    best_params_ft = None
    best_alpha_ft = None

    for params in ParameterGrid(param_grid_ft):
        alpha_scores = {alpha: [] for alpha in estimated_alphas}

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(samples_train_outer)
        ):
            samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
            samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
            omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

            sel_inner = fit_feature_selector(omics_train_raw, nbFeatures)
            omics_train = apply_feature_selector(omics_train_raw, sel_inner)
            omics_val = apply_feature_selector(omics_val_raw,   sel_inner)

            # Chargement poids pré-entraînés + freeze AE
            model = build_model(omics_train, params)
            model = load_pretrained(model, PRETRAIN_CKPT, strict=False)
            model = freeze_and_reset_optimizer(model, lr=params["lr"])
            #only train with phase 2 
            model.switch_epoch = 0

            model.fit(
                omics_train=omics_train, clinical_df=clinical_df,
                label=label, event=event, surv_time=surv_time,
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs_ft,
                verbose=False, task=task,
            )

            Z_train = model.get_latent_representation(omics_train)
            Z_val = model.get_latent_representation(omics_val)

            y_train_struct = build_survival_array(clinical_df, samples_train_inner, event, surv_time)
            y_val_struct = build_survival_array(clinical_df, samples_val_inner,   event, surv_time)

            coxnet, scaler = fit_coxnet(Z_train, y_train_struct, l1_ratio, estimated_alphas)

            for alpha in coxnet.alphas_:
                score = evaluate_survival(
                    coxnet, alpha,
                    Z_train, y_train_struct,
                    Z_val,   y_val_struct,
                    validation_function,
                )
                alpha_scores[alpha].append(score)

            print(f"    params={params} | inner fold {inner_fold} done")

        for alpha in estimated_alphas:
            mean_score = np.mean(alpha_scores[alpha])
            if mean_score < best_score_inner:
                best_score_inner = mean_score
                best_params_ft = params
                best_alpha_ft = alpha

    print(f"  Best params ft : {best_params_ft}  |  best alpha : {best_alpha_ft:.6f}")

    final_model = build_model(omics_train_outer, best_params_ft)
    final_model = load_pretrained(final_model, PRETRAIN_CKPT, strict=False)
    final_model = freeze_and_reset_optimizer(final_model, lr=best_params_ft["lr"])
    #only train with phase 2 
    final_model.switch_epoch = 0

    final_model.fit(
        omics_train=omics_train_outer, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs_ft,
        verbose=False, task=task,
    )

    Z_train_outer = final_model.get_latent_representation(omics_train_outer)
    Z_test_outer = final_model.get_latent_representation(omics_test_outer)

    coxnet, scaler = fit_coxnet(Z_train_outer, y_train_outer, l1_ratio, [best_alpha_ft])

    
    risk_scores = coxnet.predict(Z_test_outer, alpha=best_alpha_ft)
    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    survs = coxnet.predict_survival_function(Z_test_outer, alpha=best_alpha_ft)
    times = np.sort(np.unique(y_test_outer["time"]))
    upper = min(y_train_outer["time"].max(), y_test_outer["time"].max())
    times = times[times < upper]
    preds = np.vstack([fn(times) for fn in survs])
    ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

    final_model.save_figure_loss(outer_fold)

    # torch.save(
    #     final_model.state_dict(),
    #     f"results/finetuned_{TARGET_CANCER}_fold{outer_fold}.pt",
    # )

    outer_results.append({
        "fold": outer_fold,
        "cindex": c_index,
        "ibs": ibs_score,
        "best_lr": best_params_ft["lr"],
        "best_alpha": best_alpha_ft,
    })


results_df = pd.DataFrame(outer_results)
print("\n=== Résultats nested CV fine-tuning ===")
print(results_df.to_string(index=False))
print(f"C-index moyen : {results_df['cindex'].mean():.4f} ± {results_df['cindex'].std():.4f}")
print(f"IBS moyen     : {results_df['ibs'].mean():.4f} ± {results_df['ibs'].std():.4f}")

results_df.to_csv(f"{OUTPUT_DIR}/ncv_finetune_{TARGET_CANCER}_results.csv", index=False)