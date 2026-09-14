import argparse
import pandas as pd
import numpy as np
import pickle
import json
import torch
import time

from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
from utils.vvh_cv import vvh_cv
import utils.folds_utils as fold_utils

from custcox_utils import fit_feature_selector, apply_feature_selector, build_customics_model, build_survival_array, fit_coxnet, evaluate_survival

# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested cross-validation with pre-trained CustOMICS fine-tuning + CoxNet for survival prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cancer",
        type=str,
        default="COAD",
        help="Target cancer type (e.g. COAD, BRCA, LUAD)."
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default="pretrained_pan_cancer.pt",
        help="Path to the pre-trained model checkpoint."
    )
    parser.add_argument(
        "--best_params_in",
        type=str,
        default="results/best_params_source.json",
        help="Path to the JSON file with best hyperparameters from pre-training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output results."
    )
    parser.add_argument(
        "--outer_splits",
        type=int,
        default=5,
        help="Number of outer CV folds."
    )
    parser.add_argument(
        "--inner_splits",
        type=int,
        default=3,
        help="Number of inner CV folds."
    )
    parser.add_argument(
        "--n_epochs_ft",
        type=int,
        default=10,
        help="Number of fine-tuning epochs."
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.01,
        help="L1 ratio for CoxNet regularization."
    )
    parser.add_argument(
        "--use_saved_folds",
        action="store_true",
        default=False,
        help="Use pre-saved outer fold splits from splits.json instead of generating new ones."
    )

    return parser.parse_args()

# =====================================================================================
def load_pretrained(model, ckpt_path, device, strict=False):
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return model

def freeze_and_reset_optimizer(model, lr):
    model.freeze_autoencoders()
    model.update_optimizer(lr)
    return model

def unfreeze_and_reset_optimizer(model, lr):
    model.unfreeze_autoencoders()
    model.update_optimizer(lr)
    return model

def fit_transfer(model, PRETRAIN_CKPT, device, omics_train, clinical_df, label, event, surv_time, batch_size, n_epochs_central, n_epochs_all, task, learning_rate):
    model = load_pretrained(model, PRETRAIN_CKPT, device, strict=False)
    model = freeze_and_reset_optimizer(model, lr=learning_rate)
    #only train with phase 2 
    model.switch_epoch = 0
    model.fit(
        omics_train=omics_train, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs_central,
        verbose=False, task=task,
    )

    model = unfreeze_and_reset_optimizer(model, lr=learning_rate)
    model.fit(
        omics_train=omics_train, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs_all,
        verbose=False, task=task,
    )
    return model
# =====================================================================================

def main():
    args = parse_args()

    TARGET_CANCER  = args.cancer
    PRETRAIN_CKPT  = args.pretrain_ckpt
    BEST_PARAMS_IN = args.best_params_in
    OUTPUT_DIR     = args.output_dir
    n_epochs_ft    = args.n_epochs_ft
    l1_ratio       = args.l1_ratio

    print(f"\n{'='*60}")
    print(f"  Cancer         : {TARGET_CANCER}")
    print(f"  Checkpoint     : {PRETRAIN_CKPT}")
    print(f"  Best params in : {BEST_PARAMS_IN}")
    print(f"  Outer folds    : {args.outer_splits}")
    print(f"  Inner folds    : {args.inner_splits}")
    print(f"  Epochs (FT)    : {n_epochs_ft}")
    print(f"  L1 ratio       : {l1_ratio}")
    print(f"  Saved folds    : {args.use_saved_folds}")
    print(f"{'='*60}\n")

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
    switch_epoch = 0 
    # Grille réduite
    param_grid_ft = {
        "lr": [1e-3, 1e-4, 5e-5],
    }

    outer_cv = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
    inner_cv = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=0)
    outer_results = []

    if args.use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(TARGET_CANCER, src="../data/splits.json")
        outer_fold_iter = enumerate(zip(train_folds, test_folds))
    else:
        outer_fold_iter = enumerate(outer_cv.split(lt_samples, clinical_df.loc[:, event]))

    for outer_fold, (train_idx, test_idx) in outer_fold_iter:
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
        best_params["lr"] = 1e-3
        
        ref_model = build_customics_model(omics_train_outer, sources, best_params, device, hidden_dim, central_hidden,
            classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch )
        ref_model = load_pretrained(ref_model, PRETRAIN_CKPT, device, strict=False)
        # ref_model = freeze_and_reset_optimizer(ref_model, lr=1e-3)
        # #only train with phase 2 
        # ref_model.switch_epoch = 0
        # ref_model.fit(
        #     omics_train=omics_train_outer, clinical_df=clinical_df,
        #     label=label, event=event, surv_time=surv_time,
        #     omics_val=None, batch_size=batch_size, n_epochs=n_epochs_ft,
        #     verbose=False, task=task,
        # )
        ref_model = fit_transfer(ref_model, PRETRAIN_CKPT, device, omics_train_outer, clinical_df, label, event, surv_time,
                     batch_size, n_epochs_central, n_epochs_all, task, learning_rate)
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
                inner_cv.split(samples_train_outer, y_train_outer["status"])
            ):
                samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
                samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

                omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
                omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

                sel_inner = fit_feature_selector(omics_train_raw, nbFeatures)
                omics_train = apply_feature_selector(omics_train_raw, sel_inner)
                omics_val = apply_feature_selector(omics_val_raw,   sel_inner)

                # Chargement poids pré-entraînés + freeze AE
                #TODO faire automatiquement pour tous les params si je rajoute d'autre params que lr
                best_params["lr"] = params["lr"]
                model = build_customics_model(omics_train_outer, sources, best_params, device, hidden_dim, central_hidden,
                    classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch)
                model = load_pretrained(model, PRETRAIN_CKPT, device, strict=False)
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

        best_params["lr"] = best_params_ft["lr"]
        final_model = build_customics_model(omics_train_outer, sources, best_params, device, hidden_dim, central_hidden,
                    classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch)
        final_model = load_pretrained(final_model, PRETRAIN_CKPT, device, strict=False)
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


if __name__ == "__main__":
    main()