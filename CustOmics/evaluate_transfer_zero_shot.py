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
import utils.folds_utils as fold_utils

from custcox_utils import fit_feature_selector, apply_feature_selector, build_customics_model, build_survival_array, fit_coxnet, evaluate_survival
import argparse
import os
# ===== Argument Parsing ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Optuna evaluation for SOURCE pan-cancer pre-training. (no finetuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cancer", 
        type=str,
        default="COAD",
        help="Target cancer excluded from source data."
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="tl_ckpt",
        help="Directory to write the result CSV and model checkpoint."
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default="pretrained_model.pt",
        help="Path where the final model weights will be saved."
    )
    parser.add_argument(
        "--best_params_out", 
        type=str, 
        default="best_params_source.json",
        help="Path where the best config JSON will be saved."
    )
    parser.add_argument(
        "--supervised", 
        action="store_true", 
        default=False,
        help="Train in supervised mode."
    )
    parser.add_argument(
        "--use_saved_folds",
        action="store_true",
        default=False,
        help="Use pre-saved outer fold splits from splits.json instead of generating new ones."
    )

    return parser.parse_args()

# =================================================================================================


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

def load_pretrained(model, ckpt_path,device, strict=False):
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return model

# ===== Main ============================================================================================

def main():
    args = parse_args()

    TARGET_CANCER  = args.cancer
    use_saved_folds = args.use_saved_folds
    ckpt_filename = f"{TARGET_CANCER}_{os.path.basename(args.pretrain_ckpt)}"
    config_filename = f"{TARGET_CANCER}_{os.path.basename(args.best_params_out)}"

    pretrain_ckpt_path = os.path.join(args.output_dir, ckpt_filename)
    best_params_out_path = os.path.join(args.output_dir, config_filename)
    selector_path = f"{pretrain_ckpt_path}.selector.pkl"
    
    OUTPUT_DIR = "results"
    print(f"best params out path {best_params_out_path}, pretrain ckpt : {pretrain_ckpt_path}")
    with open(best_params_out_path, "r") as f:
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

    l1_ratio = 0.01

    nbFeatures = 5000
    validation_function = "vvh"

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)

    outer_results = []


    if use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(TARGET_CANCER, src="../data/splits.json")
        outer_fold_iter = enumerate(zip(train_folds, test_folds))
    else:
        outer_fold_iter = enumerate(outer_cv.split(lt_samples))

    for outer_fold, (train_idx, test_idx) in outer_fold_iter:
        print(f"\n{'='*50}")
        print(f" OUTER FOLD {outer_fold}")
        print(f"{'='*50}")

        samples_train_outer = [lt_samples[i] for i in train_idx]
        samples_test_outer = [lt_samples[i] for i in test_idx]

        omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
        omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

        with open(selector_path, "rb") as f:
            sel_outer = pickle.load(f)
        omics_train_outer = apply_feature_selector(omics_train_outer_raw, sel_outer)
        omics_test_outer = apply_feature_selector(omics_test_outer_raw,  sel_outer)

        y_train_outer = build_survival_array(clinical_df, samples_train_outer, event, surv_time)
        y_test_outer = build_survival_array(clinical_df, samples_test_outer,  event, surv_time)

        ref_model = build_customics_model_plus(
                omics_train_outer, sources, best_params, device,
                hidden_dim, central_hidden, classifier_dim,
                survival_dim, dropout, num_classes,
                unsupervised, switch_epoch=0,
            )
        ref_model = load_pretrained(ref_model, pretrain_ckpt_path,device, strict=False)
        ref_model.eval_all()
        
        print("  -> Calcul de la grille d'alphas de référence (outer train)...")

        ref_model.phase = 2
        Z_train_outer = ref_model.get_latent_representation(omics_train_outer)
        Z_test_outer = ref_model.get_latent_representation(omics_test_outer)
        coxnet, scaler = fit_coxnet(Z_train_outer, y_train_outer, l1_ratio)

        risk_scores = coxnet.predict(Z_test_outer)
        c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

        survs = coxnet.predict_survival_function(Z_test_outer)
        times = np.sort(np.unique(y_test_outer["time"]))
        upper = min(y_train_outer["time"].max(), y_test_outer["time"].max())
        times = times[times < upper]
        preds = np.vstack([fn(times) for fn in survs])
        ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

        print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

        outer_results.append({
            "fold": outer_fold,
            "cindex_default": c_index,
            "graf": ibs_score,
            "best_lr": best_params["lr"],
            "best_alpha": best_alpha_src,
        })


    results_df = pd.DataFrame(outer_results)
    print("\n=== Résultats nested CV fine-tuning ===")
    print(results_df.to_string(index=False))
    print(f"C-index moyen : {results_df['cindex_default'].mean():.4f} ± {results_df['cindex_default'].std():.4f}")
    print(f"IBS moyen     : {results_df['graf'].mean():.4f} ± {results_df['graf'].std():.4f}")

    results_df.to_csv(f"{OUTPUT_DIR}/ncv_zeroshot_{TARGET_CANCER}_results.csv", index=False)


if __name__ == "__main__":
    main()