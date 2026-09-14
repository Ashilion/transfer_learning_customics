import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import time
import optuna

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
import utils.folds_utils as fold_utils
from utils.vvh_cv import vvh_cv

from custcox_utils import (
    fit_feature_selector, apply_feature_selector, build_customics_model,
    build_survival_array, fit_coxnet, evaluate_survival, fit_scalers, apply_scalers
)
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os

# ======= Argument Parsing ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested CV with CustOMICS + CoxNet + Optuna for survival prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default="COAD",
        help="Cancer type name (e.g. COAD, BRCA, LUAD)."
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
        help="Number of inner CV folds (StratifiedKFold)."
    )
    parser.add_argument(
        "--add_clinical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate clinical features to the latent representation before CoxNet."
    )
    parser.add_argument(
        "--use_saved_folds",
        action="store_true",
        default=False,
        help="Use pre-saved outer fold splits from splits.json instead of generating new ones."
    )
    parser.add_argument(
        "--n_trials_per_worker",
        type=int,
        default=30,
        help="Number of Optuna trials per outer fold (per worker when --multiproc > 1)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optuna timeout in seconds per outer fold."
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        default=False,
        help="Train CustOMICS in supervised mode (survival loss included during training)."
    )
    parser.add_argument(
        "--limit_epochs",
        type=int,
        default=None,
        help="Limit the number of epochs for the training (and doesnt do early stopping)"
    )
    parser.add_argument(
        "--outer_fold",
        type=int,
        required=True,
        help="Index of the outer fold to process."
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="string to add to the name of the results files (an journal files)"
    )
    parser.add_argument(
        "--ridge",
        action="store_true",
        default=False,
        help="Use ridge (L2) CoxPHSurvivalAnalysis instead of Elastic Net CoxnetSurvivalAnalysis"
    )
    parser.add_argument(
        "--nb_features",
        type=int,
        default=5000,
        help="Number of maximum features per omics."
    )
    parser.add_argument(
        "--linear_decoder",
        action="store_true",
        default=False,
        help="Remove activation layers from the central decoder to push the model to have a 'linear' latent representation"
    )
    parser.add_argument(
        "--multiproc",
        type=int,
        default=1,
        help="Number of parallel worker processes used by Optuna during the search phase "
             "(kept here only for CLI compatibility with the training script; unused in this script)."
    )
    return parser.parse_args()


# ===================================================================

def load_cancer_data(cancer_name, per_cancer_dir="../data/union_separated_cancer", prefix="dict_pancancer"):
    path = f"{per_cancer_dir}/{prefix}_{cancer_name}.pickle"
    with open(path, "rb") as f:
        return pickle.load(f)

def build_customics_model_plus(omics_data, sources, params, device,
                         hidden_dim, central_hidden,
                         classifier_dim, survival_dim,
                         dropout, num_classes, unsupervised, switch_epoch,
                         linear_decoder=False, lambda_surv=5):

    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {
            'input_dim': x_dim[i],
            'hidden_dim': params["autoencoder_hidden_dims"][i],
            'latent_dim': params['rep_dim'],
            'norm': True,
            'dropout': params["dropout"],
            # 'binary': src == 'mutation' # il faudrait ne pas StandardScale les mutations pour utiliser BCE loss pour les mutations
        }
        for i, src in enumerate(sources)
    }

    central_params = {'hidden_dim': params["central_hidden_dims"], 'latent_dim': params['latent_dim'], 'norm': True, 'dropout':params["dropout"], 'beta':params["beta"], 'lambda_central': params.get("lambda_central",1) }

    classif_params = {'n_class': num_classes,'lambda': 0,'hidden_layers': classifier_dim,'dropout':params["dropout"]}

    surv_params = {'lambda': lambda_surv, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True,'dropout':params["dropout"]}

    train_params = {
        'switch': switch_epoch,
        'lr': params['lr'], 
        "modality_dropout_p" : 0.2
        }

    model = CustOMICS(source_params=source_params,central_params=central_params, classif_params=classif_params, 
            surv_params=surv_params,train_params=train_params, device=device,unsupervised=unsupervised, linear_central_decoder=linear_decoder).to(device)

    return model

# ====================================================================================
def main():
    args = parse_args()

    cancer_name = args.cancer
    n_outer = args.outer_splits
    n_inner = args.inner_splits
    add_clinical_to_cox = args.add_clinical
    use_saved_folds = args.use_saved_folds
    n_trials_per_worker = args.n_trials_per_worker
    timeout = args.timeout
    unsupervised = not args.supervised
    limit_epochs = args.limit_epochs
    outer_fold = args.outer_fold
    name_suffix = args.name_suffix
    ridge = args.ridge
    nb_features = args.nb_features
    multiproc = max(1, args.multiproc)

    print(f"\n{'='*60}")
    print(f"  Cancer       : {cancer_name}")
    print(f"  Outer folds  : {n_outer}")
    print(f"  Inner folds  : {n_inner}")
    print(f"  Add clinical : {add_clinical_to_cox}")
    print(f"  Supervised   : {args.supervised}")
    print(f"  Saved folds  : {use_saved_folds}")
    print(f"  Optuna trials: {n_trials_per_worker}  |  timeout: {timeout}s")
    print(f"  Limit Epochs : {limit_epochs}")
    print(f"  Ridge        : {ridge}")
    print(f"  Nb features  : {nb_features}")
    print(f"  Linear Decoder: {args.linear_decoder}")
    print(f"  Multiproc    : {multiproc} worker(s)")

    print(f"{'='*60}\n")

    data = load_cancer_data(cancer_name)
    clinical_df = data["clinical"]

    path = f"../data/clinical/{cancer_name}_clinical.pickle"
    with open(path, "rb") as f:
        df = pickle.load(f)
    clinical_test = df[list(set(df.columns) - {"time", "bcr_patient_barcode", "status"})]

    omics_df = {
        '_rna': data["_rna"],
        'mirna': data["mirna"],
        'cnv': data["cnv"],
        'mutation': data["mutation"],
    }
    omics_df = {k: v.astype(np.float32) for k, v in omics_df.items()}
    print(list(omics_df.keys()))

    lt_samples = list(clinical_df.index)
    offset = clinical_df.index[0] - clinical_test.index[0]

    n_epochs =  limit_epochs if limit_epochs else 1000
    cfg = dict(
        omics_df           = omics_df,
        clinical_df        = clinical_df,
        clinical_test      = clinical_test,
        offset             = offset,
        sources            = list(omics_df.keys()),
        device             = torch.device("cpu"),
        unsupervised       = unsupervised,
        hidden_dim         = [512, 256],
        central_hidden     = [512, 256],
        num_classes        = 5,
        classifier_dim     = [128, 64],
        survival_dim       = [64, 32],
        dropout            = 0.2,
        batch_size         = 32,
        n_epochs           = n_epochs,
        switch_epoch       = n_epochs//2,
        label              = 'status',
        event              = 'status',
        surv_time          = 'time',
        task               = 'survival',
        nbFeatures         = nb_features,
        validation_function= "vvh",
        l1_ratio           = 0.01,
        ridge              = ridge,
        add_clinical_to_cox= add_clinical_to_cox,
        inner_cv           = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=0),
        limit_epochs       = limit_epochs,
        linear_decoder     = args.linear_decoder
    )

    
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=0)

    if use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(cancer_name, src="../data/splits.json")
        outer_fold_iter = enumerate(zip(train_folds, test_folds))
    else:
        outer_fold_iter = enumerate(outer_cv.split(lt_samples))

   
    outer_results = []

    if use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(cancer_name, src="../data/splits.json")
        train_idx, test_idx = train_folds[outer_fold], test_folds[outer_fold]
    else:
        outer_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
        splits = list(outer_cv.split(lt_samples))
        train_idx, test_idx = splits[outer_fold]
    print(f"\n{'='*50}")
    print(f" OUTER FOLD {outer_fold}")
    print(f"{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

    selector = fit_feature_selector(omics_train_outer_raw, nbFeatures=cfg["nbFeatures"])

    omics_train_outer = apply_feature_selector(omics_train_outer_raw, selector)
    omics_test_outer = apply_feature_selector(omics_test_outer_raw,  selector)

    # SCALING
    scalers = fit_scalers(omics_train_outer)

    # 4. Transformer train et test avec les scalers du train
    omics_train_outer = apply_scalers(
        omics_train_outer,
        scalers
    )

    omics_test_outer = apply_scalers(
        omics_test_outer,
        scalers
    )

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, cfg["event"], cfg["surv_time"])
    y_test_outer = build_survival_array(clinical_df, samples_test_outer,  cfg["event"], cfg["surv_time"])

    cfg["samples_train_outer"] = samples_train_outer
    cfg["omics_train_outer"] = omics_train_outer
    cfg["y_train_outer"] = y_train_outer
    cfg["omics_test_outer"] = omics_test_outer

    # Reload the study after all workers are done
    study = optuna.load_study(
        study_name=f"journal_storage_multiprocess_{name_suffix}{cancer_name}_fold{outer_fold}",
        storage=JournalStorage(JournalFileBackend(file_path=f"optuna_journal/journal_{name_suffix}{cancer_name}_fold{outer_fold}.log")),
    )

    best_params = study.best_trial.params
    best_alpha = study.best_trial.user_attrs["best_alpha"]
    best_list_alpha = study.best_trial.user_attrs["best_list_alpha"]
    tot_time_train = study.best_trial.user_attrs["tot_time_train"]
    tot_time_validate = study.best_trial.user_attrs["tot_time_validate"]
    print(f"Optuna best params : {best_params}")
    print(f"Optuna best alpha  : {best_alpha:.6f}")

    # lambda_surv is only sampled by Optuna in supervised mode; otherwise its unused (stays fixed at 5)
    lambda_surv = best_params.get("lambda_surv", 5)

    #TODO test
    lambda_surv = 5
     # Reconstruct full params dict from trial params
    autoencoder_hidden_dims = []
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
    for i, src in enumerate(cfg["sources"]):
        key = best_params[f"hidden_dim_{i}"]
        autoencoder_hidden_dims.append(autoencoder_hidden_dims_possibilities[key])

    full_best_params = {
        **best_params,
        "autoencoder_hidden_dims": autoencoder_hidden_dims,
        "latent_dim":              32,
        "rep_dim":                 32,
        "central_hidden_dims":     [64],
        "patience":                10,
        "lambda_central": 1,
        
    }

    #TODO test (à supprimer)
    full_best_params["beta"] = 0.001

    model = build_customics_model_plus(
        omics_train_outer, cfg["sources"], full_best_params, cfg["device"],
        cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
        cfg["survival_dim"], cfg["dropout"], cfg["num_classes"], cfg["unsupervised"],
        cfg["switch_epoch"], cfg["linear_decoder"], lambda_surv
    )
    if cfg["limit_epochs"]:
        patience= None
    else:
        patience = full_best_params["patience"]
    model.fit(
        omics_train=omics_train_outer, clinical_df=clinical_df,
        label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
        omics_val=None, batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
        verbose=True, task=cfg["task"], patience=patience, min_delta=full_best_params["delta_min"],
        early_stopping_on="train", track_loss_components=True, 
    )
    model.plot_loss_detailed(save_path="results/loss_sans_tl.png")
    model.plot_loss_detailed_stacked(save_path="results/loss_sans_tl_stacked.png")


    Z_train_outer = model.get_latent_representation(omics_train_outer)
    Z_test_outer = model.get_latent_representation(omics_test_outer)

    if add_clinical_to_cox:
        train_adapted = [idx - offset for idx in samples_train_outer]
        clin_train = clinical_test.loc[train_adapted, :]
        X_train_outer = np.concatenate([clin_train, Z_train_outer], axis=1)

        test_adapted = [idx - offset for idx in samples_test_outer]
        clin_test = clinical_test.loc[test_adapted, :]
        X_test_outer = np.concatenate([clin_test, Z_test_outer], axis=1)
    else:
        X_train_outer = Z_train_outer
        X_test_outer = Z_test_outer

    coxnet, scaler = fit_coxnet(X_train_outer, y_train_outer, cfg["l1_ratio"], best_list_alpha, ridge=cfg["ridge"])
    X_test_scaled = scaler.transform(X_test_outer)

    if cfg["ridge"]:
        model = coxnet[best_alpha]
        risk_scores = model.predict(X_test_scaled)
        survs = model.predict_survival_function(X_test_scaled)
    else:
        model = coxnet
        risk_scores = model.predict(X_test_scaled, alpha=best_alpha)
        survs = model.predict_survival_function(X_test_scaled, alpha=best_alpha)

    # C-index
    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    # IBS
    times = np.sort(np.unique(y_test_outer["time"]))
    upper = min(np.max(y_train_outer["time"]), np.max(y_test_outer["time"]))
    times = times[times < upper]
    preds = np.vstack([fn(times) for fn in survs])
    ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

    if cfg["ridge"]:
        coefs = model.coef_
    else:
        alpha_idx = np.argmin(np.abs(coxnet.alphas_ - best_alpha))
        coefs = coxnet.coef_[:, alpha_idx]

    nonzero_mask = coefs != 0
    n_nonzero_coefs = int(nonzero_mask.sum())
    n_zero_coefs = int(len(coefs) - n_nonzero_coefs)

    print(f"  Coefs non-nuls : {n_nonzero_coefs} / {len(coefs)}  (nuls : {n_zero_coefs})")

    outer_results.append({
        "fold":             outer_fold,
        "cindex_default":   c_index,
        "graf":             ibs_score,
        "time_train":       tot_time_train,
        "time_eval":        tot_time_validate,
        "best_alpha":       best_alpha,
        "n_nonzero_coefs":  n_nonzero_coefs,
        "n_zero_coefs":     n_zero_coefs,
        **{f"hp_{k}": v for k, v in best_params.items()},
    })

    #TODO temporary
    name_suffix = "scaledmod_"
    out_path = f"../results/folds/ncv_custcox_optuna_{name_suffix}{cancer_name}_fold{outer_fold}.csv"
    pd.DataFrame(outer_results).to_csv(out_path, index=False)
    print(f"\nResult saved to {out_path}")

if __name__ == "__main__":
    main()