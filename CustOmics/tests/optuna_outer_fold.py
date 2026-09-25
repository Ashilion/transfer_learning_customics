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
    build_survival_array, fit_coxnet, evaluate_survival
)


# ======= Argument Parsing ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Single outer fold: CustOMICS + CoxNet + Optuna for survival prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cancer",         type=str,  default="COAD")
    parser.add_argument("--outer_fold",     type=int,  required=True,
                        help="Index of the outer fold to run (0-based).")
    parser.add_argument("--outer_splits",   type=int,  default=5)
    parser.add_argument("--inner_splits",   type=int,  default=3)
    parser.add_argument("--add_clinical",   action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_saved_folds",action="store_true", default=False)
    parser.add_argument("--n_trials",       type=int,  default=30)
    parser.add_argument("--timeout",        type=int,  default=None)
    parser.add_argument("--supervised",     action="store_true", default=False)
    parser.add_argument("--n_threads",      type=int,  default=10,
                        help="Optuna n_jobs inside a single outer fold.")
    parser.add_argument("--limit_epochs",   type=int,  default=None)
    return parser.parse_args()


# ===================================================================

def load_pancancer(path="../data/dict_pancancer_union_mutation.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_cancer_data(pancancer, cancer_name):
    return {
        name: df[pancancer["clinical"]["cancer_type"] == cancer_name]
        for name, df in pancancer.items()
    }


def build_customics_model_plus(omics_data, sources, params, device,
                               hidden_dim, central_hidden,
                               classifier_dim, survival_dim,
                               dropout, num_classes, unsupervised, switch_epoch):

    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {
            'input_dim': x_dim[i],
            'hidden_dim': params["autoencoder_hidden_dims"][i],
            'latent_dim': params['rep_dim'],
            'norm': True,
            'dropout': params["dropout"]
        }
        for i, src in enumerate(sources)
    }

    central_params = {
        'hidden_dim': params["central_hidden_dims"],
        'latent_dim': params['latent_dim'],
        'norm': True,
        'dropout': params["dropout"],
        'beta': params["beta"]
    }

    classif_params = {
        'n_class': num_classes,
        'lambda': 0,
        'hidden_layers': classifier_dim,
        'dropout': params["dropout"]
    }

    surv_params = {
        'lambda': 5,
        'dims': survival_dim,
        'activation': 'SELU',
        'l2_reg': 1e-2,
        'norm': True,
        'dropout': params["dropout"]
    }

    train_params = {'switch': switch_epoch, 'lr': params['lr']}

    model = CustOMICS(
        source_params=source_params, central_params=central_params,
        classif_params=classif_params, surv_params=surv_params,
        train_params=train_params, device=device, unsupervised=unsupervised
    ).to(device)

    return model


# =========== Optuna objective =====================================================================

def make_objective(fn_param, cfg):

    def objective(trial):
        omics_train_outer   = fn_param["omics_train_outer"]
        omics_test_outer    = fn_param["omics_test_outer"]
        samples_train_outer = fn_param["samples_train_outer"]
        y_train_outer       = fn_param["y_train_outer"]

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
        for i, src in enumerate(cfg["sources"]):
            if omics_train_outer[src].shape[1] > 1200:
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
            "dropout":     trial.suggest_float("dropout", 0, 0.3),
            "lr":          trial.suggest_float("lr", 5e-4, 1e-3, log=True),
            "beta":        trial.suggest_float("beta", 0, 1),
            "delta_min":   trial.suggest_float("delta_min", 1e-8, 1e-3, log=True),
        }

        # ===== Estimate alpha grid on the full outer train fold ============================
        model = build_customics_model_plus(
            omics_train_outer, cfg["sources"], params, cfg["device"],
            cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
            cfg["survival_dim"], cfg["dropout"], cfg["num_classes"], cfg["unsupervised"],
            cfg["switch_epoch"]
        )
        patience = None if cfg["limit_epochs"] else params["patience"]
        model.fit(
            omics_train=omics_train_outer, clinical_df=cfg["clinical_df"],
            label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
            omics_val=omics_test_outer, batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
            verbose=True, task=cfg["task"], patience=patience, min_delta=params["delta_min"],
            early_stopping_on="train"
        )
        Z_train_outer = model.get_latent_representation(omics_train_outer)

        if cfg["add_clinical_to_cox"]:
            samples_adapted = [idx - cfg["offset"] for idx in samples_train_outer]
            clin_outer = cfg["clinical_test"].loc[samples_adapted, :]
            X_for_alpha = np.concatenate([clin_outer, Z_train_outer], axis=1)
        else:
            X_for_alpha = Z_train_outer

        coxnet_ref, scaler       = fit_coxnet(X_for_alpha, y_train_outer, cfg["l1_ratio"])
        estimated_alphas = coxnet_ref.alphas_
        # ==================================================================================

        alpha_scores      = {a: [] for a in estimated_alphas}
        tot_time_train    = 0
        tot_time_validate = 0
        best_score        = np.inf
        best_alpha        = estimated_alphas[0]
        best_list_alpha   = estimated_alphas

        for inner_train_idx, inner_val_idx in cfg["inner_cv"].split(
            samples_train_outer, y_train_outer["status"]
        ):
            start_train = time.time()

            samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
            samples_val_inner   = [samples_train_outer[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
            omics_val_raw   = get_sub_omics_df(cfg["omics_df"], samples_val_inner)

            sel        = fit_feature_selector(omics_train_raw, nbFeatures=cfg["nbFeatures"])
            omics_train = apply_feature_selector(omics_train_raw, sel)
            omics_val   = apply_feature_selector(omics_val_raw,   sel)

            model = build_customics_model_plus(
                omics_train, cfg["sources"], params, cfg["device"],
                cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
                cfg["survival_dim"], cfg["dropout"], cfg["num_classes"], cfg["unsupervised"],
                cfg["switch_epoch"]
            )
            patience = None if cfg["limit_epochs"] else params["patience"]
            model.fit(
                omics_train=omics_train, clinical_df=cfg["clinical_df"],
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=omics_val, batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
                verbose=False, task=cfg["task"], patience=patience, min_delta=params["delta_min"],
                early_stopping_on="train"
            )

            Z_train = model.get_latent_representation(omics_train)
            Z_val   = model.get_latent_representation(omics_val)

            y_train_struct = build_survival_array(cfg["clinical_df"], samples_train_inner, cfg["event"], cfg["surv_time"])
            y_val_struct   = build_survival_array(cfg["clinical_df"], samples_val_inner,   cfg["event"], cfg["surv_time"])

            if cfg["add_clinical_to_cox"]:
                train_adapted = [idx - cfg["offset"] for idx in samples_train_inner]
                clin_train    = cfg["clinical_test"].loc[train_adapted, :]
                X_cox         = np.concatenate([clin_train, Z_train], axis=1)

                val_adapted = [idx - cfg["offset"] for idx in samples_val_inner]
                clin_val    = cfg["clinical_test"].loc[val_adapted, :]
                X_val       = np.concatenate([clin_val, Z_val], axis=1)
            else:
                X_cox = Z_train
                X_val = Z_val

            coxnet, scaler = fit_coxnet(X_cox, y_train_struct, cfg["l1_ratio"], estimated_alphas)
            tot_time_train += time.time() - start_train

            start_validate = time.time()
            for alpha in coxnet.alphas_:
                score = evaluate_survival(
                    coxnet, alpha, X_cox, y_train_struct, X_val, y_val_struct,
                    cfg["validation_function"]
                )
                alpha_scores[alpha].append(score)
            tot_time_validate += time.time() - start_validate

        for alpha in estimated_alphas:
            mean_score = np.mean(alpha_scores[alpha])
            print(f"Params {params} -> alpha {alpha:.6f} -> score {mean_score:.4f}")
            if mean_score < best_score:
                best_score      = mean_score
                best_alpha      = alpha
                best_list_alpha = estimated_alphas

        trial.set_user_attr("best_alpha",        float(best_alpha))
        trial.set_user_attr("tot_time_train",    tot_time_train)
        trial.set_user_attr("tot_time_validate", tot_time_validate)
        trial.set_user_attr("best_list_alpha",   best_list_alpha)
        return best_score

    return objective


# ====================================================================================

def main():
    args = parse_args()

    cancer_name        = args.cancer
    outer_fold         = args.outer_fold
    n_outer            = args.outer_splits
    n_inner            = args.inner_splits
    add_clinical_to_cox = args.add_clinical
    use_saved_folds    = args.use_saved_folds
    n_trials           = args.n_trials
    timeout            = args.timeout
    unsupervised       = not args.supervised
    n_threads          = args.n_threads
    limit_epochs       = args.limit_epochs

    print(f"\n{'='*60}")
    print(f"  Cancer       : {cancer_name}")
    print(f"  Outer fold   : {outer_fold} / {n_outer}")
    print(f"  Inner folds  : {n_inner}")
    print(f"  Add clinical : {add_clinical_to_cox}")
    print(f"  Supervised   : {args.supervised}")
    print(f"  Saved folds  : {use_saved_folds}")
    print(f"  Optuna trials: {n_trials}  |  timeout: {timeout}s")
    print(f"  n_threads    : {n_threads}")
    print(f"  Limit Epochs : {limit_epochs}")
    print(f"{'='*60}\n")

    pancancer   = load_pancancer()
    data        = get_cancer_data(pancancer, cancer_name)
    clinical_df = data["clinical"]

    path = f"../data/{cancer_name}_clinical.pickle"
    with open(path, "rb") as f:
        df = pickle.load(f)
    clinical_test = df[list(set(df.columns) - {"time", "bcr_patient_barcode", "status"})]

    omics_df = {
        'protein':   data["_rna"],
        'gene_exp':  data["mirna"],
        'methyl':    data["cnv"],
        'mutation':  data["mutation"],
    }

    lt_samples = list(clinical_df.index)
    offset     = clinical_df.index[0] - clinical_test.index[0]

    n_epochs = limit_epochs if limit_epochs else 1000

    cfg = dict(
        omics_df            = omics_df,
        clinical_df         = clinical_df,
        clinical_test       = clinical_test,
        offset              = offset,
        sources             = list(omics_df.keys()), 
        device              = torch.device("cpu"),
        unsupervised        = unsupervised,
        hidden_dim          = [512, 256],
        central_hidden      = [512, 256],
        num_classes         = 5,
        classifier_dim      = [128, 64],
        survival_dim        = [64, 32],
        dropout             = 0.2,
        batch_size          = 32,
        n_epochs            = n_epochs,
        switch_epoch        = n_epochs // 2,
        label               = 'status',
        event               = 'status',
        surv_time           = 'time',
        task                = 'survival',
        nbFeatures          = 5000,
        validation_function = "vvh",
        l1_ratio            = 0.01,
        add_clinical_to_cox = add_clinical_to_cox,
        inner_cv            = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=0),
        limit_epochs        = limit_epochs,
    )

    # ===== Select the single outer fold ===============================================
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=0)

    if use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(cancer_name, src="../data/splits.json")
        train_idx = train_folds[outer_fold]
        test_idx  = test_folds[outer_fold]
    else:
        all_splits = list(outer_cv.split(lt_samples))
        train_idx, test_idx = all_splits[outer_fold]

    print(f"\n{'='*50}")
    print(f" OUTER FOLD {outer_fold}")
    print(f"{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer  = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw  = get_sub_omics_df(omics_df, samples_test_outer)

    selector = fit_feature_selector(omics_train_outer_raw, nbFeatures=cfg["nbFeatures"])

    omics_train_outer = apply_feature_selector(omics_train_outer_raw, selector)
    omics_test_outer  = apply_feature_selector(omics_test_outer_raw,  selector)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, cfg["event"], cfg["surv_time"])
    y_test_outer  = build_survival_array(clinical_df, samples_test_outer,  cfg["event"], cfg["surv_time"])

    fn_params = {
        "samples_train_outer": samples_train_outer,
        "omics_train_outer":   omics_train_outer,
        "y_train_outer":       y_train_outer,
        "omics_test_outer":    omics_test_outer,
    }

    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(fn_params, cfg),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_threads,
    )

    best_trial      = study.best_trial
    best_params     = best_trial.params
    best_alpha      = best_trial.user_attrs["best_alpha"]
    best_list_alpha = best_trial.user_attrs["best_list_alpha"]
    tot_time_train  = best_trial.user_attrs["tot_time_train"]
    tot_time_val    = best_trial.user_attrs["tot_time_validate"]

    print(f"Optuna best params : {best_params}")
    print(f"Optuna best alpha  : {best_alpha:.6f}")

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
        "patience":                5,
    }

    # ===== Retrain on full outer train set ============================================
    model = build_customics_model_plus(
        omics_train_outer, cfg["sources"], full_best_params, cfg["device"],
        cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
        cfg["survival_dim"], cfg["dropout"], cfg["num_classes"], cfg["unsupervised"],
        cfg["switch_epoch"]
    )
    patience = None if cfg["limit_epochs"] else full_best_params["patience"]
    model.fit(
        omics_train=omics_train_outer, clinical_df=clinical_df,
        label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
        omics_val=None, batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
        verbose=False, task=cfg["task"], patience=patience,
        min_delta=full_best_params["delta_min"], early_stopping_on="train"
    )

    Z_train_outer = model.get_latent_representation(omics_train_outer)
    Z_test_outer  = model.get_latent_representation(omics_test_outer)

    if add_clinical_to_cox:
        train_adapted  = [idx - offset for idx in samples_train_outer]
        clin_train     = clinical_test.loc[train_adapted, :]
        X_train_outer  = np.concatenate([clin_train, Z_train_outer], axis=1)

        test_adapted   = [idx - offset for idx in samples_test_outer]
        clin_test_data = clinical_test.loc[test_adapted, :]
        X_test_outer   = np.concatenate([clin_test_data, Z_test_outer], axis=1)
    else:
        X_train_outer = Z_train_outer
        X_test_outer  = Z_test_outer

    coxnet, scaler = fit_coxnet(X_train_outer, y_train_outer, cfg["l1_ratio"], best_list_alpha)

    # C-index
    risk_scores = coxnet.predict(X_test_outer, alpha=best_alpha)
    c_index     = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    # IBS
    survs  = coxnet.predict_survival_function(X_test_outer, alpha=best_alpha)
    times  = np.sort(np.unique(y_test_outer["time"]))
    upper  = min(np.max(y_train_outer["time"]), np.max(y_test_outer["time"]))
    times  = times[times < upper]
    preds  = np.vstack([fn(times) for fn in survs])
    ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

    model.save_figure_loss(outer_fold)

    result = {
        "fold":           outer_fold,
        "cindex_default": c_index,
        "graf":           ibs_score,
        "time_train":     tot_time_train,
        "time_eval":      tot_time_val,
    }

    out_path = f"../results/folds/ncv_custcox_optuna_multiworkers_{cancer_name}_fold{outer_fold}.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()