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
from multiprocessing import Pool
import multiprocessing as mp
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
# os.system(f"taskset -p 0x{'f'*64} %d" % os.getpid())
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
        help="Number of parallel worker processes for Optuna (each runs n_trials_per_worker trials, "
             "sharing the same JournalStorage file). 1 = sequential, no multiprocessing."
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
        src: {'input_dim': x_dim[i], 'hidden_dim': params["autoencoder_hidden_dims"][i], 'latent_dim': params['rep_dim'], 'norm': True, 'dropout':params["dropout"]} for i, src in enumerate(sources)
    }

    central_params = {'hidden_dim': params["central_hidden_dims"], 'latent_dim': params['latent_dim'], 'norm': True, 'dropout':params["dropout"], 'beta':params["beta"], 'lambda_central': params["lambda_central"] }

    classif_params = {'n_class': num_classes,'lambda': 0,'hidden_layers': classifier_dim,'dropout':params["dropout"]}

    surv_params = {'lambda': lambda_surv, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True,'dropout':params["dropout"]}

    train_params = {'switch': switch_epoch, 'lr': params['lr']}

    model = CustOMICS(source_params=source_params,central_params=central_params, classif_params=classif_params, 
            surv_params=surv_params,train_params=train_params, device=device,unsupervised=unsupervised, linear_central_decoder=linear_decoder).to(device)

    return model
# =========== Optuna objective =====================================================================

def make_objective(cfg):
    
    def objective(trial):
        omics_train_outer = cfg["omics_train_outer"]
        omics_test_outer  = cfg["omics_test_outer"]
        samples_train_outer = cfg["samples_train_outer"]
        y_train_outer = cfg["y_train_outer"]

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
            "latent_dim": 32,
            "rep_dim": 32,
            "central_hidden_dims" : [64],
            "patience": trial.suggest_int("patience", 5, 50, step=5),
            "autoencoder_hidden_dims" : autoencoder_hidden_dims,
            # "lambda_central": trial.suggest_float("lambda_central", 1, 1e3),
            "lambda_central":1,
            "dropout" : trial.suggest_float("dropout", 0, 0.3),
            "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
            "beta": trial.suggest_float("beta", 1e-4, 1e4, log=True),
            "delta_min": trial.suggest_float("delta_min", 1e-8, 1e-3, log=True),
        }

        # ===== Estimate alpha grid on the full outer train fold ===========================
        if cfg["unsupervised"]:
            lambda_surv = 5
        else:
            lambda_surv = trial.suggest_float("lambda_surv", 1, 1e3)

        if cfg["limit_epochs"]:
            effective_patience = None
            n_epochs_eff = cfg["limit_epochs"]
        else:
            effective_patience = params["patience"]
            n_epochs_eff = cfg["n_epochs"]
        
        model = build_customics_model_plus(
            omics_train_outer, cfg["sources"], params, cfg["device"],
            cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
            cfg["survival_dim"], cfg["dropout"], cfg["num_classes"], cfg["unsupervised"],
            cfg["switch_epoch"], cfg["linear_decoder"], lambda_surv
        )

        model.fit(
            omics_train=omics_train_outer, clinical_df=cfg["clinical_df"],
            label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
            omics_val=None, batch_size=cfg["batch_size"], n_epochs=n_epochs_eff,
            verbose=True, task=cfg["task"],patience=effective_patience,min_delta=params["delta_min"],
            early_stopping_on="train"
        )
        with torch.no_grad():
            Z_train_outer = model.get_latent_representation(omics_train_outer)

        if cfg["add_clinical_to_cox"]:
            samples_adapted = [idx - cfg["offset"] for idx in samples_train_outer]
            clin_outer = cfg["clinical_test"].loc[samples_adapted, :]
            X_for_alpha = np.concatenate([clin_outer, Z_train_outer], axis=1)
        else:
            X_for_alpha = Z_train_outer
        
        n_alphas = 50
        # ici elastic net dans tous les cas pour estimer les alphas
        coxnet_ref, scaler = fit_coxnet(X_for_alpha, y_train_outer, cfg["l1_ratio"], n_alphas=n_alphas)
        estimated_alphas = coxnet_ref.alphas_
        
        # =========================================================================

        alpha_scores = {a: [] for a in estimated_alphas}
        tot_time_train = 0
        tot_time_validate = 0
        best_score = np.inf
        best_alpha = estimated_alphas[0]

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(cfg["inner_cv"].split(
            samples_train_outer, y_train_outer["status"]
        )):
            start_train = time.time()

            samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
            samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(cfg["omics_df"], samples_train_inner)
            omics_val_raw = get_sub_omics_df(cfg["omics_df"], samples_val_inner)

            sel = fit_feature_selector(omics_train_raw, nbFeatures=cfg["nbFeatures"])
            omics_train = apply_feature_selector(omics_train_raw, sel)
            omics_val = apply_feature_selector(omics_val_raw,   sel)

            model = build_customics_model_plus(
                omics_train, cfg["sources"], params, cfg["device"],
                cfg["hidden_dim"], cfg["central_hidden"], cfg["classifier_dim"],
                cfg["survival_dim"], cfg["dropout"], cfg["num_classes"], cfg["unsupervised"],
                cfg["switch_epoch"],cfg["linear_decoder"],lambda_surv
            )
            if cfg["limit_epochs"]:
                patience= None
            else:
                patience = params["patience"]
            model.fit(
                omics_train=omics_train, clinical_df=cfg["clinical_df"],
                label=cfg["label"], event=cfg["event"], surv_time=cfg["surv_time"],
                omics_val=omics_val, batch_size=cfg["batch_size"], n_epochs=n_epochs_eff,
                verbose=False, task=cfg["task"], patience=effective_patience,min_delta=params["delta_min"],
                early_stopping_on="train"
            )
            with torch.no_grad():
                Z_train = model.get_latent_representation(omics_train)
                Z_val = model.get_latent_representation(omics_val)

            y_train_struct = build_survival_array(cfg["clinical_df"], samples_train_inner, cfg["event"], cfg["surv_time"])
            y_val_struct = build_survival_array(cfg["clinical_df"], samples_val_inner,   cfg["event"], cfg["surv_time"])

            if cfg["add_clinical_to_cox"]:
                train_adapted = [idx - cfg["offset"] for idx in samples_train_inner]
                clin_train = cfg["clinical_test"].loc[train_adapted, :]
                X_cox = np.concatenate([clin_train, Z_train], axis=1)

                val_adapted = [idx - cfg["offset"] for idx in samples_val_inner]
                clin_val = cfg["clinical_test"].loc[val_adapted, :]
                X_val = np.concatenate([clin_val, Z_val], axis=1)
            else:
                X_cox = Z_train
                X_val = Z_val

            coxnet, scaler = fit_coxnet(X_cox, y_train_struct, cfg["l1_ratio"], estimated_alphas,  ridge=cfg["ridge"])
            X_val_scaled  = scaler.transform(X_val)
            tot_time_train += time.time() - start_train

            start_validate = time.time()
            for alpha in estimated_alphas:
                score = evaluate_survival(
                    coxnet, alpha, X_cox, y_train_struct, X_val_scaled, y_val_struct,
                    cfg["validation_function"], ridge=cfg["ridge"]
                )
                alpha_scores[alpha].append(score)
            tot_time_validate += time.time() - start_validate
            del omics_train_raw, omics_val_raw, omics_train, omics_val, model, Z_train, Z_val
            import gc; gc.collect()

            #pruning 
            if cfg["trial_pruning"]:
                intermediate_best_score = np.inf
                for alpha in estimated_alphas:
                    mean_score = np.mean(alpha_scores[alpha])
                    if mean_score < intermediate_best_score:
                        intermediate_best_score = mean_score
                trial.report(intermediate_best_score, inner_fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        for alpha in estimated_alphas:
            mean_score = np.mean(alpha_scores[alpha])
            print(f"Params {params} -> alpha {alpha:.6f} -> score {mean_score:.4f}")
            if mean_score < best_score:
                best_score = mean_score
                best_alpha = alpha
                best_list_alpha = estimated_alphas

        trial.set_user_attr("best_alpha",         float(best_alpha))
        trial.set_user_attr("tot_time_train",     tot_time_train)
        trial.set_user_attr("tot_time_validate",  tot_time_validate)
        trial.set_user_attr("best_list_alpha", list(best_list_alpha))
        return best_score

    return objective



# ===========================================================================================

_CFG = None
_NAME_SUFFIX = None
_CANCER_NAME = None
_OUTER_FOLD = None
_N_TRIALS_PER_WORKER = None


def run_optimization(worker_id):
    # torch.set_num_threads(2)
    # torch.set_num_interop_threads(1) 
    print(f"[worker {worker_id}] starting in process {os.getpid()}")
    # os.system(f"taskset -p 0x{'f'*64} %d" % os.getpid())
    study = optuna.create_study(
        study_name=f"journal_storage_multiprocess_{_NAME_SUFFIX}{_CANCER_NAME}_fold{_OUTER_FOLD}",
        storage=JournalStorage(JournalFileBackend(
            file_path=f"optuna_journal/journal_{_NAME_SUFFIX}{_CANCER_NAME}_fold{_OUTER_FOLD}.log"
        )),
        load_if_exists=True,  
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2, interval_steps=1)
    )
    study.optimize(
        make_objective(_CFG),
        n_trials=_N_TRIALS_PER_WORKER,
        catch=(Exception,)
    )
    print(f"[worker {worker_id}] done")


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

    n_epochs =  limit_epochs if limit_epochs else 400
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
        linear_decoder     = args.linear_decoder,
        trial_pruning      = True,
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

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, cfg["event"], cfg["surv_time"])
    y_test_outer = build_survival_array(clinical_df, samples_test_outer,  cfg["event"], cfg["surv_time"])

    cfg["samples_train_outer"] = samples_train_outer
    cfg["omics_train_outer"] = omics_train_outer
    cfg["y_train_outer"] = y_train_outer
    cfg["omics_test_outer"] = omics_test_outer

    os.makedirs("optuna_journal", exist_ok=True)

    debut = time.time()

    if multiproc > 1:
        # ---- Fill module-level globals before forking the Pool, so worker processes inherit them via copy-on-write instead of pickling
        global _CFG, _NAME_SUFFIX, _CANCER_NAME, _OUTER_FOLD, _N_TRIALS_PER_WORKER
        _CFG = cfg
        _NAME_SUFFIX = name_suffix
        _CANCER_NAME = cancer_name
        _OUTER_FOLD = outer_fold
        _N_TRIALS_PER_WORKER = n_trials_per_worker

        with mp.Pool(processes=multiproc) as pool:
            pool.map(run_optimization, range(multiproc))
    else:
        study = optuna.create_study(
            study_name=f"journal_storage_multiprocess_{name_suffix}{cancer_name}_fold{outer_fold}",
            storage=JournalStorage(JournalFileBackend(
                file_path=f"optuna_journal/journal_{name_suffix}{cancer_name}_fold{outer_fold}.log"
            )),
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2, interval_steps=1)
        )
        study.optimize(make_objective(cfg), n_trials=n_trials_per_worker, catch=(Exception,))

    time_study = time.time() - debut
    print(f"time study : {time_study}")

if __name__ == "__main__":
    main()