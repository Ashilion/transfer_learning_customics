import argparse
import pandas as pd
import numpy as np
import pickle
import copy
import torch
import time

from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys 
sys.path.append('..')
import utils.folds_utils as fold_utils
from utils.vvh_cv import vvh_cv

from custcox_utils import fit_feature_selector, apply_feature_selector, build_customics_model, build_survival_array, fit_coxnet, evaluate_survival


# ===== Arguments ================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested cross-validation with CustOMICS + CoxNet for survival prediction.",
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
        help="Number of inner CV folds."
    )
    parser.add_argument(
        "--add_clinical",
        action=argparse.BooleanOptionalAction,   # --add_clinical / --no_add_clinical
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
        "--supervised",
        action="store_true",
        default=False,
        help="Train CustOMICS in supervised mode (survival loss included during training)."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Répertoire de sortie pour les résultats CSV."
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="ncv_custcox",
        help="Préfixe du nom du fichier CSV de sortie."
    )
    return parser.parse_args()

# =====================================================================================

def load_pancancer(path="../data/dict_pancancer_union_mutation.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_cancer_data(pancancer, cancer_name):
    return {
        name: df[pancancer["clinical"]["cancer_type"] == cancer_name]
        for name, df in pancancer.items()
    }

# =====================================================================================

def main():
    args = parse_args()

    cancer_name = args.cancer
    n_outer = args.outer_splits
    n_inner = args.inner_splits
    add_clinical_to_cox = args.add_clinical
    use_saved_folds = args.use_saved_folds
    unsupervised = not args.supervised
    
    print(f"\n{'='*60}")
    print(f"  Cancer       : {cancer_name}")
    print(f"  Outer folds  : {n_outer}")
    print(f"  Inner folds  : {n_inner}")
    print(f"  Add clinical : {add_clinical_to_cox}")
    print(f"  Saved folds  : {use_saved_folds}")
    print(f"  Supervised   : {args.supervised}")
    print(f"{'='*60}\n")

    # ====== Load data ==================================================================================
    pancancer = load_pancancer()
    data = get_cancer_data(pancancer, cancer_name)
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
    offset = clinical_df.index[0] - clinical_test.index[0]

    # =========== Model / training config =================================================================
    device = torch.device("cpu")
    batch_size = 32
    n_epochs = 10
    switch_epoch = n_epochs // 2 
    label = 'status'
    event = 'status'
    surv_time = 'time'
    task = 'survival'

    sources = omics_df.keys()
    hidden_dim = [1024, 512, 256]
    central_hidden = [512, 256]
    num_classes = 5
    classifier_dim = [128, 64]
    survival_dim = [64, 32]
    dropout = 0.2

    param_grid = {
        "latent_dim": [64, 128],
        "rep_dim":    [64, 128],
        "lr":         [1e-3, 1e-4],
    }

    nbFeatures = 5000
    validation_function = "vvh"
    l1_ratio = 0.01

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=0)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=0)

    if use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(cancer_name, src="../data/splits.json")
        outer_fold_iter = enumerate(zip(train_folds, test_folds))
    else:
        outer_fold_iter = enumerate(outer_cv.split(lt_samples, clinical_df.loc[:, event]))

    outer_results = []

    for outer_fold, (train_idx, test_idx) in outer_fold_iter:

        print(f"\n{'='*50}")
        print(f" OUTER FOLD {outer_fold}")
        print(f"{'='*50}")

        samples_train_outer = [lt_samples[i] for i in train_idx]
        samples_test_outer = [lt_samples[i] for i in test_idx]

        best_score = np.inf
        best_params = None

        # Outer fold data
        omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
        omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

        selector = fit_feature_selector(omics_train_outer_raw, nbFeatures=nbFeatures)

        omics_train_outer = apply_feature_selector(omics_train_outer_raw, selector)
        omics_test_outer = apply_feature_selector(omics_test_outer_raw,  selector)

        y_train_outer = build_survival_array(clinical_df, samples_train_outer, event, surv_time)
        y_test_outer = build_survival_array(clinical_df, samples_test_outer,  event, surv_time)

        
        estimated_alphas_per_parameter = {}
        #get the list of alpha for this params on the outer train fold
        for params in ParameterGrid(param_grid):
            survival_dim[0] = params["latent_dim"]
            print("survival_dim  : ",survival_dim)

            model = build_customics_model(omics_train_outer,sources,params, device, hidden_dim, central_hidden,
                        classifier_dim, survival_dim, dropout, num_classes, unsupervised,switch_epoch )

            model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
            omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)
            Z_train_outer = model.get_latent_representation(omics_train_outer)

            if add_clinical_to_cox:
                samples_train_outer_adapted = [idx - offset for idx in samples_train_outer]
                clinical_data_outer = clinical_test.loc[samples_train_outer_adapted, :]
                X_concat = np.concatenate([clinical_data_outer, Z_train_outer], axis=1)
                coxnet, scaler = fit_coxnet(X_concat, y_train_outer, l1_ratio)
            else:
                coxnet, scaler = fit_coxnet(Z_train_outer, y_train_outer, l1_ratio)

            estimated_alphas_per_parameter[tuple(params.values())] = coxnet.alphas_

        # Inner CV
        for params in ParameterGrid(param_grid):

            estimated_alphas = estimated_alphas_per_parameter[tuple(params.values())]
            alpha_scores = {alpha: [] for alpha in estimated_alphas}

            tot_time_train = 0
            tot_time_validate = 0

            for inner_train_idx, inner_val_idx in inner_cv.split(samples_train_outer, y_train_outer["status"]):
                start_train = time.time()

                samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
                samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

                omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
                omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

                sel = fit_feature_selector(omics_train_raw, nbFeatures=nbFeatures)

                omics_train = apply_feature_selector(omics_train_raw, sel)
                omics_val = apply_feature_selector(omics_val_raw,   sel)

                survival_dim[0] = params["latent_dim"]
                print("survival_dim  : ",survival_dim)
                model = build_customics_model(omics_train,sources,params, device, hidden_dim, central_hidden,
                            classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch )

                model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                            omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

                Z_train = model.get_latent_representation(omics_train)
                Z_val = model.get_latent_representation(omics_val)

                y_train_struct = build_survival_array(clinical_df, samples_train_inner, event, surv_time)
                y_val_struct = build_survival_array(clinical_df, samples_val_inner,   event, surv_time)

                if add_clinical_to_cox:
                    samples_train_inner_adapted = [idx - offset for idx in samples_train_inner]
                    clinical_data_inner = clinical_test.loc[samples_train_inner_adapted, :]
                    X_cox = np.concatenate([clinical_data_inner, Z_train], axis=1)

                    samples_val_inner_adapted = [idx - offset for idx in samples_val_inner]
                    clinical_data_val = clinical_test.loc[samples_val_inner_adapted, :]
                    X_val = np.concatenate([clinical_data_val, Z_val], axis=1)
                else:
                    X_cox = Z_train
                    X_val = Z_val

                coxnet, scaler = fit_coxnet(X_cox, y_train_struct, l1_ratio, estimated_alphas)
                stop_train = time.time()
                tot_time_train += stop_train - start_train

                start_validate = time.time()
                for alpha in coxnet.alphas_:
                    score = evaluate_survival(
                        coxnet, alpha, X_cox, y_train_struct, X_val, y_val_struct,
                        validation_function
                    )
                    alpha_scores[alpha].append(score)
                stop_validate = time.time()
                tot_time_validate += stop_validate - start_validate

            for alpha in estimated_alphas:
                mean_score = np.mean(alpha_scores[alpha])
                print(f"Params {params} -> alpha {alpha:.6f} -> score {mean_score:.4f}")

                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params
                    best_alpha = alpha

        print("Best params:", best_params)

        # ========== Final outer model ======================================================================
        best_estimated_alphas = estimated_alphas_per_parameter[tuple(best_params.values())]

        survival_dim[0] = best_params["latent_dim"]
        model = build_customics_model(omics_train_outer,sources,best_params, device,hidden_dim, central_hidden,
                classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch)

        model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
                omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)


        Z_train_outer = model.get_latent_representation(omics_train_outer)
        Z_test_outer = model.get_latent_representation(omics_test_outer)

        if add_clinical_to_cox:
            samples_train_outer_adapted = [idx - offset for idx in samples_train_outer]
            clinical_data_outer = clinical_test.loc[samples_train_outer_adapted, :]
            X_train_outer = np.concatenate([clinical_data_outer, Z_train_outer], axis=1)

            samples_test_outer_adapted = [idx - offset for idx in samples_test_outer]
            clinical_data_test = clinical_test.loc[samples_test_outer_adapted, :]
            X_test_outer = np.concatenate([clinical_data_test, Z_test_outer], axis=1)
        else:
            X_train_outer = Z_train_outer
            X_test_outer = Z_test_outer

        coxnet, scaler = fit_coxnet(X_train_outer, y_train_outer, l1_ratio, best_estimated_alphas)

        # C-index
        risk_scores = coxnet.predict(X_test_outer, alpha=best_alpha)
        c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

        # IBS
        survs = coxnet.predict_survival_function(X_test_outer, alpha=best_alpha)
        times = np.sort(np.unique(y_test_outer["time"]))
        upper = min(np.max(y_train_outer["time"]), np.max(y_test_outer["time"]))
        times = times[times < upper]
        preds = np.vstack([fn(times) for fn in survs])
        ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

        print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

        model.save_figure_loss(outer_fold)

        outer_results.append({
            "fold":           outer_fold,
            "cindex_default": c_index,
            "graf":           ibs_score,
            "time_train":     tot_time_train,
            "time_eval":      tot_time_validate,
        })

    # ========== Summary =========================================================================================
    results_df = pd.DataFrame(outer_results)
    print("\n=== Résultats nested CV ===")
    print(results_df.to_string(index=False))
    print(f"C-index moyen : {results_df['cindex_default'].mean():.4f} ± {results_df['cindex_default'].std():.4f}")
    print(f"IBS moyen     : {results_df['graf'].mean():.4f} ± {results_df['graf'].std():.4f}")

    run_name = args.output_name if args.output_name else "ncv_custcox"
    out_path = f"{args.output_dir}/{run_name}_{cancer_name}.csv"
    
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()