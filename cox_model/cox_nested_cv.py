import pandas as pd
import numpy as np
import pickle
import copy
import pyreadr
import glob
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, ParameterGrid, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.datasets import load_breast_cancer
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import warnings
import time
import json
import sys 

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
sys.path.append(wkd_path)
from utils.vvh_cv import vvh_cv
import utils.folds_utils as fold_utils


validation_function = "vvh" # "vvh" or "ibs"


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

cancer_name = "COAD"

train_folds, test_folds = fold_utils.get_folds(cancer_name)

path = f"data/{cancer_name}_clinical.pickle"

with open(path, "rb") as f:
    df = pickle.load(f)

Xt = df[list(set(df.columns)-set(["time", "bcr_patient_barcode","status"]))]
y = np.array([(bool(i),j ) for i, j in zip(df["status"], df["time"])],dtype=[('status', 'bool_'), ('time', '<f4')])
print(list(df["status"]))

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

outer_results = []

l1_ratio = 0.01

# for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(Xt,y["status"])):
for outer_fold, (train_idx, test_idx) in enumerate(zip(train_folds, test_folds)):
    
    print(f" OUTER FOLD {outer_fold}")
    X_train_outer = Xt.iloc[train_idx,:]
    X_test_outer = Xt.iloc[test_idx,:]

    y_train_outer = y[train_idx] 
    # get list of alphas in the outer fold to have the same alphas for the inner folds
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(n_alphas=100,l1_ratio=l1_ratio, alpha_min_ratio=0.001, max_iter=100, fit_baseline_model=True))
    coxnet_pipe.fit(X_train_outer, y[train_idx])
    print("quantité censure", sum([e== 0 for e in y[train_idx]["status"]])/len(y[train_idx]))
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

    alpha_scores = {alpha: [] for alpha in estimated_alphas}

    tot_time_train = 0
    tot_time_validate = 0 
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y[train_idx]["status"]):

        X_train_inner = X_train_outer.iloc[inner_train_idx,:]
        X_val_inner = X_train_outer.iloc[inner_val_idx,:]
        
        y_train_inner = y_train_outer[inner_train_idx]
        y_val_inner = y_train_outer[inner_val_idx]     
        start_train = time.time()

        coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,alphas=estimated_alphas, fit_baseline_model=True))
        est = coxnet_pipe.fit(X_train_inner, y_train_inner)
        model = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]
        scaler = coxnet_pipe.named_steps["standardscaler"]
        X_val_scaled  = scaler.transform(X_val_inner)
        stop_train = time.time()
        tot_time_train += stop_train - start_train 
        
        # evaluate with integrated Brier Score , TODO with Van Houwelingen
        start_validate = time.time()
        for i, alpha in enumerate(model.alphas_):

            ### dans la doc (user guide) ils utilisent mais très lent : preds = np.asarray([[fn(t) for t in times] for fn in survs]) 

            if validation_function == "ibs":
                # extract survival functions for alpha i
                survs = model.predict_survival_function(X_val_scaled, alpha = alpha)
                times = np.sort(np.unique(y_val_inner["time"]))
                upper = min(
                    np.max( y_train_inner["time"]),
                    np.max(y_val_inner["time"])
                )

                times = times[times < upper]
                preds = np.vstack([fn(times) for fn in survs])
                score = integrated_brier_score(y_val_inner,y_val_inner,preds,times)
            elif validation_function == "vvh":
                coefs = model.coef_[:, i]
                nonzero_mask = coefs != 0
                # print(f"non zero mask {nonzero_mask.sum()}, {nonzero_mask.sum()} / {len(coefs)}")
                if(nonzero_mask.sum()<3):
                    score = np.inf
                else:
                    score = vvh_cv(model, alpha, X_train_inner,y_train_inner, X_val_scaled, y_val_inner)
            
            alpha_scores[alpha].append(score)
        
        stop_validate = time.time()
        tot_time_validate += stop_validate - start_validate 
    mean_alpha_scores = {
        alpha: np.mean(scores)
        for alpha, scores in alpha_scores.items()
    }
    best_score = np.inf
    best_alpha = None
    for alpha, score in mean_alpha_scores.items():
        # print(f"score: {score} alpha:{alpha} best_score:{best_score}")
        
        if score < best_score:
            best_alpha = alpha
            best_score = score

    #retrain the model on the train + validation data with the best alpha + evaluate on the test data 
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,alphas=[best_alpha], fit_baseline_model=True))
    est = coxnet_pipe.fit(X_train_outer, y[train_idx])
    scaler = coxnet_pipe.named_steps["standardscaler"]
    X_test_scaled = scaler.transform(X_test_outer)
    model = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]

    # === DEBUG : modèle réentraîné sur outer fold ===
    print(f"\n--- DEBUG Outer Fold {outer_fold} ---")
    print(f"  list alpha : {estimated_alphas}")
    print(f"  best_alpha       : {best_alpha:.6f}")
    print(f"  best_score (CV)  : {mean_alpha_scores[best_alpha]:.6f}")
    
    # Coefficients non nuls
    # model.coef_ shape: (n_features, n_alphas) — on sélectionne la colonne de best_alpha
    alpha_idx = list(model.alphas_).index(best_alpha)
    coefs = model.coef_[:, alpha_idx]
    nonzero_mask = coefs != 0
    nonzero_features = X_train_outer.columns[nonzero_mask]
    nonzero_coefs = coefs[nonzero_mask]
    print(f"  Nb features sélectionnées : {nonzero_mask.sum()} / {len(coefs)}")
    print(f"  Coefficients non nuls :")
    for feat, coef in sorted(zip(nonzero_features, nonzero_coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:40s} : {coef:.6f}")
    print(f"--- FIN DEBUG ---\n")

    survs = model.predict_survival_function(X_test_scaled, alpha = best_alpha)
    times = np.sort(np.unique(y[test_idx]["time"]))
    upper = min(
        np.max( y[train_idx]["time"]),
        np.max(y[test_idx]["time"])
    )

    times = times[times < upper]
    print(min(y[train_idx]["time"]),max(y[train_idx]["time"]))

    ### dans la doc ils utilisent mais très lent : preds = np.asarray([[fn(t) for t in times] for fn in survs]) 
    preds = np.vstack([fn(times) for fn in survs])

    ibs_score = integrated_brier_score(y[test_idx],y[test_idx],preds,times)
    

    preds_for_cindex = model.predict(X_test_scaled, alpha=best_alpha)
    print("preds_for_cindex",preds_for_cindex)
    c_index = concordance_index_ipcw(y[test_idx],y[test_idx], preds_for_cindex)[0]
    
    
    step_values = {
        "fold": outer_fold,
        "cindex_default": c_index,
        "graf": ibs_score, 
        "time_train": tot_time_train,
        "time_eval": tot_time_validate
    }
    print(step_values)
    outer_results.append(step_values)


results_df = pd.DataFrame(outer_results)
results_df.to_csv("results/outer_cv_results_vvh_COAD.csv", index=False)