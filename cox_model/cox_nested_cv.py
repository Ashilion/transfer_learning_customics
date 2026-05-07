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


validation_function = "ibs" # "vvh" or "ibs"


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

cancer_name = "COAD"

train_folds, test_folds = fold_utils.get_folds(cancer_name)

path = f"data/{cancer_name}_clinical.pickle"

with open(path, "rb") as f:
    df = pickle.load(f)

Xt = df[list(set(df.columns)-set(["time", "bcr_patient_barcode","status"]))]
y = np.array([(bool(i),j ) for i, j in zip(df["status"], df["time"])],dtype=[('status', 'bool_'), ('time', '<f4')])


outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

outer_results = []

l1_ratio = 0.0001

# for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(Xt,y["status"])):
for outer_fold, (train_idx, test_idx) in enumerate(zip(train_folds, test_folds)):
    
    print(f" OUTER FOLD {outer_fold}")
    X_train_outer = Xt.iloc[train_idx,:]
    X_test_outer = Xt.iloc[test_idx,:]

    # get list of alphas in the outer fold to have the same alphas for the inner folds
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(n_alphas=100,l1_ratio=l1_ratio, alpha_min_ratio=0.001, max_iter=100, fit_baseline_model=True))
    coxnet_pipe.fit(X_train_outer, y[train_idx])

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

    alpha_scores = {alpha: [] for alpha in estimated_alphas}

    tot_time_train = 0
    tot_time_validate = 0 
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y[train_idx]["status"]):

        X_train_inner = X_train_outer.iloc[inner_train_idx,:]
        X_val_inner = X_train_outer.iloc[inner_val_idx,:]
        
       
        start_train = time.time()

        coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,alphas=estimated_alphas, fit_baseline_model=True))
        est = coxnet_pipe.fit(X_train_inner, y[inner_train_idx])
        model = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]

        stop_train = time.time()
        tot_time_train += stop_train - start_train 
        
        # evaluate with integrated Brier Score , TODO with Van Houwelingen
        start_validate = time.time()
        for i, alpha in enumerate(model.alphas_):
            # extract survival functions for alpha i
            survs = model.predict_survival_function(X_val_inner, alpha = alpha)
            t_min = y[inner_val_idx]["time"].min()
            t_max = min(1492, y[inner_val_idx]["time"].max())
            times = np.arange(t_min, t_max)

            ### dans la doc (user guide) ils utilisent mais très lent : preds = np.asarray([[fn(t) for t in times] for fn in survs]) 

            if validation_function == "ibs":
                preds = np.vstack([fn(times) for fn in survs])
                score = integrated_brier_score(y[inner_train_idx],y[inner_val_idx],preds,times)
            elif validation_function == "vvh":
                score = vvh_cv(model, alpha, X_train_inner,y[inner_train_idx], X_val_inner, y[inner_val_idx])
            
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
        if score < best_score:
            best_alpha = alpha

    #retrain the model on the train + validation data with the best alpha + evaluate on the test data 
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,alphas=[best_alpha], fit_baseline_model=True))
    est = coxnet_pipe.fit(X_train_outer, y[train_idx])
    model = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]
    survs = model.predict_survival_function(X_test_outer, alpha = best_alpha)
    t_min = y[test_idx]["time"].min()
    t_max = min(1492, y[test_idx]["time"].max())
    times = np.arange(t_min, t_max)

    ### dans la doc ils utilisent mais très lent : preds = np.asarray([[fn(t) for t in times] for fn in survs]) 
    preds = np.vstack([fn(times) for fn in survs])

    ibs_score = integrated_brier_score(y[train_idx],y[test_idx],preds,times)
    

    preds_for_cindex = model.predict(X_test_outer, alpha=best_alpha)
    c_index = concordance_index_ipcw(y[train_idx],y[test_idx], preds_for_cindex)[0]
    
    
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
results_df.to_csv("results/outer_cv_results.csv", index=False)