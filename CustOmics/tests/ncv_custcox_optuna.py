import pandas as pd
import numpy as np
import pickle
import copy
import torch
import time
import optuna

from sklearn.model_selection import KFold, ParameterGrid,StratifiedKFold
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
#load the pancancer object
path = "../data/dict_pancancer_union_mutation.pickle"
with open(path, "rb") as f:
    pancancer = pickle.load(f)

def get_cancer_data(cancer_name):
    source = {}
    for name,df in pancancer.items():
        source[name] = df[pancancer["clinical"]["cancer_type"]==cancer_name]
    return source

cancer_name = "COAD"
train_folds, test_folds = fold_utils.get_folds(cancer_name, src="../data/splits.json")

data = get_cancer_data(cancer_name)
clinical_df = data["clinical"]

omics_df = {
    'protein': data["_rna"],
    'gene_exp': data["mirna"],
    'methyl': data["cnv"],
    'mutation': data["mutation"]
}

lt_samples = list(clinical_df.index)

device = torch.device("cpu")
batch_size = 32
n_epochs = 10
switch_epoch = n_epochs // 2

label = 'status'
event = 'status'
surv_time = 'time'
task = 'survival'


sources = omics_df.keys()

hidden_dim = [512, 256]
central_hidden = [512, 256]

num_classes = 5

classifier_dim = [128, 64]
survival_dim = [64, 32]

unsupervised = True
dropout = 0.2

param_grid = {
    "latent_dim": [64, 128],
    "rep_dim": [64, 128],
    "lr": [1e-3, 1e-4]
}

alpha = 0.01

nbFeatures = 5000
validation_function = "vvh"

outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
# inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

outer_results = []



#optuna optimize cross validation meilleure moyenne de paramètre des inner folds
def objective_with_params(trial, fn_param):
    params = {
        "latent_dim": trial.suggest_int("latent_dim", 64, 128),
        "rep_dim": trial.suggest_int("rep_dim", 64, 128),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    }
    best_score = np.inf
    #============================= get the list of alpha for this params on the outer train fold ======================================
    model = build_customics_model(fn_param["omics_train_outer"],sources,params, device, hidden_dim, central_hidden,
                    classifier_dim, survival_dim, dropout, num_classes, unsupervised,switch_epoch )
    model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
            omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)
    Z_train_outer = model.get_latent_representation(fn_param["omics_train_outer"])

    coxnet, scaler = fit_coxnet(Z_train_outer, y_train_outer, l1_ratio)
    estimated_alphas = coxnet.alphas_
    #==================================================================================================================================
    alpha_scores = {alpha: [] for alpha in estimated_alphas}

    tot_time_train = 0
    tot_time_validate = 0 
    
    for inner_train_idx, inner_val_idx in inner_cv.split(fn_param["samples_train_outer"], fn_param["y_train_outer"]["status"]):
        start_train = time.time()
        samples_train_inner = [fn_param["samples_train_outer"][i] for i in inner_train_idx]
        samples_val_inner = [fn_param["samples_train_outer"][i] for i in inner_val_idx]

        omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
        omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

        selector = fit_feature_selector(omics_train_raw, nbFeatures=nbFeatures)

        omics_train = apply_feature_selector(omics_train_raw, selector)
        omics_val = apply_feature_selector(omics_val_raw, selector)

        model = build_customics_model(omics_train,sources,params, device, hidden_dim, central_hidden,
                    classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch )

        model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
            omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

        
        Z_train = model.get_latent_representation(omics_train)
        Z_val   = model.get_latent_representation(omics_val)
        
        y_train_struct = build_survival_array(clinical_df, samples_train_inner, event, surv_time)
        y_val_struct = build_survival_array(clinical_df, samples_val_inner, event, surv_time)

        # Try all values of lambda for this model / latent representation 
        # Cox model
        #====================================================================================================================
        
        coxnet, scaler = fit_coxnet(Z_train, y_train_struct,l1_ratio, estimated_alphas)
        stop_train = time.time()
        tot_time_train += stop_train - start_train 
        start_validate = time.time()

        for i, alpha in enumerate(coxnet.alphas_):
            score = evaluate_survival(coxnet, alpha,Z_train,y_train_struct, Z_val, y_val_struct,validation_function)
            alpha_scores[alpha].append(score)
            
        #====================================================================================================================
        stop_validate = time.time()
        tot_time_validate += stop_validate - start_validate 
    for i , alpha in enumerate(estimated_alphas):

        mean_score = np.mean(alpha_scores[alpha])
        print(f"Params {params} -> alpha {alpha} -> score {mean_score:.4f}")

        if mean_score < best_score:
            best_score = mean_score
            best_alpha = alpha
    
    # save the best alpha for this trial
    trial.set_user_attr("best_alpha", float(best_alpha))
    trial.set_user_attr("tot_time_train", tot_time_train)
    trial.set_user_attr("tot_time_validate", tot_time_validate)
    return best_score


outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(lt_samples)):
# for outer_fold, (train_idx, test_idx) in enumerate(zip(train_folds, test_folds)):

    print(f" OUTER FOLD {outer_fold}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    best_score = -np.inf
    best_params = None

    # outer fold data preparation / do it before inner because we need to train a model on the "big" train to generate the list of alphas
    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

    selector = fit_feature_selector(omics_train_outer_raw, nbFeatures=nbFeatures)

    omics_train_outer = apply_feature_selector(omics_train_outer_raw, selector)
    omics_test_outer = apply_feature_selector(omics_test_outer_raw, selector)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, event, surv_time)
    y_test_outer = build_survival_array(clinical_df, samples_test_outer, event, surv_time)
    
    fn_params= {"samples_train_outer":samples_train_outer, "omics_train_outer":omics_train_outer,"y_train_outer":y_train_outer}    
    def objective(trial):
        return objective_with_params(trial, fn_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, timeout=600)
    
    best_params = study.best_trial.params
    #how to get the best alpha ?
    best_alpha = study.best_trial.user_attrs["best_alpha"]
    tot_time_train = study.best_trial.user_attrs["tot_time_train"]
    tot_time_validate = study.best_trial.user_attrs["tot_time_validate"]
    print(f"optuna best params  : {best_params} \n optuna best alpha : {best_alpha}")

    model = build_customics_model(omics_train_outer,sources,best_params, device,hidden_dim, central_hidden,
                         classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch )

    model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

    Z_train_outer = model.get_latent_representation(omics_train_outer)
    Z_test_outer  = model.get_latent_representation(omics_test_outer)
    
    coxnet, scaler = fit_coxnet(Z_train_outer, y_train_outer,l1_ratio, [best_alpha])

    #cindex calcul
    risk_scores = coxnet.predict(Z_test_outer, alpha=best_alpha)
    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    #ibs calcul
    survs = coxnet.predict_survival_function(Z_test_outer, alpha = best_alpha)
    times = np.sort(np.unique(y_test_outer["time"]))
    upper = min(
        np.max( y_train_outer["time"]),
        np.max(y_test_outer["time"])
    )
    times = times[times < upper]
    preds = np.vstack([fn(times) for fn in survs])
    ibs_score = integrated_brier_score(y_train_outer,y_test_outer,preds,times)
    


    print(f"Outer test score: {c_index:.4f}")

    model.save_figure_loss(outer_fold)

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
print("Mean score:", np.mean(results_df["cindex_default"]))
print("Std:", np.std(results_df["cindex_default"]))
results_df.to_csv("results/ncv_custcox_mult_alpha_results.csv", index=False)