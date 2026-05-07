import pandas as pd
import numpy as np
import pickle
import copy
import torch
import time

from sklearn.model_selection import KFold, ParameterGrid
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
inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

outer_results = []


def fit_feature_selector(cancer_dataset, nbFeatures):
    selector = {}

    for name, df in cancer_dataset.items():
        if name in ["_rna", "mirna"]:
            col_std = df.std(axis=0)
            df = df.loc[:, col_std != 0]

        if name == "mutation":
            freq = df.sum(axis=0) / df.shape[0]
            df = df.loc[:, freq > 0.01]

        if df.shape[1] > nbFeatures:
            col_std = df.std(axis=0)
            top_cols = col_std.sort_values(ascending=False).index[:nbFeatures]
        else:
            top_cols = df.columns

        selector[name] = top_cols

    return selector

def apply_feature_selector(cancer_dataset, selector):
    filtered = {}
    for name, df in cancer_dataset.items():
        cols = selector[name]
        filtered[name] = df.loc[:, df.columns.intersection(cols)]
    return filtered

def build_customics_model(omics_data, sources, params, device,
                         hidden_dim, central_hidden,
                         classifier_dim, survival_dim,
                         dropout, num_classes, unsupervised):

    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': params['rep_dim'], 'norm': True, 'dropout': dropout} for i, src in enumerate(sources)
    }

    central_params = {'hidden_dim': central_hidden, 'latent_dim': params['latent_dim'], 'norm': True, 'dropout': dropout, 'beta': 1}

    classif_params = {'n_class': num_classes,'lambda': 0,'hidden_layers': classifier_dim,'dropout': dropout}

    surv_params = {'lambda': 5, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True,'dropout': dropout}

    train_params = {'switch': 5, 'lr': params['lr']}

    model = CustOMICS(source_params=source_params,central_params=central_params, classif_params=classif_params, 
            surv_params=surv_params,train_params=train_params, device=device,unsupervised=unsupervised).to(device)

    return model

def build_survival_array(clinical_df, samples, event, time):
    return np.array(
        [(bool(e), t) for e, t in zip(
            clinical_df.loc[samples, event],
            clinical_df.loc[samples, time]
        )],
        dtype=[('status', 'bool'), ('time', 'float')]
    )

def fit_coxnet(X, y, alphas=None):
    if alphas is None:
        model = CoxnetSurvivalAnalysis(n_alphas=10, l1_ratio=0.0001, alpha_min_ratio=0.00001, max_iter=100, fit_baseline_model=True)
    else:
        model = CoxnetSurvivalAnalysis(l1_ratio=0.0001, alphas=alphas, fit_baseline_model=True)

    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X, y)

    return pipe.named_steps["coxnetsurvivalanalysis"]

def evaluate_survival(coxnet, alpha, Z_train, y_train, Z_val, y_val, mode):
    if mode == "ibs":
        survs = coxnet.predict_survival_function(Z_val, alpha=alpha)
        t_min = y_val["time"].min()
        t_max = min(1492, y_val["time"].max())
        times = np.arange(t_min, t_max)
        preds = np.vstack([fn(times) for fn in survs])
        return integrated_brier_score(y_train, y_val, preds, times)

    elif mode == "vvh":
        return vvh_cv(coxnet, alpha, Z_train, y_train, Z_val, y_val)

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

    y_train_outer = build_survival_array(clinical_df, sample_train_outer, event, surv_time)
    y_test_outer = build_survival_array(clinical_df, samples_test_outer, event, surv_time)

    for params in ParameterGrid(param_grid):

        inner_scores = []

        #============================= get the list of alpha for this params on the outer train fold ======================================
        model = build_customics_model(omics_train_outer,sources,params, device, hidden_dim, central_hidden,
                        classifier_dim, survival_dim, dropout, num_classes, unsupervised )

        Z_train_outer = model.get_latent_representation(omics_train_outer)

        coxnet = fit_coxnet(Z_train_outer, y_train_outer)
        estimated_alphas = coxnet.alphas_
        #==================================================================================================================================
        alpha_scores = {alpha: [] for alpha in estimated_alphas}

        tot_time_train = 0
        tot_time_validate = 0 
        #TODO stratify
        for inner_train_idx, inner_val_idx in inner_cv.split(samples_train_outer):
            start_train = time.time()
            samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
            samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
            omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

            selector = fit_feature_selector(omics_train_raw, nbFeatures=nbFeatures)

            omics_train = apply_feature_selector(omics_train_raw, selector)
            omics_val = apply_feature_selector(omics_val_raw, selector)

            model = build_customics_model(omics_train,sources,params, device, hidden_dim, central_hidden,
                        classifier_dim, survival_dim, dropout, num_classes, unsupervised )

            model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

            
            Z_train = model.get_latent_representation(omics_train)
            Z_val   = model.get_latent_representation(omics_val)
            
            y_train_struct = build_survival_array(clinical_df, samples_train_inner, event, surv_time)
            y_val_struct = build_survival_array(clinical_df, samples_val_inner, event, surv_time)

            # Try all values of lambda for this model / latent representation 
            # Cox model
            #====================================================================================================================
            
            coxnet = fit_coxnet(Z_train, y_train_struct, estimated_alphas)
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

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_alpha = alpha

    print("Best params:", best_params)

    

    model = build_customics_model(omics_train,sources,best_params, device,hidden_dim, central_hidden,
                         classifier_dim, survival_dim, dropout, num_classes, unsupervised )

    model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

    Z_train_outer = model.get_latent_representation(omics_train_outer)
    Z_test_outer  = model.get_latent_representation(omics_test_outer)
    
    coxnet = fit_coxnet(Z_train_outer, y_train_outer, [best_alpha])

    #cindex calcul
    risk_scores = coxnet.predict(Z_test_outer, alpha=best_alpha)
    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    #ibs calcul
    survs = coxnet.predict_survival_function(Z_test_outer, alpha = best_alpha)
    t_min = y_test_outer["time"].min()
    t_max = min(1492, y_test_outer["time"].max())
    times = np.arange(t_min, t_max)
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