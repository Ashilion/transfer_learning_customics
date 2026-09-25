import pandas as pd
import numpy as np
import pickle
import copy
import torch
import time

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
print("taille donnée :", len(lt_samples))

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

# outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


outer_results = []
l1_ratio= 0.01

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


outer_results = []

# for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(lt_samples)):
for outer_fold, (train_idx, test_idx) in enumerate(zip(train_folds, test_folds)):

    
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

    y_train_outer = np.array(
        [(bool(e), t) for e, t in zip(clinical_df.loc[samples_train_outer, event],
                                    clinical_df.loc[samples_train_outer, surv_time])],
        dtype=[('status', 'bool'), ('time', 'float')]
    )

    y_test_outer = np.array(
        [(bool(e), t) for e, t in zip(clinical_df.loc[samples_test_outer, event],
                                    clinical_df.loc[samples_test_outer, surv_time])],
        dtype=[('status', 'bool'), ('time', 'float')]
    )

    for params in ParameterGrid(param_grid):

        inner_scores = []

        #============================= get the list of alpha for this params on the outer train fold ======================================
         # rebuild params
        source_params = {}
        
        x_dim = [omics_train_outer[src].shape[1] for src in sources]
        for i, src in enumerate(sources):
            source_params[src] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': params['rep_dim'], 'norm': True, 'dropout':dropout}

        central_params = {'hidden_dim': central_hidden, 'latent_dim': params['latent_dim'], 'norm': True, 'dropout':dropout,'beta': 1 }

        classif_params = {'n_class': num_classes, 'lambda': 0, 'hidden_layers': classifier_dim, 'dropout':dropout}

        surv_params = {'lambda': 5, 'dims': survival_dim,'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout':dropout }

        train_params = {'switch': 5,'lr': params['lr']}

        model = CustOMICS(source_params=source_params,central_params=central_params, classif_params=classif_params, 
            surv_params=surv_params,train_params=train_params, device=device,unsupervised=unsupervised).to(device)

        model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
            omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

        Z_train_outer = model.get_latent_representation(omics_train_outer)
        Z_test_outer  = model.get_latent_representation(omics_test_outer)

        coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(n_alphas=10,l1_ratio=l1_ratio, alpha_min_ratio=0.00001, max_iter=100, fit_baseline_model=True))
        coxnet_pipe.fit(Z_train_outer, y_train_outer)
        coxnet = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]
        estimated_alphas = coxnet.alphas_
        #==================================================================================================================================
        alpha_scores = {alpha: [] for alpha in estimated_alphas}

        tot_time_train = 0
        tot_time_validate = 0 
        #TODO stratify
        for inner_train_idx, inner_val_idx in inner_cv.split(samples_train_outer, y_train_outer["status"]):
            start_train = time.time()
            samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
            samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

            omics_train_raw = get_sub_omics_df(omics_df, samples_train_inner)
            omics_val_raw = get_sub_omics_df(omics_df, samples_val_inner)

            selector = fit_feature_selector(omics_train_raw, nbFeatures=nbFeatures)

            omics_train = apply_feature_selector(omics_train_raw, selector)
            omics_val = apply_feature_selector(omics_val_raw, selector)

            x_dim = [omics_train[src].shape[1] for src in sources]
            source_params = {}
            for i, src in enumerate(sources):
                source_params[src] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': params['rep_dim'], 'norm': True, 'dropout':dropout}

            central_params = {'hidden_dim': central_hidden, 'latent_dim': params['latent_dim'], 'norm': True, 'dropout':dropout,'beta': 1 }

            classif_params = {'n_class': num_classes, 'lambda': 0, 'hidden_layers': classifier_dim, 'dropout':dropout}

            surv_params = {'lambda': 5, 'dims': survival_dim,'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout':dropout }

            train_params = {'switch': 5,'lr': params['lr']}

            model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                        surv_params=surv_params, train_params=train_params, device=device, unsupervised=unsupervised).to(device)

            model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)
            
            # score = model.evaluate(omics_test=omics_val, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
            #     task=task, batch_size=1024, plot_roc=False)

            # cox model sur la latent representation / faut il ajouter lambda (alpha) aux hyperparamètres ?
            
            Z_train = model.get_latent_representation(omics_train)
            Z_val   = model.get_latent_representation(omics_val)
            
            y_train_struct = np.array(
                [(bool(e), t) for e, t in zip(clinical_df.loc[samples_train_inner, event],
                                            clinical_df.loc[samples_train_inner, surv_time])],
                dtype=[('status', 'bool'), ('time', 'float')]
            )

            y_val_struct = np.array(
                [(bool(e), t) for e, t in zip(clinical_df.loc[samples_val_inner, event],
                                            clinical_df.loc[samples_val_inner, surv_time])],
                dtype=[('status', 'bool'), ('time', 'float')]
            )

            # Try all values of lambda for this model / latent representation 
            # TODO elastic net vs l2 regression only
            # Cox model 
            #====================================================================================================================
            coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,alphas=estimated_alphas, fit_baseline_model=True))        
            est = coxnet_pipe.fit(Z_train, y_train_struct)
            coxnet = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]

            stop_train = time.time()
            tot_time_train += stop_train - start_train 
            start_validate = time.time()

            for i, alpha in enumerate(coxnet.alphas_):
                # extract survival functions for alpha i
                survs = coxnet.predict_survival_function(Z_val, alpha = alpha)
                t_min = y_val_struct["time"].min()
                t_max = min(1492, y_val_struct["time"].max())
                times = np.arange(t_min, t_max)

                if validation_function == "ibs":
                    preds = np.vstack([fn(times) for fn in survs])
                    score = integrated_brier_score(y_train_struct,y_val_struct,preds,times)
                elif validation_function == "vvh":
                    score = vvh_cv(coxnet, alpha, Z_train,y_train_struct, Z_val, y_val_struct)
                
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

    

    # rebuild params
    source_params = {}
    for i, src in enumerate(sources):
        source_params[src] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': best_params['rep_dim'], 'norm': True,'dropout': dropout}

    central_params = {'hidden_dim': central_hidden, 'latent_dim': best_params['latent_dim'],'norm': True, 'dropout': dropout,'beta': 1}

    classif_params = {'n_class': num_classes, 'lambda': 0, 'hidden_layers': classifier_dim, 'dropout': dropout}

    surv_params = {'lambda': 5,'dims': survival_dim,'activation': 'SELU','l2_reg': 1e-2,'norm': True,'dropout': dropout}

    train_params = {'switch': 5,'lr': best_params['lr']}

    model = CustOMICS(source_params=source_params,central_params=central_params, classif_params=classif_params, 
        surv_params=surv_params,train_params=train_params, device=device,unsupervised=unsupervised).to(device)

    model.fit(omics_train=omics_train_outer, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time, 
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs, verbose=False, task=task)

    Z_train_outer = model.get_latent_representation(omics_train_outer)
    Z_test_outer  = model.get_latent_representation(omics_test_outer)
    
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,alphas=[best_alpha], fit_baseline_model=True))        
    coxnet_pipe.fit(Z_train_outer, y_train_outer)
    coxnet = coxnet_pipe.named_steps["coxnetsurvivalanalysis"]

    #cindex calcul
    risk_scores = coxnet.predict(Z_test_outer, alpha=best_alpha)
    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    #ibs calcul
    survs = coxnet.predict_survival_function(Z_test_outer, alpha = best_alpha)
    times = np.sort(np.unique(y[test_idx]["time"]))
    upper = min(
        np.max( y[train_idx]["time"]),
        np.max(y[test_idx]["time"])
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