import pandas as pd
import numpy as np
import pickle
import copy
import torch

from sklearn.model_selection import KFold, ParameterGrid

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

path = "../data/source_preprocessed.pickle"

with open(path, "rb") as f:
    data = pickle.load(f)

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

x_dim = [omics_df[src].shape[1] for src in sources]

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

outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

outer_results = []


for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(lt_samples)):
    
    print(f" OUTER FOLD {outer_fold}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):

        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(samples_train_outer):

            samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
            samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

            omics_train = get_sub_omics_df(omics_df, samples_train_inner)
            omics_val = get_sub_omics_df(omics_df, samples_val_inner)

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
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=True, task=task)
            
            score = model.evaluate(omics_test=omics_val, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                task=task, batch_size=1024, plot_roc=False)

            inner_scores.append(score)
            print(f"inner_score -> {score}")
        mean_score = np.mean(inner_scores)
        print(f"Params {params} -> score {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print("Best params:", best_params)

    
    omics_train_outer = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer = get_sub_omics_df(omics_df, samples_test_outer)

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

    test_score = model.evaluate(omics_test=omics_test_outer,clinical_df=clinical_df,label=label,event=event,surv_time=surv_time,
        task=task, batch_size=1024 )

    print(f"Outer test score: {test_score:.4f}")
    outer_results.append(test_score)



print("Mean score:", np.mean(outer_results))
print("Std:", np.std(outer_results))