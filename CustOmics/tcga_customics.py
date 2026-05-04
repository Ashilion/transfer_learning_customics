import pandas as pd
import numpy as np
import os
import pickle
from src.network.customics import CustOMICS
from src.tools.prepare_dataset import prepare_dataset
from src.tools.utils import get_sub_omics_df
from sklearn.model_selection import train_test_split
import torch

path = "../data/dict_pancancer_preprocessed.pickle"

### load the preprocessed data
with open(path, "rb") as f:
    data = pickle.load(f)

for name_omic, df in data.items():
    print("head of " + name_omic)
    print(df.head())

clinical_df = data["clinical"]

omics_df = {'protein': data["_rna"] ,
            'gene_exp': data["mirna"],
            'methyl': data["cnv"],
            'mutation':data["mutation"]
            }
### create train test val
lt_samples = list(clinical_df.index)

batch_size = 32


samples_train, samples_test = train_test_split(lt_samples, test_size=0.2)
samples_train, samples_val = train_test_split(samples_train, test_size=0.2)

if len(samples_train)%batch_size == 1:
    samples_train= samples_train[:-1]
print("size sample train :" , len(samples_train))
print("reste division par batch size :" , len(samples_train)%batch_size)

omics_train = get_sub_omics_df(omics_df, samples_train)
omics_val = get_sub_omics_df(omics_df, samples_val)
omics_test = get_sub_omics_df(omics_df, samples_test)




x_dim = [omics_df[omic_source].shape[1] for omic_source in omics_df.keys()]

#### Defining Hyperparameters

n_epochs = 10
device = torch.device('cpu')
label = 'status'
event = 'status'
surv_time = 'time'

task = 'survival'
sources = ['gene_exp', 'methyl', 'protein','mutation']

hidden_dim = [512, 256]
central_dim = [512, 256]
rep_dim = 128
latent_dim= 128
num_classes = 5
dropout = 0.2
beta = 1
lambda_classif = 0
classifier_dim = [128, 64]
lambda_survival = 5
survival_dim = [64,32]

source_params = {}
central_params = {'hidden_dim': central_dim, 'latent_dim': latent_dim, 'norm': True, 'dropout': dropout, 'beta': beta}
classif_params = {'n_class': num_classes, 'lambda': lambda_classif, 'hidden_layers': classifier_dim, 'dropout': dropout}
surv_params = {'lambda': lambda_survival, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout': dropout}
for i, source in enumerate(sources):
    source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': rep_dim, 'norm': True, 'dropout': 0.2}
train_params = {'switch': 5, 'lr': 1e-3}

unsupervised = True
#### Training the model

model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                        surv_params=surv_params, train_params=train_params, device=device, unsupervised=unsupervised).to(device)
print('Number of Parameters: ', model.get_number_parameters())
model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
            omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=True, task=task)
metric = model.evaluate(omics_test=omics_test, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                task=task, batch_size=1024, plot_roc=False)
model.plot_loss()
