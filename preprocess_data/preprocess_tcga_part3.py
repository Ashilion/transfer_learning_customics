import pandas as pd
import numpy as np
import os
import pickle

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
path = wkd_path + "data/dict_pancancer_union_mutation.pickle"

#load the pancancer object
with open(path, "rb") as f:
    pancancer = pickle.load(f)

nbFeatures = 5000
cancerToRmv = "COAD"


source = {}
target = {}
for name,df in pancancer.items():
    source[name] = df[pancancer["clinical"]["cancer_type"]!=cancerToRmv]
    target[name] = df[pancancer["clinical"]["cancer_type"]==cancerToRmv]

def get_cancer_data(cancer_name):
    source = {}
    for name,df in pancancer.items():
        source[name] = df[pancancer["clinical"]["cancer_type"]==cancer_name]
    return source

def get_selected_features(preprocessed_dataset):
    res= {}
    for name, df in preprocessed_dataset.items():
        res[name] = list(preprocessed_dataset.columns)
    return res

def preprocess_dataset(cancer_dataset):
    ### remove zero variance samples
    cleaned_cancer_dataset = {}
    for name,df in cancer_dataset.items():
        if name== "_rna" or name == "mirna":
            print(df.head())
            col_std = df.std(axis= 0)
            cleaned_df = df.loc[ : , col_std != 0]
            
            cleaned_cancer_dataset[name] = cleaned_df
        else:
            cleaned_cancer_dataset[name] = df

    ### remove genes from SNV matrices if the mutation rate across samples was ≤ 1%
    mutation = cleaned_cancer_dataset["mutation"]
    freq = mutation.sum(axis=0) / mutation.shape[0]
    mutation = mutation.loc[:, freq > 0.01]
    cleaned_cancer_dataset["mutation"] = mutation

    ### filter all omics to include only the "nbFeatures" most variable features.
    cancer_dataset_filtered = {}
    for name,df in cleaned_cancer_dataset.items():
        if df.shape[1]>nbFeatures:
            col_std =df.std(axis = 0)
            top_cols = col_std.sort_values(ascending=False).index[:nbFeatures]

            cancer_dataset_filtered[name] = df[top_cols]
        else:
            cancer_dataset_filtered[name] = df

        # print head of each block
        print(cancer_dataset_filtered[name].head())

    return cancer_dataset_filtered

def save_dataset(dataset, path):
    tmp_path = path + ".tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(dataset, f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path) 

#preprocess for both source and target
preprocessed_source = preprocess_dataset(source)
preprocessed_target = preprocess_dataset(target)

# save dataset #####################################################################
source_path = wkd_path + "data/source_preprocessed.pickle"
target_path = wkd_path + "data/target_preprocessed.pickle"
save_dataset(preprocessed_source, source_path)
save_dataset(preprocessed_target, target_path)