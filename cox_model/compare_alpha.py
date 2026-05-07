import pandas as pd
import glob
import numpy as np
import os
import pickle
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cancer_name = "COAD"
wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
path = f"{wkd_path}data/{cancer_name}_clinical.pickle"

with open(path, "rb") as f:
    df = pickle.load(f)
print(df.head())
print(df.columns)
Xt = df[list(set(df.columns)-set(["time", "bcr_patient_barcode","status"]))]
y = np.array([(bool(i),j ) for i, j in zip(df["status"], df["time"])],dtype=[('status', 'bool_'), ('time', '<f4')])

print(Xt.head())
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(Xt),
    columns=Xt.columns,
    index=Xt.index
)

# elastic net
cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.001, alpha_min_ratio=0.0001, n_alphas=100)
cox_elastic_net.fit(X_scaled, y)  

alphas = cox_elastic_net.alphas_
print("nb alpha : ", len(alphas))
print(alphas)

#save X_scaled et y pour lire en R et comparer , save alphas
if True:
    data_path = f"{wkd_path}data/compare_alpha/{cancer_name}"

    X_scaled.to_csv(f"{data_path}_Xt.csv", index=False)

    y_df = pd.DataFrame({
        "time": df["time"],
        "status": df["status"].astype(int)  # important: 0/1
    })

    y_df.to_csv(f"{data_path}_y.csv", index=False)

    alphas_df = pd.DataFrame({
        "alpha": cox_elastic_net.alphas_
    })

    alphas_df.to_csv(f"{data_path }_alphas_python_sksurv_dev_modified.csv", index=False)