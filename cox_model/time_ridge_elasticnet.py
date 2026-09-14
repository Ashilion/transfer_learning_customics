import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw
import pickle
import time 

import sys 

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
sys.path.append(wkd_path)

from utils.vvh_cv import vvh_cv

cancer_name = "COAD"
wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
path = f"{wkd_path}data/{cancer_name}_clinical.pickle"
data_path = f"{wkd_path}data/compare_alpha/{cancer_name}"

with open(path, "rb") as f:
    df = pickle.load(f)

X = df.drop(columns=["time", "status",  "bcr_patient_barcode"])
y_struct = np.array(
    [(bool(s), t) for s, t in zip(df["status"], df["time"])],
    dtype=[("status", "bool"), ("time", "float")]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_struct, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = CoxnetSurvivalAnalysis(l1_ratio=0.01, alpha_min_ratio=0.0001, n_alphas=100, tol=1e-15,fit_baseline_model=True)
model.fit(X_train, y_train)

alphas = model.alphas_

start_train = time.time()
model = CoxnetSurvivalAnalysis(l1_ratio=0.01, alphas= alphas, tol=1e-15,fit_baseline_model=True)
stop_train = time.time()
time_train = stop_train - start_train 

start_train_ridge = time.time()
for a in alphas:
    model = CoxPHSurvivalAnalysis(alpha=a, tol=1e-15)
    model.fit(X_train, y_train)
stop_train_ridge = time.time()
time_train_ridge = stop_train_ridge - start_train_ridge

print("time train elastic", time_train)
print("time train ridge", time_train_ridge)
