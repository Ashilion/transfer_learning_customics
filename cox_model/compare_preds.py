import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw
import pickle

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

pd.DataFrame(X_train).to_csv(f"{data_path}_X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{data_path}_X_test.csv", index=False)

pd.DataFrame({
    "time": [t for (_, t) in y_train],
    "status": [int(s) for (s, _) in y_train]
}).to_csv(f"{data_path}_y_train.csv", index=False)

pd.DataFrame({
    "time": [t for (_, t) in y_test],
    "status": [int(s) for (s, _) in y_test]
}).to_csv(f"{data_path}_y_test.csv", index=False)

pd.DataFrame({"lambda": alphas}).to_csv(f"{data_path}_lambda_grid.csv", index=False)


preds = []
vvh_score = []
for a in alphas:
    preds.append(model.predict(X_test, alpha=a))

    score = vvh_cv(
        model,
        a,
        X_train,
        y_train,
        X_test,
        y_test
    )
    vvh_score.append(score)

preds_df = pd.DataFrame(preds)
vvh_df = pd.DataFrame(vvh_score)
print(preds_df.head())
print(vvh_df.head())

preds_df.to_csv(f"{data_path}_pred_python_multi_full.csv", index=False)
vvh_df.to_csv(f"{data_path}_vvh_python_full.csv", index=False)

# save model coefs
coef_df = pd.DataFrame(model.coef_)
print(coef_df.head())
coef_df.to_csv(f"{data_path}_coef_python.csv")