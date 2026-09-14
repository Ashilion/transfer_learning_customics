import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw
import pickle

import json
import sys 

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
sys.path.append(wkd_path)
from utils.vvh_cv import vvh_cv

cancer_name = "COAD"
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

print(len(y_test))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


with open("data/compare_alpha/preds_array_survival_fn.json", "r") as f:
    data = json.load(f)

with open("data/compare_alpha/preds_array.json", "r") as f:
    data_preds = json.load(f)

ibs_results = []
cindex_results = []
print(np.max(y_test["time"]))
print(y_test)
print(y_test["status"][np.argmax(y_test["time"])])
for i, entry in enumerate(data):
    
    lambda_value = entry["lambda"]

    times = np.array(entry["times"])
    t_min = min(times)
    t_max = max(times)
    preds = np.array(entry["preds"]).T

    times = np.array(times[:])
    preds = preds[:, :]
    ibs = integrated_brier_score(
        y_test,
        y_test,
        preds,
        times
    )

    ibs_results.append(ibs)

    cindex_preds = np.array(data_preds[i]["preds"]).T
    c_index = concordance_index_ipcw(y_train,y_test, cindex_preds)[0]
    cindex_results.append(c_index)

    print(f"lambda={lambda_value} IBS={ibs} Cindex={c_index}")

print(ibs_results)
#============================================================================== save ibs

ibs_df = pd.DataFrame({
    "ibs": ibs_results
})

ibs_df.to_csv(f"{data_path}_ibs_python.csv", index=False)

#============================================================================== save cindex
cindex_df = pd.DataFrame({
    "cindex": cindex_results
})

cindex_df.to_csv(f"{data_path}_cindex_python.csv", index=False)