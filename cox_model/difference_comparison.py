import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cancer_name = "COAD"
wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
data_path = f"{wkd_path}data/compare_alpha/{cancer_name}"

plt.figure(figsize=(10, 6))
### lambdas ==================================================================

py = pd.read_csv(f"{data_path}_alphas_python_sksurv_dev_modified.csv")
r_same = pd.read_csv(f"{data_path}_lambda_R.csv")
r_ridge = pd.read_csv(f"{data_path}_lambda_R_l2.csv")

# harmoniser les noms
py.columns = ["value"]
r_same.columns = ["value"]
r_ridge.columns = ["value_r_ridge"]

# df = pd.concat([py, r_same, r_ridge], ignore_index=True)
print(py.head())
print(r_same.head())
compare_df = py.merge(
    r_same,
    suffixes=("_python", "_r_same"),
    left_index=True, right_index=True
).merge(
    r_ridge,
    left_index= True,  right_index=True
)

print(compare_df.columns)
print(compare_df.head())
compare_df["same_params_diff"] = np.abs(compare_df["value_python"] - compare_df["value_r_same"])
compare_df["ridge_diff"] = np.abs(compare_df["value_python"] - compare_df["value_r_ridge"])

plt.subplot(2, 3, 1)
plt.xlabel("index lambda/alpha ")
plt.ylabel("difference valeur lambda/alpha")
plt.plot(compare_df["same_params_diff"])

# plt.show()

### Weights =======================================================
py_coefs_df = pd.read_csv(f"{data_path}_coef_python.csv")
r_coefs_df = pd.read_csv(f"{data_path}_coef_r.csv")

diff_coef_df = pd.DataFrame()
for i in range(len(py_coefs_df.columns)-1):
    diff_coef_df[f"diff_{i}"] = np.abs(py_coefs_df[f"{i}"]-r_coefs_df[f"s{i}"])

means_diff_coefs = diff_coef_df.mean(axis=0)
plt.subplot(2, 3, 2)
plt.plot(range(len(means_diff_coefs)), means_diff_coefs)
plt.xlabel("index lambda/alpha")
plt.ylabel("difference coefs")

### Preds =======================================================
py_preds = pd.read_csv(f"{data_path}_pred_python_multi_full.csv")
r_preds = pd.read_csv(f"{data_path}_pred_r_multi_full.csv")

print(py_preds.columns)
print(r_preds.columns)

compare_preds_df = py_preds.merge(
    r_preds,
    left_index=True, right_index=True
)

diff_preds_df = pd.DataFrame()
for i in range(len(py_preds.columns)):
    diff_preds_df[f"diff_{i}"] = np.abs(compare_preds_df[f"{i}"] - compare_preds_df[f"V{i+1}"])

means_diff = diff_preds_df.mean(axis=1)
print(means_diff)

plt.subplot(2, 3, 3)
plt.plot(means_diff)
plt.xlabel("index lambda/alpha ")
plt.ylabel("difference prediction")

# plt.show()
### VVH =======================================================
py_vvh = pd.read_csv(f"{data_path}_vvh_python_full.csv")
r_vvh = pd.read_csv(f"{data_path}_vvh_r_full.csv")

print(py_vvh.head())
print(r_vvh.head())
vvh_merge = py_vvh.merge(
    r_vvh,
    left_index=True, right_index=True
)
print(vvh_merge.head())
vvh_merge["vvh_diff"] = np.abs(vvh_merge["0"] - vvh_merge["vvh_score"])

plt.subplot(2, 3, 4)
plt.plot(vvh_merge["vvh_diff"])
plt.xlabel("index lambda/alpha")
plt.ylabel("difference vvh")
# plt.show()

### IBS =======================================================

py_ibs_df = pd.read_csv(f"{data_path}_ibs_python.csv")
r_ibs_df = pd.read_csv(f"{data_path}_ibs_r.csv")

ibs_merge_df = py_ibs_df.merge(
    r_ibs_df,
    left_index=True, right_index=True
)
print(ibs_merge_df.head())
ibs_merge_df["ibs_diff"] = np.abs(ibs_merge_df["ibs"] - ibs_merge_df["graf"])

plt.subplot(2, 3, 5)
plt.plot(ibs_merge_df["ibs_diff"])
plt.xlabel("index lambda/alpha")
plt.ylabel("difference ibs")
# plt.show()



### Cindex =======================================================
py_cindex_df = pd.read_csv(f"{data_path}_cindex_python.csv")
r_cindex_df = pd.read_csv(f"{data_path}_cindex_r.csv")

cindex_merge_df = py_cindex_df.merge(
    r_cindex_df,
    left_index=True, right_index=True,
    suffixes=("_py", "_r"),
)
print(cindex_merge_df.head())
cindex_merge_df["cindex_diff"] = np.abs(cindex_merge_df["cindex_py"] - cindex_merge_df["cindex_r"])

plt.subplot(2, 3, 6)
plt.plot(cindex_merge_df["cindex_diff"])
plt.xlabel("index lambda/alpha")
plt.ylabel("difference cindex")

plt.tight_layout()
plt.savefig("difference_r_python_cox_and_metrics.png")   
# plt.show()

