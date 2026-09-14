import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cancer_name = "COAD"
wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
data_path = f"{wkd_path}data/compare_alpha/{cancer_name}"

py = pd.read_csv(f"{data_path}_pred_python_multi.csv")
r = pd.read_csv(f"{data_path}_pred_r_multi.csv")


for col in py.columns:
    plt.figure()
    plt.scatter(py[col], r[col])
    plt.xlabel("Python")
    plt.ylabel("R")
    plt.title(f"Comparison for {col}")
    plt.show()

for col in py.columns:
    diff = np.abs(py[col] - r[col]).mean()
    print(f"{col}: mean abs diff = {diff:.6f}")

    print(py[col])
    print(r[col])

### IBS =======================================================

py_df = pd.read_csv(f"{data_path}_ibs_python.csv")
r_df = pd.read_csv(f"{data_path}_ibs_r.csv")

compare_df = py_df.merge(
    r_df,
    on=["alpha_index"],
    suffixes=("_python", "_r")
)

compare_df["abs_diff"] = np.abs(
    compare_df["ibs_normalized_python"]
    - compare_df["ibs_normalized_r"]
)

compare_df["relative_diff"] = (
    compare_df["abs_diff"]
    / np.maximum(compare_df["ibs_normalized_python"], 1e-12)
)

print(compare_df.head())

print("\nMax absolute difference:")
print(compare_df["abs_diff"].max())

print("\nMean absolute difference:")
print(compare_df["abs_diff"].mean())