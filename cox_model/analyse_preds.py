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