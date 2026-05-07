import pandas as pd
import matplotlib.pyplot as plt

cancer_name = "COAD"
wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
data_path = f"{wkd_path}data/compare_alpha/{cancer_name}"
py = pd.read_csv(f"{data_path}_alphas_python.csv")
r_same = pd.read_csv(f"{data_path}_lambda_R.csv")
r_ridge = pd.read_csv(f"{data_path}_lambda_R_l2.csv")

# harmoniser les noms
py.columns = ["value"]
r_same.columns = ["value"]
r_ridge.columns = ["value"]

py["source"] = "Python"
r_same["source"] = "R (same params)"
r_ridge["source"] = "R (ridge)"

df = pd.concat([py, r_same, r_ridge], ignore_index=True)


for source, group in df.groupby("source"):
    plt.hist(group["value"], bins=50, alpha=0.5, label=source)

plt.xscale("log")
plt.xlabel("lambda / alpha (log scale)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure()

for source, group in df.groupby("source"):
    values = group["value"].values
    plt.plot(range(len(values)), values, label=source)

plt.yscale("log")
plt.xlabel("Index")
plt.ylabel("lambda / alpha (log scale)")
plt.legend()
plt.show()
