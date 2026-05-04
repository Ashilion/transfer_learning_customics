import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_results(path_ref, path_new, type_ref="ref"):
    
    #read ref results
    if type_ref =="ref":
        df = pd.read_csv(path_ref)

        ref_df = df[df["learner_id"] == "cox_ref"]
        ref_df = ref_df[ref_df["task_id"]=="COAD"]
        ref_df = ref_df[['nr', 'task_id','iteration', 'cindex_default', 'graf','time_train', 'time_both']]
    else :
        ref_df = pd.read_csv(path_ref)
    #read own results
    df = pd.read_csv(path_new)

    print(ref_df.head())
    print(df.head())

    ref_df = ref_df.rename(columns={"iteration": "fold"})

    #merge and analyse
    merged = pd.merge(
        ref_df,
        df,
        on="fold",
        suffixes=("_ref", "_new")
    )

    merged["cindex_diff"] = merged["cindex_default_new"] - merged["cindex_default_ref"]
    merged["graf_diff"] = merged["graf_new"] - merged["graf_ref"]

    print(merged[[
        "fold",
        "cindex_default_ref", "cindex_default_new", "cindex_diff",
        "graf_ref", "graf_new", "graf_diff"
    ]])

    print("C-index difference (mean):", merged["cindex_diff"].mean())
    print("Graf difference (mean):", merged["graf_diff"].mean())

    plt.figure()
    plt.plot(merged["fold"], merged["cindex_default_ref"], label="ref")
    plt.plot(merged["fold"], merged["cindex_default_new"], label="new")
    plt.legend()
    plt.title("C-index comparison")
    plt.show()

    plt.figure()
    plt.plot(merged["fold"], merged["graf_ref"], label="ref")
    plt.plot(merged["fold"], merged["graf_new"], label="new")
    plt.legend()
    plt.title("IBS comparison")
    plt.show()

    return merged

# path_ref = "/env/cnrgh/proj/math_stats/scratch/hlegrand/results/ref_results.csv"
path_new = "/env/cnrgh/proj/math_stats/scratch/hlegrand/results/outer_cv_results_vvh.csv"
path_ref = "/env/cnrgh/proj/math_stats/scratch/hlegrand/results/ref_results.csv"

merged = compare_results(path_ref, path_new, type_ref="ref")