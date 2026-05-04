import pandas as pd
import numpy as np

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/results/ref_results.csv"

df = pd.read_csv(wkd_path)

cox_df = df[df["learner_id"] == "cox_ref"]
cox_df = cox_df[cox_df["task_id"]=="COAD"]
cox_df = cox_df[['nr', 'task_id','iteration', 'cindex_default', 'graf','time_train', 'time_both']]

print(cox_df[:1000])