import pyreadr
import pandas as pd
import glob
import numpy as np
import os
import pickle

cancer_name = "LGG"

#get the file of specific cancer

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
data_path = "data/herrmann_data"

file_path = f"{wkd_path}{data_path}/{cancer_name}.RData"

res = pyreadr.read_r(file_path)
df = res["dat"]


#get only clinical data
regex = "clinical|time|status|bcr_patient_barcode"
df = df.filter(regex=f"({regex})$")

print(df.head())

final_path = f"{wkd_path}data/{cancer_name}_clinical.pickle"
tmp_path = f"{final_path}.tmp"

with open(tmp_path, "wb") as f:
    pickle.dump(df, f)
    f.flush()
    os.fsync(f.fileno())

os.replace(tmp_path, final_path) 