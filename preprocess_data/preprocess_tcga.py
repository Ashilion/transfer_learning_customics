import pyreadr
import pandas as pd
import glob
import numpy as np
import os
import pickle

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
data_path = "data/herrmann_data"

# print(df.head())
# print(df.columns)

list_cancers_csv = glob.glob(wkd_path+ data_path + "/*.RData")
print(list_cancers_csv)

pancancer = {}

blocks = ["clinical|time|status|bcr_patient_barcode", "mutation", "_rna", "mirna", "cnv"]
for file_path in list_cancers_csv:
    if "minilgg" in file_path:
        continue
    print("reading file" + file_path)
    result = pyreadr.read_r(file_path)
    df = result["dat"]

    block_dict = {
        bl: df.filter(regex=f"({bl})$")
        for bl in blocks
    }
    cancer_type = os.path.basename(file_path).replace(".RData","")
    print("cancer type : ", cancer_type)
    block_dict["clinical"] = block_dict["clinical|time|status|bcr_patient_barcode"] 
    print(block_dict["clinical"].columns)   
    block_dict["clinical"]["cancer_type"] = cancer_type
    del block_dict["clinical|time|status|bcr_patient_barcode"]
    for modality in block_dict:
        print(modality + ":" )
        print(block_dict[modality].head())

    pancancer[cancer_type] = block_dict

print(len(pancancer))
tmp_path = wkd_path + "data/dict_pancancer.pickle.tmp"
final_path = wkd_path + "data/dict_pancancer.pickle"

with open(tmp_path, "wb") as f:
    pickle.dump(pancancer, f)
    f.flush()
    os.fsync(f.fileno())

os.replace(tmp_path, final_path) 


