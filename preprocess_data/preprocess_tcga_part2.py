import pandas as pd
import numpy as np
import os
import pickle

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
path = wkd_path + "data/dict_pancancer.pickle"
#load the pancancer object
with open(path, "rb") as f:
    pancancer = pickle.load(f)

#test if loaded correctly
if pancancer is None:
    print("Error reading file")

blocks = pancancer["BRCA"].keys()
pancancer_intersect = {}
for bl in blocks:
    if bl == "mutation":
        # union_features = set.union(*[set(pancancer[ca][bl]) for ca in pancancer.keys()])
        pancancer_list = [pancancer[ca][bl] for ca in pancancer.keys()]
        pancancer_intersect[bl] = pd.concat(pancancer_list, ignore_index=True).fillna(0)

    else:
        common_features = set.intersection(*[set(pancancer[ca][bl]) for ca in pancancer.keys()])
        
        pancancer_filtered = [pancancer[ca][bl][list(common_features)] for ca in pancancer.keys()]
        pancancer_intersect[bl] = pd.concat(pancancer_filtered, ignore_index=True)


# Convert some clinical variables into numerics
cols = ["time", "status", "age_clinical"]
pancancer_intersect["clinical"][cols] = (
    pancancer_intersect["clinical"][cols]
    .apply(pd.to_numeric)
)
# print(pancancer_intersect["clinical"].columns)

# Save pan-cancer data-set
tmp_path = wkd_path + "data/dict_pancancer_union_mutation.pickle.tmp"
final_path = wkd_path + "data/dict_pancancer_union_mutation.pickle"

with open(tmp_path, "wb") as f:
    pickle.dump(pancancer_intersect, f)
    f.flush()
    os.fsync(f.fileno())

os.replace(tmp_path, final_path) 
