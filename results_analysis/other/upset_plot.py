import pandas as pd
import numpy as np
import os
import pickle
from upsetplot import UpSet, from_indicators,from_contents,from_memberships
from collections import defaultdict
import matplotlib.pyplot as plt

path = "../data/dict_pancancer.pickle"

with open("../data/dict_pancancer.pickle", "rb") as f:
    pancancer = pickle.load(f)

if pancancer is None:
    print("Error reading file")

blocks = pancancer["BRCA"].keys()

columns_dict = defaultdict(lambda: defaultdict(list))
for ca in pancancer.keys():
    for bl in blocks:
        if bl == "mutation":
            continue
        else:
            columns_dict[bl][ca].append(pancancer[ca][bl].columns)

#for each modality do upsetplot
for bl in blocks:
    if bl != "mutation":
        data = from_contents(columns_dict[bl])
        print(data)
        ax_dict = UpSet(data, subset_size="count").plot()
        plt.title("upset plot for " + bl)
        plt.show()