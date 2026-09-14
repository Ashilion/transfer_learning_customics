import pyreadr
import pandas as pd
import glob
import numpy as np
import os
import pickle

wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"
data_path = "data/herrmann_data"

rdata_files = glob.glob(os.path.join(wkd_path, data_path, "*.RData"))

if not rdata_files:
    print(f"No .RData files found in {os.path.join(wkd_path, data_path)}")
else:
    print(f"Found {len(rdata_files)} cancer dataset(s): {[os.path.basename(f).replace('.RData','') for f in rdata_files]}")

for file_path in rdata_files:
    cancer_name = os.path.basename(file_path).replace(".RData", "")
    print(f"\nProcessing {cancer_name}...")

    try:
        res = pyreadr.read_r(file_path)
        df = res["dat"]

        regex = "clinical|time|status|bcr_patient_barcode"
        df = df.filter(regex=f"({regex})$")

        print(f"  Shape after filtering: {df.shape}")

        final_path = os.path.join(wkd_path, "data/clinical", f"{cancer_name}_clinical.pickle")
        tmp_path = f"{final_path}.tmp"

        with open(tmp_path, "wb") as f:
            pickle.dump(df, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, final_path)
        print(f"  Saved to {final_path}")

    except Exception as e:
        print(f"  ERROR processing {cancer_name}: {e}")

print("\nDone.")