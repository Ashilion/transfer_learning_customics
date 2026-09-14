import pandas as pd
from pathlib import Path

# Dossier contenant les fichiers
input_dir = Path("../CustOmics/results_mtl")
output_dir = Path("../results/")
output_dir.mkdir(exist_ok=True)

# Lire tous les fichiers des folds
dfs = []
for file in input_dir.glob("mtl_pancancer_deepsurv_fold*.csv"):
    df = pd.read_csv(file)

    # Si outer_fold n'est pas déjà dans le fichier, on le récupère depuis le nom
    if "outer_fold" not in df.columns:
        fold = int(file.stem.split("fold")[-1])
        df["outer_fold"] = fold

    dfs.append(df)

# Concaténer tous les folds
all_results = pd.concat(dfs, ignore_index=True)

# Renommer les colonnes
all_results = all_results.rename(columns={
    "outer_fold": "fold",
    "cindex": "cindex_default",
    "ibs": "graf"
})

# Sauvegarder un fichier par cancer
for cancer, df_cancer in all_results.groupby("cancer"):
    df_cancer = df_cancer.sort_values("fold")
    df_cancer.to_csv(output_dir / f"mtl_pancancer_deepsurv_{cancer}.csv", index=False)

print(f"{all_results['cancer'].nunique()} fichiers créés dans {output_dir}")