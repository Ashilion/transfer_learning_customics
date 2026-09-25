import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import pickle
import torch
import sys
sys.path.append("..")
sys.path.append("../..")

from sklearn.model_selection import KFold
from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df
from custcox_utils import fit_feature_selector, apply_feature_selector

from analyse_latent_space_save_model import build_model,run_latent_analysis
from analyse_latent_space_statistics import run_latent_statistics
from analyse_multiomics_latent.py import run_extended_analysis

if __name__ == "__main__":
   
 
    CANCER        = "COAD"
    OUTER_FOLD    = 0
    OUTER_SPLITS  = 5
    NB_FEATURES   = 5000
    CKPT_PATH     = f"models/{CANCER}_fold{OUTER_FOLD}_trial0_best.pt"
    DATA_PATH     = "../data/dict_pancancer_union_mutation.pickle"
    CLINICAL_DIR  = "../data/clinical"
 
    # ── 1. Données omiques (pancancer pickle) ─────────────────────────────────
    print("Chargement du pancancer pickle …")
    with open(DATA_PATH, "rb") as f:
        pancancer = pickle.load(f)
 
    # Filtrer sur COAD
    mask = pancancer["clinical"]["cancer_type"] == CANCER
    data = {name: df[mask] for name, df in pancancer.items()}
 
    clinical_survival = data["clinical"]   # contient 'status' et 'time'
 
    omics_df_full = {
        "protein":  data["_rna"],
        "gene_exp": data["mirna"],
        "methyl":   data["cnv"],
        "mutation": data["mutation"],
    }
    sources    = list(omics_df_full.keys())
    lt_samples = list(clinical_survival.index)
 
    # ── 2. Données cliniques EXTERNES (ne vont pas dans le modèle) ───────────
    # Contient les covariables cliniques riches (stade, âge, type histologique…)
    print("Chargement des données cliniques externes …")
    clin_path = f"{CLINICAL_DIR}/{CANCER}_clinical.pickle"
    with open(clin_path, "rb") as f:
        df_clin_raw = pickle.load(f)
 
    # On retire les colonnes de survie déjà présentes dans clinical_survival
    clinical_ext = df_clin_raw[
        list(set(df_clin_raw.columns) - {"time", "bcr_patient_barcode", "status"})
    ]
    # Aligner l'index : clinical_ext peut avoir un offset d'index vs clinical_survival
    offset = clinical_survival.index[0] - clinical_ext.index[0]
 
    # ── 3. Split outer fold 0 (reproduit exactement le split du training) ─────
    print(f"Split outer fold {OUTER_FOLD} …")
    outer_cv  = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=0)
    splits    = list(outer_cv.split(lt_samples))
    train_idx, test_idx = splits[OUTER_FOLD]
 
    samples_train = [lt_samples[i] for i in train_idx]
    samples_test  = [lt_samples[i] for i in test_idx]
    # On analyse sur TOUS les patients (train + test) pour avoir une vue globale
    samples_all   = lt_samples
 
    # ── 4. Feature selection (sur train uniquement, comme pendant l'entraînement)
    print(f"Feature selection (top {NB_FEATURES}) …")
    omics_train_raw = get_sub_omics_df(omics_df_full, samples_train)
    selector        = fit_feature_selector(omics_train_raw, nbFeatures=NB_FEATURES)
 
    # Appliquer le selector sur tous les patients
    omics_all   = apply_feature_selector(get_sub_omics_df(omics_df_full, samples_all), selector)
    omics_train = apply_feature_selector(omics_train_raw, selector)
 
    # ── 5. Charger le modèle sauvegardé ───────────────────────────────────────
    print(f"Chargement du modèle : {CKPT_PATH} …")
    device = torch.device("cpu")
    ckpt   = torch.load(CKPT_PATH, map_location=device)
 
    model = build_model(
        omics_data   = omics_train,
        sources      = sources,
        params       = ckpt["params"],
        device       = device,
        unsupervised = True,
        switch_epoch = ckpt["switch_epoch"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval_all()
    print("Modèle chargé.")
 
    # ── 6. Construire le clinical_df aligné pour l'analyse ───────────────────
    # On fusionne survie + covariables cliniques externes sur les patients communs
    samples_ext_idx = [idx - offset for idx in samples_all]   # index dans clinical_ext
    clinical_ext_aligned = clinical_ext.loc[samples_ext_idx].copy()
    clinical_ext_aligned.index = samples_all   # réindexer sur les IDs omics
 
    # Ajouter les colonnes de survie (status, time) depuis clinical_survival
    clinical_full = clinical_ext_aligned.join(
        clinical_survival.loc[samples_all, ["status", "time"]]
    )
 
    # ── 7. Définir les variables cliniques à visualiser ───────────────────────
    # Adapte les noms de colonnes à ce qui est réellement dans clinical_ext
    available_cols = set(clinical_full.columns)
    print(f"\nColonnes cliniques disponibles :\n  {sorted(available_cols)}\n")
 
    # Filtre automatique : ne garde que les colonnes qui existent réellement
    candidates = [
        {"col": "lymphatic_invasion_YES_clinical", "label": "lymphatic invasion", "type": "cat"},
        {"col": "number_of_lymphnodes_positive_by_he_clinical", "label": "number_of_lymphnodes", "type": "cont"},
        {"col": "age_clinical",  "label": "Âge diagnostic",      "type": "cont"},
        # {"col": "gender_MALE_clinical",                       "label": "Sexe",                "type": "cat"},
        # {"col": "venous_invasion_YES_clinical",            "label": "venous_invasion",       "type": "cat"}
    ]
    clinical_vars = [v for v in candidates if v["col"] in available_cols]
    clinical_vars += [
        {"col": "time",   "label": "Temps de survie (jours)", "type": "cont"},
        {"col": "status", "label": "Censure", "type": "cat"},
    ]
    if not clinical_vars:
        # Fallback : prendre les 4 premières colonnes catégorielles disponibles
        cat_cols = [c for c in clinical_full.columns
                    if clinical_full[c].dtype == object and c not in ("status", "time")]
        clinical_vars = [
            {"col": c, "label": c, "type": "cat"} for c in cat_cols[:4]
        ]
    print(f"Variables cliniques sélectionnées : {[v['col'] for v in clinical_vars]}")
 
    # ── 8. Lancer le pipeline d'analyse ──────────────────────────────────────
    results = run_latent_analysis(
        model         = model,
        omics_df      = omics_all,
        clinical_df   = clinical_full,
        samples       = samples_all,
        event_col     = "status",
        time_col      = "time",
        clinical_vars = clinical_vars,
        k_range       = range(2, 8),
        save_dir      = f"figures_latent_{CANCER}",
        km_xlim       = 2500,
    )
 
    # ── 9. Résultats ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Z shape    : {results['Z'].shape}")
    print(f"  best_k     : {results['best_k']}")
    print(f"  KM p-value : {results['km_pvalue']:.4g}")
    print(f"  Figures    → figures_latent_{CANCER}/")
    print(f"{'='*50}")

    # ── 10. Latent Statistics ──────────────────────────────────────
    results = run_latent_statistics(
        model         = model,
        omics_df      = omics_all,
        clinical_df   = clinical_full,
    )

    # ── 11. Omics Latent Space analysis ──────────────────────────────────────
    results = run_extended_analysis(
        model         = model,
        omics_df      = omics_all,
        clinical_df   = clinical_full,
        samples       = samples_all,
        event_col     = "status",
        time_col      = "time",
        save_dir      = f"figures_latent_multiomics_{CANCER}",
    )