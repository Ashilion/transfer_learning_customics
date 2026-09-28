import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import time
import optuna

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys
sys.path.append('..')
import utils.folds_utils as fold_utils
from utils.vvh_cv import vvh_cv

from custcox_utils import (
    fit_feature_selector, apply_feature_selector, build_customics_model,
    build_survival_array, fit_coxnet, evaluate_survival, fit_scalers, apply_scalers
)
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os

# Missing data utils ==========================================================================================


def simulate_missing_modalities(sample_ids, sources, missing_rate, repetition_seed):
    """
    Détermine, pour une répétition donnée (PAS le fold), quelle modalité unique
    est manquante pour chaque patient. Le même repetition_seed doit être utilisé
    pour les 5 folds d'une même répétition -> pattern identique garanti.

    sample_ids : tous les patients du cancer considéré (union train+test de la répétition)
    sources : ex. ['_rna', 'mirna', 'cnv', 'mutation']
    missing_rate : fraction de patients (0.3 / 0.5 / 0.7) avec une modalité manquante
    repetition_seed : seed = index de répétition (0..9), PAS l'outer_fold

    Retourne : dict sample_id -> nom de la modalité supprimée, ou None si complet.
    """
    rng = np.random.RandomState(repetition_seed)
    sample_ids_sorted = sorted(sample_ids)  # déterminisme indépendant de l'ordre d'entrée
    n_missing = int(round(missing_rate * len(sample_ids_sorted)))
    missing_patients = rng.choice(sample_ids_sorted, size=n_missing, replace=False)

    assignment = {sid: None for sid in sample_ids_sorted}
    for sid in missing_patients:
        assignment[sid] = sources[rng.randint(len(sources))]
    return assignment


def apply_missing_modalities(omics_df, assignment):
    """
    Zéro-impute les dataframes omiques selon `assignment` (à appliquer APRES
    feature selection + scaling, pour que 0 = valeur moyenne dans l'espace scalé)
    et construit le masque d'observation (1 = observé, 0 = manquant).

    omics_df : dict[str, pd.DataFrame] déjà scalé
    assignment : dict sample_id -> modalité manquante ou None

    Retourne : (omics_df_masked, modality_mask_df)
    modality_mask_df : DataFrame index=sample_id, colonnes=sources, valeurs 0/1
    """
    sources = list(omics_df.keys())
    masked = {src: df.copy() for src, df in omics_df.items()}
    rows = []

    for sid in omics_df[sources[0]].index:
        missing_src = assignment.get(sid, None)
        row = {'sample_id': sid}
        for src in sources:
            observed = 1
            if src == missing_src:
                masked[src].loc[sid, :] = 0.0
                observed = 0
            row[src] = observed
        rows.append(row)

    modality_mask_df = pd.DataFrame(rows).set_index('sample_id')[sources]
    return masked, modality_mask_df

# ==========================================================================================
