import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kruskal, false_discovery_control
from sklearn.feature_selection import f_regression
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")


def extract_embeddings(model, omics_df):
    return model.get_latent_representation(omics_df)


# ──────────────────────────────────────────────────────────────────────────
# 1. Kruskal-Wallis (variables catégorielles / discrètes)
# ──────────────────────────────────────────────────────────────────────────
def latent_kruskal(Z, clinical):
    res = {}
    for col in clinical.columns:
        if clinical[col].dtype == "O" or clinical[col].nunique() < 10:
            p = []
            vals = clinical[col]
            for d in range(Z.shape[1]):
                groups = [Z.loc[vals[vals == g].index, d].values
                          for g in pd.unique(vals.dropna())]
                groups = [g for g in groups if len(g) > 1]
                _, pv = kruskal(*groups) if len(groups) > 1 else (None, np.nan)
                p.append(pv)
            res[col] = p
    return pd.DataFrame(res, index=[f"z{i}" for i in range(Z.shape[1])])


# ──────────────────────────────────────────────────────────────────────────
# 2. f_regression scikit-learn (variables continues)
# ──────────────────────────────────────────────────────────────────────────
def latent_f_regression(Z, clinical):
    res = {}
    num = clinical.select_dtypes(include=np.number)

    for col in num.columns:
        if num[col].nunique() > 10:  # variable continue
            vals = num[col]
            ok = vals.notna()

            _, pvals = f_regression(
                Z.loc[ok].values,
                vals.loc[ok].values.astype(float)
            )
            res[col] = pvals

    return pd.DataFrame(res, index=[f"z{i}" for i in range(Z.shape[1])])


# ──────────────────────────────────────────────────────────────────────────
# 3. CoxPH (lifelines) — une dimension latente à la fois vs survie
# ──────────────────────────────────────────────────────────────────────────
def latent_coxph(Z, clinical, duration_col="time", event_col="status"):
    ok = clinical[[duration_col, event_col]].notna().all(axis=1)
    common_idx = Z.index.intersection(clinical.loc[ok].index)

    pvals = []
    for d in range(Z.shape[1]):
        df = pd.DataFrame({
            "z": Z.loc[common_idx, d].values,
            duration_col: clinical.loc[common_idx, duration_col].astype(float).values,
            event_col: clinical.loc[common_idx, event_col].astype(float).values,
        })

        cph = CoxPHFitter()
        try:
            cph.fit(df, duration_col=duration_col, event_col=event_col)
            p = cph.summary.loc["z", "p"]
        except Exception:
            p = np.nan

        pvals.append(p)

    return pd.DataFrame({"CoxPH": pvals}, index=[f"z{i}" for i in range(Z.shape[1])])


# ──────────────────────────────────────────────────────────────────────────
# 4. Correction multiple : Benjamini-Hochberg (FDR)
# ──────────────────────────────────────────────────────────────────────────
def correct_pvalues_fdr(pvals_df, method="bh"):
    """
    Corrige les p-values d'un DataFrame avec la methode Benjamini-Hochberg
    via scipy.stats.false_discovery_control.

    - Les NaN sont ignores (masques) puis remis a leur place apres correction.
    - La correction est appliquee sur l'ensemble aplati du DataFrame
      `correct_pvalues_fdr_per_column`.
    """
    values = pvals_df.values.astype(float)
    flat = values.ravel()

    mask = ~np.isnan(flat)
    corrected = np.full_like(flat, np.nan)

    if mask.sum() > 0:
        corrected[mask] = false_discovery_control(flat[mask], method=method)

    corrected = corrected.reshape(values.shape)
    return pd.DataFrame(corrected, index=pvals_df.index, columns=pvals_df.columns)


def correct_pvalues_fdr_per_column(pvals_df, method="bh"):
    """
    Variante : correction BH appliquee independamment a chaque colonne
    (chaque variable clinique testee separement sur les dimensions latentes).
    """
    out = pvals_df.copy().astype(float)
    for col in out.columns:
        vals = out[col].values
        mask = ~np.isnan(vals)
        if mask.sum() > 0:
            vals[mask] = false_discovery_control(vals[mask], method=method)
        out[col] = vals
    return out


# ──────────────────────────────────────────────────────────────────────────
# Plot : les 3 heatmaps cote a cote (-log10(p))
# ──────────────────────────────────────────────────────────────────────────
import seaborn as sns

def plot_three_tests(kw, freg, cox, save_dir, suffix=""):

    kw_log = -np.log10(kw.astype(float))
    freg_log = -np.log10(freg.astype(float))
    cox_log = -np.log10(cox.astype(float))

    dfs = [kw_log, freg_log, cox_log]

    titles = [
        f"Kruskal-Wallis{suffix}\n-log10(p)",
        f"f_regression{suffix}\n-log10(p)",
        f"CoxPH{suffix}\n-log10(p)"
    ]

    # Meme echelle de couleurs pour les trois heatmaps
    finite_max = [df.replace([np.inf, -np.inf], np.nan).max().max() for df in dfs]
    vmax = np.nanmax(finite_max)

    widths = [max(3, df.shape[1]) for df in dfs]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(sum(widths) + 2, max(5, kw.shape[0] * 0.45)),
        gridspec_kw={"width_ratios": widths}
    )

    for ax, df, title in zip(axes, dfs, titles):
        df = df.rename(columns={
            "venous_invasion_YES_clinical": "Venous invasion",
            "stage_event_pathologic_stage_Stage.III_clinical": "Stage III",
            "lymphatic_invasion_YES_clinical": "Lymphatic invasion",
            "stage_event_pathologic_stage_Stage.II_clinical": "Stage II",
            "gender_MALE_clinical": "Male",
            "status": "Status",
            "age_clinical": "Age",
            "number_of_lymphnodes_positive_by_he_clinical": "Positive lymph nodes",
            "time": "Time",
            "hemoglobin_result_Normal_clinical" : "hemoglobin normal",
            "white_cell_count_result_Normal_clinical": "white cell count",  
            "laterality_Right_clinical": "laterality",  
            "stage_event_pathologic_stage_Stage.III_clinical":"stage III",
        })
        
        ax.grid(False)
        sns.heatmap(
            df,
            ax=ax,
            cmap="viridis",
            mask=df < 1.301,
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            square=False,
            annot=df.shape[0] <= 15 and df.shape[1] <= 10,
            fmt=".1f",
            annot_kws={"size": 8}
        )

        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_yticks(np.arange(df.shape[0]) + 0.5)

        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticklabels(df.index)

    plt.tight_layout()

    fname = "KIRP_kruskal_fregression_coxph_sidebyside"
    if suffix:
        fname += suffix.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(
        f"{save_dir}/{fname}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────
def run_latent_statistics(model, omics_df, clinical_df, save_dir="latent_statistics",
                           duration_col="time", event_col="status",
                           fdr_method="bh", fdr_per_column=False):
    os.makedirs(save_dir, exist_ok=True)

    Z_arr = extract_embeddings(model, omics_df)
    Z = pd.DataFrame(Z_arr, index=clinical_df.index)

    # Clinical sans les colonnes de survie, pour Kruskal et f_regression
    clinical_no_surv = clinical_df.drop(columns=[duration_col, event_col], errors="ignore")

    kw = latent_kruskal(Z, clinical_no_surv)
    freg = latent_f_regression(Z, clinical_no_surv)
    cox = latent_coxph(Z, clinical_df, duration_col=duration_col, event_col=event_col)

    # ── p-values brutes ────────────────────────────────────────────────
    kw.to_csv(f"{save_dir}/kruskal_pvalues.csv")
    freg.to_csv(f"{save_dir}/f_regression_pvalues.csv")
    cox.to_csv(f"{save_dir}/coxph_pvalues.csv")

    plot_three_tests(kw, freg, cox, save_dir)

    # ── p-values corrigees (Benjamini-Hochberg / FDR) ──────────────────
    correct_fn = correct_pvalues_fdr_per_column if fdr_per_column else correct_pvalues_fdr

    kw_adj = correct_fn(kw, method=fdr_method)
    freg_adj = correct_fn(freg, method=fdr_method)
    cox_adj = correct_fn(cox, method=fdr_method)

    kw_adj.to_csv(f"{save_dir}/kruskal_pvalues_fdr.csv")
    freg_adj.to_csv(f"{save_dir}/f_regression_pvalues_fdr.csv")
    cox_adj.to_csv(f"{save_dir}/coxph_pvalues_fdr.csv")

    plot_three_tests(kw_adj, freg_adj, cox_adj, save_dir, suffix=" (FDR)")

    print("Analysis complete.")
    return dict(
        Z=Z,
        kruskal=kw,
        f_regression=freg,
        coxph=cox,
        kruskal_fdr=kw_adj,
        f_regression_fdr=freg_adj,
        coxph_fdr=cox_adj,
    )


if __name__ == "__main__":
    import pickle
    import torch
    import sys
    sys.path.append("..")

    from sklearn.model_selection import KFold
    from src.network.customics import CustOMICS
    from src.tools.utils import get_sub_omics_df
    from custcox_utils import fit_feature_selector, apply_feature_selector
    from analyse_latent_space_save_model import build_model   # reutilise la fonction du 2e script

    CANCER        = "KIRP"
    OUTER_FOLD    = 0
    OUTER_SPLITS  = 5
    NB_FEATURES   = 5000
    # CKPT_PATH     = f"models/{CANCER}_fold{OUTER_FOLD}_trial0_best.pt"
    # CKPT_PATH     = f"models/{CANCER}_supridge_fold{OUTER_FOLD}_best.pt"
    CKPT_PATH     = f"models/{CANCER}_fold{OUTER_FOLD}_best.pt"
    
    DATA_PATH     = "../data/dict_pancancer_union_mutation.pickle"
    CLINICAL_DIR  = "../data/clinical"

    # ── 1. Donnees omiques (pancancer pickle) ─────────────────────────────────
    print("Chargement du pancancer pickle")
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

    # ── 2. Donnees cliniques EXTERNES (ne vont pas dans le modele) ───────────
    print("Chargement des donnees cliniques externes")
    clin_path = f"{CLINICAL_DIR}/{CANCER}_clinical.pickle"
    with open(clin_path, "rb") as f:
        df_clin_raw = pickle.load(f)

    clinical_ext = df_clin_raw[
        list(set(df_clin_raw.columns) - {"time", "bcr_patient_barcode", "status"})
    ]
    offset = clinical_survival.index[0] - clinical_ext.index[0]

    # ── 3. Split outer fold 0 (reproduit exactement le split du training) ─────
    print(f"Split outer fold {OUTER_FOLD}")
    outer_cv  = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=0)
    splits    = list(outer_cv.split(lt_samples))
    train_idx, test_idx = splits[OUTER_FOLD]

    samples_train = [lt_samples[i] for i in train_idx]
    samples_test  = [lt_samples[i] for i in test_idx]
    samples_all   = lt_samples

    # ── 4. Feature selection (sur train uniquement, comme pendant l'entrainement)
    print(f"Feature selection (top {NB_FEATURES})")
    omics_train_raw = get_sub_omics_df(omics_df_full, samples_train)
    selector        = fit_feature_selector(omics_train_raw, nbFeatures=NB_FEATURES)

    omics_all   = apply_feature_selector(get_sub_omics_df(omics_df_full, samples_all), selector)
    omics_train = apply_feature_selector(omics_train_raw, selector)
    
    # ── 5. Charger le modele sauvegarde ───────────────────────────────────────
    print(f"Chargement du modele : {CKPT_PATH}")
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
    print("Modele charge.")

    # ── 6. Construire le clinical_df aligne pour l'analyse ───────────────────
    samples_ext_idx = [idx - offset for idx in samples_all]
    clinical_ext_aligned = clinical_ext.loc[samples_ext_idx].copy()
    clinical_ext_aligned.index = samples_all

    clinical_full = clinical_ext_aligned.join(
        clinical_survival.loc[samples_all, ["status", "time"]]
    )
    print("dbug", clinical_full.index)

    # ── 7. Lancer le pipeline d'analyse ──────────────────────────────────────
    results = run_latent_statistics(
        model         = model,
        omics_df      = omics_all,
        clinical_df   = clinical_full,
    )

    print("Summary")
    print(results)