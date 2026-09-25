import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, r2_score

warnings.filterwarnings("ignore")


def extract_embeddings(model, omics_df):
    return model.get_latent_representation(omics_df)

def latent_pearson(Z, clinical):
    res = {}
    num = clinical.select_dtypes(include=np.number)

    for col in num.columns:
        if num[col].nunique() > 10:   # variable continue
            r = []
            vals = num[col]
            ok = vals.notna()

            for d in range(Z.shape[1]):
                coef, _ = pearsonr(
                    Z.loc[ok, d].values,
                    vals.loc[ok].values.astype(float)
                )
                r.append(coef)

            res[col] = r

    return pd.DataFrame(res, index=[f"z{i}" for i in range(Z.shape[1])])

def latent_spearman(Z, clinical):
    res = {}
    num = clinical.select_dtypes(include=np.number)

    for col in num.columns:
        if num[col].nunique() > 10:
            r = []
            vals = num[col]
            ok = vals.notna()

            for d in range(Z.shape[1]):
                coef, _ = spearmanr(
                    Z.loc[ok, d].values,
                    vals.loc[ok].values.astype(float)
                )
                r.append(coef)

            res[col] = r

    return pd.DataFrame(res, index=[f"z{i}" for i in range(Z.shape[1])])

def latent_anova(Z, clinical):
    res = {}
    for col in clinical.columns:
        if clinical[col].dtype == "O" or clinical[col].nunique() < 10:
            p = []
            vals = clinical[col]
            for d in range(Z.shape[1]):
                groups = [Z.loc[vals[vals==g].index, d].values 
                          for g in pd.unique(vals.dropna())]
                groups = [g for g in groups if len(g) > 1]
                _,pv = f_oneway(*groups) if len(groups) > 1 else (None, np.nan)
                p.append(pv)
            res[col] = p
    return pd.DataFrame(res, index=[f"z{i}" for i in range(Z.shape[1])])


def latent_kruskal(Z, clinical):
    res = {}
    for col in clinical.columns:
        if clinical[col].dtype == "O" or clinical[col].nunique() < 10:
            p = []
            vals = clinical[col]
            for d in range(Z.shape[1]):
                groups = [Z.loc[vals[vals==g].index, d].values 
                          for g in pd.unique(vals.dropna())]
                groups = [g for g in groups if len(g) > 1]
                _, pv = kruskal(*groups) if len(groups) > 1 else (None, np.nan)
                p.append(pv)
            res[col] = p
    return pd.DataFrame(res, index=[f"z{i}" for i in range(Z.shape[1])])


def latent_mutual_information(Z, clinical):
    out = {}
    for col in clinical.columns:
        y = clinical[col]
        ok = ~pd.isna(y)
        if y.dtype == "O" or y.nunique() < 10:
            out[col] = mutual_info_classif(Z.loc[ok].values, pd.Categorical(y[ok]).codes)
        else:
            out[col] = mutual_info_regression(Z.loc[ok].values, y[ok].astype(float))
    return pd.DataFrame(out, index=[f"z{i}" for i in range(Z.shape[1])])


def linear_probe(Z, y):
    ok = ~pd.isna(y)
    common_idx = Z.index.intersection(y[ok].index)
    X = Z.loc[common_idx].values
    y_clean = y.loc[common_idx]
    if y_clean.dtype == "O" or y_clean.nunique() < 10:
        y_clean = pd.Categorical(y_clean).codes
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        return cross_val_score(model, X, y_clean, cv=cv, scoring="accuracy").mean()
    cv = KFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), Ridge())
    return cross_val_score(model, X, y_clean.astype(float), cv=cv,
                           scoring=make_scorer(r2_score)).mean()


def latent_cca(Z, clinical, n_components=2):
    num = clinical.select_dtypes(include=np.number).dropna(axis=1)
    num_clean = num.dropna()
    common_idx = Z.index.intersection(num_clean.index)

    X = StandardScaler().fit_transform(Z.loc[common_idx].values)
    Y = StandardScaler().fit_transform(num_clean.loc[common_idx].values)

    cca = CCA(n_components=min(n_components, X.shape[1], Y.shape[1]))
    Xc, Yc = cca.fit_transform(X, Y)
    corr = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(Xc.shape[1])]
    return corr, cca


def plot_heatmap(df, title, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]), max(4, df.shape[0]/2)))

    im = ax.imshow(
        df.values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)

    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_linear_probe(probe_scores, save_dir):
    scores = pd.Series(probe_scores).sort_values(ascending=True)
    colors = ["steelblue" if v > 0 else "salmon" for v in scores]
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(scores) * 0.4)))
    bars = ax.barh(scores.index, scores.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Score (accuracy ou R²)")
    ax.set_title("Linear Probe — prédictibilité depuis l'espace latent")
    
    for bar, val in zip(bars, scores.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=6)
    
    plt.tight_layout()
    fig.savefig(f"{save_dir}/linear_probe_barplot.png", dpi=150)
    plt.close(fig)

def plot_cca(corr, cca, Z, clinical, save_dir):
    # 1. Bar plot des corrélations canoniques
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"CC{i+1}" for i in range(len(corr))], corr, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Corrélation canonique")
    ax.set_title("CCA — corrélations par composante canonique")
    for i, v in enumerate(corr):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/cca_correlations_barplot.png", dpi=150)
    plt.close(fig)

    # 2. Scatter plot CC1 vs CC2
    num = clinical.select_dtypes(include=np.number).dropna(axis=1)
    num_clean = num.dropna()
    common_idx = Z.index.intersection(num_clean.index)
    X = StandardScaler().fit_transform(Z.loc[common_idx].values)
    Y = StandardScaler().fit_transform(num_clean.loc[common_idx].values)
    Xc, Yc = cca.transform(X, Y)

    fig, axes = plt.subplots(1, len(corr), figsize=(5 * len(corr), 4))
    if len(corr) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.scatter(Xc[:, i], Yc[:, i], alpha=0.5, s=20, color="steelblue")
        ax.set_xlabel(f"Latent CC{i+1}")
        ax.set_ylabel(f"Clinical CC{i+1}")
        ax.set_title(f"CC{i+1} — r={corr[i]:.2f}")
        # Droite de régression
        m, b = np.polyfit(Xc[:, i], Yc[:, i], 1)
        x_line = np.linspace(Xc[:, i].min(), Xc[:, i].max(), 100)
        ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/cca_scatter.png", dpi=150)
    plt.close(fig)

def run_latent_statistics(model,omics_df,clinical_df,save_dir="latent_statistics"):
    os.makedirs(save_dir,exist_ok=True)

    Z_arr = extract_embeddings(model, omics_df)
    Z = pd.DataFrame(Z_arr, index=clinical_df.index)

    an=latent_anova(Z,clinical_df)
    kw=latent_kruskal(Z,clinical_df)
    mi=latent_mutual_information(Z,clinical_df)
    # an.to_csv(f"{save_dir}/anova_pvalues.csv")
    kw.to_csv(f"{save_dir}/kruskal_pvalues.csv")
    mi.to_csv(f"{save_dir}/mutual_information.csv")
    plot_heatmap(-np.log10(an),"ANOVA -log10(p)",f"{save_dir}/anova_heatmap.png")
    plot_heatmap(-np.log10(kw),"Kruskal -log10(p)",f"{save_dir}/kruskal_heatmap.png")
    plot_heatmap(mi,"Mutual Information",f"{save_dir}/mi_heatmap.png")

    pear = latent_pearson(Z, clinical_df)
    spear = latent_spearman(Z, clinical_df)

    pear.to_csv(f"{save_dir}/pearson_correlations.csv")
    spear.to_csv(f"{save_dir}/spearman_correlations.csv")

    plot_heatmap(
        pear,
        "Pearson correlation",
        f"{save_dir}/pearson_heatmap.png"
    )

    plot_heatmap(
        spear,
        "Spearman correlation",
        f"{save_dir}/spearman_heatmap.png"
    )

    probe={c:linear_probe(Z,clinical_df[c]) for c in clinical_df.columns}
    pd.Series(probe,name="score").to_csv(f"{save_dir}/linear_probe_scores.csv")
    plot_linear_probe(probe, save_dir)
    
    corr, cca = latent_cca(Z, clinical_df)
    # pd.Series(corr, name="CCA correlation").to_csv(f"{save_dir}/cca_correlations.csv")
    plot_cca(corr, cca, Z, clinical_df, save_dir)
    print("Analysis complete.")
    return dict(
        Z=Z,
        anova=an,
        kruskal=kw,
        mi=mi,
        pearson=pear,
        spearman=spear,
        probe=probe,
        cca=corr,
    )


if __name__ == "__main__":
    import pickle
    import torch
    import sys
    sys.path.append("..")
    sys.path.append("../..")
    from sklearn.model_selection import KFold
    from src.network.customics import CustOMICS
    from src.tools.utils import get_sub_omics_df
    from custcox_utils import fit_feature_selector, apply_feature_selector
    from analyse_latent_space_save_model import build_model   # réutilise la fonction du 2e script
 
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
 
    # ── 8. Lancer le pipeline d'analyse ──────────────────────────────────────
    results = run_latent_statistics(
        model         = model,
        omics_df      = omics_all,
        clinical_df   = clinical_full,
    )