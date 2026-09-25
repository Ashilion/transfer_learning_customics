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


def extract_embeddings(model, omics_df):
    """
    Retourne Z (n_samples, latent_dim) depuis un modèle CustOMICS entraîné.

    Parameters
    ----------
    model    : CustOMICS   instance entraînée
    omics_df : dict        {source_name: pd.DataFrame}
    """
    Z = model.get_latent_representation(omics_df)   # numpy (n, d)
    return Z


# ===================================================================================
# 1.  RÉDUCTION DE DIMENSIONNALITÉ
# ===================================================================================

def compute_reductions(Z, random_state=42, umap_n_neighbors=15,
                       umap_min_dist=0.1, tsne_perplexity=30):
    """
    Calcule PCA, UMAP et t-SNE sur l'espace latent Z.

    Returns
    -------
    dict avec clés 'pca', 'umap', 'tsne', chacune de shape (n, 2)
    + 'pca_var' : variance expliquée par composante
    """
    print("PCA …")
    pca_full = PCA(random_state=random_state)
    pca_full.fit(Z)
    pca_2d = pca_full.transform(Z)[:, :2]

    print("UMAP …")
    reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist, random_state=random_state)
    umap_2d = reducer.fit_transform(Z)

    print("t-SNE …")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity,
                random_state=random_state, max_iter=1000)
    tsne_2d = tsne.fit_transform(Z)

    print("Réductions terminées.")
    return {
        "pca":     pca_2d,
        "umap":    umap_2d,
        "tsne":    tsne_2d,
        "pca_var": pca_full.explained_variance_ratio_,
    }


def plot_pca_variance(pca_var, n_components=20, save_path=None):
    """Scree plot de la variance expliquée par PCA."""
    n = min(n_components, len(pca_var))
    cumvar = np.cumsum(pca_var[:n]) * 100
    indvar = pca_var[:n] * 100

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(range(1, n + 1), indvar, color="#5DCAA5", alpha=0.8, label="Variance individuelle")
    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), cumvar, color="#D85A30", marker="o",
             markersize=4, linewidth=2, label="Variance cumulée")
    ax2.axhline(80, linestyle="--", color="#888780", linewidth=1)
    ax2.set_ylabel("Variance cumulée (%)", color="#D85A30")
    ax1.set_xlabel("Composante principale")
    ax1.set_ylabel("Variance expliquée (%)", color="#5DCAA5")
    ax1.set_title("PCA — variance expliquée")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    _save_or_show(fig, save_path, "pca_variance.png")


# ===================================================================================
# 2.  COLORATION PAR VARIABLES CLINIQUES
# ===================================================================================

def plot_latent_colored_by_clinical(
    reductions,
    clinical_df,
    samples,
    clinical_vars,
    methods=("pca", "umap", "tsne"),
    save_dir=".",
    figsize_per_panel=(4, 4),
):
    """
    Pour chaque variable clinique x méthode de réduction, trace un scatter
    coloré par la valeur clinique.

    Parameters
    ----------
    reductions   : dict retourné par compute_reductions()
    clinical_df  : pd.DataFrame   index = sample IDs, colonnes = variables cliniques
    samples      : list           liste ordonnée des sample IDs correspondant à Z
    clinical_vars: list[dict]
        Chaque dict : {
            'col'   : str,              # nom de la colonne dans clinical_df
            'label' : str,              # titre lisible
            'type'  : 'cat' | 'cont',  # catégorielle ou continue
        }
    methods      : tuple   sous-ensemble de ('pca', 'umap', 'tsne')
    """
    method_labels = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}
    clin_aligned = clinical_df.loc[samples]   # aligne sur l'ordre de Z

    n_methods = len(methods)
    n_vars    = len(clinical_vars)

    fig, axes = plt.subplots(
        n_vars, n_methods,
        figsize=(figsize_per_panel[0] * n_methods, figsize_per_panel[1] * n_vars),
        squeeze=False,
    )

    for row, var_cfg in enumerate(clinical_vars):
        col   = var_cfg["col"]
        label = var_cfg.get("label", col)
        vtype = var_cfg.get("type", "cat")
        values = clin_aligned[col].values

        for c_idx, method in enumerate(methods):
            ax = axes[row][c_idx]
            coords = reductions[method]

            if vtype == "cat":
                categories = pd.Categorical(values)
                codes = categories.codes.astype(float)
                n_cat = len(categories.categories)
                cmap = plt.get_cmap("tab10", n_cat)
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=codes, cmap=cmap, s=18, alpha=0.8,
                                vmin=-0.5, vmax=n_cat - 0.5)
                # Légende catégorielle
                handles = [
                    Patch(color=cmap(i / max(n_cat - 1, 1)), label=str(cat))
                    for i, cat in enumerate(categories.categories)
                ]
                ax.legend(handles=handles, fontsize=7, title=label,
                          title_fontsize=7, loc="best",
                          markerscale=0.8, handlelength=1)
            else:
                # Variable continue
                mask_valid = ~pd.isna(values)
                sc = ax.scatter(coords[mask_valid, 0], coords[mask_valid, 1],
                                c=values[mask_valid].astype(float),
                                cmap="viridis", s=18, alpha=0.8)
                plt.colorbar(sc, ax=ax, shrink=0.7, label=label)

            ax.set_xlabel(f"{method_labels[method]} 1", fontsize=9)
            ax.set_ylabel(f"{method_labels[method]} 2", fontsize=9)
            ax.set_title(f"{method_labels[method]} — {label}", fontsize=10)
            ax.tick_params(labelsize=7)

    plt.suptitle("Espace latent coloré par variables cliniques", fontsize=12, y=1.02)
    plt.tight_layout()
    path = f"{save_dir}/latent_clinical_coloring.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f" Sauvegardé : {path}")
    plt.show()


# ===================================================================================
# 3.  CLUSTERING + SILHOUETTE + KAPLAN-MEIER
# ===================================================================================

def optimal_kmeans(Z, k_range=range(2, 9), random_state=42):
    """
    Recherche du k optimal par Silhouette score sur l'espace latent complet.

    Returns
    -------
    best_k      : int
    labels_best : np.array  (n_samples,)
    silhouette_scores : dict {k: score}
    """
    scores = {}
    labels_all = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(Z)
        s = silhouette_score(Z, lbl)
        scores[k] = s
        labels_all[k] = lbl
        print(f"   k={k}  →  silhouette = {s:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nMeilleur k : {best_k}  (silhouette = {scores[best_k]:.4f})")
    return best_k, labels_all[best_k], scores, labels_all


def plot_silhouette_scores(silhouette_scores, best_k, save_path=None):
    """Bar plot du Silhouette score en fonction de k."""
    ks = list(silhouette_scores.keys())
    vals = [silhouette_scores[k] for k in ks]
    colors = ["#D85A30" if k == best_k else "#9FE1CB" for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ks, vals, color=colors, edgecolor="#5F5E5A", linewidth=0.5)
    ax.axhline(silhouette_scores[best_k], linestyle="--",
               color="#D85A30", linewidth=1.2, label=f"Meilleur k={best_k}")
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Sélection du k optimal (K-Means sur espace latent)")
    ax.set_xticks(ks)
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path, "silhouette_scores.png")


def plot_silhouette_per_sample(Z, labels, best_k, save_path=None):
    """
    Silhouette plot par échantillon (un bloc coloré par cluster).
    Utile pour détecter les clusters mal séparés.
    """
    sample_scores = silhouette_samples(Z, labels)
    fig, ax = plt.subplots(figsize=(8, max(4, len(Z) // 60)))
    y_lower = 10
    cmap = plt.get_cmap("tab10", best_k)

    for k in range(best_k):
        vals = np.sort(sample_scores[labels == k])
        size = len(vals)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=cmap(k), alpha=0.8, label=f"Cluster {k}")
        y_lower = y_upper + 10

    ax.axvline(silhouette_score(Z, labels), linestyle="--",
               color="black", linewidth=1.2, label="Moyenne globale")
    ax.set_xlabel("Silhouette score individuel")
    ax.set_ylabel("Échantillons (triés par cluster)")
    ax.set_title(f"Silhouette plot — k={best_k}")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    _save_or_show(fig, save_path, "silhouette_per_sample.png")


def plot_umap_clusters(reductions, labels, best_k, save_path=None):
    """UMAP coloré par cluster K-Means."""
    coords = reductions["umap"]
    cmap = plt.get_cmap("tab10", best_k)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(best_k):
        mask = labels == k
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(k)], s=20, alpha=0.8, label=f"Cluster {k}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP — clusters K-Means (k={best_k})")
    ax.legend(fontsize=8, markerscale=1.2)
    plt.tight_layout()
    _save_or_show(fig, save_path, "umap_clusters.png")


def kaplan_meier_by_cluster(
    labels,
    samples,
    clinical_df,
    event_col,
    time_col,
    save_path=None,
    title="Kaplan-Meier par cluster latent",
    xlim=None,
):
    """
    Trace les courbes KM pour chaque cluster + p-value (log-rank multivarié).

    Parameters
    ----------
    labels      : np.array  (n_samples,)  — sorties de K-Means
    samples     : list      — IDs dans le même ordre que Z / labels
    clinical_df : pd.DataFrame avec colonnes event_col et time_col
    event_col   : str   colonne événement (0/1)
    time_col    : str   colonne durée
    """
    clin = clinical_df.loc[samples, [event_col, time_col]].copy()
    clin["cluster"] = labels

    best_k = len(np.unique(labels))
    cmap   = plt.get_cmap("tab10", best_k)

    # Log-rank multivarié
    mlr = multivariate_logrank_test(
        event_durations = clin[time_col].values.astype(float),
        groups     = clin["cluster"].values,
        event_observed = clin[event_col].values.astype(float),
    )
    p_val = mlr.p_value

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(best_k):
        mask = clin["cluster"] == k
        kmf  = KaplanMeierFitter(label=f"Cluster {k}  (n={mask.sum()})")
        kmf.fit(
            durations       = clin.loc[mask, time_col].astype(float),
            event_observed  = clin.loc[mask, event_col].astype(float),
        )
        kmf.plot_survival_function(ax=ax, ci_show=True,
                                   color=cmap(k), linewidth=2)

    if xlim:
        ax.set_xlim(0, xlim)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Probabilité de survie")
    ax.set_title(f"{title}\np-value log-rank = {p_val:.3g}", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_path, "km_clusters.png")
    print(f"KM tracé  |  p-value log-rank multivarié = {p_val:.4g}")
    return p_val


# ===================================================================================
# PIPELINE COMPLET
# ===================================================================================

def run_latent_analysis(
    model,
    omics_df,
    clinical_df,
    samples,
    event_col,
    time_col,
    clinical_vars,
    k_range=range(2, 8),
    save_dir="figures_latent",
    km_xlim=None,
):
    """
    Lance les 3 points d'analyse sur l'espace latent.

    Parameters
    ----------
    model         : CustOMICS  entraîné
    omics_df      : dict  {source: pd.DataFrame}
    clinical_df   : pd.DataFrame  (données cliniques EXTERNES)
    samples       : list   IDs patients dans l'ordre de omics_df
    event_col     : str    colonne événement dans clinical_df
    time_col      : str    colonne durée dans clinical_df
    clinical_vars : list[dict]  cf. plot_latent_colored_by_clinical()
    k_range       : range  k à tester pour le clustering
    save_dir      : str    dossier de sortie
    km_xlim       : int|None   limite axe x des courbes KM

    Returns
    -------
    dict {
        'Z'          : np.array,
        'reductions' : dict,
        'labels'     : np.array,
        'best_k'     : int,
        'silhouette' : dict,
        'km_pvalue'  : float,
    }
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # ── 0. Embeddings ──────────────────────────────────────────────────────────
    print("\n═══ 0. Extraction des embeddings ═══")
    Z = extract_embeddings(model, omics_df)
    print(f"   Z shape : {Z.shape}")

    # ── 1. Réductions ─────────────────────────────────────────────────────────
    print("\n═══ 1. Réductions de dimensionnalité ═══")
    reductions = compute_reductions(Z)
    plot_pca_variance(reductions["pca_var"], save_path=save_dir)

    # ── 2. Coloration clinique ────────────────────────────────────────────────
    print("\n═══ 2. Coloration par variables cliniques ═══")
    plot_latent_colored_by_clinical(
        reductions   = reductions,
        clinical_df  = clinical_df,
        samples      = samples,
        clinical_vars= clinical_vars,
        methods      = ("pca", "umap", "tsne"),
        save_dir     = save_dir,
    )

    # ── 3. Clustering + Silhouette + KM ───────────────────────────────────────
    print("\n═══ 3. Clustering K-Means + Silhouette ═══")
    best_k, labels, sil_scores, _ = optimal_kmeans(Z, k_range=k_range)

    plot_silhouette_scores(sil_scores, best_k,
                           save_path=f"{save_dir}/silhouette_scores.png")
    plot_silhouette_per_sample(Z, labels, best_k,
                               save_path=f"{save_dir}/silhouette_per_sample.png")
    plot_umap_clusters(reductions, labels, best_k,
                       save_path=f"{save_dir}/umap_clusters.png")

    print("\n═══ 3b. Kaplan-Meier par cluster latent ═══")
    p_val = kaplan_meier_by_cluster(
        labels      = labels,
        samples     = samples,
        clinical_df = clinical_df,
        event_col   = event_col,
        time_col    = time_col,
        save_path   = f"{save_dir}/km_clusters.png",
        xlim        = km_xlim,
    )

    print("\nAnalyse complète terminée.")
    return dict(
        Z          = Z,
        reductions = reductions,
        labels     = labels,
        best_k     = best_k,
        silhouette = sil_scores,
        km_pvalue  = p_val,
    )


# ===================================================================================
# HELPERS
# ===================================================================================

def _save_or_show(fig, save_path, default_name):
    if save_path:
        path = save_path if save_path.endswith(".png") else f"{save_path}/{default_name}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Sauvegardé : {path}")
    else:
        plt.show()
    plt.close(fig)


# ===================================================================================

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