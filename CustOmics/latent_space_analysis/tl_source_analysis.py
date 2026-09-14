import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score,
)

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


def extract_embeddings(model, omics_df):
    """
    Retourne Z (n_samples, latent_dim) depuis un modèle CustOMICS entraîné.

    Parameters
    ----------
    model    : CustOMICS   instance entraînée (mono ou pan-cancer)
    omics_df : dict        {source_name: pd.DataFrame}
    """
    Z = model.get_latent_representation(omics_df)   # numpy (n, d)
    return Z


# ===========================================================================================================
# 1.  RÉDUCTION DE DIMENSIONNALITÉ
# ===========================================================================================================

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


# ===========================================================================================================
# 2.  COLORATION PAR VARIABLES (générique — sert pour cancer_type et/ou clinique)
# ===========================================================================================================

def plot_latent_colored_by_vars(
    reductions,
    meta_df,
    samples,
    variables,
    methods=("pca", "umap", "tsne"),
    save_dir=".",
    figsize_per_panel=(4, 4),
    filename="latent_coloring.png",
    suptitle="Espace latent coloré",
):
    """
    Pour chaque variable x méthode de réduction, trace un scatter coloré.

    Parameters
    ----------
    reductions : dict retourné par compute_reductions()
    meta_df    : pd.DataFrame   index = sample IDs, colonnes = variables à afficher
                 (peut être clinical_df, ou juste une Series 'cancer_type' passée
                 en DataFrame à une colonne)
    samples    : list   liste ordonnée des sample IDs correspondant à Z
    variables  : list[dict]  {'col':..., 'label':..., 'type': 'cat'|'cont'}
    """
    method_labels = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}
    meta_aligned = meta_df.loc[samples]

    n_methods = len(methods)
    n_vars    = len(variables)

    fig, axes = plt.subplots(
        n_vars, n_methods,
        figsize=(figsize_per_panel[0] * n_methods, figsize_per_panel[1] * n_vars),
        squeeze=False,
    )

    for row, var_cfg in enumerate(variables):
        col   = var_cfg["col"]
        label = var_cfg.get("label", col)
        vtype = var_cfg.get("type", "cat")
        values = meta_aligned[col].values

        for c_idx, method in enumerate(methods):
            ax = axes[row][c_idx]
            coords = reductions[method]

            if vtype == "cat":
                categories = pd.Categorical(values)
                codes = categories.codes.astype(float)
                n_cat = len(categories.categories)
                # tab20 pour supporter davantage de catégories (types de cancer)
                cmap_name = "tab10" if n_cat <= 10 else "tab20"
                cmap = plt.get_cmap(cmap_name, n_cat)
                ax.scatter(coords[:, 0], coords[:, 1],
                          c=codes, cmap=cmap, s=16, alpha=0.8,
                          vmin=-0.5, vmax=n_cat - 0.5)
                handles = [
                    Patch(color=cmap(i / max(n_cat - 1, 1)), label=str(cat))
                    for i, cat in enumerate(categories.categories)
                ]
                ax.legend(handles=handles, fontsize=6, title=label,
                          title_fontsize=7, loc="best",
                          markerscale=0.8, handlelength=1, ncol=1 if n_cat <= 12 else 2)
            else:
                mask_valid = ~pd.isna(values)
                sc = ax.scatter(coords[mask_valid, 0], coords[mask_valid, 1],
                                c=values[mask_valid].astype(float),
                                cmap="viridis", s=18, alpha=0.8)
                plt.colorbar(sc, ax=ax, shrink=0.7, label=label)

            ax.set_xlabel(f"{method_labels[method]} 1", fontsize=9)
            ax.set_ylabel(f"{method_labels[method]} 2", fontsize=9)
            ax.set_title(f"{method_labels[method]} — {label}", fontsize=10)
            ax.tick_params(labelsize=7)

    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    path = f"{save_dir}/{filename}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f" Sauvegardé : {path}")
    plt.show()


# ===========================================================================================================
# 3.  CLUSTERING + SILHOUETTE
# ===========================================================================================================

def optimal_kmeans(Z, k_range=range(2, 9), random_state=42):
    """Recherche du k optimal par Silhouette score sur l'espace latent complet."""
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
    ks = list(silhouette_scores.keys())
    vals = [silhouette_scores[k] for k in ks]
    colors = ["#D85A30" if k == best_k else "#9FE1CB" for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ks, vals, color=colors, edgecolor="#5F5E5A", linewidth=0.5)
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


# ===========================================================================================================
# 4.  CLUSTERS  ↔  TYPE DE CANCER 
# ===========================================================================================================

def cluster_vs_cancer_type(labels, samples, cancer_types, save_path=None):
    """
    Mesure si les clusters K-Means (sur l'espace latent) récupèrent
    simplement les types de cancer, ou capturent une structure transversale.

    - Tableau croisé cluster x cancer_type (heatmap, normalisé par cluster)
    - ARI (Adjusted Rand Index) et NMI (Normalized Mutual Info) entre
      labels de cluster et labels de cancer_type

    Parameters
    ----------
    labels       : np.array   labels K-Means (n_samples,)
    samples      : list       IDs dans le même ordre que labels
    cancer_types : pd.Series  index = sample IDs, valeurs = type de cancer
                   (ou array aligné sur `samples`)

    Returns
    -------
    dict {'ari':..., 'nmi':..., 'crosstab': pd.DataFrame}
    """
    if isinstance(cancer_types, pd.Series):
        ct = cancer_types.loc[samples].values
    else:
        ct = np.asarray(cancer_types)

    ari = adjusted_rand_score(ct, labels)
    nmi = normalized_mutual_info_score(ct, labels)
    print(f"ARI (cluster vs cancer_type)  = {ari:.4f}")
    print(f"NMI (cluster vs cancer_type)  = {nmi:.4f}")
    print("  -> proche de 1 : les clusters latents ~ redécoupent les types de cancer")
    print("  -> proche de 0 : les clusters latents capturent autre chose (ex: sous-types "
          "transversaux, signatures moléculaires partagées entre cancers)")

    ctab = pd.crosstab(pd.Series(labels, name="cluster"),
                       pd.Series(ct, name="cancer_type"))
    ctab_norm = ctab.div(ctab.sum(axis=1), axis=0)   # proportion par cluster

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * ctab.shape[1]), max(4, 0.6 * ctab.shape[0])))
    im = ax.imshow(ctab_norm.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(ctab.shape[1]))
    ax.set_xticklabels(ctab.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(ctab.shape[0]))
    ax.set_yticklabels([f"Cluster {c}" for c in ctab.index], fontsize=8)
    for i in range(ctab.shape[0]):
        for j in range(ctab.shape[1]):
            n = ctab.values[i, j]
            if n > 0:
                ax.text(j, i, str(n), ha="center", va="center",
                       fontsize=7, color="white" if ctab_norm.values[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion du cluster")
    ax.set_title(f"Composition des clusters par type de cancer\nARI={ari:.3f}  NMI={nmi:.3f}",
                fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path, "cluster_vs_cancer_type.png")

    return {"ari": ari, "nmi": nmi, "crosstab": ctab}


# ===========================================================================================================
# 5.  KAPLAN-MEIER (générique : par cluster OU par type de cancer)
# ===========================================================================================================

def kaplan_meier_by_group(
    group_labels,
    samples,
    clinical_df,
    event_col,
    time_col,
    group_name="groupe",
    save_path=None,
    title=None,
    xlim=None,
):
    """
    Trace les courbes KM pour chaque valeur de `group_labels` + p-value
    (log-rank multivarié). Fonctionne aussi bien avec des clusters K-Means
    (int) qu'avec des types de cancer (str).

    Parameters
    ----------
    group_labels : array-like (n_samples,)  — clusters OU cancer_type
    samples      : list      — IDs dans le même ordre que Z / group_labels
    clinical_df  : pd.DataFrame avec colonnes event_col et time_col
    """
    clin = clinical_df.loc[samples, [event_col, time_col]].copy()
    clin["group"] = np.asarray(group_labels)

    groups = sorted(clin["group"].unique(), key=str)
    n_groups = len(groups)
    cmap = plt.get_cmap("tab10" if n_groups <= 10 else "tab20", n_groups)

    mlr = multivariate_logrank_test(
        event_durations = clin[time_col].values.astype(float),
        groups          = clin["group"].values,
        event_observed  = clin[event_col].values.astype(float),
    )
    p_val = mlr.p_value

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(groups):
        mask = clin["group"] == g
        kmf  = KaplanMeierFitter(label=f"{g}  (n={mask.sum()})")
        kmf.fit(
            durations      = clin.loc[mask, time_col].astype(float),
            event_observed = clin.loc[mask, event_col].astype(float),
        )
        kmf.plot_survival_function(ax=ax, ci_show=(n_groups <= 6),
                                   color=cmap(i), linewidth=2)

    if xlim:
        ax.set_xlim(0, xlim)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Probabilité de survie")
    title = title or f"Kaplan-Meier par {group_name}"
    ax.set_title(f"{title}\np-value log-rank = {p_val:.3g}", fontsize=11)
    ax.legend(fontsize=7, ncol=1 if n_groups <= 8 else 2)
    plt.tight_layout()
    _save_or_show(fig, save_path, f"km_{group_name}.png")
    print(f"KM par {group_name} tracé  |  p-value log-rank multivarié = {p_val:.4g}")
    return p_val


# ===========================================================================================================
# PIPELINE COMPLET — VERSION PAN-CANCER
# ===========================================================================================================

def run_latent_analysis_pancancer(
    model,
    omics_df,
    clinical_df,
    samples,
    event_col,
    time_col,
    cancer_type_col="cancer_type",
    extra_clinical_vars=None,
    k_range=range(2, 8),
    save_dir="figures_latent_pancancer",
    km_xlim=None,
):
    """
    Version adaptée : compare l'espace latent au TYPE DE CANCER plutôt
    qu'à des covariables cliniques riches.

    Parameters
    ----------
    model            : CustOMICS  entraîné sur plusieurs cancers
    omics_df         : dict  {source: pd.DataFrame}  — tous cancers confondus
    clinical_df      : pd.DataFrame  doit contenir cancer_type_col, event_col, time_col
    samples          : list   IDs patients dans l'ordre de omics_df
    cancer_type_col  : str    colonne du type de cancer dans clinical_df
    extra_clinical_vars : list[dict] ou None  variables cliniques additionnelles
                         à afficher en plus du cancer_type (optionnel)

    Returns
    -------
    dict avec Z, reductions, labels, best_k, silhouette, cluster_vs_cancer, km_pvalue_cluster, km_pvalue_cancer
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # -- 0. Embeddings -----------------------------------------------------------------
    print("\n=== 0. Extraction des embeddings ===")
    Z = extract_embeddings(model, omics_df)
    print(f"   Z shape : {Z.shape}")
    print(f"   Types de cancer présents : "
          f"{sorted(clinical_df.loc[samples, cancer_type_col].unique())}")

    # -- 1. Réductions ----------------------------------------------------------------
    print("\n=== 1. Réductions de dimensionnalité ===")
    reductions = compute_reductions(Z)
    plot_pca_variance(reductions["pca_var"], save_path=save_dir)

    # -- 2. Coloration par type de cancer (+ variables cliniques optionnelles) -------
    print("\n=== 2. Coloration par type de cancer ===")
    variables = [{"col": cancer_type_col, "label": "Type de cancer", "type": "cat"}]
    if extra_clinical_vars:
        variables += extra_clinical_vars

    plot_latent_colored_by_vars(
        reductions = reductions,
        meta_df    = clinical_df,
        samples    = samples,
        variables  = variables,
        methods    = ("pca", "umap", "tsne"),
        save_dir   = save_dir,
        filename   = "latent_cancer_type_coloring.png",
        suptitle   = "Espace latent coloré par type de cancer",
    )

    # -- 3. Clustering + Silhouette ---------------------------------------------------
    print("\n=== 3. Clustering K-Means + Silhouette ===")
    best_k, labels, sil_scores, _ = optimal_kmeans(Z, k_range=k_range)

    plot_silhouette_scores(sil_scores, best_k,
                           save_path=f"{save_dir}/silhouette_scores.png")
    plot_silhouette_per_sample(Z, labels, best_k,
                               save_path=f"{save_dir}/silhouette_per_sample.png")
    plot_umap_clusters(reductions, labels, best_k,
                       save_path=f"{save_dir}/umap_clusters.png")

    # -- 4. Clusters vs type de cancer -----------------------------------------
    print("\n=== 4. Association clusters ↔ type de cancer ===")
    assoc = cluster_vs_cancer_type(
        labels       = labels,
        samples      = samples,
        cancer_types = clinical_df.loc[samples, cancer_type_col],
        save_path    = f"{save_dir}/cluster_vs_cancer_type.png",
    )

    # -- 5. Kaplan-Meier : par cluster ET par type de cancer -------------------
    print("\n=== 5a. Kaplan-Meier par cluster latent ===")
    p_val_cluster = kaplan_meier_by_group(
        group_labels = labels,
        samples      = samples,
        clinical_df  = clinical_df,
        event_col    = event_col,
        time_col     = time_col,
        group_name   = "cluster",
        save_path    = f"{save_dir}/km_clusters.png",
        xlim         = km_xlim,
    )

    print("\n=== 5b. Kaplan-Meier par type de cancer (référence) ===")
    p_val_cancer = kaplan_meier_by_group(
        group_labels = clinical_df.loc[samples, cancer_type_col].values,
        samples      = samples,
        clinical_df  = clinical_df,
        event_col    = event_col,
        time_col     = time_col,
        group_name   = "cancer_type",
        save_path    = f"{save_dir}/km_cancer_type.png",
        xlim         = km_xlim,
    )

    print("\nAnalyse pan-cancer terminée.")
    return dict(
        Z                 = Z,
        reductions        = reductions,
        labels            = labels,
        best_k            = best_k,
        silhouette        = sil_scores,
        cluster_vs_cancer = assoc,
        km_pvalue_cluster = p_val_cluster,
        km_pvalue_cancer  = p_val_cancer,
    )


# ===========================================================================================================
# HELPERS
# ===========================================================================================================

def _save_or_show(fig, save_path, default_name):
    if save_path:
        path = save_path if save_path.endswith(".png") else f"{save_path}/{default_name}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Sauvegardé : {path}")
    else:
        plt.show()
    plt.close(fig)


# ===========================================================================================================

if __name__ == "__main__":
    import pickle
    import torch
    import sys
    sys.path.append("..")
    sys.path.append("../..")
    
    from src.network.customics import CustOMICS
    from src.tools.utils import get_sub_omics_df
    from custcox_utils import fit_feature_selector, apply_feature_selector

    # Réutilise le builder du script d'entraînement pan-cancer (source),
    # qui gère aussi le domain-adversarial si besoin.
    from tl_custcox_source_eval_all import build_customics_model_plus 

    CKPT_PATH      = "../tl_ckpt/KIRP_lossdann_pretrained_model.pt"          # modèle pan-cancer (ex: exclut COAD)
    CONFIG_PATH    = "../tl_ckpt/KIRP_lossdann_best_params_source.json"
    DATA_PATH      = "../../data/dict_pancancer_union_mutation.pickle"
    NB_FEATURES    = 5000

    # -- 1. Charger la config sauvegardée pendant l'entraînement source --------
    print("Chargement de la config du modèle pan-cancer …")
    import json
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    best_params_full = config["best_params"]
    arch             = config["architecture"]
    domain_cfg       = config["domain_adversarial"]

    # -- 2. Charger les données pancancer (TOUS les cancers utilisés en source)
    print("Chargement du pancancer pickle …")
    with open(DATA_PATH, "rb") as f:
        pancancer = pickle.load(f)

    clinical_all = pancancer["clinical"]

    # Ici on prend tous les cancers, y compris la cible, pour voir où elle tombe :
    samples_all = list(clinical_all.index)
    clinical_df = clinical_all.loc[samples_all]

    omics_df_full = {
        "protein":  pancancer["_rna"].loc[samples_all],
        "gene_exp": pancancer["mirna"].loc[samples_all],
        "methyl":   pancancer["cnv"].loc[samples_all],
        "mutation": pancancer["mutation"].loc[samples_all],
    }
    sources = list(omics_df_full.keys())

    # -- 3. Ré-appliquer le MÊME sélecteur de features que l'entraînement ------
    print("Application du sélecteur de features (sauvegardé au training) …")
    selector_path = f"{CKPT_PATH}.selector.pkl"
    with open(selector_path, "rb") as f:
        selector = pickle.load(f)
    omics_all = apply_feature_selector(omics_df_full, selector)

    # -- 4. Reconstruire le modèle et charger les poids -------------------------
    print(f"Chargement du modèle : {CKPT_PATH} …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_params = None
    if domain_cfg.get("enabled"):
        domain_params = {
            "n_domains":     domain_cfg["n_domains"],
            "lambda":        domain_cfg["lambda_domain"],
            "hidden_layers": domain_cfg["domain_hidden_dim"],
            "dropout":       arch["dropout"],
        }

    model = build_customics_model_plus(
        omics_data      = omics_all,
        sources         = sources,
        params          = best_params_full,
        device          = device,
        hidden_dim      = arch["hidden_dim"],
        central_hidden  = arch["central_hidden"],
        classifier_dim  = arch["classifier_dim"],
        survival_dim    = arch["survival_dim"],
        dropout         = arch["dropout"],
        num_classes     = arch["num_classes"],
        unsupervised    = arch["unsupervised"],
        switch_epoch    = 0,   
        domain_params   = domain_params,
    )
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval_all()
    print("Modèle chargé.")

    # -- 5. (optionnel) quelques variables cliniques en plus du cancer_type ----
    # extra_clinical_vars = [
    #     {"col": "age_clinical", "label": "Âge diagnostic", "type": "cont"},
    # ]
    extra_clinical_vars = []
    extra_clinical_vars = [v for v in extra_clinical_vars if v["col"] in clinical_df.columns]

    # -- 6. Lancer le pipeline d'analyse pan-cancer -----------------------------
    results = run_latent_analysis_pancancer(
        model               = model,
        omics_df            = omics_all,
        clinical_df         = clinical_df,
        samples             = samples_all,
        event_col           = "status",
        time_col            = "time",
        cancer_type_col     = "cancer_type",
        extra_clinical_vars = extra_clinical_vars,
        k_range             = range(2, 12),   # plus large : plusieurs cancers = potentiellement plus de clusters naturels
        save_dir            = "figures_latent_pancancer_dann",
        km_xlim             = 2500,
    )

    # -- 7. Résultats -------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  Z shape              : {results['Z'].shape}")
    print(f"  best_k               : {results['best_k']}")
    print(f"  ARI cluster/cancer   : {results['cluster_vs_cancer']['ari']:.4f}")
    print(f"  NMI cluster/cancer   : {results['cluster_vs_cancer']['nmi']:.4f}")
    print(f"  KM p-value (cluster) : {results['km_pvalue_cluster']:.4g}")
    print(f"  KM p-value (cancer)  : {results['km_pvalue_cancer']:.4g}")
    print(f"  Figures              → figures_latent_pancancer/")
    print(f"{'='*50}")