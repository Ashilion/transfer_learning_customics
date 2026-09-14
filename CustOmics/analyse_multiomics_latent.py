# -*- coding: utf-8 -*-
"""
Extension de l'analyse de l'espace latent CustOMICS — adaptée à la vraie
architecture (cf. src/network/customics.py) :

  A. Contribution par omique     -> model.get_per_source_representation(x)
  B. Cross-modal alignment       -> embeddings par omique avant fusion
  C. Feature attribution (IG)    -> sur le z par-source (autoencoders[i])
                                     et sur le z fusionné (central_encoder),
                                     en complément du SHAP déjà présent dans
                                     model.explain() qui cible le classifieur.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors

from lifelines.statistics import multivariate_logrank_test

from analyse_latent_space import _save_or_show 

def _sources_in_order(omics_df):
    """Ordre des omiques tel que CustOMICS l'attend (= ordre du dict)."""
    return list(omics_df.keys())


def _to_tensor_list(model, omics_df):
    sources = _sources_in_order(omics_df)
    return sources, [torch.Tensor(omics_df[s].values).to(model.device) for s in sources]


# ═══════════════════════════════════════════════════════════════════════════
# A. CONTRIBUTION PAR OMIQUE
# ═══════════════════════════════════════════════════════════════════════════

def extract_per_omic_embeddings(model, omics_df):
    """
    Embedding propre à chaque omique (avant fusion, via les autoencodeurs
    intermédiaires) + embedding fusionné (z central).

    Returns
    -------
    dict {source_1: np.array(n, d1), ..., 'fused': np.array(n, d_central)}
    """
    model.eval_all()
    sources, x = _to_tensor_list(model, omics_df)
    with torch.no_grad():
        lt_rep = model.get_per_source_representation(x)   # liste, ordre = sources
    embeddings = {s: z.cpu().numpy() for s, z in zip(sources, lt_rep)}
    embeddings["fused"] = model.get_latent_representation(omics_df)
    return embeddings


def compare_modality_contributions(
    embeddings, samples, clinical_df, event_col, time_col,
    k_range=range(2, 8), random_state=42, save_dir=".",
):
    """
    Pour chaque omique seule + le latent fusionné : K-Means optimal (silhouette)
    puis log-rank multivarié sur les clusters obtenus. Répond à :
    "quelle omique porte le plus de signal pronostique, et la fusion
    apporte-t-elle plus que la meilleure omique seule ?"
    """
    rows = []
    clin = clinical_df.loc[samples, [event_col, time_col]]

    for source, Z in embeddings.items():
        best_score, best_k, best_labels = -1, None, None
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            lbl = km.fit_predict(Z)
            s = silhouette_score(Z, lbl)
            if s > best_score:
                best_score, best_k, best_labels = s, k, lbl

        mlr = multivariate_logrank_test(
            event_durations=clin[time_col].values.astype(float),
            groups=best_labels,
            event_observed=clin[event_col].values.astype(float),
        )
        rows.append({
            "modalite": source, "best_k": best_k, "silhouette": best_score,
            "km_pvalue": mlr.p_value,
            "neg_log10_p": -np.log10(max(mlr.p_value, 1e-300)),
        })
        print(f"  {source:10s}  k={best_k}  silhouette={best_score:.3f}  "
              f"KM p={mlr.p_value:.3g}")

    result = pd.DataFrame(rows).set_index("modalite")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#9FE1CB" if m != "fused" else "#D85A30" for m in result.index]

    axes[0].bar(result.index, result["silhouette"], color=colors)
    axes[0].set_ylabel("Silhouette score (meilleur k)")
    axes[0].set_title("Qualité du clustering par modalité")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(result.index, result["neg_log10_p"], color=colors)
    axes[1].axhline(-np.log10(0.05), linestyle="--", color="black", linewidth=1,
                     label="p = 0.05")
    axes[1].set_ylabel("-log10(p-value log-rank)")
    axes[1].set_title("Signal pronostique par modalité")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()

    plt.tight_layout()
    _save_or_show(fig, save_dir, "modality_contribution.png")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# B. CROSS-MODAL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def cross_modal_retrieval_accuracy(emb_a, emb_b, k=1):
    """
    Pour chaque patient, son embedding en modalité A est-il parmi les k plus
    proches voisins de son PROPRE embedding en modalité B (parmi tous les
    patients) ? -> mesure l'alignement au niveau individuel.
    """
    n = emb_a.shape[0]
    nn_model = NearestNeighbors(n_neighbors=k).fit(emb_b)
    _, idx = nn_model.kneighbors(emb_a)
    hits = sum(i in idx[i] for i in range(n))
    return hits / n


def cross_modal_alignment(embeddings, samples, pairs=None, n_components=2, save_dir="."):
    """
    Compare les embeddings mono-omiques deux à deux :
      - CCA : alignement linéaire global entre les deux espaces
      - retrieval accuracy top-1 / top-5 : alignement au niveau patient

    `embeddings` : sortie de extract_per_omic_embeddings (la clé 'fused' est
    ignorée automatiquement).
    """
    sources = [s for s in embeddings if s != "fused"]
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(sources) for b in sources[i + 1:]]

    rows = []
    for a, b in pairs:
        Za, Zb = embeddings[a], embeddings[b]
        n_comp = min(n_components, Za.shape[1], Zb.shape[1])
        cca = CCA(n_components=n_comp)
        Za_c, Zb_c = cca.fit_transform(Za, Zb)
        corr = np.mean([np.corrcoef(Za_c[:, i], Zb_c[:, i])[0, 1] for i in range(n_comp)])

        top1 = cross_modal_retrieval_accuracy(Za, Zb, k=1)
        top5 = cross_modal_retrieval_accuracy(Za, Zb, k=min(5, len(Za) - 1))

        rows.append({"source_a": a, "source_b": b, "cca_corr": corr,
                      "top1_acc": top1, "top5_acc": top5})
        print(f"  {a} ↔ {b}   CCA corr={corr:.3f}   top1={top1:.2%}   top5={top5:.2%}")

    result = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r.source_a}↔{r.source_b}" for r in result.itertuples()]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, result["cca_corr"], width, label="CCA corr", color="#5DCAA5")
    ax.bar(x, result["top1_acc"], width, label="Top-1 retrieval", color="#D85A30")
    ax.bar(x + width, result["top5_acc"], width, label="Top-5 retrieval", color="#888780")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Alignement cross-modal au niveau patient")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "cross_modal_alignment.png")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# C. FEATURE ATTRIBUTION (Integrated Gradients)
# ═══════════════════════════════════════════════════════════════════════════

def attribute_source_latent(model, omics_df, source, latent_dims=None,
                             n_steps=50, baseline="zero"):
    """
    Attribution des features d'UNE omique sur SES PROPRES dimensions latentes
    (avant fusion), via model.autoencoders[idx].

    Returns
    -------
    pd.DataFrame (n_features, n_latent_dims) — |IG| moyen sur tous les patients
    """
    from captum.attr import IntegratedGradients

    sources = _sources_in_order(omics_df)
    idx = sources.index(source)
    autoencoder = model.autoencoders[idx]
    autoencoder.eval()

    df = omics_df[source]
    X = torch.Tensor(df.values).to(model.device)
    X_baseline = torch.zeros_like(X) if baseline == "zero" else X.mean(0, keepdim=True).expand_as(X)

    def forward_fn(x):
        return autoencoder(x)[1]   # [1] = z, cf. AutoEncoder.forward -> (x_hat, z)

    with torch.no_grad():
        latent_dim = forward_fn(X[:1]).shape[1]
    if latent_dims is None:
        latent_dims = list(range(latent_dim))

    ig = IntegratedGradients(forward_fn)
    attributions = np.zeros((X.shape[1], len(latent_dims)))
    for j, d in enumerate(latent_dims):
        attr = ig.attribute(X, baselines=X_baseline, target=d, n_steps=n_steps)
        attributions[:, j] = attr.abs().mean(dim=0).cpu().numpy()

    return pd.DataFrame(attributions, index=df.columns,
                         columns=[f"{source}_latent_{d}" for d in latent_dims])


def attribute_fused_latent(model, omics_df, source, latent_dims=None,
                            n_steps=50, baseline="zero"):
    """
    Attribution des features d'UNE omique sur les dimensions latentes
    FUSIONNÉES (z central), les autres omiques étant fixées à leur valeur
    réelle. Répond à : "quels gènes de telle omique influencent le plus
    chaque dimension de l'espace latent final partagé ?"
    """
    from captum.attr import IntegratedGradients
 
    sources, x = _to_tensor_list(model, omics_df)
    idx = sources.index(source)
    model.eval_all()
 
    # Représentations fixes des autres sources (pas de gradient dessus)
    with torch.no_grad():
        lt_rep_fixed = model.get_per_source_representation(x)
 
    X = x[idx].clone().requires_grad_(False)
    X_baseline = torch.zeros_like(X) if baseline == "zero" else X.mean(0, keepdim=True).expand_as(X)
    n_orig = X.shape[0]
 
    def forward_fn(x_source):
        # captum/IG retile en interne le batch (n_steps répétitions par
        # échantillon) pour interpoler baseline -> input : x_source peut donc
        # avoir un batch_size = n_orig * n_steps, pas n_orig. Il faut retiler
        # les représentations fixes des autres sources à l'identique pour
        # que torch.cat fonctionne.
        batch = x_source.shape[0]
        repeat = batch // n_orig
        lt_rep = []
        for i, fixed in enumerate(lt_rep_fixed):
            if i == idx:
                lt_rep.append(model.autoencoders[idx](x_source)[1])
            else:
                lt_rep.append(fixed.repeat_interleave(repeat, dim=0) if repeat > 1 else fixed)
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = model.central_encoder(central_concat)
        return mean
 
    # Probe la dimension latente sans passer par forward_fn (pour éviter le
    # même problème de taille de batch sur un appel à 1 seul échantillon)
    with torch.no_grad():
        latent_dim = model.get_latent_representation(omics_df).shape[1]
    if latent_dims is None:
        latent_dims = list(range(latent_dim))
 
    ig = IntegratedGradients(forward_fn)
    df = omics_df[source]
    attributions = np.zeros((X.shape[1], len(latent_dims)))
    for j, d in enumerate(latent_dims):
        attr = ig.attribute(X, baselines=X_baseline, target=d, n_steps=n_steps)
        attributions[:, j] = attr.abs().mean(dim=0).cpu().numpy()
 
    return pd.DataFrame(attributions, index=df.columns,
                         columns=[f"fused_latent_{d}" for d in latent_dims])


def plot_top_features_per_dim(attribution_df, top_n=15, save_dir=".", source=""):
    """Bar plot horizontal des top_n features les plus influentes par dimension latente."""
    n_dims = attribution_df.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, max(4, top_n * 0.3)), squeeze=False)
    for j, col in enumerate(attribution_df.columns):
        ax = axes[0][j]
        top = attribution_df[col].sort_values(ascending=True).tail(top_n)
        ax.barh(top.index, top.values, color="#5DCAA5")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=7)
    plt.suptitle("Features les plus influentes par dimension latente (Integrated Gradients)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_dir, f"feature_attribution_{source}.png")


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════════════════

def run_extended_analysis(model, omics_df, clinical_df, samples, event_col, time_col,
                           save_dir="figures_latent_extended"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("\n═══ A. Embeddings par omique ═══")
    embeddings = extract_per_omic_embeddings(model, omics_df)
    for s, z in embeddings.items():
        print(f"   {s:10s} -> shape {z.shape}")

    print("\n═══ A. Contribution par omique (clustering + survie) ═══")
    contrib = compare_modality_contributions(embeddings, samples, clinical_df,
                                              event_col, time_col, save_dir=save_dir)

    print("\n═══ B. Cross-modal alignment ═══")
    alignment = cross_modal_alignment(embeddings, samples, save_dir=save_dir)

    print("\n═══ C. Feature attribution (1ère omique, latent propre + latent fusionné) ═══")
    sources = [s for s in embeddings if s != "fused"]
    for s in sources :
        attr_own   = attribute_source_latent(model, omics_df, s, latent_dims=list(range(5)))
        attr_fused = attribute_fused_latent(model, omics_df, s, latent_dims=list(range(5)))
        
        dir_own   = f"{save_dir}/attribution_own"
        dir_fused = f"{save_dir}/attribution_fused"
        os.makedirs(dir_own, exist_ok=True)
        os.makedirs(dir_fused, exist_ok=True)
        plot_top_features_per_dim(attr_own, save_dir=dir_own, source=s)
        plot_top_features_per_dim(attr_fused, save_dir=dir_fused, source=s)

    return dict(
        embeddings=embeddings,
        modality_contribution=contrib,
        cross_modal_alignment=alignment,
        attribution_own=attr_own,
        attribution_fused=attr_fused,
    )

# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pickle
    import torch
    import sys
    sys.path.append("..")
 
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
        "rna":  data["_rna"],
        "mirna": data["mirna"],
        "cnv":   data["cnv"],
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
    results = run_extended_analysis(
        model         = model,
        omics_df      = omics_all,
        clinical_df   = clinical_full,
        samples       = samples_all,
        event_col     = "status",
        time_col      = "time",
        save_dir      = f"figures_latent_multiomics_{CANCER}",
    )