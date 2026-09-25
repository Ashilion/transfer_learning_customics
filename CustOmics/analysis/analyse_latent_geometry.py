import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

def extract_embeddings(model, omics_df):
    return model.get_latent_representation(omics_df)
# ================================================================================
# 1. DIMENSIONNALITÉ INTRINSÈQUE
# ================================================================================

def intrinsic_dim_twonn(Z: np.ndarray) -> float:
    """
    Two-NN estimator (Facco et al. 2017).
    Ratio r = dist(2nd NN) / dist(1st NN) → MLE de la dim intrinsèque.
    Ne dépend pas du nombre de dimensions de Z, seulement de la géométrie locale.
    """
    nn = NearestNeighbors(n_neighbors=3).fit(Z)
    dists, _ = nn.kneighbors(Z)
    # dists[:, 0] = 0 (soi-même), [:, 1] = 1er voisin, [:, 2] = 2e voisin
    r = np.clip(dists[:, 2] / (dists[:, 1] + 1e-12), 1 + 1e-6, None)
    # MLE : d = -N / sum(log(r_i))
    d_hat = -len(r) / np.sum(np.log(r + 1e-12))
    return float(d_hat)


def participation_ratio(Z: np.ndarray) -> float:
    """
    Participation ratio = (sum λ_i)² / sum(λ_i²)
    où λ_i sont les valeurs propres de la covariance de Z.
    Interprétation : nombre effectif de dimensions utilisées.
    PR proche de Z.shape[1] → toutes les dims sont exploitées.
    PR << Z.shape[1] → collapse sur peu de directions.
    """
    Z_centered = Z - Z.mean(axis=0)
    cov = np.cov(Z_centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    pr = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return float(pr)


def plot_intrinsic_dim(Z: np.ndarray, save_dir: str):
    d_twonn = intrinsic_dim_twonn(Z)
    pr      = participation_ratio(Z)
    n_dims  = Z.shape[1]

    # Variance expliquée par dim (pour visualiser la concentration)
    Z_c = Z - Z.mean(axis=0)
    var_per_dim = np.var(Z_c, axis=0)
    var_sorted  = np.sort(var_per_dim)[::-1]
    cum_var     = np.cumsum(var_sorted) / var_sorted.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Courbe de variance cumulée
    axes[0].plot(range(1, n_dims + 1), cum_var, marker="o", markersize=3)
    axes[0].axhline(0.9, color="red", linestyle="--", label="90 % variance")
    axes[0].set_xlabel("Nombre de dims (triées par variance)")
    axes[0].set_ylabel("Variance cumulée")
    axes[0].set_title(
        f"Variance cumulée\nTwo-NN d={d_twonn:.1f}  |  PR={pr:.1f}/{n_dims}"
    )
    axes[0].legend()

    # Variance par dimension
    axes[1].bar(range(n_dims), var_sorted, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Dimension (triée)")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Variance par dimension latente")

    plt.tight_layout()
    fig.savefig(f"{save_dir}/intrinsic_dimensionality.png", dpi=150)
    plt.close(fig)
    return {"two_nn_dim": d_twonn, "participation_ratio": pr}


# ================================================================================
# 2. INTERPOLATION LATENTE (AE / VAE)
# ================================================================================

def interpolate_latent(
    model,
    z_a: np.ndarray,
    z_b: np.ndarray,
    n_steps: int = 10,
) -> list[np.ndarray]:
    """
    Interpole linéairement entre z_a et z_b dans l'espace latent,
    décode chaque point, et retourne les reconstructions omiques.

    Suppose que model.decode(z) ou model.decoder(z) existe.
    Adapte le nom selon ton architecture CustOMICS.
    """
    import torch
    alphas = np.linspace(0, 1, n_steps)
    decoded = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z_a + alpha * z_b
        z_tensor = torch.tensor(z_interp, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # ← adapte selon l'API de CustOMICS
            recon = model.decode(z_tensor)
        decoded.append(recon.squeeze(0).numpy())
    return decoded


def plot_interpolation(
    model,
    Z: pd.DataFrame,
    omics_df: dict,           # dict source → DataFrame omique original
    idx_a: int = 0,
    idx_b: int = 1,
    n_steps: int = 12,
    save_dir: str = "latent_statistics",
):
    """
    Trace deux diagnostics d'interpolation :
    1. Distance cosinus entre reconstructions successives (smoothness de trajectoire).
    2. Heatmap de l'évolution d'une source omique le long du chemin.
    """
    z_a = Z.iloc[idx_a].values
    z_b = Z.iloc[idx_b].values
    recons = interpolate_latent(model, z_a, z_b, n_steps=n_steps)
    recons_arr = np.stack(recons)   # (n_steps, n_features)

    # Distance cosinus entre pas successifs
    step_dists = [
        cdist(recons_arr[i : i + 1], recons_arr[i + 1 : i + 2], metric="cosine")[0, 0]
        for i in range(n_steps - 1)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(range(1, n_steps), step_dists, marker="o", color="steelblue")
    axes[0].set_xlabel("Étape d'interpolation")
    axes[0].set_ylabel("Distance cosinus")
    axes[0].set_title(
        f"Smoothness de la trajectoire\n(patient {idx_a} → patient {idx_b})"
    )
    axes[0].axhline(np.mean(step_dists), color="red", linestyle="--", label="moyenne")
    axes[0].legend()

    # Heatmap : évolution des 50 premières features le long du chemin
    n_show = min(50, recons_arr.shape[1])
    im = axes[1].imshow(
        recons_arr[:, :n_show].T,
        aspect="auto",
        cmap="RdBu_r",
    )
    axes[1].set_xlabel("Étape d'interpolation")
    axes[1].set_ylabel("Feature (50 premières)")
    axes[1].set_title("Évolution des features le long du chemin latent")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    fig.savefig(f"{save_dir}/interpolation_path.png", dpi=150)
    plt.close(fig)
    return step_dists


# ================================================================================
# 3. SMOOTHNESS — voisins latents ↔ voisins omiques
# ================================================================================

def latent_smoothness(
    Z: np.ndarray,
    X_omics: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Pour chaque patient, trouve ses k plus proches voisins dans Z.
    Mesure si ces mêmes voisins sont aussi proches dans l'espace omique.

    Deux métriques :
    - neighbor_overlap : % de k-NN latents qui sont aussi k-NN omiques
    - profile_corr     : corrélation moyenne des profils omiques entre voisins latents
    """
    Z_scaled = StandardScaler().fit_transform(Z)
    X_scaled = StandardScaler().fit_transform(X_omics)

    nn_latent = NearestNeighbors(n_neighbors=k + 1).fit(Z_scaled)
    nn_omics  = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)

    _, idx_lat  = nn_latent.kneighbors(Z_scaled)
    _, idx_omic = nn_omics.kneighbors(X_scaled)

    overlaps = []
    corrs    = []
    for i in range(len(Z)):
        lat_neighbors  = set(idx_lat[i, 1:])   # exclut soi-même
        omic_neighbors = set(idx_omic[i, 1:])
        overlaps.append(len(lat_neighbors & omic_neighbors) / k)

        # Corrélation moyenne des profils omiques entre voisins latents
        neighbor_profiles = X_scaled[list(lat_neighbors)]
        ref_profile       = X_scaled[i]
        c = np.corrcoef(ref_profile, neighbor_profiles)[0, 1:]
        corrs.append(np.nanmean(c))

    return {
        "mean_neighbor_overlap": float(np.mean(overlaps)),
        "mean_profile_corr":     float(np.nanmean(corrs)),
        "per_sample_overlap":    overlaps,
        "per_sample_corr":       corrs,
    }


def plot_smoothness(smoothness_res: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(smoothness_res["per_sample_overlap"], bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(smoothness_res["mean_neighbor_overlap"], color="red", linestyle="--",
                    label=f"moy={smoothness_res['mean_neighbor_overlap']:.2f}")
    axes[0].set_xlabel("Overlap k-NN (latent ∩ omique) / k")
    axes[0].set_ylabel("Nombre de patients")
    axes[0].set_title("Chevauchement des voisinages")
    axes[0].legend()

    axes[1].hist(smoothness_res["per_sample_corr"], bins=20, color="darkorange", edgecolor="white")
    axes[1].axvline(smoothness_res["mean_profile_corr"], color="red", linestyle="--",
                    label=f"moy={smoothness_res['mean_profile_corr']:.2f}")
    axes[1].set_xlabel("Corrélation profil omique avec voisins latents")
    axes[1].set_ylabel("Nombre de patients")
    axes[1].set_title("Cohérence biologique des voisinages")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(f"{save_dir}/smoothness.png", dpi=150)
    plt.close(fig)


# ================================================================================
# 4. DEAD DIMENSIONS
# ================================================================================

def dead_dimensions(
    Z: np.ndarray,
    var_threshold: float = 0.01,
    model=None,              # optionnel : VAE pour récupérer le KL par dim
) -> dict:
    """
    Détecte les dimensions avec variance quasi-nulle (non utilisées).
    Si model est un VAE avec .encode() renvoyant (mu, log_var),
    calcule aussi le KL par dimension pour voir celles qui restent
    proches du prior N(0,1).
    """
    var_per_dim = np.var(Z, axis=0)
    dead_mask   = var_per_dim < var_threshold
    n_dead      = int(dead_mask.sum())

    result = {
        "var_per_dim":     var_per_dim,
        "dead_mask":       dead_mask,
        "n_dead":          n_dead,
        "pct_dead":        100 * n_dead / Z.shape[1],
        "threshold_used":  var_threshold,
    }

    # KL par dimension (VAE uniquement)
    # Nécessite que model.encode() retourne (mu, log_var)
    if model is not None and hasattr(model, "encode"):
        try:
            import torch
            with torch.no_grad():
                mu, log_var = model.encode(...)  #TODO modifier Customics ajouter fonction
            mu_np      = mu.numpy()
            log_var_np = log_var.numpy()
            # KL analytique = -0.5 * (1 + log_var - mu² - exp(log_var))
            kl_per_dim = -0.5 * (1 + log_var_np - mu_np**2 - np.exp(log_var_np))
            result["kl_per_dim"] = kl_per_dim.mean(axis=0)
        except Exception as e:
            print(f"[dead_dimensions] KL non calculable : {e}")

    return result


def plot_dead_dimensions(dead_res: dict, save_dir: str):
    var  = dead_res["var_per_dim"]
    mask = dead_res["dead_mask"]
    thr  = dead_res["threshold_used"]
    n    = len(var)

    colors = ["salmon" if m else "steelblue" for m in mask]

    fig, axes = plt.subplots(1, 2 if "kl_per_dim" in dead_res else 1,
                             figsize=(14 if "kl_per_dim" in dead_res else 8, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].bar(range(n), var, color=colors, alpha=0.8)
    axes[0].axhline(thr, color="red", linestyle="--", label=f"seuil={thr}")
    axes[0].set_xlabel("Dimension latente")
    axes[0].set_ylabel("Variance")
    axes[0].set_title(
        f"Variance par dimension — {dead_res['n_dead']} mortes "
        f"({dead_res['pct_dead']:.1f} %)"
    )
    axes[0].legend()
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color="salmon", label=f"dead (var<{thr})"),
        Patch(color="steelblue", label="active"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"seuil={thr}"),
    ])

    if "kl_per_dim" in dead_res:
        kl = dead_res["kl_per_dim"]
        axes[1].bar(range(len(kl)), kl, color=["salmon" if mask[i] else "steelblue"
                                                for i in range(len(kl))], alpha=0.8)
        axes[1].set_xlabel("Dimension latente")
        axes[1].set_ylabel("KL moyen")
        axes[1].set_title("KL divergence par dimension (VAE)\nKL ≈ 0 → dim ignorée")

    plt.tight_layout()
    fig.savefig(f"{save_dir}/dead_dimensions.png", dpi=150)
    plt.close(fig)


# ================================================================================
# RUNNER : intégration dans run_latent_statistics
# ================================================================================

def run_latent_geometry(
    model,
    Z: pd.DataFrame,
    omics_df,           # dict source→DataFrame OU DataFrame concaténé
    save_dir: str = "latent_statistics",
    k_smooth: int = 10,
    interp_pairs: list[tuple] = [(0, 1), (0, 2)],
):
    """
    Lance les 4 analyses géométriques sur Z et sauvegarde les figures.
    Appelle cette fonction après run_latent_statistics().
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    Z_arr = Z.values

    print("── Dimensionnalité intrinsèque …")
    dim_res = plot_intrinsic_dim(Z_arr, save_dir)
    print(f"   Two-NN d̂ = {dim_res['two_nn_dim']:.2f}  |  "
          f"PR = {dim_res['participation_ratio']:.2f} / {Z_arr.shape[1]}")

    # Construire la matrice omique concaténée pour la smoothness
    if isinstance(omics_df, dict):
        X_concat = pd.concat(omics_df.values(), axis=1).loc[Z.index].values
    else:
        X_concat = omics_df.loc[Z.index].values

    print("── Smoothness …")
    smooth_res = latent_smoothness(Z_arr, X_concat, k=k_smooth)
    plot_smoothness(smooth_res, save_dir)
    print(f"   Overlap k-NN = {smooth_res['mean_neighbor_overlap']:.3f}  |  "
          f"Corr profil = {smooth_res['mean_profile_corr']:.3f}")

    print("── Dead dimensions …")
    dead_res = dead_dimensions(Z_arr, model=model)
    plot_dead_dimensions(dead_res, save_dir)
    print(f"   {dead_res['n_dead']} dimensions mortes ({dead_res['pct_dead']:.1f} %)")

    # print("── Interpolation latente …")
    # for idx_a, idx_b in interp_pairs:
    #     try:
    #         plot_interpolation(model, Z, omics_df,
    #                            idx_a=idx_a, idx_b=idx_b, save_dir=save_dir)
    #         print(f"   Interpolation {idx_a}↔{idx_b} OK")
    #     except AttributeError:
    #         print("   [skip] model.decode() non disponible — adapte l'appel à CustOMICS")

    print("Analyse géométrique terminée.")
    return {"intrinsic_dim": dim_res, "smoothness": smooth_res, "dead": dead_res}


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
    Z_arr = extract_embeddings(model, omics_all)
    Z = pd.DataFrame(Z_arr, index=clinical_full.index)

    run_latent_geometry(
        model     = model,
        Z         = Z,
        omics_df  = omics_all, 
        save_dir  = "latent_geometry",
        k_smooth  = 10,
        interp_pairs = [(0, 1), (5, 10)],
    )