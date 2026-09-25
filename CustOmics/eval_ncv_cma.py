"""Post-hyperparameter evaluation of the nested CV pipeline, using the
cross-modal attention model (cross_modal_attention_survival.py) INSTEAD OF
CustOmics, for UN outer fold donné (--outer_fold).

Reprend exactement la méthodologie d'évaluation du script original
(ncv_eval_no_tl-style) : sélection de features, simulation de modalités
manquantes, scaling, extraction de représentation latente, Cox net
(coxnet/ridge) sur la latente, C-index (IPCW) et IBS. Seule la partie
"modèle de fusion multi-omique" change : CustOmics -> cross-modal attention
(pairwise ou cls_token, --fusion_type).

Les modalités manquantes ne sont plus gérées par un masque externe passé à
CustOmics : elles sont gérées nativement par le masquage d'attention du
nouveau modèle (present_mask), cf. cross_modal_attention_survival.py.

Vous pouvez lancer ce script deux fois (--fusion_type pairwise puis
--fusion_type cls_token) sur le même outer_fold pour comparer les deux
architectures de fusion à méthodologie d'évaluation strictement identique.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

sys.path.append('..')
import utils.folds_utils as fold_utils

from custcox_utils import (
    fit_feature_selector, apply_feature_selector, build_survival_array, fit_coxnet,
    fit_scalers, apply_scalers,
)
from src.tools.utils import get_sub_omics_df
from missing_data_load_all import simulate_missing_modalities

from pipeline_utils.cli import (
    base_parser, add_cancer_arg, add_outer_cv_args, add_training_args, add_cox_args,
    add_missing_modality_args,
)
from pipeline_utils.cox_cv import maybe_concat_clinical
from pipeline_utils.data import load_cancer_data, load_clinical_test, build_omics_dict

# module livré séparément — mettez-le dans le même dossier que ce script
# (ou ajustez le sys.path.append ci-dessous vers son emplacement)
from cross_modal_attention_survival import (
    CrossModalAttentionSurvivalModel, present_mask_from_missing,
)


def build_knn_feature_matrix(clinical_test, aligned_idx, cols=None, exclude_cols=None, verbose=True):
    """Construit la matrice de features clinique pour le mode 'knn'.

    - cols=None (défaut) : utilise toutes les colonnes de clinical_test, sauf
      celles de `exclude_cols`.
    - Colonnes numériques : gardées telles quelles (imputation par la médiane
      si NaN) ; standardisées ensuite par l'appelant.
    - Colonnes non numériques : one-hot (NaN -> catégorie "__missing__").
    - Colonnes catégorielles à très forte cardinalité (>50% de valeurs
      uniques, typiquement un identifiant) sont écartées automatiquement --
      un one-hot dessus n'apporterait qu'une notion de similarité binaire
      quasi inutile, avec un coût dimensionnel énorme.

    Retourne (raw_feats: np.ndarray [n, d], used_columns: list[str]).
    """
    df = clinical_test.loc[aligned_idx].copy()
    exclude = set(exclude_cols or [])
    use_cols = cols if cols is not None else [c for c in df.columns if c not in exclude]
    df = df[use_cols]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    n = len(df)
    high_cardinality = [c for c in categorical_cols if df[c].nunique(dropna=True) > 0.5 * n]
    if high_cardinality:
        if verbose:
            print(f"  [knn clinical features] colonnes catégorielles à cardinalité trop élevée, "
                  f"écartées automatiquement (probables identifiants) : {high_cardinality}")
        categorical_cols = [c for c in categorical_cols if c not in high_cardinality]

    if verbose:
        print(f"  [knn clinical features] colonnes numériques utilisées ({len(numeric_cols)}) : {numeric_cols}")
        print(f"  [knn clinical features] colonnes catégorielles (one-hot) utilisées "
              f"({len(categorical_cols)}) : {categorical_cols}")

    parts = []
    if numeric_cols:
        num = df[numeric_cols].astype(float)
        num = num.fillna(num.median())
        parts.append(num)
    if categorical_cols:
        cat = df[categorical_cols].astype("object")
        cat = cat.where(cat.notna(), "__missing__")
        dummies = pd.get_dummies(cat, columns=categorical_cols)
        parts.append(dummies.astype(float))

    if not parts:
        raise ValueError("Aucune colonne clinique exploitable pour le mode 'knn' après exclusion "
                          "(vérifiez --clinical_knn_cols / --clinical_knn_exclude_cols).")

    feats = pd.concat(parts, axis=1)
    return feats.values.astype(np.float64), list(feats.columns)


def parse_args():
    parser = base_parser("Nested CV evaluation with a cross-modal attention fusion model (no CustOmics).")
    add_cancer_arg(parser)
    add_outer_cv_args(parser, outer_fold_required=True)
    add_training_args(parser)
    add_cox_args(parser)
    add_missing_modality_args(parser)

    parser.add_argument("--nb_features", type=int, default=5000,
        help="Number of maximum features per omics (feature selection, identique au script original).")
    parser.add_argument("--eval_patience", type=int, default=10,
        help="Patience de l'early stopping du modèle de fusion.")
    parser.add_argument("--list_clinical_columns", action="store_true",
        help="Charge juste clinical_test pour ce cancer, affiche ses colonnes, et quitte "
             "(rien n'est entraîné). Utile pour choisir --clinical_stage_col / --clinical_knn_cols.")

    # --- hyperparamètres propres au modèle cross-modal attention ---
    parser.add_argument("--fusion_type", type=str, default="pairwise", choices=["pairwise", "cls_token"],
        help="Stratégie de fusion cross-modale : 'pairwise' (chaque modalité interroge les autres) "
             "ou 'cls_token' (un token de fusion attend sur toutes les modalités).")
    parser.add_argument("--latent_dim", type=int, default=32,
        help="Dimension de la représentation latente fusionnée (comparable au latent_dim de CustOmics).")
    parser.add_argument("--n_heads", type=int, default=4,
        help="Nombre de têtes d'attention.")
    parser.add_argument("--dropout", type=float, default=0.2,
        help="Dropout dans les encodeurs/décodeurs et l'attention.")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate (Adam).")
    parser.add_argument("--lambda_recon", type=float, default=1.0,
        help="Poids de la perte de reconstruction.")
    parser.add_argument("--lambda_surv", type=float, default=5.0,
        help="Poids de la perte de survie (Cox, Breslow). 0 = purement non supervisé.")
    parser.add_argument("--switch_epoch", type=int, default=None,
        help="Époque à partir de laquelle la perte de survie s'active (None = dès le début, ou, "
             "si --two_phase_training, dès le début de la phase 2). "
             "Équivalent du switch_epoch unsupervised->supervised de CustOmics.")

    # --- entraînement en deux phases (désactivé par défaut) ---
    parser.add_argument("--two_phase_training", action="store_true",
        help="Entraîne d'abord les autoencodeurs par modalité seuls (encoder_s -> decoder_s, "
             "sans fusion ni tête de survie), puis le reste (fusion + survie + contrastifs) "
             "pendant --phase1_epochs puis le reste des époques. Désactivé par défaut : sans ce "
             "flag, tout est entraîné conjointement dès le début, comme avant.")
    parser.add_argument("--phase1_epochs", type=int, default=100,
        help="Nombre d'époques de la phase 1 (autoencodeurs seuls) si --two_phase_training.")
    # --- contrastif modalités (pré-fusion), désactivé par défaut ---
    parser.add_argument("--lambda_contrastive_modality", type=float, default=0.0,
        help="Poids de la perte contrastive d'alignement inter-modalités (0 = désactivé). "
             "Rapproche, dans l'espace pré-fusion, les embeddings des différentes modalités "
             "d'un même patient (InfoNCE bidirectionnel, négatifs intra-batch).")
    parser.add_argument("--contrastive_modality_temperature", type=float, default=0.1,
        help="Température de la perte contrastive d'alignement inter-modalités.")

    # --- contrastif clinique (post-fusion), désactivé par défaut ---
    parser.add_argument("--lambda_contrastive_clinical", type=float, default=0.0,
        help="Poids de la perte contrastive guidée par la similarité clinique (0 = désactivé). "
             "À utiliser comme ablation, pas comme réglage par défaut -- voir la discussion "
             "méthodologique dans le README/la conversation (risque de redondance avec "
             "lambda_surv + maybe_concat_clinical, et de diluer l'apport propre de l'omique).")
    parser.add_argument("--clinical_contrastive_mode", type=str, default="stage", choices=["stage", "knn"],
        help="'stage' : positifs = même stade clinique (nécessite --clinical_stage_col). "
             "'knn' : positifs = k plus proches voisins dans l'espace clinique continu "
             "(nécessite --clinical_knn_cols).")
    parser.add_argument("--clinical_stage_col", type=str, default=None,
        help="Nom de la colonne de clinical_test contenant le stade (mode 'stage'). "
             "Les valeurs sont encodées en entiers (pd.factorize) avant l'entraînement. "
             "Utilisez --list_clinical_columns pour voir les colonnes disponibles.")
    parser.add_argument("--clinical_knn_cols", type=str, default=None,
        help="Liste de colonnes de clinical_test séparées par des virgules, à utiliser comme "
             "espace clinique continu pour le mode 'knn' (ex: 'age,tumor_size,nb_nodes_positive'). "
             "Si omis (défaut), TOUTES les colonnes de clinical_test sont utilisées (numériques "
             "standardisées + catégorielles one-hot), sauf celles listées dans "
             "--clinical_knn_exclude_cols. Utilisez --list_clinical_columns pour voir les colonnes disponibles.")
    parser.add_argument("--clinical_knn_exclude_cols", type=str, default="time,status",
        help="Colonnes à exclure de l'espace clinique 'knn' quand --clinical_knn_cols n'est pas "
             "précisé (séparées par des virgules). Par défaut 'time,status' pour éviter de fuir "
             "l'outcome dans le guidage contrastif -- si clinical_test contient la survie sous "
             "d'autres noms (ex: 'OS.time', 'vital_status'), ajoutez-les ici explicitement.")
    parser.add_argument("--clinical_knn_k", type=int, default=10,
        help="Nombre de voisins pour le mode 'knn'.")
    parser.add_argument("--clinical_contrastive_temperature", type=float, default=0.5,
        help="Température de la perte contrastive clinique (softmax sur la similarité latente).")
    parser.add_argument("--clinical_knn_temperature", type=float, default=1.0,
        help="Température du softmax des distances cliniques dans le mode 'knn'.")

    parser.add_argument("--fixed_params_file", type=str, default=None,
        help="JSON optionnel pour surcharger les hyperparamètres ci-dessus (mêmes clés : "
             "fusion_type, latent_dim, n_heads, dropout, lr, weight_decay, lambda_recon, "
             "lambda_surv, switch_epoch, batch_size, modality_hidden_dims, "
             "lambda_contrastive_modality, contrastive_modality_temperature, "
             "lambda_contrastive_clinical, clinical_contrastive_mode, clinical_knn_k, "
             "clinical_contrastive_temperature, clinical_knn_temperature). Utile pour brancher "
             "une recherche Optuna dédiée à ce modèle plus tard.")
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="String to add to the study name and journal filename.",
    )
    return parser.parse_args()


def load_hyperparams(args):
    """Renvoie (hp: dict, source_label: str). Les clés de hp sont directement
    les kwargs attendus par CrossModalAttentionSurvivalModel / .fit()."""
    hp = dict(
        fusion_type=args.fusion_type,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_recon=args.lambda_recon,
        lambda_surv=args.lambda_surv,
        switch_epoch=args.switch_epoch,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 32,
        modality_hidden_dims=None,
        lambda_contrastive_modality=args.lambda_contrastive_modality,
        contrastive_modality_temperature=args.contrastive_modality_temperature,
        lambda_contrastive_clinical=args.lambda_contrastive_clinical,
        clinical_contrastive_mode=args.clinical_contrastive_mode,
        clinical_knn_k=args.clinical_knn_k,
        clinical_contrastive_temperature=args.clinical_contrastive_temperature,
        clinical_knn_temperature=args.clinical_knn_temperature,
        two_phase_training=args.two_phase_training,
        phase1_epochs=args.phase1_epochs,
    )
    if not args.fixed_params_file:
        return hp, "CLI args"

    with open(args.fixed_params_file, "r") as f:
        fixed = json.load(f)
    hp.update(fixed)
    return hp, f"fixed params ({args.fixed_params_file})"


def main():
    args = parse_args()
    hp, hp_source = load_hyperparams(args)

    need_clinical_test = args.add_clinical or hp["lambda_contrastive_clinical"] > 0 or args.list_clinical_columns

    print(f"\n{'='*60}")
    print(f"  Cancer        : {args.cancer}")
    print(f"  Outer fold    : {args.outer_fold} / {args.outer_splits}")
    print(f"  Add clinical  : {args.add_clinical}")
    print(f"  Saved folds   : {args.use_saved_folds}")
    print(f"  Ridge         : {args.ridge}")
    print(f"  Nb features   : {args.nb_features}")
    print(f"  Fusion model  : cross-modal attention ({args.fusion_type})")
    print(f"  Two-phase     : {args.two_phase_training}"
          + (f" (phase1_epochs={args.phase1_epochs})" if args.two_phase_training else ""))
    print(f"  Eval patience : {args.eval_patience}")
    print(f"  Missing rate  : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"{'='*60}\n")

    data = load_cancer_data(args.cancer)
    clinical_df = data["clinical"]
    omics_df = build_omics_dict(data, naming="raw", cast_float32=True)
    sources = list(omics_df.keys())
    print(sources)

    clinical_test, offset = None, None
    if need_clinical_test:
        clinical_test = load_clinical_test(args.cancer)
        offset = clinical_df.index[0] - clinical_test.index[0]

    if args.list_clinical_columns:
        print(f"\nColonnes de clinical_test pour '{args.cancer}' "
              f"({len(clinical_test)} lignes, offset vs clinical_df = {offset}) :")
        print(clinical_test.dtypes.to_string())
        print("\nAperçu (5 premières lignes) :")
        print(clinical_test.head().to_string())
        print("\n-> relancez avec --clinical_stage_col <col> (mode 'stage') ou "
              "--clinical_knn_cols <col1,col2,...> (mode 'knn').")
        return

    lt_samples = list(clinical_df.index)

    repetition_id = args.outer_fold // args.outer_splits
    missing_assignment = None
    if args.missing_rate > 0:
        missing_assignment = simulate_missing_modalities(
            sample_ids=lt_samples, sources=sources,
            missing_rate=args.missing_rate, repetition_seed=repetition_id,
        )
        n_missing = sum(1 for v in missing_assignment.values() if v is not None)
        print(f"  -> {n_missing}/{len(lt_samples)} patients avec une modalité manquante simulée "
              f"(repetition {repetition_id})")

    if args.use_saved_folds:
        train_folds, test_folds = fold_utils.get_folds(args.cancer, src="../data/splits.json")
        train_idx, test_idx = train_folds[args.outer_fold], test_folds[args.outer_fold]
    else:
        outer_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=0)
        splits = list(outer_cv.split(lt_samples))
        train_idx, test_idx = splits[args.outer_fold]

    print(f"\n{'='*50}\n OUTER FOLD {args.outer_fold}\n{'='*50}")

    samples_train_outer = [lt_samples[i] for i in train_idx]
    samples_test_outer = [lt_samples[i] for i in test_idx]

    omics_train_outer_raw = get_sub_omics_df(omics_df, samples_train_outer)
    omics_test_outer_raw = get_sub_omics_df(omics_df, samples_test_outer)

    selector = fit_feature_selector(omics_train_outer_raw, nbFeatures=args.nb_features)
    omics_train_outer = apply_feature_selector(omics_train_outer_raw, selector)
    omics_test_outer = apply_feature_selector(omics_test_outer_raw, selector)

    # ===== Simulation de modalités manquantes (avant le scaling) =====
    # On zero-fille comme dans le script original (cohérent avec le scaling
    # ensuite), mais le present_mask ci-dessous est ce que le modèle
    # utilisera réellement pour masquer l'attention.
    present_mask_train = np.ones((len(samples_train_outer), len(sources)), dtype=bool)
    if missing_assignment is not None:
        present_mask_train = present_mask_from_missing(samples_train_outer, sources, missing_assignment)
        for i, sid in enumerate(samples_train_outer):
            for j, s in enumerate(sources):
                if not present_mask_train[i, j]:
                    omics_train_outer[s].loc[sid, :] = 0.0
    present_mask_test = np.ones((len(samples_test_outer), len(sources)), dtype=bool)  # pas de missing simulé au test

    # ===== Scaling (fit sur train uniquement) =====
    scalers = fit_scalers(omics_train_outer)
    omics_train_outer = apply_scalers(omics_train_outer, scalers)
    omics_test_outer = apply_scalers(omics_test_outer, scalers)

    y_train_outer = build_survival_array(clinical_df, samples_train_outer, "status", "time")
    y_test_outer = build_survival_array(clinical_df, samples_test_outer, "status", "time")

    print(f"Hyperparams source : {hp_source}")
    print(f"Hyperparams : { {k: v for k, v in hp.items() if k != 'modality_hidden_dims'} }")

    input_dims = {s: omics_train_outer[s].shape[1] for s in sources}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalAttentionSurvivalModel(
        input_dims=input_dims, latent_dim=hp["latent_dim"],
        modality_hidden_dims=hp["modality_hidden_dims"], fusion_type=hp["fusion_type"],
        n_heads=hp["n_heads"], dropout=hp["dropout"], device=device,
    )

    omics_train_np = {s: omics_train_outer[s].loc[samples_train_outer].values for s in sources}
    omics_test_np = {s: omics_test_outer[s].loc[samples_test_outer].values for s in sources}

    # ===== Données pour la perte contrastive clinique (optionnelle) =====
    # Source = clinical_test (pas clinical_df), comme demandé, construites
    # UNIQUEMENT sur le train de l'outer fold pour ne pas fuir d'information
    # du test dans l'entraînement.
    #
    # ATTENTION -- hypothèse d'alignement d'index non vérifiée de mon côté :
    # clinical_test a son propre index, décalé de `offset` par rapport à
    # celui de clinical_df (offset = clinical_df.index[0] - clinical_test.index[0]),
    # exactement comme pour maybe_concat_clinical(...) plus bas dans ce script.
    # J'utilise ici la même convention (clinical_test.index = sample_id - offset).
    # Si les colonnes sorties contiennent beaucoup de NaN, c'est probablement
    # que cette convention ne correspond pas à celle de maybe_concat_clinical
    # dans pipeline_utils.cox_cv -- vérifiez-la à cette source avant de faire confiance aux résultats.
    clinical_stage_train = None
    clinical_features_train = None
    if hp["lambda_contrastive_clinical"] > 0:
        aligned_idx = [s - offset for s in samples_train_outer]
        if hp["clinical_contrastive_mode"] == "stage":
            if not args.clinical_stage_col:
                raise ValueError("--clinical_contrastive_mode stage nécessite --clinical_stage_col "
                                  "(voir --list_clinical_columns pour les noms disponibles dans clinical_test).")
            raw_stage = clinical_test.loc[aligned_idx, args.clinical_stage_col]
            clinical_stage_train = pd.factorize(raw_stage)[0]  # encodage entier, NaN -> -1
        else:
            knn_cols = [c.strip() for c in args.clinical_knn_cols.split(",")] if args.clinical_knn_cols else None
            exclude_cols = [c.strip() for c in args.clinical_knn_exclude_cols.split(",")] \
                if args.clinical_knn_exclude_cols else []
            raw_feats, used_cols = build_knn_feature_matrix(
                clinical_test, aligned_idx, cols=knn_cols, exclude_cols=exclude_cols)
            mu, sigma = raw_feats.mean(axis=0), raw_feats.std(axis=0)
            sigma[sigma == 0] = 1.0
            clinical_features_train = ((raw_feats - mu) / sigma).astype(np.float32)

    n_epochs = args.limit_epochs if args.limit_epochs else 1000
    history = model.fit(
        omics_train=omics_train_np, present_mask=present_mask_train,
        clinical_df=clinical_df, samples=samples_train_outer, event="status", surv_time="time",
        batch_size=hp["batch_size"], n_epochs=n_epochs, lr=hp["lr"], weight_decay=hp["weight_decay"],
        lambda_recon=hp["lambda_recon"], lambda_surv=hp["lambda_surv"], switch_epoch=hp["switch_epoch"],
        patience=args.eval_patience, verbose=True,
        lambda_contrastive_modality=hp["lambda_contrastive_modality"],
        contrastive_modality_temperature=hp["contrastive_modality_temperature"],
        lambda_contrastive_clinical=hp["lambda_contrastive_clinical"],
        clinical_contrastive_mode=hp["clinical_contrastive_mode"],
        clinical_stage=clinical_stage_train,
        clinical_features=clinical_features_train,
        clinical_knn_k=hp["clinical_knn_k"],
        clinical_contrastive_temperature=hp["clinical_contrastive_temperature"],
        clinical_knn_temperature=hp["clinical_knn_temperature"],
        two_phase_training=hp["two_phase_training"],
        phase1_epochs=hp["phase1_epochs"],
    )

    loss_plot_dir = "results"
    os.makedirs(loss_plot_dir, exist_ok=True)
    tag = f"{args.name_suffix}{args.cancer}_fold{args.outer_fold}_{hp['fusion_type']}"
    model.plot_loss_detailed(save_path=f"{loss_plot_dir}/loss_cma_{tag}.png")
    model.plot_loss_detailed_stacked(save_path=f"{loss_plot_dir}/loss_cma_stacked_{tag}.png")

    Z_train_outer = model.get_latent_representation(omics_train_np, present_mask_train)
    Z_test_outer = model.get_latent_representation(omics_test_np, present_mask_test)

    X_train_outer = maybe_concat_clinical(Z_train_outer, samples_train_outer, offset, clinical_test)
    X_test_outer = maybe_concat_clinical(Z_test_outer, samples_test_outer, offset, clinical_test)

    coxnet, scaler = fit_coxnet(X_train_outer, y_train_outer, args.l1_ratio, None, ridge=args.ridge)
    X_test_scaled = scaler.transform(X_test_outer)

    # NOTE méthodologique — IMPORTANT :
    # Dans le script original, best_alpha venait de la recherche Optuna
    # (best_trial.user_attrs["best_alpha"]), donc sélectionné via la CV
    # interne (inner folds), PAS sur le train de l'outer fold. Ici, comme ce
    # script ne fait plus de recherche d'hyperparamètres, l'alpha du Cox net
    # est choisi en maximisant le C-index sur le train lui-même (ci-dessous),
    # ce qui est optimiste par rapport à l'original et n'est PAS équivalent
    # méthodologiquement. Si vous voulez une sélection propre, il faut soit
    # rebrancher une CV interne sur alpha (inner_splits), soit fixer
    # best_alpha explicitement via --fixed_params_file.
    X_train_scaled = scaler.transform(X_train_outer)
    alphas = coxnet.alphas_ if not args.ridge else np.asarray(list(coxnet.alphas_))
    alpha_scores = []
    for a in alphas:
        if args.ridge:
            r = coxnet[a].predict(X_train_scaled)
        else:
            r = coxnet.predict(X_train_scaled, alpha=a)
        alpha_scores.append(concordance_index_ipcw(y_train_outer, y_train_outer, r)[0])
    best_alpha = alphas[int(np.argmax(alpha_scores))]

    if args.ridge:
        fitted = coxnet[best_alpha]
        risk_scores = fitted.predict(X_test_scaled)
        survs = fitted.predict_survival_function(X_test_scaled)
        coefs = fitted.coef_
    else:
        risk_scores = coxnet.predict(X_test_scaled, alpha=best_alpha)
        survs = coxnet.predict_survival_function(X_test_scaled, alpha=best_alpha)
        alpha_idx = np.argmin(np.abs(coxnet.alphas_ - best_alpha))
        coefs = coxnet.coef_[:, alpha_idx]

    c_index = concordance_index_ipcw(y_train_outer, y_test_outer, risk_scores)[0]

    times = np.sort(np.unique(y_test_outer["time"]))
    upper = min(np.max(y_train_outer["time"]), np.max(y_test_outer["time"]))
    times = times[times < upper]
    preds = np.vstack([fn(times) for fn in survs])
    ibs_score = integrated_brier_score(y_train_outer, y_test_outer, preds, times)

    print(f"  C-index : {c_index:.4f}  |  IBS : {ibs_score:.4f}")

    nonzero_mask = coefs != 0
    n_nonzero_coefs = int(nonzero_mask.sum())
    n_zero_coefs = int(len(coefs) - n_nonzero_coefs)
    print(f"  Coefs non-nuls : {n_nonzero_coefs} / {len(coefs)}  (nuls : {n_zero_coefs})")

    result = [{
        "fold": args.outer_fold,
        "fusion_type": hp["fusion_type"],
        "cindex_default": c_index,
        "graf": ibs_score,
        "best_alpha": best_alpha,
        "n_nonzero_coefs": n_nonzero_coefs,
        "n_zero_coefs": n_zero_coefs,
        "missing_rate": args.missing_rate,
        "missing_strategy": args.missing_strategy,
        "hp_source": hp_source,
        **{f"hp_{k}": v for k, v in hp.items() if k != "modality_hidden_dims"},
    }]

    out_dir = "../results/folds"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/ncv_cma_{args.name_suffix}{args.cancer}_fold{args.outer_fold}_{hp['fusion_type']}.csv"
    pd.DataFrame(result).to_csv(out_path, index=False)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()