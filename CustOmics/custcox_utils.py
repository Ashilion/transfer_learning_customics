import pandas as pd
import numpy as np
import pickle
import copy
import torch
import time

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.network.customics import CustOMICS
from src.tools.utils import get_sub_omics_df

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

import sys 
sys.path.append('..')
import utils.folds_utils as fold_utils
from utils.vvh_cv import vvh_cv
import copy
#==========================================================================================================================================

def fit_feature_selector(cancer_dataset, nbFeatures):
    selector = {}

    for name, df in cancer_dataset.items():
        if name in ["_rna", "mirna"]:
            col_std = df.std(axis=0)
            df = df.loc[:, col_std != 0]

        if name == "mutation":
            freq = df.sum(axis=0) / df.shape[0]
            df = df.loc[:, freq > 0.01]

        if df.shape[1] > nbFeatures:
            col_std = df.std(axis=0)
            top_cols = col_std.sort_values(ascending=False).index[:nbFeatures]
        else:
            top_cols = df.columns

        selector[name] = top_cols

    return selector

#==========================================================================================================================================

def apply_feature_selector(cancer_dataset, selector):
    filtered = {}
    for name, df in cancer_dataset.items():
        cols = selector[name]
        filtered[name] = df.loc[:, df.columns.intersection(cols)]
    return filtered

#==========================================================================================================================================

def fit_scalers(cancer_dataset):
    scalers = {}

    for name, df in cancer_dataset.items():
        scaler = StandardScaler()

        # fit uniquement sur le train
        scaler.fit(df)

        scalers[name] = scaler

    return scalers

#==========================================================================================================================================

def apply_scalers(cancer_dataset, scalers):
    scaled = {}

    for name, df in cancer_dataset.items():
        scaler = scalers[name]

        scaled_values = scaler.transform(df)

        scaled[name] = pd.DataFrame(
            scaled_values,
            index=df.index,
            columns=df.columns
        )

    return scaled

#==========================================================================================================================================

def build_customics_model(omics_data, sources, params, device,
                         hidden_dim, central_hidden,
                         classifier_dim, survival_dim,
                         dropout, num_classes, unsupervised, switch_epoch):

    x_dim = [omics_data[src].shape[1] for src in sources]

    source_params = {
        src: {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': params['rep_dim'], 'norm': True, 'dropout': dropout} for i, src in enumerate(sources)
    }

    central_params = {'hidden_dim': central_hidden, 'latent_dim': params['latent_dim'], 'norm': True, 'dropout': dropout, 'beta':1 }

    classif_params = {'n_class': num_classes,'lambda': 0,'hidden_layers': classifier_dim,'dropout': dropout}

    surv_params = {'lambda': 5, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True,'dropout': dropout}

    train_params = {'switch': switch_epoch, 'lr': params['lr']}

    model = CustOMICS(source_params=source_params,central_params=central_params, classif_params=classif_params, 
            surv_params=surv_params,train_params=train_params, device=device,unsupervised=unsupervised).to(device)

    return model

#==========================================================================================================================================


def build_survival_array(clinical_df, samples, event, time):
    return np.array(
        [(bool(e), t) for e, t in zip(
            clinical_df.loc[samples, event],
            clinical_df.loc[samples, time]
        )],
        dtype=[('status', 'bool'), ('time', 'float')]
    )

#==========================================================================================================================================

def fit_coxnet(X, y, l1_ratio, alphas=None, standardize_cox=False, n_alphas=10, ridge=False):
    if not ridge:
        if alphas is None:
            model = CoxnetSurvivalAnalysis(n_alphas=n_alphas, l1_ratio=l1_ratio, alpha_min_ratio=0.0001, max_iter=100, fit_baseline_model=True)
        else:
            model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, fit_baseline_model=True)

        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X, y)

        return pipe.named_steps["coxnetsurvivalanalysis"], pipe.named_steps["standardscaler"]

    else:
        if alphas is None:
            raise ValueError("alphas doit être fourni quand ridge=True (à estimer au préalable avec ridge=False)")

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        models = {}
        for alpha in alphas:
            m = CoxPHSurvivalAnalysis(alpha=alpha)
            m.fit(X_scaled, y)
            models[alpha] = m

        return models, scaler

#==========================================================================================================================================

def evaluate_survival(coxnet, alpha, Z_train, y_train, Z_val, y_val, mode, ridge=False):
    model = coxnet[alpha] if ridge else coxnet

    if mode == "ibs":
        if ridge:
            survs = model.predict_survival_function(Z_val)
        else:
            survs = model.predict_survival_function(Z_val, alpha=alpha)
        times = np.sort(np.unique(y_val["time"]))
        upper = min(
            np.max(y_train["time"]),
            np.max(y_val["time"])
        )
        times = times[times < upper]
        preds = np.vstack([fn(times) for fn in survs])
        return integrated_brier_score(y_train, y_val, preds, times)

    elif mode == "vvh":
        return vvh_cv(model, alpha, Z_train, y_train, Z_val, y_val, ridge=ridge)


# ===== LOCO Splitter ================================================================================

class LeaveOneCancerOutCV:
    """
    Leave-One-Cancer-Out cross-validator.
    Compatible avec l'interface sklearn (split / get_n_splits).
    
    Pour chaque fold, tous les samples d'un cancer type constituent le val set,
    le reste forme le train set. On peut limiter le nombre de folds avec n_folds.
    """

    def __init__(self, clinical_df, cancer_col="cancer_type", n_folds=None, random_state=None):
        self.clinical_df  = clinical_df
        self.cancer_col   = cancer_col
        self.n_folds      = n_folds
        self.random_state = random_state

    def _get_cancer_groups(self, sample_indices):
        """Retourne {cancer_type: [positions dans sample_indices]}."""
        idx_series = pd.Series(sample_indices)
        cancer_of_sample = self.clinical_df.loc[sample_indices, self.cancer_col]

        groups = {}
        for pos, (_, sample_id) in enumerate(idx_series.items()):
            cancer = cancer_of_sample.loc[sample_id]
            groups.setdefault(cancer, []).append(pos)
        return groups

    def split(self, X, y=None, groups=None):
        """
        X  : liste/array des sample indices (lt_samples)
        y  : ignoré (gardé pour compatibilité sklearn)
        """
        sample_indices = list(X)
        cancer_groups  = self._get_cancer_groups(sample_indices)
        cancer_types   = sorted(cancer_groups.keys())

        if self.n_folds is not None and self.n_folds < len(cancer_types):
            rng = np.random.RandomState(self.random_state)
            cancer_types = list(rng.choice(cancer_types, size=self.n_folds, replace=False))

        all_positions = np.arange(len(sample_indices))
        for cancer in cancer_types:
            val_positions   = np.array(cancer_groups[cancer])
            train_positions = np.setdiff1d(all_positions, val_positions)
            yield train_positions, val_positions

    def get_n_splits(self, X=None, y=None, groups=None):
        if self.n_folds is not None:
            return self.n_folds
        sample_indices = list(X) if X is not None else list(self.clinical_df.index)
        cancer_groups  = self._get_cancer_groups(sample_indices)
        return len(cancer_groups)
        
# ========== Transfer Learning Utils ============================================================================

def load_pretrained(model, ckpt_path, device, strict=False):
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    return model

def load_pretrained_expand(model, ckpt_path, device, verbose=True):
    """
    Charge le checkpoint pré-entraîné en copiant les anciens poids dans le
    sous-bloc [0:old_dim, ...] des nouveaux tenseurs (potentiellement plus
    larges). Les neurones/connexions ajoutés gardent l'initialisation
    aléatoire déjà faite par le constructeur du modèle.
    - Formes identiques -> copie directe.
    - Nouvelle forme >= ancienne sur toutes les dims -> copie dans le coin,
      reste = random (init du modèle, non touché).
    - Nouvelle couche (absente du checkpoint) ou forme incompatible -> ignorée,
      reste random.
    """
    pretrained_state = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()
    new_state = {k: v.clone() for k, v in model_state.items()}

    n_exact, n_expanded, n_skipped = 0, 0, 0
    for name, old_param in pretrained_state.items():
        if name not in model_state:
            n_skipped += 1
            continue
        new_param = model_state[name]

        if old_param.shape == new_param.shape:
            new_state[name] = old_param.clone()
            n_exact += 1
        elif old_param.dim() == new_param.dim() and all(
            o <= n for o, n in zip(old_param.shape, new_param.shape)
        ):
            expanded = new_param.clone()
            slices = tuple(slice(0, o) for o in old_param.shape)
            expanded[slices] = old_param
            new_state[name] = expanded
            n_expanded += 1
        else:
            n_skipped += 1  # nouvelle couche ou rétrécissement -> reste random

    model.load_state_dict(new_state, strict=False)
    if verbose:
        print(f"  -> poids : {n_exact} identiques, {n_expanded} élargis "
              f"(anciens + neurones random ajoutés), {n_skipped} ignorés (nouvelle couche)")
    return model


# =============================================================
def freeze_and_reset_optimizer(model, lr):
    model.freeze_autoencoders()
    model.update_optimizer(lr)
    return model

def unfreeze_and_reset_optimizer(model, lr):
    model.unfreeze_autoencoders()
    model.update_optimizer(lr)
    return model

#===============================================================

def fit_transfer(model, ckpt_path, device,
                 omics_train, clinical_df, label, event, surv_time,
                 batch_size, n_epochs_central, n_epochs_all, task,
                 lr1, lr2, patience=None, min_delta=None,
                 expand_load=False, track_loss_components=False,
                 modality_mask_train=None, missing_strategy="impute"):
    if expand_load:
        model = load_pretrained_expand(model, ckpt_path, device)
    else:
        model = load_pretrained(model, ckpt_path, device, strict=False)

    # Phase 1 : central only (identique pour les 4 variantes)
    model = freeze_and_reset_optimizer(model, lr=lr1)
    model.switch_epoch = 0
    model.phase = 2
    model.fit(
        omics_train=omics_train, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs_central,
        verbose=True, task=task,
        patience=patience, min_delta=min_delta,
        early_stopping_on="train", track_loss_components=track_loss_components,
        modality_mask_train=modality_mask_train, missing_strategy=missing_strategy,
    )
    if track_loss_components:
        model.plot_loss_detailed_stacked(save_path="results/loss_detailed_A.png", log_scale=True)
        model.plot_loss_recon_surv(save_path="results/loss_reconsurv_ft_A.png", log_scale=True)
    # Phase 2 : all layers
    model = unfreeze_and_reset_optimizer(model, lr=lr2)
    model.phase = 2
    model.fit(
        omics_train=omics_train, clinical_df=clinical_df,
        label=label, event=event, surv_time=surv_time,
        omics_val=None, batch_size=batch_size, n_epochs=n_epochs_all,
        verbose=True, task=task,
        patience=patience, min_delta=min_delta,
        early_stopping_on="train", track_loss_components=track_loss_components,
        modality_mask_train=modality_mask_train, missing_strategy=missing_strategy,
    )
    if track_loss_components:
        model.plot_loss_detailed_stacked(save_path="results/loss_detailed_B.png", log_scale=True)
        model.plot_loss_recon_surv(save_path="results/loss_reconsurv_ft_B.png", log_scale=True)
    return model

# =============================================================


def get_finetune_architecture(base_arch, variant):
    """
    base_arch: dict avec hidden_dim, central_hidden, classifier_dim,
               survival_dim, dropout, num_classes
    variant: "full_retrain" | "widen_10" | "widen_25" | "add_layer16"
    """
    arch = copy.deepcopy(base_arch)

    def scale_dims(dims, factor):
        if isinstance(dims, (list, tuple)):
            return [scale_dims(d, factor) for d in dims]
        return max(1, int(round(dims * factor)))

    if variant == "full_retrain":
        pass
    elif variant == "widen_10":
        arch["hidden_dim"] = scale_dims(arch["hidden_dim"], 1.10)
        arch["central_hidden"] = scale_dims(arch["central_hidden"], 1.10)
    elif variant == "widen_25":
        arch["hidden_dim"] = scale_dims(arch["hidden_dim"], 1.25)
        arch["central_hidden"] = scale_dims(arch["central_hidden"], 1.25)
    elif variant == "add_layer16":
        if isinstance(arch["central_hidden"], list):
            arch["central_hidden"] = arch["central_hidden"] + [16]
        else:
            arch["central_hidden"] = [arch["central_hidden"], 16]
    else:
        raise ValueError(f"finetune_arch inconnu : {variant}")

    return arch