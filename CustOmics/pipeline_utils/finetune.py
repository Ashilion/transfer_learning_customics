"""Wrapper autour de la logique de fine-tuning (architecture variante +
chargement du checkpoint pré-entraîné), commun à `search_target_finetune.py`
et `eval_target_finetune.py`.
"""
from custcox_utils import fit_transfer, get_finetune_architecture

from pipeline_utils.model import build_customics_model


def resolve_finetune_architecture(base_arch, finetune_arch="full_retrain"):
    """base_arch : dict avec hidden_dim, central_hidden, classifier_dim,
    survival_dim, dropout, num_classes (architecture du modèle pré-entraîné).

    Renvoie (arch_variant, expand_load) où expand_load indique si le modèle
    doit être élargi par rapport au checkpoint chargé (False pour
    "full_retrain").
    """
    arch_variant = get_finetune_architecture(base_arch, finetune_arch)
    expand_load = finetune_arch != "full_retrain"
    return arch_variant, expand_load


def build_and_finetune(omics_train, sources, pretrain_params, device,
                        arch_variant, unsupervised, ckpt_path,
                        clinical_df, label, event, surv_time,
                        batch_size, n_epochs_ft, task,
                        lr1, lr2, patience, min_delta, expand_load,
                        lambda_surv=5, track_loss_components=False,
                        optimizer="adam", weight_decay=0.0,
                        modality_dropout_p=None, md_mode="exclude",
                        modality_mask_train=None, missing_strategy="impute"):
    """Construit un modèle avec l'architecture `arch_variant`, puis lance
    `fit_transfer` depuis le checkpoint `ckpt_path`. Renvoie le modèle
    fine-tuné.
    """
    model = build_customics_model(
        omics_train, sources, pretrain_params, device,
        arch_variant["classifier_dim"], arch_variant["survival_dim"],
        arch_variant["dropout"], arch_variant["num_classes"],
        unsupervised, switch_epoch=0, lambda_surv=lambda_surv,
        modality_dropout_p=modality_dropout_p, md_mode=md_mode,
        optimizer=optimizer, weight_decay=weight_decay
    )

    n_epochs_central = n_epochs_ft // 2
    n_epochs_all = n_epochs_ft - n_epochs_central

    extra_kwargs = {"track_loss_components": True} if track_loss_components else {}

    return fit_transfer(
        model, ckpt_path, device,
        omics_train, clinical_df,
        label, event, surv_time,
        batch_size, n_epochs_central, n_epochs_all, task,
        lr1=lr1, lr2=lr2, patience=patience, min_delta=min_delta,
        expand_load=expand_load,
        modality_mask_train=modality_mask_train, missing_strategy=missing_strategy,
        **extra_kwargs,
    )