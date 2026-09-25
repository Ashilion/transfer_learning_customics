"""Boucle de CV interne (CoxNet) factorisée.

Les scripts de recherche Optuna d'origine répétaient presque à l'identique :

  1. estimation d'une grille d'alphas sur l'ensemble outer train,
  2. boucle sur les folds internes : entraînement du modèle, embedding,
     fit CoxNet, scoring pour chaque alpha, pruning Optuna optionnel,
  3. sélection du meilleur alpha (score moyen minimal sur les folds internes).

`estimate_alpha_grid` couvre (1), `run_inner_cv_cox` couvre (2)+(3). La seule
partie qui diffère vraiment d'un script à l'autre — "comment entraîne-t-on le
modèle sur ce fold" (from scratch, avec ou sans early stopping, ou
fine-tuning depuis un checkpoint) — reste déléguée à l'appelant via
`fit_and_embed_fn`.

Pour le script "pré-entraînement source" avec `validation_function == "loss"`
(pas de CoxNet, juste la loss de validation du modèle), voir
`run_inner_cv_loss`.
"""
import gc

import numpy as np
import optuna

from custcox_utils import fit_coxnet, evaluate_survival, build_survival_array


def maybe_concat_clinical(Z, samples, offset, clinical_test):
    """Concatène les features cliniques brutes à la représentation latente Z
    si `clinical_test` est fourni (mode --add_clinical). Sinon renvoie Z tel quel.
    """
    if clinical_test is None:
        return Z
    adapted = [idx - offset for idx in samples]
    clin = clinical_test.loc[adapted, :]
    return np.concatenate([clin, Z], axis=1)


def estimate_alpha_grid(Z, samples, y_struct, l1_ratio, offset=None,
                         clinical_test=None, n_alphas=None):
    """Fit un CoxNet de référence sur l'ensemble (outer) train pour obtenir
    une grille d'alphas candidats, réutilisée pour tous les folds internes
    (comme dans les scripts d'origine, qui n'estiment cette grille qu'une
    fois par trial plutôt qu'une fois par fold interne).
    """
    X = maybe_concat_clinical(Z, samples, offset, clinical_test)
    kwargs = {} if n_alphas is None else {"n_alphas": n_alphas}
    coxnet_ref, _ = fit_coxnet(X, y_struct, l1_ratio, **kwargs)
    return coxnet_ref.alphas_


def run_inner_cv_cox(inner_cv, samples_train_outer, y_train_outer,
                      estimated_alphas, l1_ratio, validation_function,
                      fit_and_embed_fn, clinical_df, event, surv_time,
                      offset=None, clinical_test=None, ridge=False,
                      trial=None, trial_pruning=False, gc_collect=False):
    """Boucle de CV interne pour la validation basée sur CoxNet (vvh / ibs).

    fit_and_embed_fn(samples_train_inner, samples_val_inner) -> (Z_train, Z_val)
        doit encapsuler tout l'entraînement du modèle CustOMICS pour ce fold
        (from-scratch ou fine-tuning) et renvoyer les représentations
        latentes train/val.

    Renvoie (best_alpha, best_score, alpha_scores) où alpha_scores est le
    dict {alpha: [score_fold_0, score_fold_1, ...]} (utile pour
    trial.set_user_attr comme dans les scripts d'origine).
    """
    alpha_scores = {a: [] for a in estimated_alphas}

    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(samples_train_outer, y_train_outer["status"])
    ):
        samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
        samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

        Z_train, Z_val = fit_and_embed_fn(samples_train_inner, samples_val_inner)

        y_train_struct = build_survival_array(clinical_df, samples_train_inner, event, surv_time)
        y_val_struct = build_survival_array(clinical_df, samples_val_inner, event, surv_time)

        X_train = maybe_concat_clinical(Z_train, samples_train_inner, offset, clinical_test)
        X_val = maybe_concat_clinical(Z_val, samples_val_inner, offset, clinical_test)

        coxnet, scaler = fit_coxnet(X_train, y_train_struct, l1_ratio, estimated_alphas, ridge=ridge)
        X_val_scaled = scaler.transform(X_val)

        for alpha in estimated_alphas:
            score = evaluate_survival(
                coxnet, alpha, X_train, y_train_struct, X_val_scaled, y_val_struct,
                validation_function, ridge=ridge,
            )
            alpha_scores[alpha].append(score)

        if gc_collect:
            del Z_train, Z_val
            gc.collect()

        if trial_pruning and trial is not None:
            best_so_far = min(np.mean(v) for v in alpha_scores.values())
            trial.report(best_so_far, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    best_alpha = min(alpha_scores, key=lambda a: np.mean(alpha_scores[a]))
    best_score = float(np.mean(alpha_scores[best_alpha]))
    return best_alpha, best_score, alpha_scores


def run_inner_cv_loss(inner_cv, samples_train_outer, y_status, fit_and_eval_fn,
                       trial=None, trial_pruning=False):
    """Variante de la CV interne pour `validation_function == "loss"` : pas de
    CoxNet, le score de chaque fold est directement la loss de validation du
    modèle CustOMICS.

    fit_and_eval_fn(samples_train_inner, samples_val_inner) -> loss (tensor)
    """
    loss_scores = []
    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(samples_train_outer, y_status)
    ):
        samples_train_inner = [samples_train_outer[i] for i in inner_train_idx]
        samples_val_inner = [samples_train_outer[i] for i in inner_val_idx]

        loss_scores.append(fit_and_eval_fn(samples_train_inner, samples_val_inner))

        if trial_pruning and trial is not None:
            intermediate = float(np.mean([l.detach().cpu().numpy() for l in loss_scores]))
            trial.report(intermediate, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean([l.detach().cpu().numpy() for l in loss_scores]))