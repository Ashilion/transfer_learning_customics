"""
Wrappers communs pour la persistance Optuna (JournalStorage + pruning).
"""
import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def get_or_create_study(study_name, journal_file, direction="minimize",
                         pruning=True, n_startup_trials=10, n_warmup_steps=2,
                         interval_steps=1):
    """Crée (ou recharge si elle existe déjà) une étude Optuna appuyée sur un
    JournalFileBackend. `journal_file` est le chemin complet du fichier .log.
    """
    parent = os.path.dirname(journal_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

    storage = JournalStorage(JournalFileBackend(file_path=journal_file))
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )
        if pruning else optuna.pruners.NopPruner()
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        pruner=pruner,
    )


def load_best_trial(study_name, journal_file):
    """Recharge une étude terminée et renvoie (study, study.best_trial)."""
    storage = JournalStorage(JournalFileBackend(file_path=journal_file))
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study, study.best_trial