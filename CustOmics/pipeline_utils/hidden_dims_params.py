"""Grille d'hyperparamètres de l'auto-encodeur CustOMICS + helpers Optuna.

Les 4 scripts de recherche redéfinissaient chacun les deux dicts
`autoencoder_hidden_dims_possibilities(_small)` à l'identique, ainsi que la
boucle qui pioche dedans pour chaque source omique, et la boucle inverse
(reconstruire la liste de dims à partir de `best_trial.params`) dans les
scripts d'évaluation. Tout est ici.
"""

AUTOENCODER_HIDDEN_DIMS_LARGE = {
    "(1024, 512, 256, 128)": (1024, 512, 256, 128),
    "(1024, 256)":           (1024, 256),
    "(512, 128)":            (512, 128),
    "(1024, 256, 128)":      (1024, 256, 128),
    "(1024, 512, 128)":      (1024, 512, 128),
}

AUTOENCODER_HIDDEN_DIMS_SMALL = {
    "(512, 256, 128)": (512, 256, 128),
    "(256, 128)":      (256, 128),
    "(512, 256)":      (512, 256),
}

# Union des deux grilles : utilisée pour reconstruire les dims à partir des
# best_params d'une étude Optuna déjà terminée, quand on ne sait plus a
# priori quelle grille (large/small) a été utilisée pour chaque source.
AUTOENCODER_HIDDEN_DIMS_ALL = {**AUTOENCODER_HIDDEN_DIMS_LARGE, **AUTOENCODER_HIDDEN_DIMS_SMALL}

DEFAULT_WIDE_SOURCE_THRESHOLD = 1200


def suggest_autoencoder_hidden_dims(trial, omics_data, sources,
                                     wide_threshold=DEFAULT_WIDE_SOURCE_THRESHOLD):
    """Suggère, pour chaque source omique, une architecture d'auto-encodeur.
    Les sources avec plus de `wide_threshold` colonnes piochent dans la
    grille "large", les autres dans la grille "small".
    """
    dims = []
    for i, src in enumerate(sources):
        possibilities = (
            AUTOENCODER_HIDDEN_DIMS_LARGE
            if omics_data[src].shape[1] > wide_threshold
            else AUTOENCODER_HIDDEN_DIMS_SMALL
        )
        key = trial.suggest_categorical(f"hidden_dim_{i}", list(possibilities.keys()))
        dims.append(possibilities[key])
    return dims


def reconstruct_autoencoder_hidden_dims(best_params, sources):
    """Reconstruit la liste `autoencoder_hidden_dims` à partir des
    `best_trial.params` d'une étude Optuna terminée.
    """
    return [
        AUTOENCODER_HIDDEN_DIMS_ALL[best_params[f"hidden_dim_{i}"]]
        for i in range(len(sources))
    ]