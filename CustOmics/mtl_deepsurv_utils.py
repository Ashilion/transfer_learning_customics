"""
Utilitaires pour l'evaluation "DeepSurv pur" (sans CoxPH) :
  - estimateur de Breslow pour obtenir une fonction de survie a partir d'un
    simple score de risque (hazard relatif) predit par le reseau
  - sampler de batchs equilibres par cancer, pour que chaque tete de survie
    voie un risk-set correct a chaque batch (indispensable en multitask, vu
    que 18 cancers pooles dans un batch_size classique donnent trop peu
    d'echantillons par cancer).
"""
import numpy as np
from torch.utils.data import Sampler


# ===== Sampler equilibre par cancer =========================================

class TaskBalancedBatchSampler(Sampler):
    """
    A chaque batch, tire ~ n_per_task echantillons pour CHAQUE cancer present
    (avec remplacement si un cancer a moins d'echantillons que n_per_task).
    Garantit que chaque tete de survie a un risk-set exploitable a (presque)
    chaque batch.
    """
    def __init__(self, cancer_ids, n_per_task=6, n_batches=None, seed=0):
        """
        cancer_ids : array-like, longueur = nb d'echantillons du dataset,
                     id de cancer de chaque echantillon (meme ordre que le
                     Dataset / lt_samples).
        n_per_task : nb d'echantillons tires par cancer et par batch.
        n_batches  : nb de batchs par epoque (defaut : de quoi couvrir une
                     fois chaque cancer selon son plus petit sous-groupe).
        """
        self.cancer_ids = np.asarray(cancer_ids)
        self.tasks = np.unique(self.cancer_ids)
        self.idx_by_task = {t: np.where(self.cancer_ids == t)[0] for t in self.tasks}
        self.n_per_task = n_per_task
        self.rng = np.random.RandomState(seed)

        if n_batches is None:
            sizes = [len(v) for v in self.idx_by_task.values()]
            n_batches = max(1, int(np.mean(sizes) / n_per_task))
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for t in self.tasks:
                pool = self.idx_by_task[t]
                replace = len(pool) < self.n_per_task
                chosen = self.rng.choice(pool, size=self.n_per_task, replace=replace)
                batch.extend(chosen.tolist())
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ===== Estimateur de Breslow (baseline hazard) ==============================

def breslow_baseline_cumhazard(risk_train, time_train, event_train):
    """
    H0(t) = somme_{t_i <= t, event} d_i / somme_{j in risk-set(t_i)} exp(risk_j)

    risk_train  : (n,) log-hazard relatif predit par le reseau (sortie brute
                  de la tete de survie, pas de CoxPH)
    time_train  : (n,) temps de suivi
    event_train : (n,) indicateur d'evenement (1) / censure (0)

    Retourne (unique_event_times, cumhazard) : fonction en escalier de H0.
    """
    time_train = np.asarray(time_train, dtype=float)
    event_train = np.asarray(event_train, dtype=float)
    exp_risk = np.exp(np.asarray(risk_train, dtype=float).ravel())

    event_times = np.unique(time_train[event_train == 1])
    cumhazard = np.zeros_like(event_times, dtype=float)

    H = 0.0
    for i, t in enumerate(event_times):
        at_risk = exp_risk[time_train >= t].sum()
        d = np.sum((time_train == t) & (event_train == 1))
        H += d / max(at_risk, 1e-12)
        cumhazard[i] = H
    return event_times, cumhazard


def predict_survival_function(risk_test, event_times, cumhazard, eval_times):
    """
    S(t | x) = exp(- H0(t) * exp(risk(x)))   (modele a hasards proportionnels)

    risk_test  : (m,) log-hazard relatif predit pour les sujets test
    eval_times : (T,) grille de temps sur laquelle evaluer S (pour IBS)

    Retourne un array (m, T) de probabilites de survie.
    """
    eval_times = np.asarray(eval_times, dtype=float)
    H0_at_eval = np.array([
        cumhazard[event_times <= t][-1] if np.any(event_times <= t) else 0.0
        for t in eval_times
    ])
    exp_risk_test = np.exp(np.asarray(risk_test, dtype=float).ravel())
    S = np.exp(-np.outer(exp_risk_test, H0_at_eval))
    return S