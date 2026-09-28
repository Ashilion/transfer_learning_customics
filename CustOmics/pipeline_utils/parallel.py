"""Lancement d'une étude Optuna, séquentielle ou multi-process, avec affinité CPU optionnelle."""
import multiprocessing as mp
import os

import torch

from pipeline_utils.affinity import split_cpus_ht_pairs
from pipeline_utils.optuna_utils import get_or_create_study

# Rempli avant le fork : les workers en héritent en copy-on-write (rien n'est picklé
# à part worker_id), donc on peut y mettre des closures et de gros DataFrames.
_STATE = {}


def _pin(cpus):
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(len(cpus))


def _worker(worker_id):
    s = _STATE
    if s["cpu_chunks"] is not None:
        _pin(s["cpu_chunks"][worker_id])
    print(f"[worker {worker_id}] pid {os.getpid()} | cpus {sorted(os.sched_getaffinity(0))}")
    study = get_or_create_study(s["study_name"], s["journal_file"])
    study.optimize(s["objective_factory"](), n_trials=s["n_trials"],
                   timeout=s["timeout"], catch=(Exception,))
    print(f"[worker {worker_id}] done")


def run_study(objective_factory, study_name, journal_file, n_trials, timeout=None,
              multiproc=1, affinity=False, initial_affinity=None):
    """objective_factory : callable sans argument qui renvoie la fonction objective."""
    multiproc = max(1, multiproc)
    if affinity and initial_affinity is None:
        raise ValueError("--affinity nécessite initial_affinity (capturé avant import torch)")

    if multiproc == 1:
        if affinity:
            _pin(initial_affinity)
        study = get_or_create_study(study_name, journal_file)
        study.optimize(objective_factory(), n_trials=n_trials, timeout=timeout, catch=(Exception,))
        return

    cpu_chunks = None
    if affinity:
        cpu_chunks = split_cpus_ht_pairs(initial_affinity, multiproc)
        os.sched_setaffinity(0, initial_affinity)
        print(f"  -> affinité : {cpu_chunks}")

    _STATE.update(objective_factory=objective_factory, study_name=study_name,
                  journal_file=journal_file, n_trials=n_trials, timeout=timeout,
                  cpu_chunks=cpu_chunks)
    with mp.get_context("fork").Pool(processes=multiproc) as pool:
        pool.map(_worker, range(multiproc))