#!/bin/bash
#SBATCH -n 100               # Nombre de workers PMC parallèles
#SBATCH -c 1                  # Coeurs par worker (cf. n_trials_per_worker)
#SBATCH -t 24:00:00           # Budget temps global du pool
#SBATCH -o logs/pmc_%j.out
#SBATCH -e logs/pmc_%j.out

module load pegasus

module load python

cd /env/cnrgh/proj/math_stats/scratch/hlegrand/CustOmics

source ../venv_sksurv_dev/bin/activate
# pegasus-mpi-cluster lit tasks.dag et distribue les TASK sur les workers
# en respectant les dépendances EDGE. -m 5 : marge de checkpoint (comme
# dans l'exemple fourni) ; ajuster si besoin.
srun -n $((SLURM_NTASKS + 1)) --overcommit \
    pegasus-mpi-cluster -m 5 ../tasks.dag