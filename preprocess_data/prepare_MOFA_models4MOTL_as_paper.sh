#!/bin/bash
#MSUB -c 10
#MSUB -n 7
#MSUB -q normal
#MSUB -T 43200
#MSUB -E --mem-per-cpu=7G
#MSUB -o logs/prepare_MOFA_models4MOTL_as_paper.%I.out  # Job standard output (%I is the job ID)
#MSUB -e logs/prepare_MOFA_models4MOTL_as_paper.%I.err  # Job error output (%I is the job ID)

module load r
module load pegasus
module load bcftools
source /env/cnrgh/proj/math_stats/scratch/gloaguen/TLOMICS/functions2makeDAG.sh

TASK=7
DAG=logs/prepare_MOFA_models4MOTL_as_paper.${SLURM_JOB_ID}.dag
nb_features=5000
prior_K=50
nb_CPU=10

cancer_list=("SARC" "LAML" "COAD" "ESCA" "LIHC" "PAAD" "KIRP")
for cancer in "${cancer_list[@]}"; do
        pmc_task ${cancer} "Rscript prepare_MOFA_models4MOTL_as_paper.R ${nb_features} ${cancer} ${prior_K}" ${nb_CPU}>>${DAG}
done

# Start icarust simulation
mpirun -oversubscribe -bind-to none -n "$((TASK+1))" pegasus-mpi-cluster --keep-affinity "${DAG}"

