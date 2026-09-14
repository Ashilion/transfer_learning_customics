#!/bin/bash


for f in data/clinical/*_clinical.pickle; do
    cancer=$(basename "$f" _clinical.pickle)
    sbatch --nodelist="inti[000-003,005-006]" scripts/tl_source_optu_multi.sh $cancer
done
