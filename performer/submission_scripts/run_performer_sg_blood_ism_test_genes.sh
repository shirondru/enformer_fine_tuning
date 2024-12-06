#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" #cd into the directory containing this script
echo | pwd
n_subsets=10
for subset in 0 1 2 3 4 5 6 7 8 9; do
    sbatch ./submit_performer_sg_blood_ism_test_genes.sh $subset $n_subsets
done


