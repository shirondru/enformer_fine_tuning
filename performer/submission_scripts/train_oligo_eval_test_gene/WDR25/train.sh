#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/train/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/train/slurm_stderr/%j.err
#SBATCH --job-name=WDR25



eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source /pollard/home/sdrusinsky/miniforge3/bin/activate test_pt231
export CUBLAS_WORKSPACE_CONFIG=:4096:8

## from https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
## Get absolute path of this script
SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
cd "$(dirname "${SCRIPT_PATH}")"
cd ../../..
echo $SCRIPT_PATH
echo $(pwd)


test_gene=WDR25
base_dir=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/train_oligo_eval_test_gene/$test_gene
config_path=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/performer/configs/blood_config.yaml
script_path=./train_oligogene.py
for fold in 0 1 2; do
    for seed in 0 1 2; do
        for dir in "$base_dir/2_genes" "$base_dir/10_genes"; do
            for file in "$dir"/*.txt; do
                python $script_path --config_path $config_path --fold $fold --seed $seed --train_gene_path $file --test_gene $test_gene
            done
        done
    done
done
