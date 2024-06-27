#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/ISM/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/ISM/slurm_stderr/%j.err
#SBATCH --job-name=EnformerISM

eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source /pollard/home/sdrusinsky/miniforge3/bin/activate test_pt231
export CUBLAS_WORKSPACE_CONFIG=:4096:8

## from https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
## Get absolute path of this script
SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
cd "$(dirname "${SCRIPT_PATH}")"
cd ..
echo $SCRIPT_PATH
echo $(pwd)

script_path=./ism_enformer.py
tissues_to_train='Brain - Cortex,Whole Blood'
path_to_genes_file=../data/genes/all_brain_blood_train_test_genes
n_center_bins=3
for desired_seq_len in 49152 196608; do
    python $script_path --path_to_genes_file $path_to_genes_file --desired_seq_len $desired_seq_len --tissues_to_train "$tissues_to_train" --n_center_bins $n_center_bins
done