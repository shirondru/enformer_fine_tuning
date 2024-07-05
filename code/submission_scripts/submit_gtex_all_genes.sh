#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=1000G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/train/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/train/slurm_stderr/%j.err
#SBATCH --job-name=AllGenes

eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source /pollard/home/sdrusinsky/miniforge3/bin/activate test_pt231
export CUBLAS_WORKSPACE_CONFIG=:4096:8

script_path=
path_to_yaml=
fold=0
srun python $script_path --path_to_yaml $path_to_yaml --fold $fold