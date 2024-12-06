#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=1000G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/train/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/train/slurm_stderr/%j.err
#SBATCH --job-name=EvalAllGenes

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

script_path=./tmp_eval_all_gene.py

srun python $script_path
