#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stderr/%j.err
#SBATCH --job-name=EvalSGTestGenes

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

subset=$1
n_subsets=$2
script_path=./eval_performer_on_other_genes.py
path_to_metadata=./metadata_from_past_runs/wandb_export_FinalPaperWholeBlood_SingleGene.csv


python $script_path --path_to_metadata $path_to_metadata --donor_pickiness all_donors --path_to_test_gene_file ../data/genes/Whole_Blood/test_genes.txt --subset $subset --n_subsets $n_subsets