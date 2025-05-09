#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stderr/%j.err
#SBATCH --job-name=attr
#SBATCH --time=36:00:00

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


attribution=grad_input
#### multi gene models ####
# evaluate multi gene models on the 15 high h2 genes and 15 other genes, as well as 100 test genes
path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/wandb_export_FinalPaperWholeBlood_MultiGene.csv
model_name=MultiGene

#evaluate multi gene model on 15 high h2 genes
# path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/high_h2_genes.txt
# python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution

#repeat multi gene models with other genes
# path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/other_genes.txt
# python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution

#repeat multi gene models with test genes
# path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/test_genes.txt
# python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution

#repeat with remaining 271 train genes using *** only ref genome grad ***
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/remaining_genes.txt
python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution ref_only_grad_input

#### single gene models. Don't pass in gene file so the genes used during training each of the single gene models are used instead ####
# path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/revision_attributions_FinalPaperWholeBlood_SingleGene.csv
# model_name=SingleGene
# python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --attribution $attribution


### Enformer####
model_name=Enformer
enformer_seq_len=49152
# 15 high h2 genes
# path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/high_h2_genes.txt
# python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution

# # 15 other genes
# path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/other_genes.txt
# python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution

# # test genes
# path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/test_genes.txt
# python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution

#remaining 271 train genes using *** only ref genome grad ***
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/remaining_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution ref_only_grad_input