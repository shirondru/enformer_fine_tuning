#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stderr/%j.err
#SBATCH --job-name=saturation_mutagenesis
#SBATCH --time=167:00:00
#SBATCH --nodelist=arrietty-h100-gpu02


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




#### THIS SCRIPT LAUNCHES SATURATION MUTAGENESIS EXPERIMENTS FOR DRIVER SNPS FOR MULTIGENE,SINGLEGENE, AND ENFORMER MODELS ####
attribution=saturation_mutagenesis
width=10

#### multi gene models ####
# evaluate multi gene models on the 15 high h2 genes and 15 other genes, as well as 100 test genes
position_df=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionSaturationMutagenesis/PerformerDrivers.csv
model_name=MultiGene
path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/wandb_export_FinalPaperWholeBlood_MultiGene.csv

#evaluate multi gene model on 15 high h2 genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/high_h2_genes.txt
python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

#repeat multi gene models with other genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/other_genes.txt
python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

#repeat multi gene models with test genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/test_genes.txt
python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

### Enformer 49kb on the same genes ####
model_name=Enformer
enformer_seq_len=49152
position_df=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionSaturationMutagenesis/EnformerDrivers_49kb_3CenterBins_WholeBlood.csv

# 15 high h2 genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/high_h2_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

# 15 other genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/other_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

# test genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/test_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

## Repeat with 196kb Enformer ##
enformer_seq_len=196608
position_df=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionSaturationMutagenesis/EnformerDrivers_196kb_3CenterBins_WholeBlood.csv

# 15 high h2 genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/high_h2_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

# 15 other genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/attributions_revisions/other_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df

# test genes
path_to_only_genes_file=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/genes/Whole_Blood/test_genes.txt
python ./new_attr.py --model_name $model_name --enformer_seq_len $enformer_seq_len --path_to_only_genes_file $path_to_only_genes_file --attribution $attribution --width $width --position_df $position_df


### Single Gene Models ###
#### single gene models. Don't pass in gene file so the genes used during training each of the single gene models are used instead ####
path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/revision_attributions_FinalPaperWholeBlood_SingleGene.csv
model_name=SingleGene
position_df=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionSaturationMutagenesis/PerformerDrivers.csv
python ./new_attr.py --path_to_metadata $path_to_metadata --model_name $model_name --attribution $attribution --width $width --position_df $position_df
