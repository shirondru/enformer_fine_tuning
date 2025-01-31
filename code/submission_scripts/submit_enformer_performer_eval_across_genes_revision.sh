#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stderr/%j.err
#SBATCH --job-name=CrossGene



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



# Start with Enformer
# Variformer was trained with Log2(TPM + 2) values, but Enformer wasn't. Evaluating Enformer using both transformed and un-transformed values
script_path=./eval_enformer_across_genes.py
desired_seq_len=196608 #enformer/variformer both used with 196kb for this experiment

for log_transform in 0 1; do
    outdir=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionEvalAcrossGenes/EnformerEvalAcrossGenes/LogTransform_${log_transform}
    python $script_path --gtex_tissue 'Whole Blood' --desired_seq_len $desired_seq_len --n_center_bins 3 --log_transform $log_transform --outdir $outdir
done


script_path=./eval_performer_across_genes.py
path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/wandb_revision_TPM_196kb.csv
for log_transform in 0 1; do
    outdir=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionEvalAcrossGenes/PerformerEvalAcrossGenes/LogTransform_${log_transform}
    python $script_path --path_to_metadata $path_to_metadata --outdir $outdir --log_transform $log_transform
done


#re-do eQTL-normalized models
path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/wandb_export_FinalPaperWholeBlood_MultiGene.csv
for log_transform in 0 1; do
    outdir=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionEvalAcrossGenes/PerformerEvalAcrossGenes/eQTLNormModels/LogTransform_${log_transform}
    python $script_path --path_to_metadata $path_to_metadata --outdir $outdir --log_transform $log_transform
done

path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/wandb_export_FinalPaperWholeBlood_SingleGene.csv
for log_transform in 0 1; do
    outdir=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionEvalAcrossGenes/PerformerEvalAcrossGenes/eQTLNormModels/LogTransform_${log_transform}
    python $script_path --path_to_metadata $path_to_metadata --outdir $outdir --log_transform $log_transform
done

# path_to_metadata=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/metadata_from_past_runs/wandb_revision_TPM_196kb.csv
# log_transform=0
# outdir=/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionEvalAcrossGenes/PerformerEvalAcrossGenes/LogTransform_${log_transform}
# python $script_path --path_to_metadata $path_to_metadata --outdir $outdir --log_transform $log_transform



