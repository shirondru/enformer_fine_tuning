#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/eval/slurm_stderr/%j.err
#SBATCH --job-name=EnformerEval

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

#evaluate blood
path_to_train_genes_file=../data/genes/Whole_Blood/MultiGene/300_train_genes.txt
path_to_eval_genes_file=../data/genes/Whole_Blood/test_genes.txt
config_path=configs/blood_config.yaml
name=WholeBloodTrainTestGenes
n_center_bins=3
python_script=eval_enformer_gtex.py
for desired_seq_len in 49152 196608; do
    for donor_fold in 0 1 2; do
        python $python_script --path_to_train_genes_file $path_to_train_genes_file --path_to_eval_genes_file $path_to_eval_genes_file --config_path $config_path --name $name --donor_fold $donor_fold  --n_center_bins $n_center_bins --desired_seq_len $desired_seq_len 
    done
done


#evaluate brain
path_to_train_genes_file=../data/genes/Brain_Cortex/MultiGene/300_train_genes.txt
path_to_eval_genes_file=../data/genes/Brain_Cortex/test_genes.txt
config_path=configs/brain_config.yaml
name=BrainCortexTrainTestGenes
n_center_bins=3
python_script=eval_enformer_rosmap.py
donor_fold=0 #only doing one fold because evaluating is test with all people from GTEx. So other training would use the same GTEx individuals and it is redundant
for desired_seq_len in 49152 196608; do
    python $python_script --path_to_train_genes_file $path_to_train_genes_file --path_to_eval_genes_file $path_to_eval_genes_file --config_path $config_path --name $name --donor_fold $donor_fold  --n_center_bins $n_center_bins --desired_seq_len $desired_seq_len 
done

