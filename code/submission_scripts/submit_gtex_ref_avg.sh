fold=0
seed=0
model_type=MultiGene
config_path=configs/blood_config.yaml
for monitor in mean_loss_train_genes_across_valid_donors mean_loss_valid_genes_across_valid_donors; do
    sbatch slurm_train_gtex_ref_avg.sh $config_path $fold $model_type $seed $monitor
done