config_path=configs/PPIF_196kb_blood_config.yaml
model_type=SingleGene
for fold in 0 1 2; do
    sbatch slurm_train_gtex.sh $config_path $fold $model_type
done


