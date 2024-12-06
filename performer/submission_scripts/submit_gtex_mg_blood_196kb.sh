config_path=configs/multi_gene_196kb_blood.yaml
model_type=MultiGene
for fold in 0 1 2; do
    sbatch slurm_train_gtex.sh $config_path $fold $model_type
done



