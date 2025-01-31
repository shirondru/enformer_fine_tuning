fold=0
seed=0

config_path=configs/blood_config.yaml
model_type=SingleGene
sbatch slurm_train_gtex_tpm.sh $config_path $fold $model_type $seed

config_path=configs/multi_gene_196kb_blood.yaml
model_type=MultiGene
sbatch slurm_train_gtex_tpm.sh $config_path $fold $model_type $seed


