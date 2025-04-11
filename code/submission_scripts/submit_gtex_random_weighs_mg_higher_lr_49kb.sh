# config_path=configs/blood_config.yaml
config_path=$1
model_type=MultiGene
seed=0
fold=0
random_weights=1
keep_checkpoint=1

sbatch slurm_train_gtex_random_weights.sh $config_path $fold $seed $model_type $random_weights $keep_checkpoint

