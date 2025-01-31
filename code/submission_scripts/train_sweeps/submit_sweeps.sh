#run with default settings and make sure all works
sbatch slurm_default.sh

# learning rate sweep, including an experiment with normal hyperparameters
#5e-6 is the original
#running 5e-5, 5e-4
for lr in 0.00005 0.0005 0.0000005; do
    sbatch slurm_lr_sweep.sh $lr
done

for alpha in 0 1; do
    sbatch slurm_alpha_sweep.sh $alpha
done

#Vary num_individuals_per_gene to adjust effective batch size (number for accumulate_grad_batches trainer paraneter)
#train_batch_size will be kept as 32
#For example, 128 // 32 means 4 gradient accumulation batches
#iterate through no grad accumulation, and effective batch size of 2
for num_individuals_per_gene in 32 64; do
    sbatch slurm_num_individuals_per_gene_sweep.sh $num_individuals_per_gene
done

#train with frozen trunk using default max_epochs and monitor
sbatch slurm_freeze_sweep.sh 1 150 mean_r2_across_train_genes_across_valid_donors

#train while monitoring valid gene pearson R
sbatch slurm_monitor_sweep.sh mean_pearsonr_across_valid_genes_across_valid_donors