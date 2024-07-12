from train_gtex import *
from datasets import CustomDataModule, CustomDistributedSampler


def load_trainer_multi_gpu(config):
    metric_logger,early_stopper,checkpoint_callback = load_callbacks(config)
    trainer = Trainer(
        max_epochs = config.max_epochs,
        precision = config.precision,
        accumulate_grad_batches = config.num_individuals_per_gene // config.train_batch_size, #accumulate as many batches as necessary to achieve num_individuals_per_gene effective samples per gradient accumulated step
        gradient_clip_val = config.gradient_clip_val,
        callbacks = [checkpoint_callback,metric_logger,early_stopper],
        logger = WandbLogger(),
        num_sanity_val_steps = 0, #don't do any validation before training, as all sorts of R2 metrics will be computed during callbacks. Could lead to error with small sample size
        log_every_n_steps = 1,
        accelerator="gpu", 
        devices=int(config.n_gpus), 
        check_val_every_n_epoch = config.valid_metrics_save_freq,
        strategy="ddp_find_unused_parameters_true", ## HeadAdapterWrapper has unused parameters
        # strategy = 'ddp
    )
    assert trainer.world_size > 1,"This script is for multi-gpu training. Increase config.n_gpus"
    return trainer
def train_gtex_multi_gpu(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list) -> None:
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
    data_module = CustomDataModule(train_ds,valid_ds,test_ds,int(config.train_batch_size))

    model = LitModelHeadAdapterWrapper(
        config.tissues_to_train.split(','),
        config.save_dir,
        train_ds,
        float(config.learning_rate),
        config.alpha,
        train_genes,
        valid_genes,
        test_genes
    )
    trainer = load_trainer_multi_gpu(config)

    if hasattr(config,'train_ckpt_resume_path'): #resume training from desired ckpt path, in case job ran out of time on SLURM or crashed etc
        trainer.fit(model = model, datamodule = data_module,ckpt_path = config.train_ckpt_resume_path) 
    else:
        trainer.fit(model = model, datamodule = data_module) 
    trainer.test(model, datamodule=data_module, ckpt_path = 'best')




def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    model_type = 'MultiGene'

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    
    

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR
    tissue = config['tissues_to_train'].replace(' -','').replace(' ','_').replace('(','').replace(')','')

    train_genes = parse_gene_files(os.path.join(DATA_DIR,'genes',tissue,'all_possible_train_genes.txt'))
    valid_genes = []
    test_genes = parse_gene_files(os.path.join(DATA_DIR,'genes',tissue,'test_genes.txt'))
    wandb_exp_name = config['experiment_name'] + f'_Fold-{fold}'
    wandb.init(
        project = 'fine_tune_enformer',
        name = wandb_exp_name,
        group = config['experiment_name'],
        config = config
    )
    wandb.config.update({'fold':fold})
    wandb.config.update({'train_genes':train_genes})
    wandb.config.update({'valid_genes':valid_genes})
    wandb.config.update({'test_genes':test_genes})
    wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}/{model_type}/AllGenes/Fold-{fold}/{wandb.run.id}")})
    pl.seed_everything(int(wandb.config.seed), workers=True)
    torch.use_deterministic_algorithms(True)
    train_gtex_multi_gpu(wandb.config,train_genes,valid_genes,test_genes)
    wandb.finish()
            


if __name__ == '__main__':
    main()