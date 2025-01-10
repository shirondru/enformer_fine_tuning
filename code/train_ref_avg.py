from train_gtex import *

class RefAvgModel(LitModelHeadAdapterWrapper):
    def __init__(self, tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test,eval_test_gene_during_validation = False):
        super().__init__(tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test,eval_test_gene_during_validation)
    def loss_fn(self,y_hat, y, alpha = 1):
        mse = masked_mse(y_hat,y)
        return mse


class RefAvgMetricLogger(MetricLogger):
    """
    Logs personal genome metrics during test but not valid epochs because valid epochs do not contain personal genomes
    """
    def __init__(self):
        super().__init__()
    
    def calc_loss(self,gene_name,tissue,donor_split,gene_split,pl_module):
        loss_fn = pl_module.loss_fn
        all_predictions = torch.cat(pl_module.pred_dict[gene_name][tissue])
        all_targets = torch.cat(pl_module.target_dict[gene_name][tissue])
        loss_val = loss_fn(all_predictions.unsqueeze(1),all_targets.unsqueeze(1)).cpu().numpy() #unsqueeze because the values per tissue were returned and have shape [batch_size]. Add a tissue dimension, which the loss function expects
        self.metrics_history['per_gene_tissue_val_loss'].append(loss_val)
        self.metrics_history['donor_split'].append(donor_split)
        self.metrics_history['gene_split'].append(gene_split)
        self.metrics_history['epoch'].append(self.epoch)
        self.metrics_history['tissue'].append(tissue)
        self.metrics_history['gene_name'].append(gene_name)
    def log_all_loss_vals(self,pl_module,donor_split):
        for gene_name in pl_module.pred_dict.keys():
            if gene_name in pl_module.genes_for_training:
                gene_split = 'train'
            elif gene_name in pl_module.genes_for_valid:
                gene_split = 'valid'
            elif gene_name in pl_module.genes_for_test:
                gene_split = 'test'
            else:
                raise Exception(f"Gene {gene_name} not in desired train set, valid set, or test set")
            for tissue_idx,tissue in enumerate(pl_module.tissues_to_train):  # loop through each tissue and calculate pearsonr
                self.calc_loss(gene_name,tissue,donor_split,gene_split,pl_module)
        metrics_history = pd.DataFrame(self.metrics_history)
        name = f"RefGenomeAvgExpression_{donor_split}DonorLoss_Epoch{self.epoch}.csv"
        metrics_history.to_csv(os.path.join(pl_module.save_dir,name), index=False)
        for gene_split in list(metrics_history['gene_split'].unique()):
            mean_epoch_loss = metrics_history[metrics_history['gene_split'] == gene_split]['per_gene_tissue_val_loss'].mean()
            epoch_dict = { f"mean_loss_{gene_split}_genes_across_{donor_split}_donors" : mean_epoch_loss}
            pl_module.log_dict(epoch_dict)
    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics_history = {'epoch': [],'gene_name': [],'tissue': [],'per_gene_tissue_val_loss': [],'gene_split':[],'donor_split': []}
        self.get_epoch(trainer,pl_module)
        self.log_predictions(trainer,pl_module,'valid') #log predictions but nothing else
        self.log_all_loss_vals(pl_module,'valid')
    def on_test_epoch_end(self,trainer,pl_module):
        self.metrics_history = {'epoch': [],'pearsonr':[], 'r2':[],'gene_name': [],'tissue': [],'per_gene_tissue_val_loss': [],'gene_split':[],'donor_split': []}
        self.log_and_save_eval(trainer,pl_module,'test')   



def load_RefAvg_trainer(config):
    """Overwrite to use `RefAvgMetricLogger` instead of `MetricLogger` """

    ##load callbacks
    checkpoint_dir = os.path.join(config.save_dir,'checkpoints')
    os.makedirs(checkpoint_dir,exist_ok = True)
    if hasattr(config,'monitor'):
        monitor = config.monitor
    else:
        monitor = 'mean_r2_across_train_genes_across_valid_donors'
    mode = 'max'

    checkpoint_callback =  ModelCheckpoint(
            dirpath = checkpoint_dir,
            save_top_k = 1,
            monitor = monitor,
            mode = mode
        )
    early_stopper = EarlyStopping(monitor = monitor, mode = mode, min_delta = 0, patience = int(config.patience))
    metric_logger = RefAvgMetricLogger()


    trainer = Trainer(
        max_epochs = config.max_epochs,
        precision = config.precision,
        accumulate_grad_batches = config.num_individuals_per_gene // config.train_batch_size, #accumulate as many batches as necessary to achieve num_individuals_per_gene effective samples per gradient accumulated step
        gradient_clip_val = config.gradient_clip_val,
        callbacks = [checkpoint_callback,metric_logger,early_stopper],
        logger = WandbLogger(),
        num_sanity_val_steps = 0, #don't do any validation before training, as all sorts of R2 metrics will be computed during callbacks. Could lead to error with small sample size
        log_every_n_steps = 1,
        check_val_every_n_epoch = config.valid_metrics_save_freq
    )
    config.update({'Gradient Accumulation Effective Batch Size': config.num_individuals_per_gene // config.train_batch_size })
    return trainer

def load_ref_avg_datasets(config,train_genes,valid_genes,test_genes):
    """
    Loads GTExRefAvg train and valid datasets, and a personal genomes test dataset.
    Also overwrite `load_trainer` to use RefAvgMetricLogger instead of MetricLogger
    """
    tissues_to_train = config.tissues_to_train.split(',') #ex: 'Whole Blood' -> ['Whole Blood]
    assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"
    tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')

    #load gene expression df, merge in gene names onto gene ids
    expression_dir = os.path.join(config.DATA_DIR,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"{tissue_str}.v8.normalized_expression.bed.gz")
    gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gene_expression_df = gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')
    

    genes_to_train = train_genes
    genes_to_validate = train_genes + valid_genes
    genes_to_test = train_genes + valid_genes + test_genes

    train_ds = GTExRefDataset(tissues_to_train, genes_to_train, config.seq_length,config.train_donor_path,gene_expression_df,config.DATA_DIR)
    valid_ds = GTExRefDataset(tissues_to_train, genes_to_validate, config.seq_length,config.valid_donor_path,gene_expression_df,config.DATA_DIR)  
    test_ds = GTExDataset(tissues_to_train, genes_to_test, config.seq_length,-1,config.test_donor_path,gene_expression_df,config.DATA_DIR)
        

    ensure_no_donor_overlap(train_ds,valid_ds,test_ds)

    return train_ds, valid_ds, test_ds
def train_ref_avg(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list,
               eval_test_gene_during_validation: bool = False,
               validate_first: bool = False) -> None:
    """
    Overwrite to change datasets (RefAvg train and valid datasets, personal genome test dataset)
    Overwrite to use RefAvgMetricLogger not metric logger (the former won't error when trying to calc personal genome metrics when there is only ref genome values)
    Overwrite to use RefAvgMetricLogger, which will not include a cross-individual contrastive term in the loss
    """
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes,eval_test_gene_during_validation)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_ref_avg_datasets(config,train_genes, valid_genes,test_genes)
    model = RefAvgModel(
        config.tissues_to_train.split(','),
        config.save_dir,
        train_ds,
        float(config.learning_rate),
        config.alpha,
        train_genes,
        valid_genes,
        test_genes,
        eval_test_gene_during_validation
        )
    trainer = load_RefAvg_trainer(config)
    if validate_first:
        trainer.validate(model = model, dataloaders = DataLoader(valid_ds, batch_size = 1))
    trainer.fit(model = model,
                train_dataloaders = DataLoader(train_ds,batch_size = config.train_batch_size),
                val_dataloaders = DataLoader(valid_ds, batch_size = 1) #code for logging and storing validation/test results expects batch size of 1 for these 
                ) 
    trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = 'best')

def main():
    """
    Changes from normal personal genome training:
    Train and validation sets include reference genome inputs and average expression targets (averaged over people from the train or validation set)
    Test is normal personal genome `GTExDataset`

    Using RefAvgMetricLogger to avoid erroring from trying to compute personal genome metrics when there aren't multiple individuals (only ref geonome) during training and validation

    Loss only includes MSE, no cross-individual contrastive term
    Monitor the loss using train genes over held out people instead (meaning that I am using train genes, but average expression among people from the validation set). Doing this instead of loss over validation genes because it is more similar to the personal genome scheme, where we monitor the R2 over train genes in held out people.
    Monitoring the train gene loss (expression averaged over unseen ppl) is the default, but args offer an option to monitor unseen genes over unseen people, which is more akin to the setting Enformer was trained in originally
    """


    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model_type",type=str)
    parser.add_argument("--seed", type = int, nargs='?')
    parser.add_argument("--monitor", type = str, nargs='?',default = 'mean_loss_train_genes_across_valid_donors')


    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    seed = args.seed
    model_type = args.model_type
    monitor = args.monitor
    assert model_type in ['SingleGene','MultiGene']
    assert monitor in ['mean_loss_train_genes_across_valid_donors','mean_loss_valid_genes_across_valid_donors']

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR
    if type(seed) == int:
        seed = int(seed) #ensure read as int if defined
        config['seed'] = seed
    else:
        assert 'seed' in config
        seed = int(config['seed'])
    config['alpha'] = 1 #Loss is overwritten to only include MSE, which is akin to an alpha of 1
    config['experiment_name'] = "FinalPaperWholeBloodRevisions_TrainRefGenome"
    config['monitor'] = monitor

    train_gene_filedir, train_gene_filenames, valid_genes, test_genes = prepare_genes(config)
    #if training a single gene model, loop through all single gene files in the dir. If its a multi gene model, there is only 1 train gene file and loop will exit after 1 iteration
    for train_gene_filename in train_gene_filenames:
        wandb_filename = f"{config['model_type']}_{train_gene_filename.strip('.txt')}"
        train_gene_path = os.path.join(os.path.join(train_gene_filedir,train_gene_filename))
        train_genes = parse_gene_files(train_gene_path) #will contain 1 gene if this is a single gene model, else it will contain ~300 genes
        wandb_exp_name = config['experiment_name'] + f'_Fold-{fold}_Seed-{seed}' + wandb_filename
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
        wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}/{model_type}/{train_gene_filename.strip('.txt')}/Fold-{fold}/Seed-{seed}/Monitor-{monitor}/{wandb.run.id}")})
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True)
        train_ref_avg(wandb.config,train_genes,valid_genes,test_genes,validate_first = True)
        wandb.finish()
            

if __name__ == '__main__':
    main()