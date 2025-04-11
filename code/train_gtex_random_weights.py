from train_gtex import *
from pl_models import LitModelHeadAdapterWrapperRandom

def load_model(random_weights,config,train_ds,train_genes,valid_genes,test_genes,eval_test_gene_during_validation):
    
    if random_weights:
        if hasattr(config,'weight_decay'):
            weight_decay = float(config.weight_decay)
        else:
            weight_decay = None
        model = LitModelHeadAdapterWrapperRandom(
        config.tissues_to_train.split(','),
        config.save_dir,
        train_ds,
        float(config.learning_rate),
        config.alpha,
        train_genes,
        valid_genes,
        test_genes,
        eval_test_gene_during_validation,
        weight_decay = weight_decay #for trying smaller weight decay with random weights
        )
        config.update({'Random Weights':True})
    else:
        model = LitModelHeadAdapterWrapper(
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
        config.update({'Random Weights':False})
    return model

def train_random_weights(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list,
               eval_test_gene_during_validation: bool = False,
               validate_first: bool = False,
               random_weights: bool = True,
               keep_checkpoint: bool = True) -> None:
    """ Overwrite to use `LitModelHeadAdapterWrapperRandom`"""
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes,eval_test_gene_during_validation)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
    model = load_model(random_weights,config,train_ds,train_genes,valid_genes,test_genes,eval_test_gene_during_validation)
    trainer = load_trainer(config)
    if validate_first:
        trainer.validate(model = model, dataloaders = DataLoader(valid_ds, batch_size = 1))
    trainer.fit(model = model,
                train_dataloaders = DataLoader(train_ds,batch_size = config.train_batch_size),
                val_dataloaders = DataLoader(valid_ds, batch_size = 1) #code for logging and storing validation/test results expects batch size of 1 for these 
                ) 
    trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = 'best')
    if not keep_checkpoint:
        delete_checkpoint(trainer)
def main():
    """
    Changes from normal personal genome training:
    Train from random weights
    """


    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model_type",type=str)
    parser.add_argument("--seed", type = int, nargs='?')
    parser.add_argument("--random_weights", type = int, nargs='?')
    parser.add_argument("--keep_checkpoint", type = int, nargs='?',default = 1)

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    seed = args.seed
    model_type = args.model_type
    random_weights = args.random_weights
    keep_checkpoint = args.keep_checkpoint
    assert model_type in ['SingleGene','MultiGene']
    assert random_weights in [0,1]
    random_weights = bool(random_weights)
    assert keep_checkpoint in [0,1]
    keep_checkpoint = bool(keep_checkpoint)

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
    config['experiment_name'] = 'FinalPaperWholeBloodRevisions_RandomWeights' #overwrite to avoid needing to make a new config file
    
    train_gene_filedir, train_gene_filenames, valid_genes, test_genes = prepare_genes(config)
    #if training a single gene model, loop through all single gene files in the dir. If its a multi gene model, there is only 1 train gene file and loop will exit after 1 iteration
    for train_gene_filename in train_gene_filenames:
        wandb_filename = f"{config['model_type']}_{train_gene_filename.strip('.txt')}"
        train_gene_path = os.path.join(os.path.join(train_gene_filedir,train_gene_filename))
        train_genes = parse_gene_files(train_gene_path) #will contain 1 gene if this is a single gene model, else it will contain ~300 genes
        wandb_exp_name = config['experiment_name'] + f'_Fold-{fold}_Seed-{seed}' + wandb_filename + f"RandomWeights{random_weights}"
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
        wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}/{model_type}/{train_gene_filename.strip('.txt')}/Fold-{fold}/Seed-{seed}/RandomWeights{random_weights}/{wandb.run.id}")})
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True)
        train_random_weights(wandb.config,train_genes,valid_genes,test_genes,random_weights = random_weights,validate_first = True, keep_checkpoint = keep_checkpoint)
        wandb.finish()
            
if __name__ == '__main__':
    main()