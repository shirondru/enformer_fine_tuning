from train_gtex import *
class LitModelHeadAdapterWrapperRandom(LitModel):
    """Same as LitModelHeadAdapterWrapper but Enformer is loaded using random weights not pre-trained weights"""
    def __init__(self, tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test,eval_test_gene_during_validation = False):
        super().__init__(tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test,eval_test_gene_during_validation)

        random_enformer = Enformer.from_hparams(
                dim = 1536,
                depth = 11,
                heads = 8,
                output_heads = dict(human = 5313, mouse = 1643),
                target_length = -1
            )

        self.model = HeadAdapterWrapper(
            enformer = random_enformer,
            num_tracks = len(self.tissues_to_train),
            post_transformer_embed = False, # important to keep False
            output_activation = nn.Identity()
        )

    def forward(self, x):
        return self.model(x, freeze_enformer = False) 

def load_model(random_weights,config,train_ds,train_genes,valid_genes,test_genes,eval_test_gene_during_validation):
    if random_weights:
        model = LitModelHeadAdapterWrapperRandom(
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
               random_weights: bool = True) -> None:
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


    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    seed = args.seed
    model_type = args.model_type
    random_weights = args.random_weights
    assert model_type in ['SingleGene','MultiGene']
    assert random_weights in [0,1]
    random_weights = bool(random_weights)

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
        train_random_weights(wandb.config,train_genes,valid_genes,test_genes,random_weights = random_weights)
        wandb.finish()
            
if __name__ == '__main__':
    main()