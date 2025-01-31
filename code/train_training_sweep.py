from train_gtex import *
import subprocess



class LitModelHeadAdapterSweep(LitModel): 
    """
    Offers option to freeze enformer during sweep
    """
    def __init__(self, tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test,eval_test_gene_during_validation = False,freeze_enformer = False):
        super().__init__(tissues_to_train,save_dir,train_dataset,learning_rate,alpha,genes_for_training,genes_for_valid,genes_for_test,eval_test_gene_during_validation)
        self.freeze_enformer = freeze_enformer
        enformer = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        )

        self.model = HeadAdapterWrapper(
            enformer = enformer,
            num_tracks = len(self.tissues_to_train),
            post_transformer_embed = False, # important to keep False
            output_activation = nn.Identity()
        )
        self.write_busid()
    def forward(self, x):
        return self.model(x, freeze_enformer = self.freeze_enformer)
    def write_busid(self):
        command = ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(result.stdout)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        with open(os.path.join(self.save_dir,"gpu_bus_ids.txt"), "w") as file:
            file.write(result.stdout)

def train_gtex_sweep(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list,
               eval_test_gene_during_validation: bool = False,
               validate_first: bool = False) -> None:
    """ Overwrite to use LitModelHeadAdapterSweep to enable freezing weights"""
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes,eval_test_gene_during_validation)
    define_donor_paths(config,'gtex')
    
    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
    os.makedirs(config.save_dir,exist_ok = True)
    model = LitModelHeadAdapterSweep(
        config.tissues_to_train.split(','),
        config.save_dir,
        train_ds,
        float(config.learning_rate),
        config.alpha,
        train_genes,
        valid_genes,
        test_genes,
        eval_test_gene_during_validation,
        config['freeze_weights']
    )
    trainer = load_trainer(config)
    if validate_first:
        trainer.validate(model = model, dataloaders = DataLoader(valid_ds, batch_size = 1))
    trainer.fit(model = model,
                train_dataloaders = DataLoader(train_ds,batch_size = config.train_batch_size),
                val_dataloaders = DataLoader(valid_ds, batch_size = 1) #code for logging and storing validation/test results expects batch size of 1 for these 
                ) 
    trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = 'best')
def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--fold",type=int, nargs='?', default = 0)
    parser.add_argument("--seed", type = int, nargs='?', default = 0)
    parser.add_argument("--lr", type = float, nargs = '?',default = 5e-6) #
    parser.add_argument("--alpha", type = float, nargs = '?',default = 0.5) #
    parser.add_argument("--num_individuals_per_gene", type = int, nargs = '?',default = 128) #
    parser.add_argument("--freeze_weights", type = int, nargs = '?',default = 0) #
    parser.add_argument("--monitor", type = str, nargs = '?', default = 'mean_r2_across_train_genes_across_valid_donors') #
    parser.add_argument("--max_epochs", type = int, nargs = '?', default = 150) #To train frozen enformer with unlimited epochs

    model_type = 'MultiGene'
    config_path = "/pollard/data/projects/sdrusinsky/enformer_fine_tuning/code/configs/blood_train_sweep_config.yaml"

    args = parser.parse_args()
    fold = int(args.fold)
    seed = args.seed
    lr = args.lr
    alpha = args.alpha
    num_individuals_per_gene = args.num_individuals_per_gene
    max_epochs = args.max_epochs
    freeze_weights = args.freeze_weights
    monitor = args.monitor

    assert freeze_weights in [0,1]
    assert monitor in ['mean_r2_across_train_genes_across_valid_donors','mean_pearsonr_across_valid_genes_across_valid_donors']
    freeze_weights = bool(freeze_weights)
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
    
    config['learning_rate'] = lr #overwrite what was in config file to be the learning rate meant to be implemented in the sweep
    config['alpha'] = alpha #likewise overwrite for the sweep. Alpha = 1 means only MSE. Alpha = 0 means only cross individual contrastive term
    config['num_individuals_per_gene'] = num_individuals_per_gene #likewise overwrite for gradient accumulatin
    assert config['num_individuals_per_gene'] % config['train_batch_size'] == 0 #should be perfectly divisable
    config['monitor'] = monitor 
    config['freeze_weights'] = freeze_weights
    config['max_epochs'] = max_epochs
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
        wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}/{model_type}/{train_gene_filename.strip('.txt')}/Fold-{fold}/Seed-{seed}/{wandb.run.id}")})
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True)
        train_gtex_sweep(wandb.config,train_genes,valid_genes,test_genes,validate_first = True)
        wandb.finish()
            


if __name__ == '__main__':
    main()