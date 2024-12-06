from train_gtex import *
def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model_type",type=str)

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    model_type = args.model_type
    assert model_type in ['SingleGene','MultiGene']
    

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    
    valid_genes = []
    test_genes = []
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR
    config['experiment_name'] = config['experiment_name'] + 'TestGenes'

    train_gene_path = os.path.join(DATA_DIR,"genes/Whole_Blood/test_genes.txt") #train on test genes
    train_gene_list = parse_gene_files(train_gene_path)
    for train_gene in train_gene_list:
        wandb_filename = f"{config['model_type']}_{train_gene}"
        train_genes = [train_gene] #expect a list even though its a single gene
        wandb_exp_name = config['experiment_name'] + f'_Fold-{fold}_' + wandb_filename
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
        wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}/{model_type}/{train_gene}/Fold-{fold}/{wandb.run.id}")})
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True)
        train_gtex(wandb.config,train_genes,valid_genes,test_genes)
        wandb.finish()
if __name__ == '__main__':
    main()