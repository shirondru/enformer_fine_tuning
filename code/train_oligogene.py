from train_gtex import *


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--seed",type = int)
    parser.add_argument("--train_gene_path",type = str)
    parser.add_argument("--test_gene",type = str)

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    seed = int(args.seed)
    train_gene_path = args.train_gene_path
    test_gene = args.test_gene
    model_type = 'OligoGene'
    

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    train_gene_filename = os.path.basename(train_gene_path).strip('.txt')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR
    config['experiment_name'] = f'train_oligogene_eval_test_gene-Train-{train_gene_filename}_Eval-{test_gene}'
    config['seed'] = seed

    train_gene_list = parse_gene_files(train_gene_path)
    wandb_exp_name = config['experiment_name'] + f'_Fold-{fold}_Seed-{seed}'
    wandb.init(
        project = 'fine_tune_enformer',
        name = wandb_exp_name,
        group = config['experiment_name'],
        config = config
    )
    wandb.config.update({'fold':fold})
    wandb.config.update({'train_genes':train_gene_list})
    wandb.config.update({'valid_genes':[test_gene]})
    wandb.config.update({'test_genes':[test_gene]})
    wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/train_oligogene_eval_test_gene/{test_gene}/{model_type}/{config['experiment_name']}/Fold-{fold}/Seed-{seed}/{wandb.run.id}")})
    pl.seed_everything(int(wandb.config.seed), workers=True)
    torch.use_deterministic_algorithms(True)
    train_gtex(wandb.config,train_gene_list,[test_gene],[test_gene],eval_test_gene_during_validation = True, validate_first = True)
    wandb.finish()
if __name__ == '__main__':
    main()