from .train_gtex import *

"""
Train with a subset of genes while using downsampled train donors 
"""


def prepare_genes(config):
    """
    Overwrite to get list of genes from SingleGeneDownsampledTrainDonors directory, which will be a smaller list of genes due to the number of extra models being trained
    """
    model_type = config['model_type']
    tissue = config['tissues_to_train'].replace(' -','').replace(' ','_').replace('(','').replace(')','')
    data_dir = config['DATA_DIR']
    
    #if desired genes to use explicitly defined in config file, use only those
 
    train_gene_filedir = os.path.join(data_dir,'genes',tissue,f"{model_type}DownsampledTrainDonors")
    train_gene_filenames = os.listdir(train_gene_filedir) #if its a single gene model, there will be 1 txt file per train gene, if multi-gene it will 1 total txt file containing all genes
    
    if model_type == 'MultiGene':
        valid_genes = []
        test_gene_path = os.path.join(data_dir,'genes',tissue,'test_genes.txt')
        test_genes = parse_gene_files(test_gene_path)
    elif model_type == 'SingleGene':
        valid_genes = []
        test_genes = []
    return train_gene_filedir, train_gene_filenames, valid_genes, test_genes

def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model_type",type=str)
    parser.add_argument("--downsample_frac",type=int, help = "Fraction (out of 4) of original train set to use")

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    model_type = args.model_type
    downsample_frac = int(args.downsample_frac)
    assert downsample_frac in [1,2,3,4]
    assert model_type in ['SingleGene','MultiGene']
    

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')

    #define path to file containing downsampled list of train donors, as well as full set of validation and test donors
    downsampled_train_donor_dir = os.path.join(DATA_DIR,'cross_validation_folds/gtex/downsampled_train_files')
    evaluation_donor_dir = os.path.join(DATA_DIR,'cross_validation_folds/gtex/cv_folds')
    train_donor_path = os.path.join(downsampled_train_donor_dir,f"person_ids-train-fold{fold}_percentage{downsample_frac}_4.txt")
    valid_donor_path = os.path.join(evaluation_donor_dir,f"person_ids-val-fold{fold}.txt")
    test_donor_path = os.path.join(evaluation_donor_dir,f"person_ids-test-fold{fold}.txt")
    

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR

    
    train_gene_filedir, train_gene_filenames, valid_genes, test_genes = prepare_genes(config)
    #if training a single gene model, loop through all single gene files in the dir. If its a multi gene model, there is only 1 train gene file and loop will exit after 1 iteration
    for train_gene_filename in train_gene_filenames:
        wandb_filename = f"{config['model_type']}_{train_gene_filename.strip('.txt')}"
        train_gene_path = os.path.join(os.path.join(train_gene_filedir,train_gene_filename))
        train_genes = parse_gene_files(train_gene_path) #will contain 1 gene if this is a single gene model, else it will contain ~300 genes
        wandb_exp_name = config['experiment_name']+ 'DownsampleTrainDonors' + f'_Fold-{fold}_DownsampleFrac{downsample_frac}_4' + wandb_filename
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
        wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}DownsampleTrainDonors/{model_type}/{train_gene_filename.strip('.txt')}/Fold-{fold}/DownsampleFrac{downsample_frac}_4/{wandb.run.id}")})
        wandb.config.update({'train_donor_path':train_donor_path})
        wandb.config.update({'valid_donor_path':valid_donor_path})
        wandb.config.update({'test_donor_path':test_donor_path})
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True)
        train_gtex(wandb.config,train_genes,valid_genes,test_genes)
        wandb.finish()
            


if __name__ == '__main__':
    main()