from train_gtex import *


def load_tpm_datasets(config,train_genes,valid_genes,test_genes):
    """
    Overwrite to use Log2TPM + 2 expression data, as opposed to the GTEx expression data normalized for eQTL studies, which are transformed to appear standard normal.
    TPM data contains information about avearge expression of the gene in a given tissue relative to other genes in the same tissue. The other dataset does not (expression is centered around 0 and emphasizes differences between people)
    """
    def instantiate_dataset(config,gene_list,donor_path,tissues_to_train,gene_expression_df,num_individuals_per_gene):
        ds = GTExDataset(
            tissues_to_train,
            gene_list,
            config.seq_length,
            num_individuals_per_gene,
            donor_path,
            gene_expression_df,
            config.DATA_DIR
        )
        return ds

    
    tissues_to_train = config.tissues_to_train.split(',') #ex: 'Whole Blood' -> ['Whole Blood]
    assert tissues_to_train == ['Whole Blood'],"This function loads only whole blood data"
    assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"
    tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')
    gene_expression_df = pd.read_csv(os.path.join(config.DATA_DIR,"gene_tpm_2017-06-05_v8_whole_blood.gct.gz"),sep = '\t',skiprows = 2)

    for col in gene_expression_df.columns:
        if col.startswith('GTEX'):
            gene_expression_df = gene_expression_df.rename(columns = {col:'-'.join(col.split('-')[:2])})
    #log2 transform expression columns
    gene_expression_df.loc[:,[col for col in gene_expression_df.columns if col.startswith('GTEX')]] = np.log2(gene_expression_df.loc[:,[col for col in gene_expression_df.columns if col.startswith('GTEX')]] + 2)
    gene_expression_df = gene_expression_df.rename(columns = {'Name':'gene_id'}) #make compatible with dataset
    #train on just train genes. When validating w/ set of validation ppl, evaluate on train genes and valid donors (valid donors can be empty)
    #when evaluating on test set of donors, evaluate on all genes. The lightning module keeps track of which is which and early stopping and checkpoint callbacks
    #can monitor different groups of genes
    genes_to_train = train_genes
    genes_to_validate = train_genes + valid_genes
    genes_to_test = train_genes + valid_genes + test_genes

    train_ds = instantiate_dataset(config,genes_to_train,config.train_donor_path,tissues_to_train,gene_expression_df,config.num_individuals_per_gene)
    #instantiate eval datasets using different donor paths, lists of genes, and set num_individuals_per_gene to -1 so all people are used. Otherwise those that don't fit a multiple of gradient accumulated batch size will be dropped
    valid_ds = instantiate_dataset(config,genes_to_validate,config.valid_donor_path,tissues_to_train,gene_expression_df,-1)
    test_ds = instantiate_dataset(config,genes_to_test,config.test_donor_path,tissues_to_train,gene_expression_df,-1)

    ensure_no_donor_overlap(train_ds,valid_ds,test_ds)

    return train_ds, valid_ds, test_ds

def train_gtex_tpm(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list,
               eval_test_gene_during_validation: bool = False,
               validate_first: bool = False) -> None:
    """
    Overwrite to load tpm datasets instead of dataset using eQTL normalized data
    """
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes,eval_test_gene_during_validation)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_tpm_datasets(config,train_genes, valid_genes,test_genes)
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
    Changes made:
    1. Force the  usage of 196kb sequences and batch size of 4 (smaller batch size to facilitate longer sequence) for revision
    2. Force the usage of Whole Blood tissue to ensure a gene expression dataframe from a different tissue is not inadvertantly used.
    3. Use Log2(TPM) + 2 data from Whole Blood as as opposed to the GTEx expression data normalized for eQTL studies, which are transformed to appear standard normal.
        TPM data contains information about avearge expression of the gene in a given tissue relative to other genes in the same tissue. The other dataset does not (expression is centered around 0 and emphasizes differences between people)
        Still using personal genome seqeunces
    4. For single gene models, train on a random subset of 25 single genes instead of all 301
    """
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model_type",type=str)
    parser.add_argument("--seed", type = int, nargs='?')

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    seed = args.seed
    model_type = args.model_type
    assert model_type in ['SingleGene','MultiGene']

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

    config['experiment_name'] = 'FinalPaperWholeBlood_RevisionTrainTPM_196kb'
    config['seq_length'] = 196608
    config['train_batch_size'] = 4
    config['tissues_to_train'] = "Whole Blood"

    train_gene_filedir, train_gene_filenames, valid_genes, test_genes = prepare_genes(config)
    train_gene_filenames = train_gene_filenames[:25] #train  25 single gene models for revision
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
        train_gtex_tpm(wandb.config,train_genes,valid_genes,test_genes)
        wandb.finish()
            


if __name__ == '__main__':
    main()