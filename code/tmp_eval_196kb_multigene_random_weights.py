from train_gtex_random_weights import *

def main():
    model_type = 'MultiGene'
    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    
    config_path = os.path.join(current_dir,"configs/multi_gene_196kb_blood.yaml")
    remaining_tests = [(0,0,'60pxxfdk',False,'epoch=19-step=24080.ckpt'),(0,0,'8jqyho84',True,'epoch=19-step=24080.ckpt')]
    for (fold,seed, run_id,random_weight,ckpt_epoch) in remaining_tests:
        ckpt_path = os.path.join(current_dir,f"../results/FinalPaperWholeBloodRevisions_RandomWeights/MultiGene/300_train_genes/Fold-{fold}/Seed-{seed}/RandomWeights{random_weight}/{run_id}/checkpoints/{ckpt_epoch}")


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
            wandb_exp_name = config['experiment_name'] + f'_Fold-{fold}_' + wandb_filename
            wandb.init(
                project = 'fine_tune_enformer',
                id = run_id,
                resume = 'must',
                name = wandb_exp_name,
                group = config['experiment_name'],
                config = config
            )

            pl.seed_everything(int(wandb.config.seed), workers=True)
            torch.use_deterministic_algorithms(True)
            

            config = wandb.config
            ensure_no_gene_overlap(train_genes,valid_genes,test_genes)
            define_donor_paths(config,'gtex')

            train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
            trainer = load_trainer(config)
            if random_weight:
                model = LitModelHeadAdapterWrapperRandom.load_from_checkpoint(ckpt_path)
            else:
                model = LitModelHeadAdapterWrapper.load_from_checkpoint(ckpt_path)
            trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = ckpt_path)
            wandb.finish()


if __name__ == '__main__':
    main()