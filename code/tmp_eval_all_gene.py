from train_gtex_all_genes import *
class AttrDict(dict):
    """A dictionary class that supports attribute-style access."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        self[key] = value
def main():
    model_type = 'MultiGene'
    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    config_path = os.path.join(current_dir,"configs/blood_config_all_genes.yaml")
    fold = 0
    ckpt_path = os.path.join(current_dir,f"../results/FinalPaperWholeBlood_MultiGPU_AllGenes/MultiGene/AllGenes/Fold-0/0zioz3o7/checkpoint_epoch_33/epoch=33-step=194208.ckpt")

    gene_dir = os.path.join(DATA_DIR,"genes/Whole_Blood")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_type'] = model_type
    config['DATA_DIR'] = DATA_DIR
    config['fold'] = fold
    save_dir = os.path.join(current_dir,f"../results/FinalPaperWholeBlood_MultiGPU_AllGenes/MultiGene/AllGenes/Fold-0/0zioz3o7/checkpoint_epoch_33")
    config['save_dir'] = save_dir


    pl.seed_everything(int(config['seed']), workers=True)
    torch.use_deterministic_algorithms(True)

    train_genes = parse_gene_files(os.path.join(gene_dir,'all_possible_train_genes.txt'))
    valid_genes = []
    test_genes = parse_gene_files(os.path.join(gene_dir,'test_genes.txt'))
    
    

    # config = wandb.config
    config = AttrDict(config)
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
    data_module = CustomDataModule(train_ds,valid_ds,test_ds,int(config.train_batch_size))

    trainer = load_trainer_multi_gpu(config)
    rank = trainer.global_rank
    save_dir = os.path.join(os.path.join(save_dir,str(rank)))
    model = LitModelHeadAdapterWrapper.load_from_checkpoint(ckpt_path)
    model.save_dir = save_dir #update so it is saved with the checkpoint that produced it
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    trainer.test(model, datamodule=data_module, ckpt_path = ckpt_path)


if __name__ == '__main__':
    main()