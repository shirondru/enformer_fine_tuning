from train_gtex import parse_gene_files,prepare_genes,ensure_no_donor_overlap, define_donor_paths, load_callbacks, load_trainer, ensure_no_gene_overlap
from pl_models import *
from datasets import *
import argparse
import yaml
import wandb
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore', '.*does not have many workers.*')
def load_rosmap_datasets(config,train_genes,valid_genes,test_genes):    
    tissues_to_train = config.tissues_to_train.split(',') #ex: 'Whole Blood' -> ['Whole Blood]
    assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"
    assert tissues_to_train == ['Brain - Cortex'], "Rosmap training only supported while evaluating on GTEx Brain - Cortex data"
    tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')

    #load gene expression df, merge in gene names onto gene ids
    expression_dir = os.path.join(config.DATA_DIR,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"{tissue_str}.v8.normalized_expression.bed.gz")
    gtex_gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gtex_gene_expression_df = gtex_gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')


    rosmap_df_path = os.path.join(config.DATA_DIR,"expression-rosmap.parquet") #not saved in data_dir because this is protected access
    rosmap_gene_expression_df = pd.read_parquet(rosmap_df_path)
    
    
    #train on just train genes. When validating w/ set of validation ppl, evaluate on train genes and valid donors (valid donors can be empty)
    #when evaluating on test set of donors, evaluate on all genes. The lightning module keeps track of which is which and early stopping and checkpoint callbacks
    #can monitor different groups of genes
    genes_to_train = train_genes
    genes_to_validate = train_genes + valid_genes
    genes_to_test = train_genes + valid_genes + test_genes

    train_ds = ROSMAPDataset(genes_to_train, config.seq_length, config.num_individuals_per_gene, config.train_donor_path, rosmap_gene_expression_df, config.DATA_DIR)
    valid_ds = ROSMAPDataset(genes_to_validate, config.seq_length, -1, config.valid_donor_path, rosmap_gene_expression_df, config.DATA_DIR)
    test_ds = GTExDataset(tissues_to_train, genes_to_test, config.seq_length, -1, config.test_donor_path, gtex_gene_expression_df, config.DATA_DIR)

    ensure_no_donor_overlap(train_ds,valid_ds,test_ds)

    return train_ds, valid_ds, test_ds

def train_rosmap(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list) -> None:
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes)
    define_donor_paths(config,'rosmap')

    train_ds, valid_ds, test_ds = load_rosmap_datasets(config,train_genes, valid_genes,test_genes)
    model = LitModelHeadAdapterWrapper(
        config.tissues_to_train.split(','),
        config.save_dir,
        train_ds,
        float(config.learning_rate),
        config.alpha,
        train_genes,
        valid_genes,
        test_genes
    )
    trainer = load_trainer(config)
    trainer.fit(model = model,
                train_dataloaders = DataLoader(train_ds,batch_size = config.train_batch_size),
                val_dataloaders = DataLoader(valid_ds, batch_size = 1) #code for logging and storing validation/test results expects batch size of 1 for these 
                ) 
    trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = 'best')


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model_type",type=str)
    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    model_type = args.model_type

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    

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
        wandb.init(
            project = 'fine_tune_enformer',
            name = config['experiment_name'] + f'_Fold-{fold}_' + wandb_filename,
            group = config['experiment_name'],
            config = config
        )
        wandb.config.update({'fold':fold})
        wandb.config.update({'train_genes':train_genes})
        wandb.config.update({'valid_genes':valid_genes})
        wandb.config.update({'test_genes':test_genes})
        wandb.config.update({'save_dir' : os.path.join(current_dir,f"../results/{config['experiment_name']}/{model_type}/{train_gene_filename.strip('.txt')}/Fold-{fold}/{wandb.run.id}")})
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True)
        train_rosmap(wandb.config,train_genes,valid_genes,test_genes)
        wandb.finish()

if __name__ == '__main__':
    main()