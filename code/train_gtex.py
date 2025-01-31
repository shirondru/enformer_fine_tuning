import argparse
from pl_models import *
from datasets import *
import yaml
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore', '.*does not have many workers.*')


def parse_gene_files(filepath):
    """
    For parsing txt files containing one gene name per row
    """
    gene_list = []
    with open(filepath,'r') as file:
        for gene in file:
            gene_list.append(gene.strip())
    return gene_list

def prepare_genes(config):
    model_type = config['model_type']
    tissue = config['tissues_to_train'].replace(' -','').replace(' ','_').replace('(','').replace(')','')
    data_dir = config['DATA_DIR']
    
    #if desired genes to use explicitly defined in config file, use only those
    if 'train_gene_dir' in config:
        train_gene_filedir = config['train_gene_dir'] #directory containing txt file(s) of genes
        train_gene_filenames = os.listdir(train_gene_filedir) #script expects a list of filenames
        valid_genes = parse_gene_files(config['valid_gene_path']) 
        test_genes = parse_gene_files(config['test_gene_path']) 
    else:
        train_gene_filedir = os.path.join(data_dir,'genes',tissue,model_type)
        train_gene_filenames = os.listdir(train_gene_filedir) #if its a single gene model, there will be 1 txt file per train gene, if multi-gene it will 1 total txt file containing all genes
        
        if model_type == 'MultiGene':
            valid_genes = []
            test_gene_path = os.path.join(data_dir,'genes',tissue,'test_genes.txt')
            test_genes = parse_gene_files(test_gene_path)
        elif model_type == 'SingleGene':
            valid_genes = []
            test_genes = []

    #no matter what, if these are defined in config, use them.
    if 'valid_gene_path' in config:
        valid_genes = parse_gene_files(config['valid_gene_path'])
    if 'test_gene_path' in config:
        test_genes = parse_gene_files(config['test_gene_path']) 
    return train_gene_filedir, train_gene_filenames, valid_genes, test_genes

def ensure_no_gene_overlap(train_genes,val_genes,test_genes,eval_test_gene_during_validation = False):
    train_gene_set = set(train_genes)
    valid_gene_set = set(val_genes)
    test_gene_set = set(test_genes)

    train_valid_overlap = train_gene_set & valid_gene_set
    train_test_overlap = train_gene_set & test_gene_set
    valid_test_overlap = valid_gene_set & test_gene_set

    assert len(list(train_valid_overlap)) == 0, f"There is overlap between genes in the train and valid set via the following genes {train_valid_overlap}"
    assert len(list(train_test_overlap)) == 0, f"There is overlap between genes in the train and test set via the following genes {train_test_overlap}"
    assert len(train_genes) > 0, "You have no genes to train on!"
    if not eval_test_gene_during_validation:
        assert len(list(valid_test_overlap)) == 0, f"There is overlap between genes in the valid and test set via the following genes {valid_test_overlap}"


def ensure_no_donor_overlap(train_ds,val_ds,test_ds):
    train_set = set(train_ds.individuals_in_split)
    val_set = set(val_ds.individuals_in_split)
    test_set = set(test_ds.individuals_in_split)

    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def define_donor_paths(config,dataset):
    #if donor paths already defined, use them, otherwise generate them based on the fold and the dataset
    if hasattr(config,'train_donor_path'):
        assert hasattr(config,'valid_donor_path')
        assert hasattr(config,'test_donor_path')
    else:
        if dataset == 'gtex':
            donor_dir = os.path.join(config.DATA_DIR,'cross_validation_folds',dataset,'cv_folds')
            config.update({'train_donor_path':os.path.join(donor_dir,f"person_ids-train-fold{config.fold}.txt")})
            config.update({'valid_donor_path':os.path.join(donor_dir,f"person_ids-val-fold{config.fold}.txt")})
            config.update({'test_donor_path':os.path.join(donor_dir,f"person_ids-test-fold{config.fold}.txt")})
        elif dataset == 'rosmap':
            #train and validation set are from rosmap. Test set will be individuals from gtex to enable cross-cohort evaluation
            rosmap_dir = os.path.join(config.DATA_DIR,'cross_validation_folds',dataset)
            config.update({'train_donor_path':os.path.join(rosmap_dir,f"person_ids-train-fold{config.fold}.txt")})
            config.update({'valid_donor_path':os.path.join(rosmap_dir,f"person_ids-val-fold{config.fold}.txt")})

            #using all individuals from gtex as test set. Dataset will keep only those with brain cortex data.
            all_gtex_donor_path = os.path.join(config.DATA_DIR,'cross_validation_folds','gtex','All_GTEx_ID_list.txt')
            config.update({'test_donor_path' : all_gtex_donor_path}) 
        else:
            raise Exception(f"Dataset: {dataset} not supported!")


def load_gtex_datasets(config,train_genes,valid_genes,test_genes):
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
    assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"
    tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')

    #load gene expression df, merge in gene names onto gene ids
    expression_dir = os.path.join(config.DATA_DIR,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"{tissue_str}.v8.normalized_expression.bed.gz")
    gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gene_expression_df = gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')
    
    
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
def load_trainer(config):
    metric_logger,early_stopper,checkpoint_callback = load_callbacks(config)
    trainer = Trainer(
        max_epochs = config.max_epochs,
        precision = config.precision,
        accumulate_grad_batches = config.num_individuals_per_gene // config.train_batch_size, #accumulate as many batches as necessary to achieve num_individuals_per_gene effective samples per gradient accumulated step
        gradient_clip_val = config.gradient_clip_val,
        callbacks = [checkpoint_callback,metric_logger,early_stopper],
        logger = WandbLogger(),
        num_sanity_val_steps = 0, #don't do any validation before training, as all sorts of R2 metrics will be computed during callbacks. Could lead to error with small sample size
        log_every_n_steps = 1,
        check_val_every_n_epoch = config.valid_metrics_save_freq
    )
    config.update({'Gradient Accumulation Effective Batch Size': config.num_individuals_per_gene // config.train_batch_size })
    return trainer
def load_callbacks(config):
    mode = 'max'
    checkpoint_dir = os.path.join(config.save_dir,'checkpoints')
    os.makedirs(checkpoint_dir,exist_ok = True)
    if hasattr(config,'monitor'):
        monitor = config.monitor
    else:
        monitor = 'mean_r2_across_train_genes_across_valid_donors'
    
    if hasattr(config,'save_top_k'):
        save_top_k = config.save_top_k
    else:
        save_top_k = 1
    if hasattr(config,'min_delta'):
        min_delta = config.min_delta
    else:
        min_delta = 0
    checkpoint_callback =  ModelCheckpoint(
            dirpath = checkpoint_dir,
            save_top_k = save_top_k,
            monitor = monitor,
            mode = mode
        )
    early_stopper = EarlyStopping(monitor = monitor, mode = mode, min_delta = min_delta, patience = int(config.patience))
    metric_logger = MetricLogger()
    return metric_logger,early_stopper,checkpoint_callback
def train_gtex(config: wandb.config,
               train_genes: list, 
               valid_genes: list,
               test_genes: list,
               eval_test_gene_during_validation: bool = False,
               validate_first: bool = False) -> None:
    ensure_no_gene_overlap(train_genes,valid_genes,test_genes,eval_test_gene_during_validation)
    define_donor_paths(config,'gtex')

    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
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
        train_gtex(wandb.config,train_genes,valid_genes,test_genes)
        wandb.finish()
            


if __name__ == '__main__':
    main()