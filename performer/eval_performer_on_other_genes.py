"""
For evaluating trained models on genes that weren't included for train/val/test during train time
"""
from .train_gtex import *

def load_test_gtex_datasets(row,test_genes,donor_pickiness,DATA_DIR):
    assert donor_pickiness in ['all_donors','only_test_donors']
    def instantiate_dataset(row,gene_list,donor_path,tissues_to_train,gene_expression_df,num_individuals_per_gene):
        ds = GTExDataset(
            tissues_to_train,
            gene_list,
            row['seq_length'],
            num_individuals_per_gene,
            donor_path,
            gene_expression_df,
            DATA_DIR
        )
        return ds

    tissues_to_train = row['tissues_to_train'].strip('"[]"').split(',') #ex: 'Whole Blood' -> ['Whole Blood]
    assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"
    tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')

    #load gene expression df, merge in gene names onto gene ids
    expression_dir = os.path.join(DATA_DIR,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"{tissue_str}.v8.normalized_expression.bed.gz")
    gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gene_expression_df = gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')
    
    if donor_pickiness == 'only_test_donors':
        test_ds = instantiate_dataset(row,test_genes,row['test_donor_path'],tissues_to_train,gene_expression_df,-1)
    else:
        cv_dir = os.path.dirname(row['test_donor_path'])
        all_gtex_file = os.path.join(os.path.join(cv_dir,'../All_GTEx_ID_list.txt'))
        test_ds = instantiate_dataset(row,test_genes,all_gtex_file,tissues_to_train,gene_expression_df,-1)

    return test_ds

def load_trainer_for_testing(row):
    metric_logger = MetricLogger()
    trainer = Trainer(
        precision = row['precision'],
        callbacks = [metric_logger],
        num_sanity_val_steps = 0,
        log_every_n_steps = 1,
        check_val_every_n_epoch = 1
    )
    return trainer
def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--path_to_metadata",type=str)
    parser.add_argument("--path_to_test_gene_file",type=str,help = "Absolute path to a line-separated .txt file containing name of every gene to be used for evaluation")
    parser.add_argument("--donor_pickiness",type=str,help = "One of ['all_donors','only_test_donors']. Whether to evaluate these genes using all ppl with data or only those from the model's test fold. For example, if the gene is in the test set, it may be appropriate to evaluate using train donors")
    parser.add_argument("--subset",type=int,help = "Metadata enumerating models will be split into n_subsets and this subset of models will be evaluated by the job.")
    parser.add_argument("--n_subsets",type=int,help = "Metadata enumerating models will be split into n_subsets.")

    
    args = parser.parse_args()
    path_to_metadata = args.path_to_metadata
    path_to_test_gene_file = args.path_to_test_gene_file
    donor_pickiness = args.donor_pickiness
    subset = int(args.subset)
    n_subsets = int(args.n_subsets)
    metadata = pd.read_csv(path_to_metadata)
    metadata = np.array_split(metadata,n_subsets)[subset]

    assert donor_pickiness in ['all_donors','only_test_donors']
    test_genes = parse_gene_files(path_to_test_gene_file)
    test_gene_filename = os.path.basename(path_to_test_gene_file)

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir,'../data')
    for idx, row in metadata.iterrows():
        seed = row['seed']
        save_path = row['save_dir']
        ckpt_epoch = os.listdir(os.path.join(save_path,'checkpoints'))
        n_ckpts = len(ckpt_epoch) 
        assert n_ckpts == 1, f"You have {n_ckpts} but this script only expects one"
        ckpt_epoch = ckpt_epoch[0]
        ckpt_path = os.path.join(save_path,f'checkpoints/{ckpt_epoch}')
        results_dir = os.path.join(save_path,f"{test_gene_filename}_{donor_pickiness}")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        pl.seed_everything(int(seed), workers=True)
        torch.use_deterministic_algorithms(True)
            
        test_ds = load_test_gtex_datasets(row,test_genes,donor_pickiness,DATA_DIR)

        trainer = load_trainer_for_testing(row)
        model = LitModelHeadAdapterWrapper.load_from_checkpoint(ckpt_path)
        model.save_dir = results_dir #results get saved to results_dir
        model.genes_for_test = test_genes

        trainer.test(model, DataLoader(test_ds,batch_size = 1), ckpt_path = ckpt_path)

if __name__ == '__main__':
    main()