from train_rosmap import *
from eval_enformer_gtex import *
torch.use_deterministic_algorithms(True)
"""
For evaluating Enformer on matched WGS & RNA-seq data where the individuals from the "test set" -- those that were used to test our Performer models, 
come from all individuals in GTEX that have brain cortex data. These models were trained on ROSMAP DLPFC data, but tested on GTEx brain cortex data

The major difference between this script and eval_enformer_gtex.py is the construction of the dataloaders. Here, the test dataset will include all individuals with GTEx data from the test set.
"""

def main():
    parser = argparse.ArgumentParser(description="For ISM")
    parser.add_argument("--path_to_train_genes_file",type=str, help = "A single txt file containing train genes you want to evaluate.",nargs = '?')
    parser.add_argument("--path_to_eval_genes_file",type=str, help = "A single txt file containing valid/test genes you want to evaluate.",nargs = '?')
    parser.add_argument("--name",type=str, help = "Part of output filename")
    parser.add_argument("--config_path",type=str)
    parser.add_argument('--donor_fold',type = int)
    parser.add_argument('--n_center_bins',type = int)
    parser.add_argument('--desired_seq_len',nargs='?')
    args = parser.parse_args()
    config_path = args.config_path
    donor_fold = int(args.donor_fold)
    path_to_train_genes_file = args.path_to_train_genes_file
    path_to_eval_genes_file = args.path_to_eval_genes_file
    name = args.name
    desired_seq_len = int(args.desired_seq_len)
    n_center_bins = int(args.n_center_bins)
    test_cohort = 'GTEx'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = AttrDict(config)
    pl.seed_everything(int(config.seed), workers=True)
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')
    config.DATA_DIR = data_dir
    config.fold = donor_fold
    config.seq_length = desired_seq_len


    experiment_name = config.experiment_name
    define_donor_paths(config,'rosmap')
    train_genes, valid_genes, test_genes = parse_genes(path_to_train_genes_file, path_to_eval_genes_file,config.DATA_DIR)

    train_ds, valid_ds, test_ds = load_rosmap_datasets(config,train_genes, valid_genes,test_genes)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False) 
    enformer_tissue_names,enformer_output_dims = get_enformer_output_dim_from_tissue(config.tissues_to_train.split(','))
    assert enformer_tissue_names == ['CAGE:brain, adult']
    output_dict, all_y, enformer_pred_full,donor_order = get_enformer_predictions(desired_seq_len,test_dataloader,enformer_tissue_names,enformer_output_dims,n_center_bins)
    save_enformer_preds_and_metrics(output_dict,all_y,enformer_pred_full,donor_order,
                                    enformer_tissue_names,enformer_output_dims,
                                    donor_fold,desired_seq_len,experiment_name,name,n_center_bins,test_cohort)
    
if __name__ == '__main__':
    main()