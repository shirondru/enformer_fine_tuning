from train_gtex import *
torch.use_deterministic_algorithms(True)

"""
For evaluating Enformer on matched WGS & RNA-seq data where the individuals from the "test set" -- those that were used to test our Performer models, come from GTEx
"""
class AttrDict(dict):
    """A dictionary class that supports attribute-style access."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        self[key] = value

def save_enformer_preds_and_metrics(output_dict,all_y,enformer_pred_full,donor_order,
                                    enformer_tissue_names,enformer_output_dims,
                                    donor_fold,desired_seq_len,experiment_name,name,n_center_bins,test_cohort):
    outdir = os.path.join(os.getcwd(),"../results/EnformerResults")
    outpath = os.path.join(outdir,experiment_name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)  

    donor_split = 'test'
    
    #save summary statistics into a dataframe
    metrics = pd.DataFrame(output_dict)
    metrics['donor_split'] = donor_split
    metrics['donor_fold'] = donor_fold
    metrics['n_center_bins'] = n_center_bins
    metrics['desired_seq_len'] = desired_seq_len
    metrics.to_csv(os.path.join(outpath,f"Enformer_{donor_split}{test_cohort}Metrics_DonorFold_{donor_fold}_{name}_{desired_seq_len}bp_{n_center_bins}CenterBins.csv"))
   
    #save original predictions and true values for each individual into a different dataframe
    model_pred_df = pd.DataFrame({'y_true':[],'model_pred':[],'gene_name':[],'donors':[],'enformer_tissue':[],'desired_seq_len':[],'donor_split':[],'donor_fold':[],'enformer_output_dim':[],'n_center_bins':[]})
    model_pred_df = model_pred_df.astype({
                'y_true': float,
                'model_pred': float,
                'gene_name': str,
                'donors': str,
                'enformer_tissue': str,
                'desired_seq_len': int,
                'donor_split': str,
                'donor_fold': int,
                'enformer_output_dim': int,
                'n_center_bins': int
    })
    for gene_name in all_y.keys():
        for t_idx,tissue in enumerate(enformer_tissue_names):
            y_vals = all_y[gene_name][tissue]
            preds = enformer_pred_full[gene_name][tissue]
            donors = donor_order[gene_name][tissue]
            for i in range(len(y_vals)):
                model_pred_df.loc[model_pred_df.shape[0],['y_true','model_pred','gene_name','donors','enformer_tissue','desired_seq_len','donor_split','donor_fold','enformer_output_dim','n_center_bins']] = [y_vals[i].item(),preds[i].item(),gene_name,donors[i],tissue,desired_seq_len,donor_split,donor_fold,enformer_output_dims[t_idx],n_center_bins]
    model_pred_df.to_csv(os.path.join(outpath,f"Enformer_{donor_split}{test_cohort}Predictions_DonorFold_{donor_fold}_{name}_{desired_seq_len}bp_{n_center_bins}CenterBins.csv"))
def get_enformer_output_dim_from_tissue(tissues_to_train):
    enformer_output_map = {
        'Brain - Cortex': ('CAGE:brain, adult',4980),
        'Whole Blood': ('CAGE:blood, adult, pool1',4950)
    }
    enformer_tissue_names = []
    enformer_output_dims = []
    for tissue in tissues_to_train:
        if tissue in enformer_output_map.keys():
            enformer_tissue = enformer_output_map[tissue][0]
            enformer_output_dim = enformer_output_map[tissue][1]
            enformer_tissue_names.append(enformer_tissue)
            enformer_output_dims.append(enformer_output_dim)
    return enformer_tissue_names,enformer_output_dims

def slice_enformer_pred(pred,n_center_bins):
    """
    Takes either the center bin and returns the array, or takes the 3 center bins and computes the sum over those 3 bins and returns the sequence
    The enformer output dimension is left untouched and will be indexed separately

    If the seq length is the full 196608, the number of bins will be 1536, not 896, because cropping is skipped. But the center bin (768) will be identical to the center bin of the cropped output (448)
    """
    assert pred.shape[0] == 1, "This does not handle multi-batch inputs"
    arr_center = pred.shape[1] // 2
    if n_center_bins == 1:
        pred = pred[0,arr_center,:].detach().cpu()
    else:
        pred = pred[0,arr_center - 1 : arr_center + 2,:].sum(axis = 0).detach().cpu() #take 3 center bins and sum over them. Keep enformer output dimension for now, it will be selected below
    return pred

def get_enformer_predictions(seq_length,test_dataloader,tissue_str_list,desired_enformer_outputs,n_center_bins):

    
    pearson = PearsonCorrCoef(num_outputs = 1).to('cpu')
    r2_score = R2Score(num_outputs = 1).to('cpu')
    # #    If the seq length is the full 196608, the number of bins will be 1536, not 896, because cropping is skipped. But the center bin (768) will be identical to the center bin of the cropped output (448)
    enformer_full = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        )

    enformer_full.eval()
    enformer_full.cuda()
    enformer_pred_full = {}
    all_y = {}
    donor_order = {}

    output_dict = {'gene_name':[],'r2':[],'pearsonr':[],'tissue':[],'enformer_seq_len':[],'enformer_output_dim':[]}
    with torch.no_grad():
        for x,y,gene_name,donor,rank in test_dataloader:
            gene_name = gene_name[0]
            donor = donor[0]
            if gene_name not in all_y.keys():
                all_y[gene_name] = {tissue:[] for tissue in tissue_str_list}
                enformer_pred_full[gene_name] = {tissue:[] for tissue in tissue_str_list}
                donor_order[gene_name] = {tissue:[] for tissue in tissue_str_list}
            
            pred = enformer_full(x.cuda())['human']
            #use only desired # of TSS-overlapping center bins for evaluation, as well as the desired output dimension
            pred = slice_enformer_pred(pred,n_center_bins)
            for tissue_str, enformer_output in zip(tissue_str_list,desired_enformer_outputs):
                enformer_pred_full[gene_name][tissue_str].append(pred[enformer_output].unsqueeze(0))
            all_y[gene_name][tissue_str].append(y.squeeze(0))
            donor_order[gene_name][tissue_str].append(donor)
                

        for gene in all_y.keys():
            for i,tissue_str in enumerate(tissue_str_list):
                all_y_t_g = torch.cat(all_y[gene][tissue_str]).cpu()
                all_pred_t_g = torch.cat(enformer_pred_full[gene][tissue_str]).cpu()

                full_pearsonr = pearson(all_pred_t_g,all_y_t_g)
                r2_val = r2_score(all_pred_t_g,all_y_t_g)
                output_dict['pearsonr'].append(full_pearsonr.item())
                output_dict['r2'].append(r2_val.item())
                output_dict['gene_name'].append(gene)
                output_dict['tissue'].append(tissue_str)
                output_dict['enformer_output_dim'].append(desired_enformer_outputs[i])
                output_dict['enformer_seq_len'].append(seq_length)
        return output_dict, all_y, enformer_pred_full,donor_order

def parse_genes(path_to_train_genes_file, path_to_eval_genes_file,DATA_DIR):
        #ensure you have either a path to train or test genes
    if path_to_eval_genes_file is None:
        assert path_to_train_genes_file is not None
        path_to_eval_genes_file = os.path.join(DATA_DIR,"genes/empty.txt") #replace with empty file, which will become an empty list after being parsed
    if path_to_train_genes_file is None:
        assert path_to_eval_genes_file is not None
        path_to_train_genes_file = os.path.join(DATA_DIR,"genes/empty.txt") #replace with empty file, which will become an empty list after being parsed

    train_genes = parse_gene_files(path_to_train_genes_file)
    valid_genes = []
    test_genes = parse_gene_files(path_to_eval_genes_file)
    return train_genes, valid_genes, test_genes


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

    experiment_name = config.experiment_name
    define_donor_paths(config,'gtex')
    train_genes, valid_genes, test_genes = parse_genes(path_to_train_genes_file, path_to_eval_genes_file,config.DATA_DIR)

    train_ds, valid_ds, test_ds = load_gtex_datasets(config,train_genes, valid_genes,test_genes)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False) 
    enformer_tissue_names,enformer_output_dims = get_enformer_output_dim_from_tissue(config.tissues_to_train.split(','))
    assert enformer_tissue_names == ['CAGE:blood, adult, pool1']

    output_dict, all_y, enformer_pred_full,donor_order = get_enformer_predictions(desired_seq_len,test_dataloader,enformer_tissue_names,enformer_output_dims,n_center_bins)
    save_enformer_preds_and_metrics(output_dict,all_y,enformer_pred_full,donor_order,
                                    enformer_tissue_names,enformer_output_dims,
                                    donor_fold,desired_seq_len,experiment_name,name,n_center_bins,test_cohort)

if __name__ == '__main__':
    main()