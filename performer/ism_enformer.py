from eval_enformer_gtex import *
from ism_performer import get_all_gtex_snps, tss_centered_sequences
torch.use_deterministic_algorithms(True)

def parse_tissues_to_train(tissues_to_train):
    tissues_to_train = [str(tissue) for tissue in tissues_to_train.split(',')]
    for tissue in tissues_to_train:
        assert tissue in ['Whole Blood','Brain - Cortex'], f"Other tissues not yet implemented. You passed in {tissue}"
    return tissues_to_train

def main():
    """
    Finds all observed SNPs in the GTEx VCF within the window of the desired length around each desired gene's TSS. Then performs in silico mutagenesis using Enformer
    This supports two GTEx tissues, Whole Blood and Brain - Cortex, by using an Enformer output dimension from a similar tissue. ISM is performed by placing each observed SNP onto the reference genome background and performing one prediction with each.
    """

    parser = argparse.ArgumentParser(description="For ISM")
    parser.add_argument("--path_to_genes_file",type=str)
    parser.add_argument("--desired_seq_len",type=str)
    parser.add_argument("--tissues_to_train",type=str, help = "determines which enformer output will be used for ISM")
    parser.add_argument("--n_center_bins",type=int)
    args = parser.parse_args()

    
    path_to_genes_file = args.path_to_genes_file
    genes_to_score = parse_gene_files(path_to_genes_file)
    genes_to_score = sorted(list(set(genes_to_score)))
    desired_seq_len = int(args.desired_seq_len)
    tissues_to_train_str = str(args.tissues_to_train)
    tissues_to_train = parse_tissues_to_train(tissues_to_train_str)
    n_center_bins = int(args.n_center_bins)

    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd,'../data')
    outdir = os.path.join(cwd,'results/EnformerISM')
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    enformer_tissue_names,enformer_output_dims = get_enformer_output_dim_from_tissue(tissues_to_train)
    enformer_regions = pd.read_csv(os.path.join(DATA_DIR,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    pl.seed_everything(0, workers=True)
    model = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        )    
    window = desired_seq_len // 2 #window is the amount of bp ahead and behind the TSS to score. Half the sequence length to score the entire sequence 
    model.eval()
    model.cuda()


    for gene_idx, gene in enumerate(genes_to_score):
        filename = os.path.join(outdir,f"{gene}_Enformer_model_ISM_{window * 2}bp_{tissues_to_train_str}_{n_center_bins}CenterBins.csv")
        variant_df = get_all_gtex_snps(gene,window)
        if (variant_df.shape[0] > 0) and (not os.path.exists(filename)): #gene must be in enformer regions, must have SNPs nearby, and must not already have been evaluated
            print(f"Performing ISM on gene {gene} {gene_idx}/{len(genes_to_score)}")
            gene_info = enformer_regions[enformer_regions['gene_name'] == gene]
            gene_start = int(gene_info['gene_start'].item())
            variant_df = variant_df[(variant_df['pos0'] - gene_start).abs() <= window] #final assurance that no SNPs are further away than you want
                        
            results_df = pd.DataFrame(columns = ['chrom', 'pos0', 'ref','alt','ref_pred','alt_pred','run_id','gene_name','region_chr','region_start','region_end','enformer_tissue','enformer_output_dim'])
            it = tss_centered_sequences(variant_df, desired_seq_len)
            
            with torch.no_grad():
                for idx, example in enumerate(it):
                    chrom = example['metadata']['chrom']
                    pos = example['metadata']['pos']
                    ref = example['metadata']['ref']
                    alt = example['metadata']['alt']
                    gene_name = example['metadata']['gene_name']
                    region_chr = example['metadata']['region_chr']
                    region_start = example['metadata']['region_start']
                    region_end = example['metadata']['region_end']
                    if idx == 0: #ref pred is the same for each gene because they are always centered around TSS. So only form prediction once
                        ref_pred = model(torch.from_numpy(example['inputs']['ref']).unsqueeze(0).cuda())['human']
                        ref_pred = slice_enformer_pred(ref_pred,n_center_bins)
                    alt_pred = model(torch.from_numpy(example['inputs']['alt']).unsqueeze(0).cuda())['human']
                    alt_pred = slice_enformer_pred(alt_pred,n_center_bins)
                    for enformer_tissue, enformer_output_dim in list(zip(enformer_tissue_names,enformer_output_dims)):
                        ref_item = ref_pred[enformer_output_dim].item()
                        alt_item = alt_pred[enformer_output_dim].item()
                        results_df.loc[results_df.shape[0],:] = [chrom,pos,ref,alt,ref_item,alt_item,'Enformer',gene_name, region_chr,region_start,region_end,enformer_tissue,enformer_output_dim]

                results_df.to_csv(filename)

if __name__ == "__main__":
    main()