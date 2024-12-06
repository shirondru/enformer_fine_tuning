import gc
import pandas as pd
import os
import argparse
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .datasets import EvalAcrossGeneDataset
from .eval_enformer_gtex import *

def main():
    parser = argparse.ArgumentParser(description="For For evaluation of enformer's ability to predict average expression of unseen genes")
    parser.add_argument("--gtex_tissue",type=str,help = "GTEx tissue to be used to compare enformer predictions against")
    parser.add_argument("--desired_seq_len",type=int,help = "Sequence length to used for predictions")
    parser.add_argument("--n_center_bins",type=int,help = "Number of bins flanking TSS to average over")
    parser.add_argument("--outdir",type=str,nargs ='?', help = 'Optional directory to save outputs')

    args = parser.parse_args()
    desired_seq_len = int(args.desired_seq_len)
    n_center_bins = int(args.n_center_bins)
    outdir = args.outdir
    gtex_tissue = args.gtex_tissue
    assert gtex_tissue in ['Whole Blood','Brain - Cortex'],"Only one of these tissues is expected"
    tissues_to_train = [gtex_tissue] #convert to a list for compatibility downstream

    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')

    df_path = os.path.join(data_dir,"GTEx_gene_tpm.csv") #this file not in github repo beacuse it is ~3GB. It can be downloaded from GTEX portal as GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
    gene_expression_df = pd.read_csv(df_path)
    

    
    if not outdir:
        outdir = os.path.join(cwd,f'../results/EnformerEvalAcrossGenes/')
    if not os.path.exists(os.path.join(outdir)):
            os.makedirs(os.path.join(outdir))
    enformer_regions = pd.read_csv(os.path.join(data_dir,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    test_genes = list(enformer_regions[(enformer_regions['set'] == 'test') & (~enformer_regions['gene_name'].str.contains('/'))]['gene_name'].unique())
    pl.seed_everything(0, workers=True)
    filename = os.path.join(outdir,f"EvalAcrossTestGenes_{desired_seq_len}bp.csv")
    enformer = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        )
    enformer.eval()
    enformer.cuda()
    ds = EvalAcrossGeneDataset(tissues_to_train,test_genes,desired_seq_len,gene_expression_df,data_dir)
    dataloader = DataLoader(ds, batch_size = 1 )
    results_df = pd.DataFrame(columns = ['model','gene','y','y_hat','enformer_tissue','enformer_output_dim','seq_len'])
    enformer_tissue_names,enformer_output_dims = get_enformer_output_dim_from_tissue(tissues_to_train)

    with torch.no_grad():
         for x,y,gene,idx in dataloader:
                assert x.shape[1] == desired_seq_len
                gene = gene[0]
                pred = enformer(x.cuda())['human']
                pred = slice_enformer_pred(pred,n_center_bins)
                for tissue_str, enformer_output in zip(enformer_tissue_names,enformer_output_dims):
                    final_pred = pred[enformer_output].item()
                results_df.loc[results_df.shape[0],:] = ['enformer',gene,y.item(),final_pred,tissue_str,enformer_output,desired_seq_len]
    results_df.to_csv(filename,index = False)


if __name__ == '__main__':
     main()