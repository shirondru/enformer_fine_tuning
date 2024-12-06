import gc
import pandas as pd
import os
import argparse
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .datasets import EvalAcrossGeneDataset
from .ism_performer import load_model,get_ckpt


def main():
    parser = argparse.ArgumentParser(description="For evaluation of performer's ability to predict average expression of unseen genes")
    parser.add_argument("--path_to_metadata",type=str,help = "Metadata from Wandb run to ensure correct specifications are used")
    parser.add_argument("--model_type",type=str, help = 'One of SingleGene or MultiGene')
    parser.add_argument("--outdir",type=str,nargs ='?', help = 'Optional directory to save outputs')
    args = parser.parse_args()
    metadata = pd.read_csv(args.path_to_metadata)
    outdir = args.outdir
    model_type = args.model_type
    metadata = metadata.rename(columns = {'ID':'run_id'})
    assert model_type in ['SingleGene','MultiGene']

    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')


  
    df_path = os.path.join(data_dir,"GTEx_gene_tpm.csv") #this file not in github repo beacuse it is ~3GB. It can be downloaded from GTEX portal as GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
    gene_expression_df = pd.read_csv(df_path)
    

    
    if not outdir:
        outdir = os.path.join(cwd,f'../results/PerformerEvalAcrossGenes/{model_type}')
    if not os.path.exists(os.path.join(outdir)):
            os.makedirs(os.path.join(outdir))
    enformer_regions = pd.read_csv(os.path.join(data_dir,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    test_genes = list(enformer_regions[(enformer_regions['set'] == 'test') & (~enformer_regions['gene_name'].str.contains('/'))]['gene_name'].unique())
    pl.seed_everything(0, workers=True)
    for idx, row in metadata.iterrows():
        results_df = pd.DataFrame(columns = ['run_id','gene','y','y_hat'])
        run_id = row['run_id']
        filename = os.path.join(outdir,f"EvalAcrossTestGenes_{run_id}.csv")
        tissues_to_train = row['tissues_to_train'].strip('"[]"').split(',') #configure as a list of strings
        assert len(tissues_to_train) == 1, "ISM on only 1 output tissue is supported"
        desired_seq_len = int(row['seq_length'])
        save_dir = row['save_dir']
        ckpt = get_ckpt(save_dir)
        if ckpt is None: #some models won't have checkpoints saved. Namely, if the gene they were meant to be trained with is incompatible and there are no other train genes available to train them, training exits and there is no ckpt
            continue

        model = load_model(ckpt, save_dir,run_id)
        model.eval()
        model.cuda()
        
        ds = EvalAcrossGeneDataset(tissues_to_train,test_genes,desired_seq_len,gene_expression_df,data_dir)
        dataloader = DataLoader(ds, batch_size = 1 )
        with torch.no_grad():
            for x,y,gene,idx2 in dataloader:
                y_pred = model(x.cuda())
                y_pred = y_pred[:,y_pred.shape[0] // 2, :].item()
                gene = gene[0]
                results_df.loc[results_df.shape[0],:] = [run_id,gene,y.item(),y_pred]
        if idx % 50 == 0:
            print(f"{idx} / {metadata.shape[0]}")
        results_df.to_csv(filename,index = False) #save progress
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()