import torch
import pandas as pd
import os
import lightning.pytorch as pl
from scipy.stats import pearsonr
from datasets import GTExDataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import numpy as np
from eval_enformer_gtex import get_enformer_output_dim_from_tissue
from enformer_pytorch import Enformer
from eval_enformer_gtex import slice_enformer_pred
from ism_performer import load_model, get_ckpt,parse_gene_files
import argparse

def instantiate_dataset(gene_list,donor_path,seq_len,tissues_to_train):
    num_individuals_per_gene = -1 #use all possible people
    data_dir = "/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data"
    
    tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')
    #load gene expression df, merge in gene names onto gene ids
    expression_dir = os.path.join(data_dir,"gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir,"gene_id_mapping.csv"))
    df_path = os.path.join(expression_dir,f"{tissue_str}.v8.normalized_expression.bed.gz")
    gene_expression_df = pd.read_csv(df_path,sep = '\t')
    gene_expression_df = gene_expression_df.merge(gene_id_mapping, left_on = 'gene_id',right_on = 'Name')
    
    

    
    ds = GTExDataset(
        tissues_to_train,
        gene_list,
        seq_len,
        num_individuals_per_gene,
        donor_path,
        gene_expression_df,
        data_dir
    )
    return ds

def test_proper_load(model,save_dir,ckpt,row,valid_ds):
    """
    Tests proper loading of model by asserting that predicted values at the epoch match what was reported during training
    """
    ckpt_epoch = int(ckpt.split('epoch=')[-1].split('-')[0])
    valid_preds = pd.read_csv(os.path.join(save_dir,f"Prediction_Results_{ckpt_epoch}_in_valid_donors.csv"))
  
    model.eval()
    model.cuda()
    dl = DataLoader(valid_ds, batch_size = 1)
    n_tests = 10
    i = 0
    logged_preds = []
    loaded_preds = []
    with torch.no_grad():
        for x, y,gene,donor,batch_idx in dl:
            y_pred = performer_predict(model,x)
            assert y_pred.dim() == 2 #only 2 dims left after removing sequence axis

            assert y.dim() == y_pred.dim() == 2
            assert y.shape[0] == 1
            assert y_pred.shape[0] == 1
            assert len(donor) == len(gene) == 1
            donor = donor[0]
            gene = gene[0]

            valid_preds_i = valid_preds[(valid_preds['gene'] == gene) & (valid_preds['donor'] == donor)]
            for i in range(y.shape[1]): #iterate over tissues, if applicable
                tissue_y = y[0,i]
                if not torch.isnan(tissue_y):
                    tissue_y_pred = y_pred[0,i]
                    tissue_str = model.tissues_to_train[i]
                    expected = valid_preds_i[valid_preds_i['tissue'] == tissue_str]
                    assert np.allclose(expected['y_pred'].item(),tissue_y_pred.cpu().item(), atol = 0.05) #offer flexible tolerance due to machine precision, different GPUs for inference, and small original values
                    assert np.allclose(expected['y_true'].item(),tissue_y.cpu().item())

                    logged_preds.append(expected['y_pred'].item())
                    loaded_preds.append(tissue_y_pred.cpu().item())

    assert pearsonr(logged_preds,loaded_preds)[0].item() > 0.95 #predictions saved during validation and those made from same people after reloading should be tightly correlated
    print("Model Loaded Successfully!")
    
def performer_predict(model,x):
    y_hat = model(x.cuda())
    y_hat = y_hat[:,y_hat.shape[1]//2,:] #keep value at center of sequence. The sequence axis is removed
    return y_hat

def enformer_predict(model,x,n_center_bins):
    pred = model(x.cuda())['human']
    pred = slice_enformer_pred(pred,n_center_bins) #remove track axis

    _,enformer_output = get_enformer_output_dim_from_tissue(['Whole Blood'])
    pred = pred[enformer_output].unsqueeze(0) #get desired output. Keep batch axis
    return pred
def subtract_mean(seq: torch.Tensor) -> torch.Tensor:
    """Subtract the mean across the nucleotide dimension."""
    assert seq.size(-1) == 4
    return seq - torch.mean(seq, dim=-1, keepdim=True)

def get_input_grads(dl,model_name,model,outdir,run_id):
    grad_list = []
    donor_list = []
    inputs_list =  []
    for idx,batch in enumerate(dl):
        inputs = batch[0]
        donor = batch[3][0].item()
        gene = batch[2][0]
    
        inputs.requires_grad = True
        if model_name.lower() == 'performer':
            outputs = performer_predict(model,inputs)
        elif model_name.lower() == 'enformer':
            outputs = enformer_predict(model,inputs,n_center_bins = 3)
        else:
            raise Exception(f"Model type {model_name} not supported!")
        (gradients,)= torch.autograd.grad(inputs=inputs, outputs=outputs)
    
        grad_list.append(gradients.detach().cpu())
        donor_list.append(donor)
        inputs_list.append(inputs)
    cat_grads = torch.cat(grad_list) #shape = (n_donors,seq_len,4)
    cat_inputs = torch.cat(inputs_list)
    donor_list = donor_list
    save_dir = f'{outdir}/gradients/{run_id}/{gene}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(cat_grads.permute(0,2,1),os.path.join(save_dir,"gradients.pt"))
    torch.save(cat_inputs.permute(0,2,1),os.path.join(save_dir,"seqs.pt"))
    with open(f"{save_dir}/donor_order.txt",'w') as f:
        f.write('\n'.join(donor_list))


    #upweight distal regions using procedure from Borzoi paper (Linder et al. (2025))
    cat_grads = subtract_mean(cat_grads)
    all_attr_sd = torch.std(cat_grads, axis = 2)
    smoothened = gaussian_filter(all_attr_sd, sigma = 1280, truncate = 2, axes = 1) 
    grad_div_smooth = cat_grads.numpy() / np.expand_dims(smoothened, axis = 2)
    np.save(os.path.join(save_dir,"grad_div_smooth.npy"),grad_div_smooth.transpose(0,2,1))

        

  
    
def check_if_done(run_id,outdir,skip_check,gene):
    if skip_check:
        return False
    path = os.path.join(outdir,'gradients',run_id,gene)
    return os.path.exists(path)

def get_enformer_grads(outdir,gene_list,tissues_to_train = ['Whole Blood'],desired_seq_len = 49152):
    skip_check = False
    model_name = 'enformer'
    assert tissues_to_train == ['Whole Blood'],"Need to update enformer_predict to accomodate different tissues"
    model = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
        )
    model.eval()
    model.cuda()

    
    for fold in range(3):
        test_donor_path = f"/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/cross_validation_folds/gtex/cv_folds/person_ids-test-fold{fold}.txt"
        run_id = f"Enformer_Fold-{fold}_{desired_seq_len}bp"
        for gene in gene_list:
            is_done = check_if_done(run_id,outdir,skip_check,gene)
            if not is_done:
                dataset = instantiate_dataset([gene], #one gene in the dataset
                                                    test_donor_path,
                                                    desired_seq_len,
                                                    tissues_to_train)
                dl = DataLoader(dataset, batch_size=1, shuffle=False) 
                get_input_grads(dl,model_name,model,outdir,run_id)
        print(f"{fold}")    
    
def get_performer_grads(metadata,outdir,genes_to_add = None,skip_check = False,only_genes = None):
    for idx, row in metadata.iterrows():
        run_id = row['ID']
        tissues_to_train = row['tissues_to_train'].strip('"[]"').split(',')
        assert tissues_to_train == ['Whole Blood'], "non-blood tissues are not currently supported"
        desired_seq_len = int(row['seq_length'])
        save_dir = row['save_dir']
        test_donor_path = row['test_donor_path']
        valid_donor_path = row['valid_donor_path']
        train_genes = row['genes_for_training'].replace('[','').replace(']','').replace('"','').split(',')
        valid_genes = row['genes_for_valid'].replace('[','').replace(']','').replace('"','').split(',')
        test_genes = row['genes_for_test'].replace('[','').replace(']','').replace('"','').split(',')
        gene_list = list(set(train_genes + valid_genes + test_genes))
        gene_list = [x for x in gene_list if x != ''] #if test or valid genes empty there will be a `''` element in the list. remove it
        if genes_to_add:
            assert only_genes is None
            gene_list.extend(genes_to_add)
            gene_list = list(set(gene_list)) #remove redundant genes if there are any
        if only_genes:
            assert genes_to_add is None
            gene_list = list(set(only_genes))
        ckpt = get_ckpt(save_dir)
        if ckpt is None: #some models won't have checkpoints saved. Namely, if the gene they were meant to be trained with is incompatible and there are no other train genes available to train them, training exits and there is no ckpt
            continue

        model = load_model(ckpt,save_dir,run_id)
        valid_dataset = instantiate_dataset([train_genes[0]],valid_donor_path,desired_seq_len, tissues_to_train)
        test_proper_load(model,save_dir,ckpt,row,valid_dataset)
        assert not model.training
        assert next(model.parameters()).is_cuda

        gene_list = sorted(gene_list)
        for idx2,gene in enumerate(gene_list):
            print(f"{run_id} {idx}/{len(metadata)}\t Gene: {idx2} / {len(gene_list)}")
            is_done = check_if_done(run_id,outdir,skip_check,gene)
            if not is_done:
                test_dataset = instantiate_dataset([gene], #one gene in the dataset
                                                test_donor_path,
                                                desired_seq_len,
                                                tissues_to_train)

                dl = DataLoader(test_dataset, batch_size=1, shuffle=False) 
                get_input_grads(dl,'performer',model,outdir,run_id)
                test_proper_load(model,save_dir,ckpt,row)
            
def main():
    parser = argparse.ArgumentParser(description="For ISM")
    parser.add_argument("--path_to_metadata",type=str,nargs = '?',help = "Metadata from Wandb run to ensure correct specifications are used")
    parser.add_argument("--model_name",type=str, help = 'One of SingleGene or MultiGene or Enformer')
    parser.add_argument("--path_to_only_genes_file",type=str,nargs ='?', help = 'Optional path to a txt file containing an explicit set of genes to evaluate. Each gene must be on its own row')
    parser.add_argument("--enformer_seq_len",type=int,nargs ='?', help = 'seq_len to use with Enformer')

    
    args = parser.parse_args()
    metadata = args.metadata
    model_name = args.model_name
    path_to_only_genes_file = args.path_to_only_genes_file
    enformer_seq_len = args.enformer_seq_len
    gene_list = parse_gene_files(path_to_only_genes_file)

    outdir = f'/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/RevisionGradInput/{model_name}'

    if model_name in ['SingleGene','MultiGene']:
        assert metadata is not None
        if len(gene_list) == 0: #if desired genes were passed, evaluate on those. Otherwise convert to None so this flag is skipped
            gene_list = None
        get_performer_grads(metadata,outdir,only_genes = gene_list)

    else:
        assert model_name == 'Enformer'
        assert metadata is None
        tissues_to_train = ['Whole Blood']
        assert enformer_seq_len in [49152, 196608]
        get_enformer_grads(outdir,gene_list,tissues_to_train,enformer_seq_len)

if __name__ == '__main__':
    main()

## TODO:
# generate a list of ~30 genes. 15 high heritability 15 low heritability and only evaluate models that include those
# include test genes separately
# MultiGene and SingleGene models need to be handled separately. MultiGene need to be evaluated on each of these genes. SingleGene models need to be evaluated on only the single train gene as well as the test genes



    