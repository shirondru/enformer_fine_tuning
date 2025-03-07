import torch
import pandas as pd
import os
import sys
import lightning.pytorch as pl
from scipy.stats import pearsonr
from datasets import GTExDataset,GTExRefDataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import numpy as np
from eval_enformer_gtex import get_enformer_output_dim_from_tissue
from enformer_pytorch import Enformer
from eval_enformer_gtex import slice_enformer_pred
from ism_performer import load_model, get_ckpt,parse_gene_files
from pl_models import LitModelHeadAdapterWrapper
import argparse
from tangermeme.ism import saturation_mutagenesis
from tangermeme.utils import characters, one_hot_encode
def sum_over_center_bins(pred,bin_length):
    target_len = pred.shape[1]
    center = target_len // 2
    min_bin = center - (bin_length // 2)
    max_bin = center + ((bin_length // 2) + 1)
    summed_pred = pred[:,min_bin:max_bin,:].sum(axis = 1)
    return summed_pred
class SliceEnformer(torch.nn.Module):
    def __init__(self,bin_length):
        """
        For using base Enformer's 'CAGE:blood, adult, pool1' output, and automatically taking the sum over the 3 central bins
        """
        super(SliceEnformer, self).__init__()
        self.model = Enformer.from_pretrained(
            'EleutherAI/enformer-official-rough',
            target_length = -1 #disable cropping for use with shorter sequences
            )
        self.blood_output_dim = 4950
        self.bin_length = bin_length
    def forward(self, X):
        pred = self.model(X)['human']
        pred = sum_over_center_bins(pred,self.bin_length)
        pred = pred[:, self.blood_output_dim:self.blood_output_dim+1]
        return pred

class PermutedSeqModel(pl.LightningModule):
    """
    To accomodate models that expect a sequence of shape (batch size, seq len, alphabet) as input.
    Tangermeme requests a permuted shape and this permutes things back internally.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,x):
        x = x.permute(0,2,1)
        return self.model(x)
class PerformerPredict(pl.LightningModule):
    def __init__(self,model):
        """
        defines forward function to take central bin for prediction, so tangermeme.ism.saturation_mutagenesis can handle this
        """

        super(pl.LightningModule, self).__init__()
        self.model = model
    def forward(self,x):
        y_hat = self.model(x.cuda())
        y_hat = y_hat[:,y_hat.shape[1]//2,:] #keep value at center of sequence. The sequence axis is removed
        return y_hat

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

    ref_ds = GTExRefDataset(tissues_to_train,
                            gene_list,
                            seq_len,
                            donor_path,
                            gene_expression_df,
                            data_dir)
    return ds,ref_ds

def test_proper_load(model,save_dir,ckpt,valid_ds,precision,atol = 1e-05):
    """
    Tests proper loading of model by asserting that predicted values at the epoch match what was reported during training
    """
    assert precision == 'bf16-mixed', "This test autocasts to torch.bfloat16 because it assumes the model was trained with this precision. The test will fail otherwise"
    ckpt_epoch = int(ckpt.split('epoch=')[-1].split('-')[0])
    valid_preds = pd.read_csv(os.path.join(save_dir,f"Prediction_Results_{ckpt_epoch}_in_valid_donors.csv"))
  
    model.eval()
    model.cuda()
    dl = DataLoader(valid_ds, batch_size = 1)
    logged_preds = []
    loaded_preds = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for x, y,gene,donor,batch_idx in dl:
                y_pred = model(x.cuda())
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
                        tissue_str = model.model.tissues_to_train[i]
                        expected = valid_preds_i[valid_preds_i['tissue'] == tissue_str]

                        assert np.allclose(expected['y_pred'].item(),tissue_y_pred.cpu().item(), atol = atol)
                        assert np.allclose(expected['y_true'].item(),tissue_y.cpu().item())

                        logged_preds.append(expected['y_pred'].item())
                        loaded_preds.append(tissue_y_pred.cpu().item())
    #visual confirmation that loaded and logged predicitions are identical
    print("Logged",logged_preds[:3])
    print("Loaded",loaded_preds[:3])
    max_diff = np.array(loaded_preds) - np.array(logged_preds)
    print('Max Difference', np.max(max_diff))
    assert pearsonr(logged_preds,loaded_preds)[0].item() > 0.95 #predictions saved during validation and those made from same people after reloading should be tightly correlated
    print("Model Loaded Successfully!")



# def subtract_mean(seq: torch.Tensor) -> torch.Tensor:
#     """Subtract the mean across the nucleotide dimension."""
#     assert seq.size(-1) == 4
#     return seq - torch.mean(seq, dim=-1, keepdim=True)
def update_outdir(outdir,analysis):
    if analysis == 'grad_input':
        outdir = os.path.join(outdir,'gradients')
    elif analysis == 'saturation_mutagenesis':
        outdir = os.path.join(outdir,analysis)
    else:
        raise Exception(f"Analysis {analysis} not supported")
    return outdir
def get_input_grads(dl,model,outdir,run_id,gene,basedir,skip_check):
    save_dir = f'{outdir}/{run_id}/{gene}/{basedir}'
    is_done = check_if_done(save_dir,skip_check)
    if is_done:
        print(f'Skipping {save_dir} Because it is already finished.')
        return
    
    grad_list = []
    donor_list = []
    inputs_list =  []
    for idx,batch in enumerate(dl):
        inputs = batch[0]
        donor = batch[3][0]
        gene = batch[2][0]
    
        inputs.requires_grad = True
        outputs = model(inputs.cuda())
        (gradients,)= torch.autograd.grad(inputs=inputs, outputs=outputs)
    
        grad_list.append(gradients.detach().cpu())
        donor_list.append(donor)
        inputs_list.append(inputs)
    cat_grads = torch.cat(grad_list) #shape = (n_donors,seq_len,4)
    cat_inputs = torch.cat(inputs_list)
    donor_list = donor_list
    save_dir = f'{outdir}/{run_id}/{gene}/{basedir}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(cat_grads.permute(0,2,1),os.path.join(save_dir,"gradients.pt"))
    torch.save(cat_inputs.permute(0,2,1),os.path.join(save_dir,"seqs.pt"))
    with open(f"{save_dir}/donor_order.txt",'w') as f:
        f.write('\n'.join(donor_list))


    # #upweight distal regions using procedure from Borzoi paper (Linder et al. (2025))
    # cat_grads = subtract_mean(cat_grads)
    # all_attr_sd = torch.std(cat_grads, axis = 2)
    # smoothened = gaussian_filter(all_attr_sd, sigma = 1280, truncate = 2, axes = 1) 
    # grad_div_smooth = cat_grads.numpy() / np.expand_dims(smoothened, axis = 2)
    # np.save(os.path.join(save_dir,"grad_div_smooth.npy"),grad_div_smooth.transpose(0,2,1))
def _test_saturation_mutagenesis(model, avg_input, position_df, y_ref,y_ism,width,dl):
    position_df = position_df.iloc[:1] #iterate through 1 row for test
    ism_dict = {}
    for i, var_row in position_df.iterrows():
        var_pos = var_row['variant_pos']
        variant = var_row['variant']
        #tangermeme handles torch.no_grad etc
        y_ref,y_ism = saturation_mutagenesis(model,avg_input,start = var_pos - width, end = var_pos + width + 1,raw_outputs=True,batch_size = 1) #use either ref genome background or average personal genome background
        assert y_ism.shape == (1,4,(width * 2) + 1,1)
        assert y_ref.shape == (1,1)
    model.eval()
    model.cuda()
    with torch.no_grad(): 
        expected_out = model(avg_input.cuda())
    assert torch.allclose(y_ref, expected_out.detach().cpu())
    del expected_out

    # ism score for ref allele should be same as normal prediction (containing all ref alleles)
    var_ref_allele = characters(avg_input[:,:,var_pos:var_pos+1])
    assert variant.split('_')[2] == var_ref_allele #sanity check
    nt_mask = avg_input[:,:,var_pos] == 1 #True at OHE position of the ref allele
    ref_allele_ism = y_ism[nt_mask][:,width,:]
    assert torch.allclose(ref_allele_ism,y_ref)
    
   
    assert width > 1
    test_pos = var_pos + 1
    before = avg_input[:,:,:test_pos]
    after =  avg_input[:,:,(test_pos + 1):]
    test_relative_pos = width + 1 #variant will be at position index equal to width
    
    for nt in ['A','C','G','T']:
        nt_ohe = one_hot_encode(nt).T.unsqueeze(-1)
        new_seq = torch.cat([before,nt_ohe,after],dim = -1)
        assert new_seq.shape[-1] == dl.dataset.desired_seq_len #sanity check
        with torch.no_grad():
            ism_pred = model(new_seq.cuda()).detach().cpu()
        nt_mask = (nt_ohe == 1).squeeze(-1)
        sat_ism_res = y_ism[nt_mask][:,test_relative_pos,:]
        assert torch.allclose(sat_ism_res,ism_pred)
  
def perform_saturation_mutagenesis(dl,
                                    model,
                                    outdir,
                                    run_id,
                                    gene,
                                    basedir,
                                    width,
                                    position_df,
                                    debug_mode,
                                    skip_check):
    """
    Input includes a dataloader that only includes one gene and either reference genomes or personal genomes from the model's test set. 
    The average among those sequences is taken (if its the reference genome, then it won't change)
    This average is used as a background against which to conduct ISM
    """
    save_dir = f'{outdir}/{run_id}/{gene}/{basedir}'
    is_done = check_if_done(save_dir,skip_check)
    if is_done:
        print(f'Skipping {save_dir} Because it is already finished.')
        return
    assert len(dl.dataset.genes_in_dataset) == 1,"This function assumes you are ref/personal genomes coming from one gene"
    model = PermutedSeqModel(model)
    #Take average of personal genomes. If iterating through one reference genome sequence, the average is the same as the original
    avg_input = None
    for idx,batch in enumerate(dl):
        inputs = batch[0]
        donor = batch[3][0]
        gene = batch[2][0]
        if avg_input == None:
            avg_input = inputs.clone()
            initial_gene = gene
        else:
            avg_input += inputs
            assert initial_gene == gene
    avg_input = avg_input / (idx + 1) #take average over all personal genome sequences (or average of single ref genome, leaving it unchanged)
    avg_input = avg_input.permute(0,2,1) #change shape to accomodate tangermeme
    position_df = position_df[(position_df['gene'] == gene) & (position_df['variant_pos'] <= dl.dataset.desired_seq_len) & (position_df['run_id'] == run_id)]
    position_df = position_df.drop_duplicates('variant').reset_index()
    ism_dict = {}
    for i, var_row in position_df.iterrows():
        var_pos = var_row['variant_pos'] #tangermeme.ism.saturation_mutagenesis performs ism assuming 0-based (python) indexing. 
        variant = var_row['variant']
        if basedir == 'ref_genome': #double check no mix-up with 0-based indexing using the ref genome as a sanity check.
            expected_ref_allele = variant.split('_')[2]
            allele_in_ref = characters(avg_input[:,:,var_pos:var_pos+1])
            assert expected_ref_allele == allele_in_ref, "If `var_pos` is 0-based, this should pass"
        #tangermeme handles torch.no_grad etc
        y_ref,y_ism = saturation_mutagenesis(model,avg_input,start = var_pos - width, end = var_pos + width + 1,raw_outputs=True,batch_size = 1) #use either ref genome background or average personal genome background
        assert y_ism.shape == (1,4,(width * 2) + 1, 1)
        assert y_ref.shape == (1,1)

        if (i == 0) and (debug_mode):
            _test_saturation_mutagenesis(model, avg_input, position_df, y_ref,y_ism,width,dl)
        if i == 0:
            ism_dict['ref'] = y_ref.float().numpy()
        ism_dict[variant] = y_ism.float().numpy()

    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(os.path.join(save_dir,f"ism_{dl.dataset.desired_seq_len}bp.npz"),**ism_dict)
    position_df.to_csv(os.path.join(save_dir,"position_df.csv"))
   
def check_if_done(save_dir,skip_check):
    if skip_check:
        return False
    return os.path.exists(save_dir)

def analyze_enformer(outdir,gene_list,desired_seq_len = 49152,analysis = 'grad_input',model_name = 'Enformer',debug = False, width = None, position_df = None,skip_check = False):
    tissues_to_train = ['Whole Blood'] #only blood predictions supported currently
    model = SliceEnformer(bin_length = 3)
    model.eval()
    model.cuda()

    
    for fold in range(3):
        test_donor_path = f"/pollard/data/projects/sdrusinsky/enformer_fine_tuning/data/cross_validation_folds/gtex/cv_folds/person_ids-test-fold{fold}.txt"
        run_id = f"Enformer_Fold-{fold}_{desired_seq_len}bp"
        for i,gene in enumerate(gene_list):
            dataset,ref_dataset = instantiate_dataset([gene], #one gene in the dataset
                                                test_donor_path,
                                                desired_seq_len,
                                                tissues_to_train)
            dl = DataLoader(dataset, batch_size=1, shuffle=False) 
            ref_dl = DataLoader(ref_dataset, batch_size=1, shuffle=False)
            if analysis == 'grad_input':
                get_input_grads(ref_dl,model,
                            outdir,
                            run_id,
                            gene,
                            'ref_genome',
                            skip_check
                            )
            
                get_input_grads(dl,model,
                                    outdir,
                                    run_id,
                                    gene,
                                    'personal_genomes',
                                    skip_check
                                    )
            else:
                perform_saturation_mutagenesis(ref_dl,model,
                                outdir,
                                run_id,
                                gene,
                                'ref_genome',
                                width,
                                position_df,
                                debug,
                                skip_check,
                                )
                
                perform_saturation_mutagenesis(dl,model,
                                        outdir,
                                        run_id,
                                        gene,
                                        'personal_genomes',
                                        width,
                                        position_df,
                                        debug,
                                        skip_check)
            if i % 25 == 0:
                print(f"Done with {i}/{len(gene_list)} Genes")
        print(f"Done with Fold-{fold}")    
    
def analyze_performer(metadata,outdir,genes_to_add = None,skip_check = False,only_genes = None,analysis = 'grad_input',debug = False,width = None, position_df = None):
    if debug:
        atol = 1 #if debugging, use a super generous atol so there won't be an assertion error when loading the model because a different GPU is being used, allowing for debugging of other functions to take place.
    else:
        atol = 1e-05
    for idx, row in metadata.iterrows():
        run_id = row['ID']
        tissues_to_train = row['tissues_to_train'].strip('"[]"').split(',')
        assert tissues_to_train == ['Whole Blood'], "non-blood tissues are not currently supported"
        desired_seq_len = int(row['seq_length'])
        save_dir = row['save_dir']
        test_donor_path = row['test_donor_path']
        valid_donor_path = row['valid_donor_path']
        precision = row['precision']
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
        model = PerformerPredict(model) #automatically return prediction at central bin
        valid_dataset,_valid_ref_dataset = instantiate_dataset([train_genes[0]],valid_donor_path,desired_seq_len, tissues_to_train)
        test_proper_load(model,save_dir,ckpt,valid_dataset,precision,atol = atol)
        assert not model.training
        assert next(model.parameters()).is_cuda
        gene_list = sorted(gene_list)
        for idx2,gene in enumerate(gene_list):
            print(f"{run_id} {idx}/{len(metadata)}\t Gene: {idx2} / {len(gene_list)}")
            sys.stdout.flush()
            test_dataset,ref_dataset = instantiate_dataset([gene], #one gene in the dataset
                                            test_donor_path,
                                            desired_seq_len,
                                            tissues_to_train)

            dl = DataLoader(test_dataset, batch_size=1, shuffle=False) 
            ref_dl =DataLoader(ref_dataset, batch_size=1, shuffle=False)  
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if analysis == 'grad_input':
                    get_input_grads(dl,
                                    model,
                                    outdir,
                                    run_id,
                                    gene,
                                    'personal_genomes',
                                    skip_check)
                    get_input_grads(ref_dl,
                                    model,
                                    outdir,
                                    run_id,
                                    gene,
                                    'ref_genome',
                                    skip_check)
                else:
                    perform_saturation_mutagenesis(dl,
                                                   model,
                                                    outdir,
                                                    run_id,
                                                    gene,
                                                    'personal_genomes',
                                                    width,
                                                    position_df,
                                                    debug,
                                                    skip_check)
                    perform_saturation_mutagenesis(ref_dl,
                                                   model,
                                                    outdir,
                                                    run_id,
                                                    gene,
                                                    'ref_genome',
                                                    width,
                                                    position_df,
                                                    debug,
                                                    skip_check)
            
def main():
    parser = argparse.ArgumentParser(description="For ISM")
    parser.add_argument("--path_to_metadata",type=str,nargs = '?',help = "Metadata from Wandb run to ensure correct specifications are used")
    parser.add_argument("--model_name",type=str, help = 'One of SingleGene or MultiGene or Enformer')
    parser.add_argument("--path_to_only_genes_file",type=str,nargs ='?', help = 'Optional path to a txt file containing an explicit set of genes to evaluate. Each gene must be on its own row')
    parser.add_argument("--enformer_seq_len",type=int,nargs ='?', help = 'seq_len to use with Enformer')
    parser.add_argument("--debug",type=int,default = False, help = "To debug on a different GPU, don't test instantiation of performer")
    parser.add_argument("--attribution",type=str,default = 'grad_input', help = "One of `grad_input' or 'saturation_mutagenesis'")
    parser.add_argument("--width",type=int,nargs = '?', help = "Width around each variant to perform saturation mutagenesis")
    parser.add_argument("--position_df",type=str,nargs = '?', help = "Absolute path to csv file containing variants to perform saturation mutagenesis around")
    parser.add_argument("--skip_check",type=int,default = 0, help = "Whether to skip checking if a gene has already been evaluated. This will cause the gene to be evaluated again")
    parser.add_argument("--outdir",type=str,nargs = '?', help = "Absolute path to an outfile directory that is preferred")

    
    args = parser.parse_args()
    path_to_metadata = args.path_to_metadata
    model_name = args.model_name
    path_to_only_genes_file = args.path_to_only_genes_file
    enformer_seq_len = args.enformer_seq_len
    debug = bool(args.debug)
    attribution = args.attribution
    width = args.width
    position_df = args.position_df
    skip_check = bool(args.skip_check)
    outdir = args.outdir


    if path_to_only_genes_file is not None:
        gene_list = parse_gene_files(path_to_only_genes_file)
    else:
        gene_list = None # When no gene path is defined, genes used during training,validation, and test are used instead (From metadata)

    if attribution == 'saturation_mutagenesis':
        assert width is not None
        assert position_df is not None
        position_df = pd.read_csv(position_df)
        if gene_list is None:
            gene_list = sorted(list(position_df['gene'].unique()))

    if outdir is None:
        if attribution == 'grad_input':
            attr_str = 'RevisionGradInput'
        elif attribution == 'saturation_mutagenesis':
            attr_str = 'RevisionSaturationMutagenesis'
        outdir = f'/pollard/data/projects/sdrusinsky/enformer_fine_tuning/results/{attr_str}/{model_name}'
    outdir = update_outdir(outdir,attribution)
    if model_name in ['SingleGene','MultiGene']:
        metadata = pd.read_csv(path_to_metadata)
        print('cuda',torch.version.cuda)
        sys.stdout.flush()
        analyze_performer(metadata,outdir,only_genes = gene_list,analysis = attribution, debug = debug,width = width,position_df = position_df, skip_check = skip_check)


    else:
        assert model_name == 'Enformer'
        assert path_to_metadata is None
        tissues_to_train = ['Whole Blood']
        assert enformer_seq_len in [49152, 196608]
        analyze_enformer(outdir,gene_list,enformer_seq_len,analysis = attribution,model_name = model_name, debug = debug, width = width,position_df = position_df,skip_check = skip_check)

if __name__ == '__main__':
    main()


    