import gc
import os
import vcfpy
import pysam
import torch
import argparse
import kipoiseq
import pandas as pd
import numpy as np
import sys
import lightning.pytorch as pl
from torch.utils.data import IterableDataset, DataLoader
from pl_models import LitModelHeadAdapterWrapper


torch.use_deterministic_algorithms(True)

def get_window_around_TSS(window,gene_info):
    """
    Returns a string denoting the start and end position of a region centered around the TSS +/- the window on each side. 
    start and end position in gene_info already have TSS centered
    """
    gene_start = int(gene_info['gene_start'].item())
    region_start = gene_start - window
    region_end = gene_start + window
    region_chr = gene_info['seqnames'].item()
    
    region = region_chr + ':' + str(region_start) + '-' + str(region_end)
    assert region_end - region_start == (window * 2) #checks the returned region is double the window size. It should be if it includes the window on each end of the TSS

    return region

def get_all_gtex_snps(gene, window = 10000):
    """
    Gets all observed GTEx SNPs within +/- window distance from gene TSS and puts into a df with position, ref and alt for ISM
    """
    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd,"../data")
    vcf_path = os.path.join(DATA_DIR,"GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz")
    vcf_reader = vcfpy.Reader.from_path(vcf_path,tabix_path = vcf_path + '.csi')
    enformer_regions = pd.read_csv(os.path.join(DATA_DIR,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    gene_info = enformer_regions[enformer_regions['gene_name'] == gene]
    variant_dict = {'region':[],'chrom':[],'pos0':[],'pos1':[],'ref':[],'alt':[]}
    if gene_info.shape[0] > 0:
    # get all observed SNPs within +/- window of TSS
        region = get_window_around_TSS(window,gene_info) #define region +/- window of TSS
        variant_dict = {'region':[],'chrom':[],'pos0':[],'pos1':[],'ref':[],'alt':[],'AF':[]}
        for i,record in enumerate(vcf_reader.fetch(region)):
            af = record.INFO['AF'][0]
            if af > 0: #Someone should have each of these SNPs at least once, but don't include any that are unobserved
                ref = record.REF
                alt = record.ALT[0].value
                chrom = record.CHROM
                variant_dict['region'].append(region)
                variant_dict['chrom'].append(chrom)
                variant_dict['pos1'].append(record.POS) #VCF is 1-based
                variant_dict['pos0'].append(record.POS - 1) #record 0-based coordinate, because pysam is 0-based
                variant_dict['ref'].append(ref)
                variant_dict['alt'].append(alt)
                variant_dict['AF'].append(record.INFO['AF'][0]) #allele frequency
                variant_dict['gene_name'] = gene
        return pd.DataFrame(variant_dict)
    else:
        return pd.DataFrame(variant_dict) #if gene not in enformer regions return empty dataframe

def get_ckpt(save_dir):
    ckpt = os.listdir(f"{save_dir}/checkpoints")
    if len(ckpt) > 0:
        assert len(ckpt) == 1
        ckpt = ckpt[0]
        return ckpt
    else:
        return None

def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)
def tss_centered_sequences(variant_df, desired_seq_len):
    """
    Generates sequences for ISM -- two one-hot encoded sequences per SNP, one with the ref allele and one with the alt allele, with the TSS of the gene at the center. 
    The SNPs themselves are not placed at the center of the sequence, the gene TSS is always centered, allowing for just one forward pass for the reference sequence per gene.
    """
    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd,"../data")
    gene = variant_df['gene_name'].unique().item()

    ref_seq_open = pysam.Fastafile(os.path.join(DATA_DIR,"hg38_genome.fa"))
    enformer_regions = pd.read_csv(os.path.join(DATA_DIR,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    gene_info = enformer_regions[enformer_regions['gene_name'] == gene]
    seq_window = desired_seq_len // 2 #the sequence used for ism has length seq_window * 2 because it is before and after the TSS. Divide by 2 to get a sequence whose length is desired_seq_len
    region = get_window_around_TSS(seq_window,gene_info)
    region_chr = region.split(':')[0]
    region_start = int(region.split(':')[1].split('-')[0])
    region_end = int(region.split(':')[1].split('-')[1])
    assert region_end - region_start == desired_seq_len
    ref_seq = ref_seq_open.fetch(region_chr, region_start, region_end).upper()

    
    for _, variant_info in variant_df.iterrows():
        index = variant_info['pos0'] - region_start
        assert ref_seq[index] == variant_info['ref'], "nucleotide in reference genome should be identical to the reference allele of the current SNP at the current position"
        alt_seq = ref_seq[:index] + variant_info['alt'] + ref_seq[index + 1:] # put alt allele in its correct position and ref seq around it
        # assert ref_seq[index - 1] == alt_seq[index - 1]
        # assert ref_seq[index + 1] == alt_seq[index + 1]
        yield {'inputs': {'ref': one_hot_encode(ref_seq),
                      'alt': one_hot_encode(alt_seq)},
               'metadata': {'chrom': variant_info.chrom,
                            'pos': variant_info.pos0,
                            'ref': variant_info.ref,
                            'alt': variant_info.alt,
                            'gene_name':gene,
                            'region_chr':region_chr,
                            'region_start':region_start,
                            'region_end':region_end}}
        #yield one hot encoded sequences and metadata
def parse_gene_files(filepath):
    gene_list = []
    with open(filepath,'r') as file:
        for gene in file:
            gene_list.append(gene.strip())
    return gene_list

def clean_genes(metadata):
    metadata['genes_for_training'] = metadata['genes_for_training'].str.strip('[]').str.replace('"','').str.split(',')
    metadata['genes_for_valid'] = metadata['genes_for_valid'].str.strip('[]').str.replace('"','').str.split(',')
    metadata['genes_for_test'] = metadata['genes_for_test'].str.strip('[]').str.replace('"','').str.split(',')


class LitModelPerformerISM(pl.LightningModule):
    """To wrap Model within LightningModule to form ISM predictions within Lightning Trainer object, for easy handling of precision among other things"""
    def __init__(self, model, run_id, ckpt):
        super().__init__()
        self.model = model
        self.results_df = pd.DataFrame(columns = ['chrom', 'pos0', 'ref','alt','ref_pred','alt_pred','run_id','gene_name','region_chr','region_start','region_end','model_ckpt'])
        self.run_id = run_id
        self.ckpt = ckpt

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement the logic for a single prediction step
        # Extract inputs from batch, perform model forward pass, and return predictions
        inputs = batch['inputs']
        ref_seq = inputs['ref']
        alt_seq = inputs['alt']
        if batch_idx == 0:
            self.ref_pred = self.model(ref_seq)
            self.ref_pred =  self.ref_pred[:,self.ref_pred.shape[1]//2,:].cpu().item() #turn into an attribute to avoid forming same prediction over and over. Get ref_pred once and store it, then ignore it. Because
            self.region_start_in_first_batch = batch['metadata']['region_start'][0].cpu().item()
            self.region_end_in_first_batch = batch['metadata']['region_end'][0].cpu().item()
            self.region_chr_in_first_batch = batch['metadata']['region_chr'][0]

        alt_pred = self.model(alt_seq)
        alt_pred =  alt_pred[:,alt_pred.shape[1]//2,:] #turn into an attribute to avoid forming same prediction over and over. Get ref_pred once and store it, then ignore it. Because
        assert alt_pred.shape[0] == 1, "Larger batch sizes not implemented yet"
        alt_pred = alt_pred.cpu().item()

        chrom = batch['metadata']['chrom']
        assert len(chrom) == 1, "Larger batch sizes not implemented yet"

        region_start = batch['metadata']['region_start'][0].cpu().item()
        region_end = batch['metadata']['region_end'][0].cpu().item()
        region_chr = batch['metadata']['region_chr'][0]
        assert region_start == self.region_start_in_first_batch, "ISM should be occuring in just one region because you are re-using ref_pred!"
        assert region_end == self.region_end_in_first_batch
        assert region_chr == self.region_chr_in_first_batch

        gene_name = batch['metadata']['gene_name'][0]
        chrom = chrom[0]
        pos = batch['metadata']['pos'][0].cpu().item()
        ref = batch['metadata']['ref'][0]
        alt = batch['metadata']['alt'][0]

        self.results_df.loc[self.results_df.shape[0],:] = [chrom,pos,ref,alt,self.ref_pred,alt_pred,self.run_id, gene_name, region_chr,region_start,region_end,self.ckpt]
        return None
class IsmDataset(IterableDataset):
    """ To wrap ISM data generator as a pytorch dataset"""
    def __init__(self, it,length):
        self.it = it
        self.length = length
    def __iter__(self):
        return self.it
    def __len__(self):
        return self.length
    

def load_model(ckpt,save_dir,run_id):
    path_to_ckpt = os.path.join(save_dir,f'checkpoints/{ckpt}')
    loaded_model = LitModelHeadAdapterWrapper.load_from_checkpoint(path_to_ckpt)
    return loaded_model
def main():
    parser = argparse.ArgumentParser(description="For ISM")
    parser.add_argument("--path_to_metadata",type=str,help = "Metadata from Wandb run to ensure correct specifications are used")
    parser.add_argument("--model_type",type=str, help = 'One of SingleGene or MultiGene or OligoGene')
    parser.add_argument("--outdir",type=str,nargs ='?', help = 'Optional directory to save outputs')
    parser.add_argument("--path_to_only_genes_file",type=str,nargs ='?', help = 'Optional path to a txt file containing an explicit set of genes to evaluate. Each gene must be on its own row')
    parser.add_argument("--subset",type=int,nargs = '?',help = "Metadata enumerating models will be split into n_subsets and this subset of models will be evaluated by the job.")
    parser.add_argument("--n_subsets",type=int,nargs = '?',help = "Metadata enumerating models will be split into n_subsets.")

    
    args = parser.parse_args()
    metadata = pd.read_csv(args.path_to_metadata)
    outdir = args.outdir
    model_type = args.model_type
    path_to_only_genes_file = args.path_to_only_genes_file
    subset = args.subset
    n_subsets = args.n_subsets
    metadata = metadata.rename(columns = {'ID':'run_id'})
    clean_genes(metadata)
    assert model_type in ['SingleGene','MultiGene','OligoGene']
    
    if n_subsets:
        n_subsets = int(n_subsets) #ensure read as int
        subset = int(subset)
        metadata = np.array_split(metadata,int(n_subsets))[int(subset)]

    


    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'../data')
    if not outdir:
        outdir = os.path.join(cwd,'../results/PerformerISM')
    enformer_regions = pd.read_csv(os.path.join(data_dir,"Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
    for idx, row in metadata.iterrows():
        seed = int(row['seed'])
        pl.seed_everything(seed, workers=True)
        run_id = row['run_id']
        tissues_to_train = row['tissues_to_train'].strip('"[]"').split(',') #configure as a list of strings
        assert len(tissues_to_train) == 1, "ISM on only 1 output tissue is supported"
        desired_seq_len = int(row['seq_length'])
        window = desired_seq_len // 2 #window is the amount of bp ahead and behind the TSS to score. Half the sequence length to score the entire sequence 
        precision = row['precision']
        save_dir = row['save_dir']
        valid_genes = row['genes_for_valid']
        train_genes = row['genes_for_training']
        test_genes = row['genes_for_test']
        if path_to_only_genes_file:
            genes_to_score = parse_gene_files(path_to_only_genes_file)
        else:
            genes_to_score = valid_genes + train_genes + test_genes
        ckpt = get_ckpt(save_dir)
        if ckpt is None: #some models won't have checkpoints saved. Namely, if the gene they were meant to be trained with is incompatible and there are no other train genes available to train them, training exits and there is no ckpt
            continue

        loaded_model = load_model(ckpt, save_dir,run_id)
        model = LitModelPerformerISM(model = loaded_model, ckpt = ckpt, run_id = run_id) #Loaded model will be used for predictions, but predict_step overwritten to perform ISM
        model.eval()
        model.cuda()

        tissue_str = tissues_to_train[0].replace(' -','').replace(' ','_').replace('(','').replace(')','')
        outpath = os.path.join(outdir,f"{tissue_str}Models",model_type,run_id)
        if not os.path.exists(os.path.join(outpath)):
            os.makedirs(os.path.join(outpath))
        
        for gene_idx,gene in enumerate(genes_to_score):
            filename = os.path.join(outpath,f"{gene}_{run_id}_model_ISM_{window * 2}bp.csv")
            variant_df = get_all_gtex_snps(gene,window)
            if (variant_df.shape[0] > 0) and (not os.path.exists(filename)): #gene must be in enformer regions, must have SNPs nearby, and must not already have been evaluated
                print(f"Performing ISM on gene {gene} {gene_idx}/{len(genes_to_score)}")
                gene_info = enformer_regions[enformer_regions['gene_name'] == gene]
                gene_start = int(gene_info['gene_start'].item())
                variant_df = variant_df[(variant_df['pos0'] - gene_start).abs() <= window] #final assurance that no SNPs are further away than you want
                it = tss_centered_sequences(variant_df, desired_seq_len)
                dataset = IsmDataset(it, length = variant_df.shape[0])
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                lit_model = LitModelPerformerISM(model, run_id, ckpt)
                trainer = pl.Trainer(precision=precision,
                    num_sanity_val_steps = 0, #check all validation data before starting to train
                    deterministic = True)  
                trainer.predict(lit_model,dataloader) #perform ISM
                lit_model.results_df.to_csv(filename)
            else:
                print(f"Skipping {os.path.join(outpath,f"{gene}_{run_id}_model_ISM_{window * 2}bp.csv")}")
            sys.stdout.flush()
        del model
        gc.collect()
        torch.cuda.empty_cache()
       
if __name__ == "__main__":
    main()