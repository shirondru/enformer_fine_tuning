import os
import pandas as pd
import numpy as np
import kipoiseq
import random
import pysam
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class GTExDataset(Dataset):
    def __init__(self, 
             tissues_to_train: list, 
             requested_regions: list, 
             desired_seq_len: int, 
             num_individuals_per_gene: int, 
             donor_list_path: str, 
             gene_expression_df: 'pandas.DataFrame', 
             DATA_DIR: str) -> None:
        """
        Args:
            tissues_to_train (list): List of GTEx tissues to be trained.
            requested_regions (list): List of desired genes.
            desired_seq_len (int): Desired length of input DNA sequences. These sequences will be TSS-centered, and this controls how big of a region around the gene's TSS will be used.
            num_individuals_per_gene (int): Number of people assigned to a gene for one effective gradient-accumulated batch. This is distinct from batch size and there can be more than one effective batch per gene. For example, if train batch size is 8 and num_individuals_per_gene is 128, then the next 128 // 8 = 16 consecutive batches will include the same gene and 128 people will be assigned to that gene for training during those 16 batches. If there are 512 total people with data for a given tissue, then there are 512 / 128 = 4 gradient accumulated batches for each gene per epoch, and these batches need not be consecutive (they occur randomly)
            donor_list_path (str): Path to a line-separated txt file denoting the GTEx donor IDs for use during training. There are multiple train/validation/test files for different cross validation splits.
            gene_expression_df (pandas.DataFrame): DataFrame containing gene expression data for the same single tissue as in tissues_to_train.
            DATA_DIR (str): Directory for data storage.
        """
       
        assert type(requested_regions) == list
        assert type(tissues_to_train) == list
        assert len(tissues_to_train) == 1, "Only single tissue training is currently supported"

        
        self.DATA_DIR = DATA_DIR
        self.desired_seq_len = desired_seq_len
        self.donor_list_path = donor_list_path
        self.gene_expression_df = gene_expression_df
        
        
        if type(tissues_to_train) == str:
            self.tissues_to_train = [tissues_to_train]
        else:
            self.tissues_to_train = tissues_to_train

        # define genomic regions to be used
        self.all_regions = pd.read_csv(os.path.join(self.DATA_DIR, "Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"))
        self.genomic_regions_df = self.all_regions[self.all_regions['gene_name'].isin(requested_regions)]
        self.genomic_regions_df = self.genomic_regions_df.reset_index(drop=True)
        self.genes_in_dataset = list(self.genomic_regions_df['gene_name'].unique())
        
        
        #select donors that have WGS & RNA-seq in the desired tissue(s) and that are meant to be used in the current CV split
        self.individuals_in_split = self.select_donors()
                
        if num_individuals_per_gene == -1: #EvalGTExDataset overwrites this class and sets this to -1 to use all people. Whereas during training, 
            self.num_individuals_per_gene = len(self.individuals_in_split)
        else:
            self.num_individuals_per_gene = num_individuals_per_gene

        # an epoch is a pass not only through all genes with one gradient accumulated step per gene (equal to num_individuals_per_gene forward steps),
        # but multiple steps per gene will be taken. As many steps will be taken as possible, given the number of accumulations desired
        self.n_gene_replicate_batches_per_epoch = len(self.individuals_in_split) // self.num_individuals_per_gene #how many times each gene will appear in a batch per epoch, 
        self.consensus_seq_dir = os.path.join(self.DATA_DIR, "ConsensusSeqs_SNPsOnlyUnphased")

        self.shuffle_and_define_epoch()
    def _get_gene_and_individual_from_idx(self, idx):
        # Determines which gene to use. If the number of individuals per gene is 10 and the idx is 0, pick the 0th gene
        # if it is 10 pick the 1st gene and so on
        region_idx = idx // self.num_individuals_per_gene
        region_row = self.region_rows_in_epoch[region_idx]

        # Determines which person to use. First indexes the list of donors that match the current gene by indexing the gene.
        #Then the modulus selects the correct individual. If there are 10 individuals per gene and the idx is 0, the modulus is 0 so pick the 0th individual
        # if the idx is 10, the gene idx is 1 and the modulus is 0, so pick the 0th individual fror the 1st gene. If the idx is 11, pick the 1st index individual for the 1st gene.
        individual_idx = idx % self.num_individuals_per_gene
        individual = self.indivs_per_epoch[region_idx][individual_idx]

        return region_row, individual
    def select_donors(self):
        """
        Returns donors from the desired Cross-Validation Split (defined by donor_list_path) that include gene expression data for the desired tissue.
        Currently this only supports single tissue training
        """
        assert len(self.tissues_to_train) == 1, "Multi Tissue Training is not supported yet."

        gene_expression_df_cols = list(self.gene_expression_df.columns)
        with open(self.donor_list_path, 'r') as f:
            file = f.read()
            GTEx_data_split_IDs = file.split("\n")
            GTEx_data_split_IDs = [ID for ID in GTEx_data_split_IDs if ID != '']  # remove any empty strings, if they exist

        donors_for_tissue = sorted(list(set([x for x in gene_expression_df_cols if x in GTEx_data_split_IDs]))) #expr columns in this gene expression matrix are donor ids. Select desired donors with expression data
        return donors_for_tissue
    @staticmethod
    def _one_hot_encode(sequence):
        ## one hot encodes DNA using the same code from the original Enformer paper. Ensures one-hot encoding is consistent with representations Enformer has already learned for more efficient transfer learning
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)
    @staticmethod
    def _reverse_complement_seq(seq):
            """
            Reverse complements the sequence while reversing the location of the gene expression signal.
            Citation: Geeksforgeeks.com
            """
            # First complement the sequence.
            seq = seq.replace("A", "t").replace(
                "C", "g").replace("T", "a").replace("G", "c")
            seq = seq.upper()
            # reverse the strand
            rc_seq = seq[::-1]
            return rc_seq
    def _one_hot_encode_diploid(self,seq1,seq2):
        """
        Returns a single one hot encoded sequence from an unphased diploid genome in order to pass in as input into the model.
        It does this by taking two haplotypes from a diploid genome, one hot encoding each, and taking the average.
        Heterozygous positions are therefore encoded as 0.5s.
        This is only appropriate for unphased sequences without indels. It should be changed for phased genomes since heterozygous SNPs can be mapped to one haplotype or the other in that case.
        """
        one_hot_seq1 = self._one_hot_encode(seq1)
        one_hot_seq2 = self._one_hot_encode(seq2)
        return (one_hot_seq1 + one_hot_seq2) / 2 
    def get_path_to_consensus_seq(self,donor_id, haplotype_num):
        return os.path.join(self.consensus_seq_dir, f"{donor_id}_consensus_H{haplotype_num}.fa")
    def _get_single_GTEx_donor_sequence(self,gtex_id,region_chr,region_start,region_end):
        consensus1_open = pysam.Fastafile(self.get_path_to_consensus_seq(gtex_id,1))
        consensus2_open = pysam.Fastafile(self.get_path_to_consensus_seq(gtex_id,2))
        seq1 = consensus1_open.fetch(region_chr, region_start, region_end).upper()
        assert len(seq1) == self.desired_seq_len, f"Seq1 should be length {self.desired_seq_len}. Offending region: {region_chr}:{region_start}-{region_end}"
        seq2 = consensus2_open.fetch(region_chr, region_start, region_end).upper()
        assert len(seq2) == self.desired_seq_len, f"Seq2 should be length {self.desired_seq_len}. Offending region: {region_chr}:{region_start}-{region_end}"
        consensus1_open.close()
        consensus2_open.close()
        
        diploid_one_hot = self._one_hot_encode_diploid(seq1, seq2)  # one hot encode a diploid DNA sequence. Heterozygosity ==> 0.5/0.5
        return diploid_one_hot
    def _get_single_GTEx_donor_expression(self,gtex_id,gene_name):
        """
        gtex_id (str): GTEx donor ID
        gene_name (list): Gene to get expression from, in the form of a list of length 1. In the scenario where multiple genes align to the same bin, multiple genes are passed in and their summed expression is used
        """
        gene_expression_vector = np.zeros((len(self.tissues_to_train)), dtype=np.float32)
        indiv_tpm = self.gene_expression_df[['#chr', 'start', 'gene_id','Description', gtex_id]]
        indiv_tpm = indiv_tpm[indiv_tpm['Description'].isin(gene_name)]
        assert indiv_tpm.shape[0] > 0, f"There is no gene expression data available for gene(s) at position {gene_name}"
        for tissue_idx,tissue in enumerate(self.tissues_to_train):
            gene_expression_vector[tissue_idx] = indiv_tpm[gtex_id].sum(min_count = 1) 
        return gene_expression_vector
    def generate_train_batch_one_gene(self,sampled_individual,region_chr,region_start,region_end,gene_name):
        # If the desired sequence length is not Enformer's original 196kb, it resets the start and end positions by trimming the ends as necessary
        if self.desired_seq_len != 196608: 
            region_center = region_end - (196608 // 2)
            region_start = region_center - (self.desired_seq_len // 2)
            region_end = region_center + (self.desired_seq_len // 2)

        gene_name = gene_name.split('/') #gene_name can be a list of multiple genes overlap TSS bin in the embedding. Splitting converts to a list of 1 gene if none overlap, or multiple genes
        dna_seq = self._get_single_GTEx_donor_sequence(sampled_individual,
                                                            region_chr,
                                                            region_start,
                                                            region_end
                                                            )
        gene_expression = self._get_single_GTEx_donor_expression(sampled_individual,gene_name)
        return dna_seq,gene_expression   
    def shuffle_and_define_epoch(self):
        """
        This method shuffles the dataset in a way that ensures each batch contains only one gene, and the same gene appears in consecutive batches until the number of desired batches for gradient accumulation has been satisfied.
        If there are enough individuals for multiple full gradient-accumulated batches, it repeats this multiple times, while allowing fo those gradient-accumulated batches to be separated
        
        For example, if num_individuals_per_gene is 128 and there are 550 people with data for the desired tissue, each gradient accumulated batch will include 128 people for the same gene. Thus, data from 128 different random people and the same gene will appear consecutively. Then this will continue for another random gene. 
        Each gene will appear in 3 more gradient accumuated batches, appearing with 512 total random individuals, and the remaining individuals will be dropped this epoch.
        """

        self.genomic_regions_df = self.genomic_regions_df.sample(frac=1).reset_index(drop=True) #shuffle dataset. Keep this here so the dataset shuffling within litmodel.on_train_epoch_end persists
        
        self.indivs_per_epoch = [] #This will be a list of lists. Each inner list will contain list of donors paired to each gene in a gradient accumulated effective batch. There will be as many inner lists as gradient accumulated batches per epoch
        self.region_rows_in_epoch = [] #will contain order of genes. True batches will yield the same gene until the accumulated batch ends, then the next gene will be the next element in this list
        for i in range(0, len(self.genomic_regions_df), 1):
            indivs_for_gene = random.sample(self.individuals_in_split,self.num_individuals_per_gene * self.n_gene_replicate_batches_per_epoch) #randomly sample ppl for this gene. If n_gene_replicate_batches_per_epoch == 1, you get enough people for one gradient accumulated batch. If this value is 4, the # of people will be equal to 4x the length of one gradient accumulated batch
            split_indivs_for_gene = np.array_split(indivs_for_gene,self.n_gene_replicate_batches_per_epoch) #split indivs_for_gene into different arrays. They are split so the same gene doesn't need to appear in consecutive accumulated batches. The arrays will be equal length (equal to the length of a gradient accumulated batch, self.num_individuals_per_gene)
            for j in range(self.n_gene_replicate_batches_per_epoch): # Create a new accumulated batch for each gene if n_gene_replicate_batches_per_epoch >1. Else if n_gene_replicate_batches_per_epoch == 1, you get just one accumulated batch per gene
                self.region_rows_in_epoch.append(self.genomic_regions_df.iloc[i])
                self.indivs_per_epoch.append(list(split_indivs_for_gene[j]))
        
        #shuffle order of genes (and match the assigned people), so each accumulated batch doesn't have the same genes back to back
        #Since region_rows_in_epoch will only iterate to the next element (likewise for indivs_per_epoch since it is a nested list) after an accumulated batch is complete
        #this won't mix genes per accumualted batch or mini batch
        shuffled_idxs = np.random.permutation(len(self.region_rows_in_epoch))
        self.region_rows_in_epoch = [self.region_rows_in_epoch[i] for i in shuffled_idxs]
        self.indivs_per_epoch = [self.indivs_per_epoch[i] for i in shuffled_idxs]
    def __getitem__(self, idx):
                       
                       
        region_row, individual = self._get_gene_and_individual_from_idx(idx)
        dna_vector, expression_vector = self.generate_train_batch_one_gene(
                                                                            individual,
                                                                            region_row['seqnames'],
                                                                            region_row['starts'],
                                                                            region_row['ends'],
                                                                            region_row['gene_name']
                                                                            )
        return dna_vector, expression_vector, region_row['gene_name'],individual,idx
    def __len__(self):
        """
        
        """
        return len(self.genomic_regions_df) * self.num_individuals_per_gene * self.n_gene_replicate_batches_per_epoch

class ROSMAPDataset(GTExDataset):
    """
    Overwrite GTExDataset to account for different location of consensus fasta files for ROSMAP WGS, and the different structure of the expression data
    """
    def __init__(self, 
             requested_regions: list, 
             desired_seq_len: int, 
             num_individuals_per_gene: int, 
             donor_list_path: str, 
             gene_expression_df: 'pandas.DataFrame', 
             DATA_DIR: str) -> None:
        """
        Args:
            requested_regions (list): List of desired genes.
            desired_seq_len (int): Desired length of input DNA sequences. These sequences will be TSS-centered, and this controls how big of a region around the gene's TSS will be used.
            num_individuals_per_gene (int): Number of people assigned to a gene for one effective gradient-accumulated batch. This is distinct from batch size and there can be more than one effective batch per gene. For example, if train batch size is 8 and num_individuals_per_gene is 128, then the next 128 // 8 = 16 consecutive batches will include the same gene and 128 people will be assigned to that gene for training during those 16 batches. If there are 512 total people with data for a given tissue, then there are 512 / 128 = 4 gradient accumulated batches for each gene per epoch, and these batches need not be consecutive (they occur randomly)
            donor_list_path (str): Path to a line-separated txt file denoting the GTEx donor IDs for use during training. There are multiple train/validation/test files for different cross validation splits.
            gene_expression_df (pandas.DataFrame): DataFrame containing gene expression data for the same single tissue as in tissues_to_train.
            DATA_DIR (str): Directory for data storage.
        """
        self.tissues_to_train = ['DLPFC'] # dorsolateral prefrontal cortex
        super().__init__(self.tissues_to_train,requested_regions,desired_seq_len,num_individuals_per_gene,donor_list_path,gene_expression_df,DATA_DIR)
        self.consensus_seq_dir = os.path.join(self.DATA_DIR, "ROSMAPConsensusSeqs_SNPsOnlyUnphased")

    def generate_train_batch_one_gene(self,sampled_individual,region_chr,region_start,region_end,gene_name):
        """
        Overwrite to change to use self._get_single_ROSMAP_donor_expression
        self._get_single_GTEx_donor_sequence will work for ROSMAP sequences.
        """
        
        if self.desired_seq_len != 196608: #if using shorter seq len, redefine start and end of region while keeping site of gene TSS centered
            region_center = region_end - (196608 // 2)
            region_start = region_center - (self.desired_seq_len // 2)
            region_end = region_center + (self.desired_seq_len // 2)

        gene_name = gene_name.split('/') #gene_name can be a list of multiple genes overlap TSS bin in the embedding. Splitting converts to a list of 1 gene if none overlap, or multiple genes
        dna_seq = self._get_single_GTEx_donor_sequence(sampled_individual,
                                                            region_chr,
                                                            region_start,
                                                            region_end
                                                            )


        gene_expression = self._get_single_ROSMAP_donor_expression(sampled_individual,gene_name)

          
        return dna_seq,gene_expression
    def get_path_to_consensus_seq(self,donor_id, haplotype_num):
        """Consensus sequences have different naming scheme between GTEx and ROSMAP. Overwrite to accomodate this."""
        return os.path.join(self.consensus_seq_dir, f"{donor_id}_H{haplotype_num}.fa")
    def _get_single_ROSMAP_donor_expression(self,donor_id,gene_name):
        """
        donor_id (str): ROSMAP donor ID
        gene_name (list): Gene to get expression from, in the form of a list of length 1. In the scenario where multiple genes align to the same bin, multiple genes are passed in and their summed expression is used
        """
        gene_expression_vector = np.zeros((1), dtype=np.float32) #the only element in the vector will be expression in the Brain Cortex tissue. No other tissues. 
       
        indiv_tpm = self.gene_expression_df.loc[gene_name,donor_id]
        assert indiv_tpm.shape[0] > 0, f"There is no gene expression data available for gene(s) at position {gene_name}"
        gene_expression_vector[0] = indiv_tpm.sum(min_count = 1)
        return gene_expression_vector

class EvalAcrossGeneDataset(GTExDataset):
    """
    A dataset that offers, for a given set of genes (requested regions) the TSS-centered reference genome sequence and average GTEx expression value
    """
    def __init__(self,tissues_to_train,
                 requested_regions,
                 desired_seq_len,
                 gene_expression_df,
                 DATA_DIR): 
        assert len(tissues_to_train) == 1, "This class only supports evaluation across genes 1 gene at a time"
        self.tissue = tissues_to_train[0]
        donor_list_path = os.path.join(DATA_DIR,"All_GTEx_ID_list.txt") #not using list of donors. Just Reference Genome. So donor list path includes all GTEx individuals so I take average expression among all of them
        num_individuals_per_gene = 1 #one reference genome per gene
        super().__init__(tissues_to_train,requested_regions,desired_seq_len,num_individuals_per_gene,donor_list_path,gene_expression_df,DATA_DIR)
        self.ref_seq_open = pysam.Fastafile(os.path.join(self.DATA_DIR,"hg38_genome.fa")) #keep ref seq open to use to switch variants to reference allele (i.e, mask them)

    def shuffle_and_define_epoch(self):
        self.genomic_regions_df = self.genomic_regions_df.sample(frac=1).reset_index(drop=True) #shuffle dataset. Keep this here so the dataset shuffling within litmodel.on_train_epoch_end persists
        
        self.region_rows_in_epoch = [] 
        for i in range(0, len(self.genomic_regions_df), 1):
            self.region_rows_in_epoch.append(self.genomic_regions_df.iloc[i])

    def _get_ref_seq_for_gene(self, region_chr,region_start,region_end):
        ref_seq = self.ref_seq_open.fetch(region_chr, region_start, region_end).upper()
        return self._one_hot_encode(ref_seq)
    def _get_avg_GTEx_gene_expr(self,gene_name):
        gene_expression_vector = np.zeros((len(self.tissues_to_train)), dtype=np.float32)
        #self.individuals_in_split are all people with data in this tissue and for which we have WGS
        gene_df = self.gene_expression_df[self.gene_expression_df['Description'].isin(gene_name)]
        assert gene_df.shape[0] > 0, f"There is no gene expression data available for gene(s) at position {gene_name}"

        #get all expression columns corresponding to donors with data in the desired tissue
        sample_ids = self.samples_per_tissue_dict[self.tissue]
        gene_df = gene_df[sample_ids]
        assert len(gene_df.columns) == len(sample_ids)
        
        gene_expression_vector[0] = np.mean(gene_df)
        return gene_expression_vector
    def generate_train_batch_one_gene(self,region_chr,region_start,region_end,gene_name):
        """
        returns reference seq and avg expression (among GTEx ppl) for desired gene
        
        """
        
        if self.desired_seq_len != 196608: #if using shorter seq len, redefine start and end of region while keeping site of gene TSS centered
            region_center = region_end - (196608 // 2)
            region_start = region_center - (self.desired_seq_len // 2)
            region_end = region_center + (self.desired_seq_len // 2)

        gene_name = gene_name.split('/') #gene_name can be a list of multiple genes overlap TSS bin in the embedding. Splitting converts to a list of 1 gene if none overlap, or multiple genes
        dna_seq = self._get_ref_seq_for_gene(region_chr,
                                            region_start,
                                            region_end)


        gene_expression = self._get_avg_GTEx_gene_expr(gene_name)

          
        return dna_seq,gene_expression
    
    def __getitem__(self, idx):
                       
                       
        region_row = self.region_rows_in_epoch[idx]
        dna_vector, expression_vector = self.generate_train_batch_one_gene(
                                                                            region_row['seqnames'],
                                                                            region_row['starts'],
                                                                            region_row['ends'],
                                                                            region_row['gene_name'],
                                                                            )
 
        return dna_vector, expression_vector, region_row['gene_name'],idx
    def __len__(self):
        return len(self.region_rows_in_epoch) #returns one reference sequence per gene

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, shuffle = False,  **kwargs):
        super().__init__(dataset, shuffle = shuffle, **kwargs)
        self.dataset = dataset
        self.num_total_genes = len(self.dataset.genomic_regions_df)
        self.num_individuals_per_gene = self.dataset.num_individuals_per_gene
        self.n_gene_replicate_batches_per_epoch_per_replica = self.dataset.n_gene_replicate_batches_per_epoch 


        self.genes_per_replica = self.num_total_genes // self.num_replicas #this calculcates the number of complete groups per GPU. Any extras are ignored
        self.genes_per_replica = self.genes_per_replica * self.n_gene_replicate_batches_per_epoch_per_replica #With gradient accumulation, the same gene is repeated in different batches until the effective batch size is achieved. Then, the same thing could repeat multiple times for the same genes if there are enough people. Thus, if there are 3 genes but enough ppl to achieve the effective batch size 4x, it is equiavlent to ther ebeing 12 genes
        self.num_samples_per_replica = self.genes_per_replica * self.num_individuals_per_gene

        print(f"Effective genes_per_replica: {self.genes_per_replica}",f"num_individuals_per_gene: {self.num_individuals_per_gene}",dataset)

        print(f"CustomDistributedSampler Expected # of devices: {self.num_replicas}")

    def __iter__(self):
        """ 
        This is called once per epoch and is not shuffled, because the order of genes and people will be shuffled by the dataset object at each epoch. Instead, these
        indices are kept the same to ensure the data ordering by the dataset is maintained, and each batch only contains samples of one gene on each gpu
        
        How this works: Suppose I have 937 genes, and want each GPU to be trained using data from 6 people for the same gene per batch. And I have 4 GPUs. 
        Each GPU will be assigned 937 //4 = 234 genes. One gene will be ignored.
        For a GPU 0 will be assigned indices 0 to 1403, GPU1 will get 1404 to 1807 etc Because:
        For GPU 0 the start group will be 0 * 234 = 0 and the end group will be 0 + 234 = 234. The indices will be extended by 6 for each element from 0->234, yielding a list that spans 0 to 1403 (inclusive)
        For GPU 1 the start group will be 234 and the end group will be 468. And so forth
        The final GPU will span 4212 to 5615 (inclusive) because the last gene is dropped, since there is not 3 other genes (one per GPU)
        Then, each batch, indices 0-6 will be sampled, then 7-12 and so on. The dataset is organized such that this yields samples for a given gene.


        if use_all_ppl is True, and dataset.n_gene_replicate_batches_per_epoch >1, it is possible that some genes won't see all people in the batch (But they will always see a multiple of num_individuals_per_gene). 
        This is beacuse, instead of certain genes being ommitted to make things even across the replicas, certain gradient accumulated batches of length (num_individuals_per_gene) for certain genes may be ommitted instead.

        If use all_pppl is true then the same gene may appear on different GPUs, but only as different gradient accumulated effective batches
        """
        # Start and end index of groups for this replica
        start_group = self.rank * self.genes_per_replica
        end_group = start_group + self.genes_per_replica

        #indices that will only include genes (and people for those genes) that are meant to be run by one GPU. Ensures you only get one gene per batch for a GPU
        indices = []
        for group in range(start_group, end_group):
            group_start_idx = group * self.num_individuals_per_gene 
            indices.extend(range(group_start_idx, group_start_idx + self.num_individuals_per_gene )) #add next self.num_individuals_per_gene indices until you reach the end of the group

        assert len(indices) == self.num_samples_per_replica, f"Length of indices per GPU ({len(indices)}) is not the same as the number of Genes x number of people ({self.num_samples_per_replica}) per gene assigned to the GPU"
        return iter(indices)
        

    def __len__(self):
        return self.num_samples_per_replica


class CustomDataModule(LightningDataModule):
    def __init__(self, train_dataset, valid_dataset,test_dataset,train_batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        print(f"world size: {self.trainer.world_size}")
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=False,sampler = CustomDistributedSampler(self.train_dataset,shuffle=False))

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size= 1, shuffle=False, sampler = CustomDistributedSampler(self.valid_dataset,shuffle=False)),
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size= 1, shuffle=False, sampler = CustomDistributedSampler(self.test_dataset,shuffle=False)),
    

if __name__ == "__main__":
    pass

   
