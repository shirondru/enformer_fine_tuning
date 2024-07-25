Welcome to the Performer repository! This repository is meant to fine-tune sequence-to-expression models using paired whole genome sequencing (WGS) and gene expression data. We focus on [Enformer](https://www.nature.com/articles/s41592-021-01252-x), although the code used here can be adapted to other models.

Our preprint is out! Find it [here]()

NOTE: Not all necessary data is provided in this repository. This is because some of the data is protected (GTEx WGS and ROSMAP data) while other data is large and easily found online (hg38 reference genome). Thus, for this repository to be used as-is, protected data must first be retrieved. This repository can still be used as a starting point for those that do not have access to protected data, or are interested in using different datasets.

# Requirements
1. Clone the repository
2. Download [miniforge3](https://github.com/conda-forge/miniforge)
3. Install the requirements
```
eval “$(/PATH/TO/MINIFORGE3/bin/conda shell.bash hook)" #specify path to miniforge3 conda executable
conda env create -f environment.yml #install requirements
```
4. (Optional) update SLURM job scripts under `code/submission_scripts` to accommodate your requirements and directory locations. For example, the stderr/stdout locations must be changed to match your file structure, as does the activation of the conda environment. If you are not using a SLURM job scheduler, update the job scripts accordingly.
5. (Optional) Follow the [Weights & Biases quickstart](https://docs.wandb.ai/quickstart). W&B is used to track training experiments and metadata from past runs is parsed in subsequent experiments. For example, when performing in silico mutagenesis, information like sequence length, training tissue, and training genes is used. Examples of these metadata files can be found under `code/metadata_from_past_runs`. If you prefer not to use W&B, then (1) all `wandb` calls in python scripts must be removed, (2) `config` files must be restructured within python training scripts, and (3) metadata files should be generated to maintain compatibility.
6. Pre-process GTEx VCF files and generate consensus sequences from each individual's variant calls. You must [apply](https://gtexportal.org/home/protectedDataAccess) for access before doing this. Code is provided to help do this, using an SGE scheduler:
   
A) Download hg38 reference genome fasta file. Install [samtools](http://www.htslib.org/) if necessary.
```
wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > ./data/hg38_genome.fa
samtools faidx ./data/hg38_genome.fa
```
B) First pre-process the VCF files, converting them to bcf files and retrieving rows corresponding to SNPs. The VCF files should be located under ./data/VCF
```
DATA_DIR="$(realpath ./data)"
vcf_path=$DATA_DIR/VCF

sh code/process_vcf/prep_VCF_for_fasta_consensus_run_job.sh $vcf_path
```
C) Take processed BCF file and generate two consensus sequences for each individual with WGS & RNA-seq data -- one per haplotype.
```
bcf_in=$vcf_path/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz #absolute path to bcf file containing only SNPs
consensus_outdir=$DATA_DIR/ConsensusSeqs_SNPsOnlyUnphased #absolute path to directory where consensus fasta files will be saved
sh ./code/process_vcf/gen_fasta_consensus_run_jobs.sh $bcf_in $consensus_outdir $DATA_DIR
```
7. Download gene expression data
```
cd ./data/gtex_eqtl_expression_matrix
wget https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL_expression_matrices.tar
tar -xvf GTEx_Analysis_v8_eQTL_expression_matrices.tar

#bring data from tissues used in paper to expected location
mv GTEx_Analysis_v8_eQTL_expression_matrices/Brain_Cortex.v8.normalized_expression.bed.gz .
mv GTEx_Analysis_v8_eQTL_expression_matrices/Whole_Blood.v8.normalized_expression.bed.gz .
```

# Example Usage
To train single-gene and multi-gene Performer models on Whole Blood GTEx data using ~300 genes, run:
```
sh .code/submission_scripts/submit_gtex.sh
```
This will launch 6 SLURM jobs (3 single-gene & 3 multi-gene; training 3 replicates per model using different train/validation/test folds). Each job will be launched via `code/submission_scripts/slurm_train_gtex.sh`. Please update `slurm_train_gtex.sh` to match your requirements (see requirement #4 above). `slurm_train_gtex.sh` will launch `train_gtex.py` using configurations (e.g., learning rate, training tissue) from `code/configs/blood_config.yaml`. `train_gtex.py` will select the ~300 genes used in the paper and fine-tune Enformer’s pre-trained weights using paired WGS & RNA-seq.

To modify training hyperparameters, update or create a new config file. Config files are read within the training script (e.g., `train_gtex.py`) and different config parameters can be added within the script. ./code/configs/blood_config.yaml is shown below, see comments for more details on how things can be changed.
```
seq_length: 49152 
learning_rate: 5e-6
train_batch_size: 32
num_individuals_per_gene: 128 #gradient accumulation combines 4 effective batches of 32 (above) into 128
alpha: 0.5 # weight given to MSE within the loss function. The second term gets the complement of this
experiment_name: "FinalPaperWholeBlood"
max_epochs: 150
precision: bf16-mixed
tissues_to_train: "Whole Blood"
gradient_clip_val: 0.05
seed: 0
patience: 20 #if no improvement to R2 evaluated on training genes from validation donors, exit training
valid_metrics_save_freq: 1 #evaluate every epoch
```
If you were to make a new config file, you could modify submit_gtex.sh to use it instead, or pass it as an argument to the job script (if using a job scheduler) or into the python training script (e.g., `train_gtex.py`) otherwise. See `./code/submission_scripts/slurm_train_gtex.sh` for an example of a SLURM job script. 

To train or evaluate on different genes:
____

## TODO:
convert GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz -> GTEx_gene_tpm.csv
