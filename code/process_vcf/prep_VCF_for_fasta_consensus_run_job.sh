DATA_DIR=$1 #path to directory containing data; e.g., Whole Genome Sequencing VCF Files
outdir=$2 #location where BCF/VCF files containing only SNPs will be saved
log_dir=$3 #where stderr/stdout will go

job_script=./prep_VCF_for_fasta_consensus_job_script.sh
job_name=CreateSNPOnlyVCFBCF
log_dir=$log_dir/gen_fasta_consensus

#create directory for stderr and stdout

if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi
mkdir $log_dir/stdout
mkdir $log_dir/stderr


qsub -cwd  -N $job_name $job_script "$DATA_DIR" "$outdir"
