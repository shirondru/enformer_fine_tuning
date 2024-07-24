# Submits multiple jobs running gen_fasta_consensus_job_script.sh in parallel, splitting up the task into multiple jobs to get it done faster

script_path="$(dirname "${BASH_SOURCE[0]}")"
echo script path $script_path
log_dir=$script_path/../../logs/gen_fasta_consensus

bcf_in=$1 #absolute path to BCF file containing WGS data from SNPs only generated in prep_VCF_for_fasta_consensus_job_script.sh
bcf_in_idx="${bcf_in}.csi"
outdir=$2 #where consensus sequences will be saved
job_name=$3 #SGE job name
job_script=$script_path/gen_fasta_consensus_job_script.sh
total_num_tasks=1676 #838 individuals x 2 haplotypes

#create outdir if it doesn't exist
if [ ! -d "$outdir" ]; then
  mkdir -p "$outdir"
fi

#create directory for stderr and stdout
if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi
mkdir $log_dir/stdout
mkdir $log_dir/stderr

# qsub -cwd -t 1-$total_num_tasks -tc 35 -N $job_name $job_script $bcf_in $bcf_in_idx $outdir
qsub -cwd -t 1-2 -tc 35 -N $job_name $job_script $bcf_in $bcf_in_idx $outdir