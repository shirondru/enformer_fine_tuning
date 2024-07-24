vcf_path=$1 #where WGS VCF files are located
script_dir="$(dirname "${BASH_SOURCE[0]}")"
log_dir=$script_dir/../../logs/gen_fasta_consensus

job_script=$script_dir/prep_VCF_for_fasta_consensus_job_script.sh
job_name=CreateSNPOnlyVCFBCF
log_dir=$log_dir/gen_fasta_consensus

#create directory for stderr and stdout

if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi
mkdir $log_dir/stdout
mkdir $log_dir/stderr


qsub -cwd  -N $job_name $job_script "$vcf_path"
