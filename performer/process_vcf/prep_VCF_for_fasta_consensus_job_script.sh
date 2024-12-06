#!/bin/bash
#$ -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/gen_fasta_consensus/stdout/$JOB_ID.o
#$ -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/gen_fasta_consensus/stderr/$JOB_ID.e
#$ -r y                                                        
#$ -l mem_free=50G                
#$ -l scratch=1000G                  
#$ -l h_rt=168:00:00  

source ~/.bashrc

vcf_dir=$1




### create bcf from vcf, if it doesn't exist already
if [ ! -f "$vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz" ]; then
	outfile=$vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz
	infile=$vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz
	echo creating large unphased bcf 
	bcftools view --output-type b --output $outfile $infile
	echo done creating large unphased bcf
	bcftools index $outfile
	echo done indexing unphased bcf
fi

# same thing for phased vcf
# if [ ! -f "$vcf_dir/phASER_WASP_GTEx_v8_merged.bcf.gz" ]; then
# 	cp $vcf_dir/phASER_WASP_GTEx_v8_merged.vcf.gz $vcf_dir/phASER_WASP_GTEx_v8_merged.vcf.gz
# 	outfile=$vcf_dir/phASER_WASP_GTEx_v8_merged.bcf.gz
# 	infile=$vcf_dir/phASER_WASP_GTEx_v8_merged.vcf.gz
# 	echo creating phased bcf
# 	bcftools view --output-type b --output $outfile $infile
# 	echo done creating phased bcf
# 	echo indexing phased bcf 
# 	bcftools index $outfile
# fi


#subset SNPs and index new files from unphased BCF
bcftools view --types snps --output-type b --output $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz
echo done subsetting SNPs from unphased bcf
##index these files
bcftools index $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz
echo done indexing subsetted bcf


##repeat for VCFs
bcftools view --types snps --output-type z --output $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz
echo done subsetting SNPs from unphased vcf
bcftools index $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz
echo done indexing subsetted vcf
