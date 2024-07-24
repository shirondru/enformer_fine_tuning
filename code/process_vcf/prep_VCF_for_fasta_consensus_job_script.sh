#!/bin/bash
#$ -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/gen_fasta_consensus/stdout/$JOB_ID.o
#$ -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/gen_fasta_consensus/stderr/$JOB_ID.e
#$ -r y                                                        
#$ -l mem_free=50G                
#$ -l scratch=1000G                  
#$ -l h_rt=168:00:00  


vcf_dir=$1

if [[ -z "$TMPDIR" ]]; then
  if [[ -d /scratch ]]; then TMPDIR=/scratch/$USER; else TMPDIR=/tmp/$USER; fi
  mkdir -p "$TMPDIR"
  export TMPDIR
fi
echo TMPDIR PATH
echo $TMPDIR
cd $TMPDIR




#copy vcf to local scratch
cp $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz


### move the large unphased bcf file to scratch. If it doesn't exist already, create it and initially store it on local scratch 
if [ ! -f "$vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz" ]; then
	outfile=$TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz
	infile=$TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz
	echo creating large unphased bcf on TMPDIR
	bcftools view --output-type b --output $outfile $infile
	echo done creating large unphased bcf on TMPDIR
	bcftools index $outfile
	echo done indexing unphased bcf on TMPDIR
	echo moving unphased bcf and index file to global dir
	mv $outfile $vcf_dir
	mv $outfile.csi $vcf_dir
	echo done moving unphased bcf and index file to global dir
else
	cp $vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz
	echo done copying large unphased bcf from global dir to TMPDIR

fi

# ### move the phased bcf file to scratch. If it doesn't exist already, create it and initially store ot on local scratch . Then convert it into a bcf and move it back to global dir
# if [ ! -f "$vcf_dir/phASER_WASP_GTEx_v8_merged.bcf.gz" ]; then
# 	cp $vcf_dir/phASER_WASP_GTEx_v8_merged.vcf.gz $TMPDIR/phASER_WASP_GTEx_v8_merged.vcf.gz
# 	outfile=$TMPDIR/phASER_WASP_GTEx_v8_merged.bcf.gz
# 	infile=$TMPDIR/phASER_WASP_GTEx_v8_merged.vcf.gz
# 	echo creating phased bcf on TMPDIR
# 	bcftools view --output-type b --output $outfile $infile
# 	echo done creating phased bcf on TMPDIR
# 	echo indexing phased bcf on TMPDIR
# 	bcftools index $outfile
# 	echo done indexing phased bcf 
# 	echo moving phased bcf and index file to global dir
# 	mv $outfile $vcf_dir
# 	mv $outfile.csi $vcf_dir
# 	echo done moving phased bcf and index file to global dir
# fi

echo done creating bcf files on local scratch
echo| ls -lhtr

#subset SNPs and index new files from unphased BCF
bcftools view --types snps --output-type b --output $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz
echo done subsetting SNPs from unphased bcf
##index these files
bcftools index $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz
echo done indexing subsetted bcf
#move them to global dir

mv $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz $vcf_dir
mv $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.bcf.gz.csi $vcf_dir



##repeat for VCFs
bcftools view --types snps --output-type z --output $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz
echo done subsetting SNPs from unphased vcf
bcftools index $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz
echo done indexing subsetted vcf
mv $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz $vcf_dir
mv $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz.csi $vcf_dir

#move unphased bcf to global dir if it wasn't already there
if [ ! -f "$vcf_dir/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz" ]; then
	mv $TMPDIR/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.bcf.gz $vcf_dir
fi
