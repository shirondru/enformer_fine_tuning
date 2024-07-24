#!/bin/bash
#$ -o /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/gen_fasta_consensus/stdout/$JOB_ID.o
#$ -e /pollard/data/projects/sdrusinsky/enformer_fine_tuning/logs/gen_fasta_consensus/stderr/$JOB_ID.e
#$ -r y                                                        
#$ -l mem_free=50G                
#$ -l scratch=1000G                  
#$ -l h_rt=96:00:00  



#specify scartch if it isn't defined
if [[ -z "$TMPDIR" ]]; then
  if [[ -d /scratch ]]; then TMPDIR=/scratch/$USER; else TMPDIR=/tmp/$USER; fi
  mkdir -p "$TMPDIR"
  export TMPDIR
fi
echo TMPDIR PATH
echo $TMPDIR
cd $TMPDIR



bcf_in=$1
bcf_in_idx=$2
outdir=$3
DATA_DIR=$4
ref_fasta=$DATA_DIR/hg38_genome.fa
duplicated_samples_file_path=$DATA_DIR/duplicated_samples_list.txt #each sample appears twice in this file. So this file can be iterated over to generate two consensus sequences, one per haplotype
echo outdir $outdir
echo $bcf_in_idx

## Create outdir if it doesn't exist
if [ ! -d "$outdir" ]; then
  mkdir -p "$outdir"
fi

## 2. Copy input files from global disk to local scratch if they are not already there
### TO DO: Does copying vcf to scratch take more time then i save by avoiding i/o from original file? Can possibly be optimized
if [ ! -f "$TMPDIR/bcf_in.bcf.gz" ]; then
   echo copying bcf_in
  cp $bcf_in "$TMPDIR/bcf_in.bcf.gz"
fi

if [ ! -f "$TMPDIR/bcf_in.bcf.gz.csi" ]; then
  echo copying bcf_in_idx
  cp $bcf_in_idx "$TMPDIR/bcf_in.bcf.gz.csi"
fi

if [ ! -f "$TMPDIR/ref_fasta.fa" ]; then
  echo copying fasta
  cp $ref_fasta "$TMPDIR/ref_fasta.fa"
fi


echo . DIR 
echo | ls -lhtr .
echo SGE_TASK_ID $SGE_TASK_ID
echo TMPDIR
echo $TMPDIR
echo| ls -lhtr $TMPDIR

sample=$(sed -n "${SGE_TASK_ID}p" $duplicated_samples_file_path) #Get sample ID corresponding to this array job task
echo sample $sample


#if task ID is even, gen fasta using haplotype =1, else use haplotype = 2. This ensures each 
if [ `expr  $SGE_TASK_ID % 2` == 0 ]; then
        haplotype=1
fi

if [ `expr  $SGE_TASK_ID % 2` != 0 ]; then
        haplotype=2
fi
echo haplotype $haplotype
## 3. Generate consensus sequence with one haplotype from the VCF.
bcftools consensus --fasta-ref=$TMPDIR/ref_fasta.fa --haplotype=$haplotype --samples=$sample $TMPDIR/bcf_in.bcf.gz -o $TMPDIR/${sample}_consensus_H${haplotype}.fa 
echo "bcftools consensus exit status: $?"
echo result | ls -lhtr . 
## 4. Move output files back to global disk
mv $TMPDIR/${sample}_consensus_H${haplotype}.fa $outdir


qstat -j $JOB_ID
