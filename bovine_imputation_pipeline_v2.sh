#!/bin/bash
############################################################
# Bovine SNP Imputation Pipeline v2
# INDUSCHIP3 -> IndiGau
# Beagle 5.0
#
# Includes:
# - SNP remapping
# - QC
# - Strand harmonization
# - Allele flip correction
# - Cross validation
# - Beagle imputation
# - INFO filtering
# - Accuracy estimation
############################################################

############################
# SOFTWARE PARAMETERS
############################
THREADS=16
PLINK=plink
BWA=bwa
BCFTOOLS=bcftools
BEAGLE=beagle.5.0.jar
GENOME=UMD3.1.fa


############################
# INPUT DATA
############################

INDU=INDUSCHIP3_raw
INDIGAU=IndiGau_raw

OUT=Imputation_Project_v2

mkdir -p $OUT/{01_mapping,02_QC,03_merge,04_validation,05_beagle,06_accuracy}


############################################################
# 1. SNP REMAPPING TO UMD3.1
############################################################
echo "Mapping SNP probes"
bwa index $GENOME
bwa mem $GENOME probe.fastq > $OUT/01_mapping/probe.sam


# retain uniquely mapped SNPs
samtools view -q 30 $OUT/01_mapping/probe.sam > $OUT/01_mapping/probe_unique.sam



# Generate:
# SNP_ID chromosome position
# from SAM output

awk '{ print $1,$3,$4 }' $OUT/01_mapping/probe_unique.sam> $OUT/01_mapping/update_map.txt



############################################################
# 2. UPDATE SNP POSITIONS
############################################################


for CHIP in $INDU $INDIGAU
do
plink \
--bfile $CHIP \
--update-map \
$OUT/01_mapping/update_map.txt \
--make-bed \
--out $OUT/01_mapping/${CHIP}_UMD3

done

############################################################
# 3. SAMPLE AND SNP QC
############################################################


for CHIP in INDUSCHIP3_raw_UMD3 IndiGau_raw_UMD3
do
plink \
--bfile $OUT/01_mapping/$CHIP \
--mind 0.05 \
--geno 0.05 \
--maf 0.05 \
--hwe 1e-6 \
--make-bed \
--out $OUT/02_QC/$CHIP

done

############################################################
# 4. REMOVE AMBIGUOUS SNPs
############################################################


echo "Removing strand ambiguous SNPs"
plink \
--bfile $OUT/02_QC/IndiGau_raw_UMD3 \
--exclude range ambiguous_ATCG.txt \
--make-bed \
--out $OUT/02_QC/IndiGau_clean

plink \
--bfile $OUT/02_QC/INDUSCHIP3_raw_UMD3 \
--exclude range ambiguous_ATCG.txt \
--make-bed \
--out $OUT/02_QC/INDUSCHIP3_clean

############################################################
# 5. ALLELE ALIGNMENT AND FLIP CORRECTION
############################################################


echo "Checking allele mismatch"
plink \
--bfile $OUT/02_QC/INDUSCHIP3_clean \
--bmerge \
$OUT/02_QC/IndiGau_clean.bed \
$OUT/02_QC/IndiGau_clean.bim \
$OUT/02_QC/IndiGau_clean.fam \
--merge-mode 6 \
> merge_check.log



# Extract strand errors

grep "Flip" merge_check.log > flip_snps.txt



# Correct strand
plink \
--bfile $OUT/02_QC/INDUSCHIP3_clean \
--flip flip_snps.txt \
--make-bed \
--out $OUT/03_merge/INDUSCHIP3_flip



############################################################
# 6. FINAL MERGE
############################################################

plink \
--bfile $OUT/03_merge/INDUSCHIP3_flip \
--bmerge \
$OUT/02_QC/IndiGau_clean.bed \
$OUT/02_QC/IndiGau_clean.bim \
$OUT/02_QC/IndiGau_clean.fam \
--make-bed \
--out $OUT/03_merge/reference_panel

############################################################
# 7. CROSS VALIDATION MASKING
############################################################


for FOLD in Fold1 Fold2 Fold3
do
mkdir $OUT/04_validation/$FOLD

# High density validation set
plink \
--bfile $OUT/03_merge/reference_panel \
--keep ${FOLD}_animals.txt \
--make-bed \
--out $OUT/04_validation/$FOLD/validation

# remove high density SNPs
# retain low density SNPs only


plink \
--bfile $OUT/04_validation/$FOLD/validation \
--extract INDUSCHIP3_marker_list.txt \
--make-bed \
--out $OUT/04_validation/$FOLD/test


# reference panel
plink \
--bfile $OUT/03_merge/reference_panel \
--remove ${FOLD}_animals.txt \
--make-bed \
--out $OUT/04_validation/$FOLD/reference

done



############################################################
# 8. Convert PLINK -> VCF
############################################################


for FOLD in Fold1 Fold2 Fold3
do
for TYPE in test reference validation
do
plink \
--bfile $OUT/04_validation/$FOLD/$TYPE \
--recode vcf bgz \
--out $OUT/04_validation/$FOLD/$TYPE
done
done

############################################################
# 9. BEAGLE PHASING
############################################################


for FOLD in Fold1 Fold2 Fold3
do
java \
-Xmx100g \
-jar $BEAGLE \
gt=$OUT/04_validation/$FOLD/reference.vcf.gz \
out=$OUT/05_beagle/$FOLD.phase \
nthreads=$THREADS

###########################################################
# 10. IMPUTATION
###########################################################

java \
-Xmx100g \
-jar $BEAGLE \
gt=$OUT/04_validation/$FOLD/test.vcf.gz \
ref=$OUT/05_beagle/$FOLD.phase.vcf.gz \
out=$OUT/05_beagle/$FOLD.imputed \
nthreads=$THREADS

done



############################################################
# 11. IMPUTATION QUALITY FILTER
############################################################

for FOLD in Fold1 Fold2 Fold3
do
bcftools query \
-f '%CHROM\t%POS\t%INFO/DR2\n' \
$OUT/05_beagle/$FOLD.imputed.vcf.gz \
> $OUT/05_beagle/$FOLD.INFO

# keep DR2 >=0.8
awk '$3>=0.8' \
$OUT/05_beagle/$FOLD.INFO \
> $OUT/05_beagle/$FOLD.high_quality
done



############################################################
# 12. GENOTYPE CONCORDANCE
############################################################


for FOLD in Fold1 Fold2 Fold3
do
bcftools gtcheck \
-g \
$OUT/04_validation/$FOLD/validation.vcf.gz \
$OUT/05_beagle/$FOLD.imputed.vcf.gz \
> $OUT/06_accuracy/${FOLD}_concordance.txt
done



############################################################
# 13. FINAL SUMMARY
############################################################
echo "======================================"
echo "Imputation Completed"

echo "Final results:"
echo $OUT/06_accuracy

echo "Quality filtered SNPs:"
echo $OUT/05_beagle

echo "============Pipeline Completed =========================="
