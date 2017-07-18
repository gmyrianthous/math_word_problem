#!/bin/bash
###############################################################################
# run evaluation for fold files created from baseline
# save results for folds to file and average results to another file
###############################################################################

# filename prefixes
pred_prefix="baseline.pred.fold"
gold_prefix="baseline.gold.fold"

# Script names
DIMSUMEVAL_SCRIPT="$HOME/data/DIMSUM/dimsum-data/scripts/dimsumeval.py"
AVG_FOLDS_SCRIPT="./avg_folds.py"

# baseline filename suffixe
suffix=".csv"

# get the amount of baselines predicted in this folder
# the quantity of CSVs prefixed with b._
baselines_quantity=$(ls -1 *csv | grep -o ^b._ | sort | uniq | wc -l)

errs_file="errs.txt"

###############################################################################
# for quantity of baselines, run evaluation averaging
###############################################################################
for baseline_num in $(seq 1 $baselines_quantity);
do
    # get the quantity of k-folds for this baseline
    pat="b""$baseline_num""_"
    K=$(ls -1 "$pat"*gold*csv | wc -l)

    # prefix for baselines files
    baseline_prefix="b""$baseline_num""_"

    # output file evaluation runs for this baseline
    outfile="$baseline_prefix""fold_evals.txt"
    # averaged results output file for this baseline
    avgoutfile="$outfile.avg.txt"

    # for each k-fold, run evaluation and append to output file for this baseline
    echo "Running evaluations for $K folds to file $outfile"
    for k in $(seq 1 $K)
    do
	# run evaluation without colour information 
	"$DIMSUMEVAL_SCRIPT" -C "$baseline_prefix$gold_prefix$k$suffix" "$baseline_prefix$pred_prefix$k$suffix" 2>> $errs_file
    done > $outfile

    # print average of folded results to averaged output file for this baseline
    echo "Average results for $K folds" | tee $avgoutfile
    $AVG_FOLDS_SCRIPT $outfile $K 2>&1 | tee -a $avgoutfile
done
