#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
OUTPUT="../scripts/kernel_9_autotune_results.txt"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="2"

TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          echo ""
          CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

          # skip configurations that don't fullfil preconditions
          config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn"
          if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
            echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
            continue
          fi
          if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
            echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
            continue
          fi
          if [[ $(( ($bm * $bk) % ( 4 * 256 ) )) -ne 0 ]]; then
            echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
            continue
          fi
          if [[ $(( ($bn * $bk) % ( 4 * 256 ) )) -ne 0 ]]; then
            echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
            continue
          fi

          # Update the parameters in the source code
          sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
          sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
          sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
          sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
          sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
          
          # Rebuild the program
          ninja 

          echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn" |& tee -a $OUTPUT
          # Run the benchmark and get the result
          # Kill the program after 4 seconds if it doesn't finish
          timeout -v 4 ./sgemm 9 | tee -a $OUTPUT
        done
      done
    done
  done
done