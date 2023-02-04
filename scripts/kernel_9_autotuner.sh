#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES="8 16 32 64"
TM_VALUES="4 8 16 32"
TN_VALUES="4 8 16 32"
BM_VALUES="64 128 256"
BN_VALUES="64 128 256"

# Keep track of best combination and its result
best_result=0
best_params=""

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
OUTPUT="../scripts/kernel_9_autotune_results.txt"

# Set GPU to use
export DEVICE="2"

# Loop through all combinations of parameters
for bk in $BK_VALUES; do
  for tm in $TM_VALUES; do
    for tn in $TN_VALUES; do
      for bm in $BM_VALUES; do
        for bn in $BN_VALUES; do
          # Update the parameters in the source code
          sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
          sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
          sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
          sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
          sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
          
          # Rebuild the program
          ninja 

          echo "BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn" | tee -a $OUTPUT
          # Run the benchmark and get the result
          # Kill the program after 4 seconds if it doesn't finish
          timeout -v 4 ./sgemm 9 | tee -a $OUTPUT
        done
      done
    done
  done
done