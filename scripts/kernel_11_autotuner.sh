#!/usr/bin/env bash

set -u 

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
WM_VALUES=(32 64 128 256)
WN_VALUES=(32 64 128 256)
WNITER_VALUES=(1 2 4 8)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
NUM_THREADS_VALUES=(128 256)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
OUTPUT="../benchmark_results/kernel_11_autotune_results.txt"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"
WARPSIZE=32


TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#WNITER_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for BK in "${BK_VALUES[@]}"; do
for BM in "${BM_VALUES[@]}"; do
for BN in "${BN_VALUES[@]}"; do
for WM in "${WM_VALUES[@]}"; do
for WN in "${WN_VALUES[@]}"; do
for WN_ITER in "${WNITER_VALUES[@]}"; do
for TM in "${TM_VALUES[@]}"; do
for TN in "${TN_VALUES[@]}"; do
for NUM_THREADS in "${NUM_THREADS_VALUES[@]}"; do
echo ""
CONFIG_NUM=$(( CONFIG_NUM + 1 ))
# skip configurations that don't fullfil preconditions
NUM_WARPS=$(( NUM_THREADS / 32 ))
if ! (( BN % WN == 0 && BM % WM == 0 )); then
  echo "Error: BN % WN must be 0 and BM % WM must be 0."
  continue
fi
if ! (( (BN / WN) * (BM / WM) == NUM_WARPS )); then
  echo "Error: (BN / WN) * (BM / WM) must be equal to NUM_WARPS."
  continue
fi
if ! (( (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0 )); then
  echo "Error: (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) must be 0."
  continue
fi
WM_ITER=$(( (WM * WN) / (WARPSIZE * TM * TN * WN_ITER) ))
if ! (( WM % WM_ITER == 0 && WN % WN_ITER == 0 )); then
  echo "Error: WM % WM_ITER must be 0 and WN % WN_ITER must be 0."
  continue
fi
if ! (( (NUM_THREADS * 4) % BK == 0 )); then
  echo "Error: (NUM_THREADS * 4) % BK must be 0."
  continue
fi
if ! (( (NUM_THREADS * 4) % BN == 0 )); then
  echo "Error: (NUM_THREADS * 4) % BN must be 0."
  continue
fi
if ! (( BN % (16 * TN) == 0 )); then
  echo "Error: BN must be a multiple of 16 * TN."
  continue
fi
if ! (( BM % (16 * TM) == 0 )); then
  echo "Error: BM must be a multiple of 16 * TM."
  continue
fi
if ! (( (BM * BK) % (4 * NUM_THREADS) == 0 )); then
  echo "Error: (BM * BK) % (4 * NUM_THREADS) must be 0."
  continue
fi
if ! (( (BN * BK) % (4 * NUM_THREADS) == 0 )); then
  echo "Error: (BN * BK) % (4 * NUM_THREADS) must be 0."
  continue
fi

# Update the parameters in the source code
sed -i "s/const uint K11_NUM_THREADS = .*/const uint K11_NUM_THREADS = $NUM_THREADS;/" $RUNNER
sed -i "s/const uint K11_BN = .*/const uint K11_BN = $BN;/" $RUNNER
sed -i "s/const uint K11_BM = .*/const uint K11_BM = $BM;/" $RUNNER
sed -i "s/const uint K11_BK = .*/const uint K11_BK = $BK;/" $RUNNER
sed -i "s/const uint K11_WM = .*/const uint K11_WM = $WM;/" $RUNNER
sed -i "s/const uint K11_WN = .*/const uint K11_WN = $WN;/" $RUNNER
sed -i "s/const uint K11_WNITER = .*/const uint K11_WNITER = $WN_ITER;/" $RUNNER
sed -i "s/const uint K11_TM = .*/const uint K11_TM = $TM;/" $RUNNER
sed -i "s/const uint K11_TN = .*/const uint K11_TN = $TN;/" $RUNNER

# Rebuild the program
make

echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$BK BM=$BM BN=$BN WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN NUM_THREADS=$NUM_THREADS" |& tee -a $OUTPUT
# Run the benchmark and get the result
# Kill the program after 8 seconds if it doesn't finish
timeout -v 8 ./sgemm 11 | tee -a $OUTPUT
done
done
done
done
done
done
done
done
done