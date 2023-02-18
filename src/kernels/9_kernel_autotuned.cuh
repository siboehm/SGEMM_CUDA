#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)
const int NUM_THREADS = 256;

template <const int BM, const int BN, const int BK, const int TBM,
          const int TBN, const int WM, const int WN, const int TM, const int TN>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  const uint kRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // iterations of threadblock tile
  constexpr int TBMITER = CEIL_DIV(BM, TBM);
  constexpr int TBNITER = CEIL_DIV(BN, TBN);

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / warpSize;
  const uint warpCol = warpIdx % (TBN / WN);
  const uint warpRow = warpIdx / (TBN / WN);

  // Placement of the thread in the warp tile
  const uint threadIdxInWarp = threadIdx.x % warpSize;
  const uint threadColInWarp = threadIdxInWarp % (WN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TBMITER * TM * TBNITER * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    for (uint tbmIdx = 0; tbmIdx < TBMITER; ++tbmIdx) {
      for (uint tbnIdx = 0; tbnIdx < TBNITER; ++tbnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (tbmIdx * TBM) + warpRow * WM +
                         threadRowInWarp * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (tbnIdx * TBN) + warpCol * WN +
                         threadColInWarp * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(tbmIdx * TM + resIdxM) * (TBNITER * TN) +
                            tbnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint tbmIdx = 0; tbmIdx < TBMITER; ++tbmIdx) {
    for (uint tbnIdx = 0; tbnIdx < TBNITER; ++tbnIdx) {
      float *C_interim = C + (tbmIdx * TBM * N) + (tbnIdx * TBN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = __ldcs(reinterpret_cast<float4 *>(
              &C_interim[(warpRow * WM + threadRowInWarp * TM + resIdxM) * N +
                         warpCol * WN + threadColInWarp * TN + resIdxN])[0]);
          // perform GEMM update in reg
          const int i =
              (tbmIdx * TM + resIdxM) * (TBNITER * TN) + tbnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(warpRow * WM + threadRowInWarp * TM + resIdxM) * N +
                         warpCol * WM + threadColInWarp * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}