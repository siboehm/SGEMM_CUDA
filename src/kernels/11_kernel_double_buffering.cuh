#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace db {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
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
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmDoubleBuffering(const int M, const int N, const int K,
                         const float alpha, float *A, float *B, float beta,
                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // setup double buffering split
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // for the loading, we're pretending like there's half as many threads
  // as there actually are
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // load first (B0)
    db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
  }
  __syncthreads();

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // process current (B0)
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // process current+1 (B1)
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
      __syncthreads();

      // load current + 2 (B0)
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
            innerRowB, innerColB);
      }
    } else {
      // load current + 1 (B1)
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
            innerColA, innerRowB, innerColB);
      }
      __syncthreads();

      // process current (B0)
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // process current+1 (B1)
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
    }

    A += 2 * BK;     // move BK columns to right
    B += 2 * BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}