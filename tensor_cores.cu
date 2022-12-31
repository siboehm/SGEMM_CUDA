#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mma.h>
#include <vector>

using namespace nvcuda;

const uint DIM = 16;
const std::string logFile = "TensorCoreMMMResult.txt";

// A single warp performing a 16x16x16 matmul at half (=16bit) precision
__global__ void tensorMatMul(half *A, half *B, half *C) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0);

  wmma::load_matrix_sync(a_frag, A, 16);
  wmma::load_matrix_sync(b_frag, B, 16);

  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

  wmma::store_matrix_sync(C, acc_frag, 16, wmma::mem_row_major);
}

void initMatrixRange(half *A, uint size) {
  for (int i = 0; i < size; ++i) {
    A[i] = i / 100.0f;
  }
}

template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << (float)A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << (float)A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

int main(int argc, char **argv) {
  half *A, *B, *C;
  half *A_d, *B_d, *C_d;

  A = (half *)malloc(DIM * DIM * sizeof(half));
  B = (half *)malloc(DIM * DIM * sizeof(half));
  C = (half *)malloc(DIM * DIM * sizeof(half));
  initMatrixRange(A, DIM * DIM);
  initMatrixRange(B, DIM * DIM);

  cudaMalloc((void **)&A_d, DIM * DIM * sizeof(half));
  cudaMalloc((void **)&B_d, DIM * DIM * sizeof(half));
  cudaMalloc((void **)&C_d, DIM * DIM * sizeof(half));

  cudaMemcpy(A_d, A, DIM * DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, DIM * DIM * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, DIM * DIM * sizeof(half), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  dim3 grid_size(1, 1, 1);
  dim3 block_size(32);

  tensorMatMul<<<grid_size, block_size>>>(A_d, B_d, C_d);

  cudaDeviceSynchronize();

  cudaMemcpy(C, C_d, DIM * DIM * sizeof(half), cudaMemcpyDeviceToHost);

  std::ofstream fs;
  fs.open(logFile);
  print_matrix<half>((half *)C, DIM, DIM, fs);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  free(A);
  free(B);
  free(C);
}
