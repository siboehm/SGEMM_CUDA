#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void kernel(uint *A, uint *B, int row) {
  auto x = threadIdx.x / 4;
  auto y = threadIdx.x % 4;
  A[x * row + y] = x;
  B[x * row + y] = y;
}

int main(int argc, char **argv) {
  uint *Xs, *Ys;
  uint *Xs_d, *Ys_d;

  uint SIZE = 4;

  Xs = (uint *)malloc(SIZE * SIZE * sizeof(uint));
  Ys = (uint *)malloc(SIZE * SIZE * sizeof(uint));

  cudaMalloc((void **)&Xs_d, SIZE * SIZE * sizeof(uint));
  cudaMalloc((void **)&Ys_d, SIZE * SIZE * sizeof(uint));

  dim3 grid_size(1, 1, 1);
  dim3 block_size(4 * 4);

  kernel<<<grid_size, block_size>>>(Xs_d, Ys_d, 4);

  cudaMemcpy(Xs, Xs_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ys, Ys_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      std::cout << "[" << Xs[row * SIZE + col] << "|" << Ys[row * SIZE + col]
                << "] ";
    }
    std::cout << "\n";
  }

  cudaFree(Xs_d);
  cudaFree(Ys_d);
  free(Xs);
  free(Ys);
}
