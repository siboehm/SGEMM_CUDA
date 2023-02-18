# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).
This repo is inspired by [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE).

## Overview

Running the kernels on a NVIDIA A100 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4092x4092:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      292   | 1.7%                             |
| 2: GMEM Coalescing                  |     3115.7 | 17.8%                            |
| 3: SMEM Caching                     |     5448.6 | 31.1%                            |
| 4: 1D Warptiling                    |    10345.5 | 59.0%                            |
| 5: 2D Warptiling                    |    14126.6 | 80.6%                            |
| 8: Avoid Bank Conflicts (Offset)    |    15056.9 | 85.9%                            |
| 7: Avoid Bank Conflicts (Linearize) |    15157.5 | 86.5%                            |
| 6: Vectorized Mem Access            |    15334.9 | 87.5%                            |
| 9: Autotuning                       |    15664.8 | 89.4%                            |
| 0: cuBLAS                           |    17521.2 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. `mkdir build && cd build && cmake .. -GNinja && ninja`
1. `DEVICE=<device_id> ./sgemm <kernel number>`

For profiling, download [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute).
