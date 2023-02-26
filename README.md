# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A100 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4092x4092:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      292.1 | 1.5%                             |
| 2: GMEM Coalescing                  |     3119.0 | 16.4%                            |
| 3: SMEM Caching                     |     5451.9 | 28.6%                            |
| 4: 1D Blocktiling                   |    10368.6 | 54.5%                            |
| 5: 2D Blocktiling                   |    14155.1 | 74.4%                            |
| 8: Avoid Bank Conflicts (Offset)    |    15086.2 | 79.3%                            |
| 7: Avoid Bank Conflicts (Linearize) |    15171.3 | 79.7%                            |
| 6: Vectorized Mem Access            |    15296.5 | 80.4%                            |
| 9: Autotuning                       |    16344.1 | 85.9%                            |
| 10: Warptiling                      |    17806.7 | 93.6%                            |
| 0: cuBLAS                           |    19029.5 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `make`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.