# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).
This repo is inspired by [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE).

## Overview

Running the kernels on a NVIDIA A6000:

![](benchmark_results.png)

GFLOPs at matrix size 4092x4092:
<!-- benchmark_results -->
| Kernel              |   GFLOPs | Performance relative to cuBLAS   |
|:--------------------|---------:|:---------------------------------|
| 1: Naive            |    307.2 | 1.3%                             |
| 2: GMEM Coalescing  |   1987.2 | 8.4%                             |
| 3: SMEM Blocktiling |   2981.3 | 12.6%                            |
| 4: 1D Warptiling    |   8508.3 | 36.0%                            |
| 5: 2D Warptiling    |  16319   | 69.0%                            |
| 6: Vectorize        |  19281.4 | 81.5%                            |
| 0: cuBLAS           |  23663.6 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt`:
    ```cmake
    set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES 86)
    ```
1. `mkdir build && cd build && cmake .. -GNinja && ninja`
1. `./sgemm <kernel number>`

For profiling, download [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute).
