# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000:

![](benchmark_results.png)

GFLOPs at matrix size 4092x4092:
<!-- benchmark_results -->

| Kernel              |   GFLOPs | Performance relative to cuBLAS   |
|:--------------------|---------:|:---------------------------------|
| 1: Naive            |    309.5 | 1.3%                             |
| 2: GMEM Coalescing  |   2006.3 | 8.2%                             |
| 3: SMEM Blocktiling |   2984   | 12.2%                            |
| 4: 1D Warptiling    |   8626.8 | 35.3%                            |
| 5: 2D Warptiling    |  16134.3 | 66.0%                            |
| 6: Vectorize        |  19130.3 | 78.3%                            |
| 0: cuBLAS           |  24441.4 | 100.0%                           |

<!-- benchmark_results -->

## Develop

1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt`:
    ```cmake
    set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES 86)
    ```
1. Install dependencies: CUDA toolkit, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. `mkdir build && cd build && cmake .. -GNinja && ninja`
1. `./sgemm <kernel number>`

For profiling, download [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute).