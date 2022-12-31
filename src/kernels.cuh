#pragma once

#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_warptiling.cuh"
#include "kernels/5_kernel_2D_warptiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/7_kernel_tensorcores.cuh"
#include "kernels/kernel_3.cuh"
#include "kernels/kernel_4.cuh"
#include "kernels/kernel_5.cuh"
#include "kernels/kernel_6.cuh"
#include "kernels/kernel_7.cuh"