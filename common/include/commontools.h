#include <stdio.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 用宏变长参数来实现
#define CUDA_CALL(...)                                       \
  {                                                          \
    cudaError_t _cuda_tep_set_not_repeat_a = (__VA_ARGS__);  \
    if (_cuda_tep_set_not_repeat_a != cudaSuccess) {         \
      printf("\nCUDA ERROR: %s (err_num=%d)\n",              \
             cudaGetErrorString(_cuda_tep_set_not_repeat_a), \
             _cuda_tep_set_not_repeat_a);                    \
      cudaDeviceReset();                                     \
      assert(0);                                             \
    }                                                        \
  }

// last error
#define CUDA_LAST_ERROR() CUDA_CALL(cudaGetLastError())