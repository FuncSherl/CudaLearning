/*
 * @Author: xh
 * @Date: 2021-07-13 00:04:22
 */ 
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using namespace std;

// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }
#define CUDA_LAST_ERROR CUDA_CALL(cudaGetLastError());

__global__ void show_blocks_idx(){
  printf ("gridDim(x,y,z): (%d,%d,%d)\n",gridDim.x, gridDim.y, gridDim.z);
  printf ("blockIdx(x,y,z): (%d,%d,%d)\n",blockIdx.x, blockIdx.y, blockIdx.z);
  printf ("blockDim(x,y,z): (%d,%d,%d)\n",blockDim.x, blockDim.y, blockDim.z);  
  printf ("threadIdx(x,y,z): (%d,%d,%d)\n",threadIdx.x, threadIdx.y, threadIdx.z);
  return;
}

int main(){
  // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
  dim3 grid_dim(2); //这里定义的dim就是grindDim的值
  dim3 block_dim(4,2);  //这里定义的dim就是blockDim的值

  // 执行kernel
  show_blocks_idx<<<grid_dim, block_dim>>>();
  CUDA_LAST_ERROR
  CUDA_CALL(cudaDeviceSynchronize());
  return 0;
}
