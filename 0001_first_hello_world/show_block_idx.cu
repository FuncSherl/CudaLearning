/*
 * @Author: xh
 * @Date: 2021-07-13 00:04:22
 */
#include <algorithm>
#include <iostream>
#include <vector>
#include "commontools.h"

using namespace std;

__global__ void show_blocks_idx() {
  printf("gridDim(x,y,z): (%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
  printf("blockIdx(x,y,z): (%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
  printf("blockDim(x,y,z): (%d,%d,%d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("threadIdx(x,y,z): (%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  return;
}

int main() {
  // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
  dim3 grid_dim(2);      //这里定义的dim就是grindDim的值
  dim3 block_dim(4, 2);  //这里定义的dim就是blockDim的值

  // 执行kernel
  show_blocks_idx<<<grid_dim, block_dim>>>();
  CUDA_LAST_ERROR();
  CUDA_CALL(cudaDeviceSynchronize());
  return 0;
}
