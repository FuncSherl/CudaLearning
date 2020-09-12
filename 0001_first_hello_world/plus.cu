/*
 * @Author: xh
 * @Date: 2020-05-30 15:04:22
 * @LastEditTime: 2020-09-12 14:10:23
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /workspaces/cuda/cuda_start/plus.cu
 */ 
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using namespace std;

// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }


__global__ void myplus(float a[], float b[], float c[], int n){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i]=a[i]+b[i];
  if (i%1000==0)printf ("gpu print:%f\n",c[i]);
}

int main(){
  int n=1024*1024;
  int size=n*sizeof(float);
  float *A=new float[size]{1,1};
  float *B=new float[size]{2,2};
  float *C=new float[size]{3,3};

  float *ga, *gb, *gc;
  // GPU端分配内存
  cudaMalloc((void**)&ga, size);
  cudaMalloc((void**)&gb, size);
  cudaMalloc((void**)&gc, size);

  // CPU的数据拷贝到GPU端
  cudaMemcpy(ga, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(gb, B, size, cudaMemcpyHostToDevice);
  //cudaMemcpy(gc, C, size, cudaMemcpyHostToDevice);

  // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
  dim3 dimBlock(512);
  dim3 dimGrid(n/512);

  // 执行kernel
  myplus<<<dimGrid, dimBlock>>>(ga, gb, gc, n);

  //cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
  cudaMemcpy(C, gc, size, cudaMemcpyDeviceToHost);

  cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);

  cout<<A[0]<<"+"<<B[0]<<"="<<C[0]<<endl;
  cout<<A[2]<<"+"<<B[2]<<"="<<C[2]<<endl;
  delete []A;
  delete []B;
  delete []C;

  return 0;
}
