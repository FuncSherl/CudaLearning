#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "commontools.h"

using namespace std;

#define SHOW_MAT(a, m, n)                                           \
  {                                                                 \
    cout << "\nShowMat: " << #a << " -> " << m << "*" << n << endl; \
    for (int i = 0; i < m; ++i) {                                   \
      for (int j = 0; j < n; ++j) {                                 \
        cout << setw(6) << a[i * n + j];                            \
      }                                                             \
      cout << endl;                                                 \
    }                                                               \
  }

#define SET_MAT(a, d, m, n)                            \
  {                                                    \
    cout << "\nsetmat: " << #a << " -> " << d << endl; \
    for (int i = 0; i < m; ++i) {                      \
      for (int j = 0; j < n; ++j) {                    \
        a[i * n + j] = d;                              \
      }                                                \
    }                                                  \
  }

#define SET_MATAUTO(a, m, n)            \
  {                                     \
    cout << "\nsetmat: " << #a << endl; \
    for (int i = 0; i < m; ++i) {       \
      for (int j = 0; j < n; ++j) {     \
        a[i * n + j] = i * n + j;       \
      }                                 \
    }                                   \
  }

#define MDIV(a, b) ((int)a % (int)b == 0 ? (int)a / (int)b : (int)a / (int)b + 1)

template <int BSIZE>
__global__ void mattrans_v1(float *ta, float *a, int m, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int ori_ind = idx * n + idy;
  int new_ind = idy * m + idx;
  if (idx >= m || idy >= n) return;

  ta[new_ind] = a[ori_ind];
}

int main() {
  //获取设备属性
  // cudaDeviceProp prop;
  // int deviceID;
  // cudaGetDevice(&deviceID);
  // cudaGetDeviceProperties(&prop, deviceID);

  // //对于每个主机线程，每次只有一个 GPU 设备处于活动状态。
  // //如要将特定的 GPU 设置为活动状态，请使用 cudaSetDevice 以及所需 GPU 的索引

  // //检查设备是否支持重叠功能
  // //支持设备重叠功能的 GPU 能够在执行一个 CUDA 核函数的同时，还能在主机和设备之间执行复制数据操作
  // if (!prop.deviceOverlap) {
  //   printf("No device will handle overlaps. so no speed up from stream.\n");
  //   return 0;
  // }

  //启动计时器
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int m = 19;
  int n = 23;

  const int BSIZE = 5;

  float *C = new float[m * n];
  SET_MATAUTO(C, m, n);
  SHOW_MAT(C, m, n);

  float *gc, *transgc;
  // GPU端分配内存
  CUDA_CALL(cudaMalloc((void **)&gc, m * n * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&transgc, m * n * sizeof(float)));

  // CPU的数据拷贝到GPU端
  cudaMemcpy(gc, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(gc, C, size, cudaMemcpyHostToDevice);

  // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
  dim3 dimBlock(BSIZE, BSIZE);
  dim3 dimGrid(MDIV(m, BSIZE), MDIV(n, BSIZE));
  cout << "start block num: " << dimGrid.x << "*" << dimGrid.y << endl;
  cout << "each block:" << dimBlock.x << "*" << dimBlock.y << endl;
  // 执行kernel
  CUDA_CALL(cudaEventRecord(start));
  int iter = 10 * 200;
  for (int i = 0; i < iter; ++i) {
    mattrans_v1<BSIZE><<<dimGrid, dimBlock>>>(transgc, gc, m, n);
  }
  CUDA_LAST_ERROR();

  CUDA_CALL(cudaEventRecord(stop));

  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

  // cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
  cudaMemcpy(C, transgc, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  CUDA_CALL(cudaFree(transgc));
  CUDA_CALL(cudaFree(gc));

  SHOW_MAT(C, n, m);
  cout << "Iter:" << iter << " UsedTime: " << elapsedTime << " ms" << endl;
  delete[] C;
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));
  return 0;
}
