#include <iostream>
#include "commontools.h"

using namespace std;

// cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b) {
  int i = threadIdx.x;
  if (i<5) c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
bool addWithCuda(int *c, const int *a, const int *b, size_t size) {
  int *dev_a = 0;
  int *dev_b = 0;
  int *dev_c = 0;

  // Choose which GPU to run on, change this on a multi-GPU system.
  CUDA_CALL(cudaSetDevice(0));

  // Allocate GPU buffers for three vectors (two input, one output) .
  CUDA_CALL(cudaMalloc((void **)&dev_c, size * sizeof(int)));

  CUDA_CALL(cudaMalloc((void **)&dev_a, size * sizeof(int)));

  CUDA_CALL(cudaMalloc((void **)&dev_b, size * sizeof(int)));

  // Copy input vectors from host memory to GPU buffers.
  CUDA_CALL(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

  // Launch a kernel on the GPU with one thread for each element.
  addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

  // cudaThreadSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  // cudaThreadSynchronize()在cuda10.0以后被弃用了，可以用 cudaDeviceSynchronize() 来代替
  CUDA_CALL(cudaDeviceSynchronize());
  // Copy output vector from GPU buffer to host memory.
  CUDA_CALL(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

Error:
  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);

  return true;
}

int main() {
  const int arraySize = 1<<20;
  const int a[5] = {1, 2, 3, 4, 5};
  const int b[5] = {10, 20, 30, 40, 50};
  int c[5] = {0};
  // Add vectors in parallel.
  addWithCuda(c, a, b, arraySize);

  cout << "length: " << arraySize << endl;
  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

  // cudaThreadExit must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete
  // traces.
  // Note that this function is deprecated because its name does not reflect its behavior.
  // Its functionality is identical to the non-deprecated function cudaDeviceReset(), which should be used instead.
  // cudaStatus = cudaThreadExit();
  CUDA_CALL(cudaDeviceReset());

  return 0;
}
