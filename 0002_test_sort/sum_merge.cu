#define CUDA_CALL(x)                                                         \
    {                                                                        \
        const cudaError_t a = (x);                                           \
        if (a != cudaSuccess) {                                              \
            printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(a), \
                   a);                                                       \
            cudaDeviceReset();                                               \
            assert(0);                                                       \
        }                                                                    \
    }

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ int do_sum_merge(int *datas){
    
}

int main(){
    CUDA_CALL(cudaSetDevice(0));
    //init
    const int length=1024;
    int a[length], b[length];
    for (int i=0;i<length;++i){
        a[i]=i;
        b[i]=i*i;
    }
    int *datas=NULL;
    CUDA_CALL(cudaMalloc((void **)&datas, length * sizeof(int)));
    //cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count,cudaMemcpyKind kind )
    //cudaMemcpyHostToHost   cudaMemcpyHossToDevice   cudaMemcpyDeviceToHost   cudaMemcpuDeviceToDevice
    CUDA_CALL(cudaMemcpy(datas, a, length * sizeof(int), cudaMemcpyHossToDevice));


    return 0;
}

