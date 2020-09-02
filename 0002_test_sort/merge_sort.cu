#include <stdio.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CALL(x) {const cudaError_t a=(x);if (a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);} }

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
    cudaMalloc((void **)&datas, length * sizeof(int));



    return 0;
}

