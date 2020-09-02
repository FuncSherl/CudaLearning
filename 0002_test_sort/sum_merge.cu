#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

using namespace std;

#define CUDA_CALL(x) {const cudaError_t a=(x);if (a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);} }

//kernel function must return void
__global__ void do_sum_merge(int *datas, int n){
    int tid=blockDim.x*threadIdx.y+threadIdx.x;
    //int idx=blockIdx.x*blockDim.x+threadIdx.x;
    //int idy=blockIdx.y*blockDim.y+threadIdx.y;
    //int bid=gridDim.x*blockDim.x*idy+idx;
    while(n>1){
        if (tid< (1+n)/2 && n-1-tid!=tid){
            datas[tid]+=datas[n-1-tid];
            printf ("%d->%d->%d\n",n,tid,datas[tid]);             
        }
        n/=2;       
        __syncthreads();
    }
}

void cuda_call(cudaError_t a){
    if (a!=cudaSuccess){
        printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(a), a); 
        cudaDeviceReset(); 
        assert(0);
    }
}

int main(){
    CUDA_CALL(cudaSetDevice(0));
    //init
    const int length=1024;
    int a[length], b[length];
    for (int i=0;i<length;++i){
        a[i]=1;
        b[i]=i;
    }
    int *datas=NULL;
    cudaMalloc((void **)&datas, length * sizeof(int));
    //cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count,cudaMemcpyKind kind )
    //cudaMemcpyHostToHost   cudaMemcpyHossToDevice   cudaMemcpyDeviceToHost   cudaMemcpuDeviceToDevice
    cuda_call( cudaMemcpy(datas,a,length*sizeof(int),cudaMemcpyHostToDevice) );

    do_sum_merge<<<1,length>>>(datas,length);
    cuda_call(cudaGetLastError());

    cuda_call( cudaThreadSynchronize());
    cuda_call( cudaMemcpy(a,datas,length*sizeof(int),cudaMemcpyDeviceToHost));
    cudaFree(datas);

    for (int i=0;i<length;++i) cout<<a[i]<<" ";

    return 0;
}

