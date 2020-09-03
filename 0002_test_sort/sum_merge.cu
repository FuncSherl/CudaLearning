#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

using namespace std;

// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }

//#define CUDA_CALL(...) cout<<__VA_ARGS__<<endl;

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
    CUDA_CALL(cudaMalloc((void **)&datas, length * sizeof(int)));
    //cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count,cudaMemcpyKind kind )
    //cudaMemcpyHostToHost   cudaMemcpyHossToDevice   cudaMemcpyDeviceToHost   cudaMemcpuDeviceToDevice
    CUDA_CALL( cudaMemcpy(datas,a,length*sizeof(int),cudaMemcpyHostToDevice) );

    do_sum_merge<<<1,length>>>(datas,length);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL( cudaThreadSynchronize());
    CUDA_CALL( cudaMemcpy(a,datas,length*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CALL( cudaFree(datas));

    for (int i=0;i<length;++i) cout<<a[i]<<" ";

    return 0;
}

