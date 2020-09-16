#include <stdio.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <algorithm>
#include <random>

using namespace std;

// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }

__global__ void merge_sort(int *datas, int n){
    int tid=blockDim.x*threadIdx.y+threadIdx.x;
    extern __shared__ int shared[];
    if (tid<n) shared[tid] = datas[tid];
    __syncthreads();
    int cnt=1;
    for (int gap=2; gap<n*2; gap<<=1, cnt++){
        if (tid%gap==0){
            int left=tid+n*((cnt+1)%2);
            int mid=tid+gap/2+n*((cnt+1)%2);
            int right=mid;
            int end=tid+gap+((cnt+1)%2)*n;
            int full_end=(1+(cnt+1)%2)*n;
            int res_ind=n*(cnt%2)+tid;

            while((left<mid && left<full_end) || (right<end && right<full_end)){
                if (!(left<mid && left<full_end)){
                    shared[res_ind]=shared[right];
                    right++;
                }else if (!(right<end && right<full_end)){
                    shared[res_ind]=shared[left];
                    left++;
                }else{
                    if (shared[right]> shared[left]){
                        shared[res_ind]=shared[left];
                        left++;
                    }else{
                        shared[res_ind]=shared[right];
                        right++;
                    }
                }
                res_ind++;
            }           
        }
        __syncthreads();
    }

    datas[tid]=shared[tid+ ((cnt+1)%2)*n];
}

int main(){
    CUDA_CALL(cudaSetDevice(0));
    //init
    const int length=1000;
    int a[length], b[length];
    for (int i=0;i<length;++i){
        a[i]=length-i;
        b[i]=i*i;
    }
    random_shuffle(begin(a), end(a));
    int *datas=NULL;
    CUDA_CALL(cudaMalloc((void **)&datas, length * sizeof(int)));

    //cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count,cudaMemcpyKind kind )
    //cudaMemcpyHostToHost   cudaMemcpyHossToDevice   cudaMemcpyDeviceToHost   cudaMemcpuDeviceToDevice
    CUDA_CALL(cudaMemcpy(datas,a,length*sizeof(int),cudaMemcpyHostToDevice) );
    //注意这里传给kernel的shared mem大小是以字节度量的
    merge_sort<<<1, length, length*2*sizeof(int)>>>(datas, length);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL( cudaThreadSynchronize());
    CUDA_CALL( cudaMemcpy(a,datas,length*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CALL( cudaFree(datas));

    for (int i=0;i<length;++i) cout<<a[i]<<" ";

    return 0;
}

