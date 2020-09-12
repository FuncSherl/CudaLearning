#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cassert>


// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }

using namespace std;

int main()
{
    int deviceCount;
    CUDA_CALL( cudaGetDeviceCount(&deviceCount));
    std::cout<< "device count: "<<deviceCount<<endl;
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        CUDA_CALL(cudaGetDeviceProperties(&devProp, i));
        std::cout << "GPU device name: " << i << " : " << devProp.name << std::endl;
        std::cout << "device total Global Mem: " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "multiProcessor(SM) Count: " << devProp.multiProcessorCount << std::endl;
        std::cout << "shared Mem Per Block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Registers PerBlock: " << devProp.regsPerBlock << std::endl;
        std::cout << "maxThreads PerMultiProcessor(EM): " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "maxWarps PerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "======================================================" << std::endl;             
    }
    return 0;
}
