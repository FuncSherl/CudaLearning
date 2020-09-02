#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"


#define CUDA_CALL(x) {const cudaError_t a=(x);if (a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); } }

using namespace std;

int main()
{
    int deviceCount;
    CUDA_CALL( cudaGetDeviceCount(&deviceCount));
    std::cout<< "device count: "<<deviceCount<<endl;
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
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
