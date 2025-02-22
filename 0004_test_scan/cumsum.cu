#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "common/include/commontools.h"

#define THREAD_BLOCK_LIMIT 1024

template <typename T>
__global__ void cumsumBellochKernel(const T *d_in, T *d_out, long n) {
    extern __shared__ T temp[];
    long numPerThread = (n + blockDim.x - 1) / blockDim.x;
    long tid = threadIdx.x;
    long block_offset = blockIdx.x * n;

    // 1. copy the data to shared memory
    for (long i = 0; i < numPerThread; i++) {
        long index = tid * numPerThread + i;
        if (index < n) {
            temp[index] = d_in[block_offset + index];
        }
    }
    __syncthreads();

    // 2. local cumsum of each block
    for (long i = 1; i < numPerThread; i++) {
        long index = tid * numPerThread + i;
        if (index < n) {
            temp[index] += temp[index - 1];
        }
    }
    __syncthreads();

    // 3. Up-sweep phase (reduce)
    for (long stride = 1; stride < n; stride *= 2) {
        long stride_idx = stride * numPerThread;
        long index = (tid + 1) * stride_idx * 2 - 1;
        if (index < n) {
            temp[index] += temp[index - stride_idx];
        }
        __syncthreads();
    }

    // 4. Down-sweep phase (down)
    for (long stride = n / 4; stride > 0; stride /= 2) {
        long stride_idx = stride * numPerThread;
        long index = (tid + 1) * stride_idx * 2 - 1;
        if (index + stride_idx < n) {
            temp[index + stride_idx] += temp[index];
        }
        __syncthreads();
    }

    // 5. cumsum each local block
    for (long i = 0; i < numPerThread - 1; i++) {
        long index = (tid + 1) * numPerThread + i;
        if (index < n) {
            temp[index] += temp[(tid + 1) * numPerThread - 1];
        }
    }
    __syncthreads();

    // 6. copy the data back to global memory
    for (long i = 0; i < numPerThread; i++) {
        long index = tid * numPerThread + i;
        if (index < n) {
            d_out[block_offset + index] = temp[index];
        }
    }
}

template <typename T>
void cumsumBelloch(const T *h_in, T *h_out, int m, int n) {
    T *d_in, *d_out;
    size_t size = m * n * sizeof(T);

    CUDA_CALL(cudaMalloc((void **)&d_in, size));
    CUDA_CALL(cudaMalloc((void **)&d_out, size));

    CUDA_CALL(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(std::min(THREAD_BLOCK_LIMIT, n));
    dim3 blocksPerGrid(m);
    clock_t start_kernel = clock();
    cumsumBellochKernel<<<blocksPerGrid, threadsPerBlock, n * sizeof(T)>>>(
        d_in, d_out, n);
    CUDA_LAST_ERROR();
    CUDA_CALL(cudaDeviceSynchronize());
    clock_t end_kernel = clock();
    double elapsed_time_kernel =
        double(end_kernel - start_kernel) * 1000.0 / CLOCKS_PER_SEC;
    std::cout << "Kernel cumsumBelloch execution time: " << elapsed_time_kernel
              << " milliseconds" << std::endl;

    CUDA_CALL(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

template <typename T>
__global__ void cumsumNaiveKernel(const T *d_in, T *d_out, int n) {
    extern __shared__ T temp[];
    long bufferOffset = n * sizeof(T);
    long numPerThread = (n + blockDim.x - 1) / blockDim.x;
    long tid = threadIdx.x;
    long block_offset = blockIdx.x * n;

    // 1. copy the data to shared memory
    for (long i = 0; i < numPerThread; i++) {
        long index = tid * numPerThread + i;
        if (index < n) {
            temp[index] = d_in[block_offset + index];
        }
    }
    __syncthreads();

    // 2. naive tree sum
    int cnt = 0;
    for (long stride = 1; stride < n; stride *= 2, cnt++) {
        int readIdx = cnt % 2;
        int writeIdx = 1 - readIdx;
        T *tempR = temp + readIdx * bufferOffset;
        T *tempW = temp + writeIdx * bufferOffset;

        for (long j = 0; j < numPerThread; j++) {
            long index = tid * numPerThread + j;

            if (index >= stride && index < n) {
                tempW[index] = tempR[index] + tempR[index - stride];
            } else {
                tempW[index] = tempR[index];
            }
        }
        __syncthreads();
    }

    // 3. copy the data back to global memory
    for (long i = 0; i < numPerThread; i++) {
        long index = tid * numPerThread + i;
        if (index < n) {
            d_out[block_offset + index] =
                temp[index + (cnt % 2) * bufferOffset];
        }
    }
}

template <typename T>
void cumsumNaive(const T *h_in, T *h_out, int m, int n) {
    T *d_in, *d_out;
    size_t size = m * n * sizeof(T);

    CUDA_CALL(cudaMalloc((void **)&d_in, size));
    CUDA_CALL(cudaMalloc((void **)&d_out, size));

    CUDA_CALL(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(std::min(THREAD_BLOCK_LIMIT, n));
    dim3 blocksPerGrid(m);
    clock_t start_kernel = clock();
    cumsumNaiveKernel<<<blocksPerGrid, threadsPerBlock, n * sizeof(T) * 2>>>(
        d_in, d_out, n);
    CUDA_LAST_ERROR();
    CUDA_CALL(cudaDeviceSynchronize());
    clock_t end_kernel = clock();
    double elapsed_time_kernel =
        double(end_kernel - start_kernel) * 1000.0 / CLOCKS_PER_SEC;
    std::cout << "Kernel cumsumNaive execution time: " << elapsed_time_kernel
              << " milliseconds" << std::endl;

    CUDA_CALL(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

void cumsumSingle(const int *h_in, int *h_out, int m, int n) {
    clock_t start = clock();
    for (int i = 0; i < m; i++) {
        h_out[i * n] = h_in[i * n];
        for (int j = 1; j < n; j++) {
            h_out[i * n + j] = h_out[i * n + j - 1] + h_in[i * n + j];
        }
    }
    clock_t end = clock();
    double elapsed_time = double(end - start) * 1000.0 / CLOCKS_PER_SEC;
    std::cout << "Single cumsum execution time: " << elapsed_time
              << " milliseconds" << std::endl;
}

bool checkDiff(const int *a, const int *b, long size) {
    for (long i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            std::cout << "Difference found at index " << i << ": " << a[i]
                      << " != " << b[i] << std::endl;
            return false;
        }
    }
    std::cout << "No differences found." << std::endl;
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <m> <n>" << std::endl;
        return 1;
    }

    const long m = std::stol(argv[1]);
    const long n = std::stol(argv[2]);

    int *h_in = new int[m * n];
    int *h_out_cpu = new int[m * n];
    int *h_out = new int[m * n];

    srand(time(0));

    std::cout << "Initializing input array..." << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_in[i * n + j] = rand() % 10;  // Random numbers between 0 and 9
        }
    }

    std::cout << "Running single-threaded cumulative sum..." << std::endl;
    cumsumSingle(h_in, h_out_cpu, m, n);

    std::cout << "Running naive parallel cumulative sum..." << std::endl;
    cumsumNaive(h_in, h_out, m, n);
    std::cout << "Checking results of naive parallel cumulative sum..."
              << std::endl;
    checkDiff(h_out, h_out_cpu, m * n);

    std::fill(h_out, h_out + m * n, 0);

    std::cout << "Running Belloch parallel cumulative sum..." << std::endl;
    cumsumBelloch(h_in, h_out, m, n);
    std::cout << "Checking results of Belloch parallel cumulative sum..."
              << std::endl;
    checkDiff(h_out, h_out_cpu, m * n);

    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out;

    return 0;
}
