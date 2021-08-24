#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iomanip>

using namespace std;

// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }
#define CUDA_LAST_ERROR() CUDA_CALL(cudaGetLastError())

#define SHOW_MAT(a, m,n) \
{\
  cout<<"\nShowMat: "<<#a<<" -> "<<m<<"*"<<n<<endl;\
  for (int i=0;i<m;++i){\
    for (int j=0;j<n;++j){\
      cout<<setw(6)<<a[i*n+j];\
    }\
    cout<<endl;\
  }\
}

#define SET_MAT(a,d, m, n)\
{\
  cout<<"\nsetmat: "<<#a<<" -> "<<d<<endl;\
  for (int i=0;i<m;++i){\
    for (int j=0;j<n;++j){\
      a[i*n+j]=d;\
    }\
  }\
}

#define SET_MATAUTO(a, m, n)\
{\
  cout<<"\nsetmat: "<<#a<<endl;\
  for (int i=0;i<m;++i){\
    for (int j=0;j<n;++j){\
      a[i*n+j]=i*n+j;\
    }\
  }\
}

#define MDIV(a,b) ((int)a%(int)b==0?(int)a/(int)b:(int)a/(int)b+1)

template<int BSIZE>
__global__ void matmult_v1(float *a, float *b, float *c, int m, int n, int k){//a-> m*k  b->k*n

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy=blockIdx.y*blockDim.y+threadIdx.y;
  int index=idy*n+idx;
  if (idx>=n || idy>=m) return;
  
  c[index]=0;
  for (int i=0;i<k;++i){
    c[index]+=a[idy*k+i]*b[idx+i*n];
  }
}

template<int BSIZE>
__global__ void matmult_v2(float *a, float *b, float*c, int m, int n, int k){
  int tx=threadIdx.x;
  int ty=threadIdx.y;

  int btx=blockIdx.x*blockDim.x+tx;
  int bty=blockIdx.y*blockDim.y+ty;
  
	//if(btx>=n || bty>=m) return;

  float res=0;
 
  for (int i=0;i<k;i+=BSIZE){
    __shared__ float skepiA[BSIZE][BSIZE];
    __shared__ float skepiB[BSIZE][BSIZE];
    skepiA[ty][tx]=(bty>=m?0:a[bty*k+i+tx]);
    skepiB[ty][tx]=(btx>=n?0:b[(i+ty)*n+btx]);

    __syncthreads();
    
    for (int j=0; j<BSIZE && j+i<k; ++j){

      res+=skepiA[ty][j]*skepiB[j][tx];
    }
		__syncthreads();
  }
  //__syncthread();
  c[bty*n+btx]=res;
}

int main(){
  //获取设备属性
	cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&prop, deviceID);
 
	//检查设备是否支持重叠功能
	if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
		return 0;
	}

  //启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


  int m=18;
  int n=24;
  int k=14;
	const int BSIZE=4;

  float *A=new float[m*k];//{1,2,3,4,5,6,7,8,9,10,11};
  float *B=new float[k*n];//{1,0,1,0,1,0,0,1,1,0,1,1,0};
  float *C=new float[m*n];
	SET_MAT(A, 1, m, k);
	SET_MATAUTO(B, k, n);

  float *ga, *gb, *gc;
  // GPU端分配内存
  cudaMalloc((void**)&ga, m*k*sizeof(float));
  cudaMalloc((void**)&gb, k*n*sizeof(float));
  cudaMalloc((void**)&gc, m*n*sizeof(float));

  // CPU的数据拷贝到GPU端
  cudaMemcpy(ga, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gb, B, k*n*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(gc, C, size, cudaMemcpyHostToDevice);

  // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
  dim3 dimBlock(BSIZE,BSIZE);
  dim3 dimGrid(MDIV(n,BSIZE),MDIV(m,BSIZE));
  cout<<"start block num: "<<dimGrid.x<<"*"<<dimGrid.y<<endl;
	cout<<"each block:"<<dimBlock.x<<"*"<<dimBlock.y<<endl;
  // 执行kernel
	CUDA_CALL(cudaEventRecord(start));
  int iter=1000*2;
	for (int i=0;i<iter;++i){
		matmult_v2<BSIZE><<<dimGrid, dimBlock>>>(ga, gb, gc, m,n,k);
		//matmult_v1<BSIZE><<<dimGrid, dimBlock>>>(ga, gb, gc, m,n,k);
	}
	CUDA_LAST_ERROR();
	
	CUDA_CALL(cudaEventRecord(stop));
  
  CUDA_CALL(cudaEventSynchronize(stop));
	CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	
  //cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
  cudaMemcpy(C, gc, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(ga);
  cudaFree(gb);
  cudaFree(gc);

  SHOW_MAT(A,m,k);
  SHOW_MAT(B,k,n);
  SHOW_MAT(C,m,n);
	cout<<"Iter:"<<iter<<" UsedTime: "<<elapsedTime<<" ms"<<endl;
  delete []A;
  delete []B;
  delete []C;

  return 0;
}
