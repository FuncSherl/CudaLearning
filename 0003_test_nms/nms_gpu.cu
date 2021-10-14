#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cassert>
#include <bitset>
#include <chrono>

using namespace cv;
using namespace std;

// 用宏变长参数来实现
#define CUDA_CALL(...) {cudaError_t _cuda_tep_set_not_repeat_a=(__VA_ARGS__);if (_cuda_tep_set_not_repeat_a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(_cuda_tep_set_not_repeat_a), _cuda_tep_set_not_repeat_a); cudaDeviceReset(); assert(0);} }
#define CUDA_LAST_ERROR() CUDA_CALL(cudaGetLastError())
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

const int imgsize=1024;
const string sample_img="nms_gpu.jpg";
const string boxfile="./boxes.txt";//x1, y1, x2, y2, score 
const int eachboxlen=5;


void split(const string & str, vector<string> &ret, const string delim=" "){
	auto firpos=str.find_first_not_of(delim, 0);
	auto secpos=str.find_first_of(delim, firpos);

	while(firpos!=string::npos || secpos!=string::npos){
		ret.push_back(str.substr(firpos, secpos-firpos));
		firpos=str.find_first_not_of(delim, secpos);
		secpos=str.find_first_of(delim, firpos);
	}
}

template<class T>
T StrtoNum(const string & s){
	istringstream iss(s);  
    T num;  
    iss >> num;  
    return num;      
}

const int blocksize=sizeof(int)*8;

__device__ inline float iou(const float* a, const float*b){
	float iou_x1=max(a[0], b[0]);
	float iou_y1=max(a[1], b[1]);
	float iou_x2=min(a[2], b[2]);
	float iou_y2=min(a[3], b[3]);

	if (iou_x1>=iou_x2 || iou_y1>=iou_y2) return 0;
	float sa=(a[2]-a[0])*(a[3]-a[1]);
	float sb=(b[2]-b[0])*(b[3]-b[1]);
	float siou=(iou_y2-iou_y1)*(iou_x2-iou_x1);
	return siou/(sa+sb-siou);
}

//x1, y1, x2,y2,score
__global__ void nms(float *boxes, unsigned int *mask, int boxnum, float thresh){
	const int row_start = blockIdx.y;
	const int col_start = blockIdx.x;

	int x_threads=min(blocksize, boxnum-col_start*blocksize);
	int y_threads=min(blocksize, boxnum-row_start*blocksize);

	float* curbox=boxes+(row_start*blocksize+threadIdx.x)*5;
	int blocknum=DIVUP(boxnum,blocksize);
	__shared__ float block_boxes[blocksize * 5];
	if (threadIdx.x<x_threads){
		block_boxes[threadIdx.x*5+0]=boxes[(col_start*blocksize+ threadIdx.x)*5+0];
		block_boxes[threadIdx.x*5+1]=boxes[(col_start*blocksize+ threadIdx.x)*5+1];
		block_boxes[threadIdx.x*5+2]=boxes[(col_start*blocksize+ threadIdx.x)*5+2];
		block_boxes[threadIdx.x*5+3]=boxes[(col_start*blocksize+ threadIdx.x)*5+3];
		block_boxes[threadIdx.x*5+4]=boxes[(col_start*blocksize+ threadIdx.x)*5+4];
	}
	__syncthreads();

	if (threadIdx.x<y_threads){
		unsigned int tep=0;
		for (int i=0;i<x_threads;++i){
			float* anobox=&block_boxes[i*5]; //boxes+(col_start*blocksize+i)*5;
			
			if(iou(curbox, anobox)>thresh){
				//printf ("box %d and %d iou: %f\n",row_start*blocksize+threadIdx.x, col_start*blocksize+i, iou(curbox, anobox));
				tep|=(1U<<i);
			}
		}
		mask[blocknum*(row_start*blocksize+threadIdx.x)+col_start]=tep;
		//printf("Mask %d = %x\n",blocknum*(row_start*blocksize+threadIdx.x)+col_start, tep);
	}
}	

const vector<Scalar> COLORS={Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255), Scalar(255,255,0), Scalar(0,255,255),Scalar(255,0,255),
							 Scalar(125,255,0), Scalar(255,125,0), Scalar(0,125,255), Scalar(0,255,125), Scalar(255,0,125), Scalar(125,0,255)};


int main(int argc, char *argv[]){
	if(argc<3){
		cout<<"Usage: command thresh addboxnum"<<endl;
		exit(1);
	}
	ifstream readf(boxfile);
	string lin;
	vector<vector<float>> boxes;//each row is : x1, y1, x2,y2,score
	while(getline(readf, lin)){
		vector<string> tep;
		split(lin, tep, " ");
		

		vector<float> onebox;
		for (auto &t: tep){
			onebox.push_back(StrtoNum<float>(t));
		}
		
		if (onebox.size()!=5) continue;//x1, y1, x2,y2,score

		boxes.push_back(onebox);
	}
	cv::Mat ori(imgsize, imgsize, CV_8UC3, Scalar(cv::uint8_t(255),cv::uint8_t(255),cv::uint8_t(255)));
	cv::Mat aft(imgsize, imgsize, CV_8UC3,Scalar(cv::uint8_t(255),cv::uint8_t(255),cv::uint8_t(255)));

	auto fun_draw=[](const vector<vector<float>> & boxes, Mat & img){
		int cnt=0;
		for (auto & b: boxes){
			int x1=b[0];
			int y1=b[1];
			int x2=b[2];
			int y2=b[3];
			float score=b[4];

			rectangle(img, Point(x1, y1), Point(x2, y2), COLORS[cnt%COLORS.size()]);
			putText(img, to_string(score), Point(x1, y1-4), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255));
			++cnt;
		}
	};
	fun_draw(boxes, ori);
	imwrite("ori_gpu_"+sample_img, ori);

	for(int i=0;i< StrtoNum<int>(string(argv[2]));++i){
		boxes.push_back(boxes[i]);
	}

	cout<<"boxes before: "<<boxes.size()<<endl;

	auto beginTime = std::chrono::high_resolution_clock::now();
	
	sort(boxes.begin(), boxes.end(), [](auto &a, auto &b){
		return a[4]>b[4];
	});
	int blocknum=DIVUP(boxes.size(), blocksize);

	dim3 blocks(blocknum,blocknum);
	dim3 threads(blocksize);

	float host_boxes[boxes.size()*5];
	float* dev_boxes=nullptr;
	unsigned int *mask_dev=nullptr;
	for (int i=0;i< boxes.size();++i){
		for (int j=0;j<boxes[i].size();++j) host_boxes[i*5+j]=boxes[i][j];
	}

	CUDA_CALL(cudaMalloc(&dev_boxes,  boxes.size()*5 * sizeof(float)));
	CUDA_CALL(cudaMalloc(&mask_dev,  boxes.size()* blocknum * sizeof(unsigned int)));

	CUDA_CALL(cudaMemcpy(dev_boxes,
                        &host_boxes[0],
                        boxes.size()*5*sizeof(float),
                        cudaMemcpyHostToDevice));

	nms<<<blocks, threads>>>(dev_boxes, mask_dev, boxes.size(),StrtoNum<float>(string(argv[1])));
	
	std::vector<unsigned int> mask_host(boxes.size()* blocknum);
	CUDA_CALL(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        boxes.size()*blocknum*sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
	// for (auto i: mask_host){
	// 	cout<<bitset<32>(i)<<endl;
	// }
	
	std::vector<unsigned int> blacklist(blocknum, 0);

	vector<vector<float>> boxes_swap;

	for (int i=0;i<boxes.size();++i){
		
		int block_ind=i/blocksize;
		int thread_ind=i%blocksize;

		//cout<<"debug2:"<<i<<" "<<block_ind<< " "<< blacklist.size() <<endl;
		if(blacklist[block_ind] & (1U<<thread_ind)) continue;
		
		//cout<<"Add box: "<<i<< " "<< boxes[i][4] <<endl;
		boxes_swap.push_back(boxes[i]);

		for (int j=block_ind; j<blocknum;++j)  blacklist[j]|=mask_host[blocknum*i+j];
	}
	boxes_swap.swap(boxes);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime-beginTime);
	double programTimes = ((double) elapsedTime.count());
	
	cout<<"boxes after: "<<boxes.size()<<" Time: "<<programTimes<<endl;

	fun_draw(boxes, aft);
	imwrite("aft_gpu_"+sample_img, aft);

	return 0;
}
