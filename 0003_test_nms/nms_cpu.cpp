#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


const int imgsize=1024;
const string sample_img="nms_cpu.jpg";
const string boxfile="./boxes.txt";//x1, y1, x2, y2, score 


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

//x1, y1, x2,y2,score
void nms(vector<vector<float>>& boxes, float thresh){
	sort(boxes.begin(), boxes.end(), [](auto &a, auto &b){
		return a[4]>b[4];
	});

	auto fun_iou=[&](int a, int b)->float{
		if(a<0 || a>=boxes.size() || b<0 || b>=boxes.size()) return 0;

		auto &ba=boxes[a];
		auto &bb=boxes[b];

		auto iou_x1=max(ba[0], bb[0]);
		auto iou_y1=max(ba[1], bb[1]);
		auto iou_x2=min(ba[2], bb[2]);
		auto iou_y2=min(ba[3], bb[3]);

		if (iou_x1>=iou_x2 || iou_y1>=iou_y2) return 0;
		auto siou=(iou_x2-iou_x1)*(iou_y2-iou_y1);

		auto sa=(ba[2]-ba[0])*(ba[3]-ba[1]);
		auto sb=(bb[2]-bb[0])*(bb[3]-bb[1]);

		return siou/(sa+sb-siou);
	};

	for (int i=0;i<boxes.size();++i){

		for (int j=i+1; j<boxes.size();++j){
			if(fun_iou(i, j)>thresh){
				boxes.erase(boxes.begin()+j);
				--j;
			}
		}
	}
}

const vector<Scalar> COLORS={Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255), Scalar(255,255,0), Scalar(0,255,255),Scalar(255,0,255),
							 Scalar(125,255,0), Scalar(255,125,0), Scalar(0,125,255), Scalar(0,255,125), Scalar(255,0,125), Scalar(125,0,255)};


int main(int argc, char *argv[]){
	if(argc<2){
		cout<<"Usage: command thresh"<<endl;
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
	imwrite("ori_"+sample_img, ori);

	cout<<"boxes before: "<<boxes.size()<<endl;
	nms(boxes, StrtoNum<float>(string(argv[1])));
	cout<<"boxes after: "<<boxes.size()<<endl;

	fun_draw(boxes, aft);
	imwrite("aft_"+sample_img, aft);

	return 0;
}
