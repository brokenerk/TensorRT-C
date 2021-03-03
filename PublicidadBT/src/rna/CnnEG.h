#ifndef CNNEG_H_
#define CNNEG_H_
#include "./../util/buffers.h"
#include "./../util/common.h"
#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
class CnnEG {
	public:
	    CnnEG();
	    void obtenerEG(cv::Mat rostro, float* e_g);
	private:
	    int __model_dims[3] = {3, 208, 208};
	    samplesCommon::BufferManager* buffers;
	    IExecutionContext* context;
};
#endif