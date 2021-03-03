#ifndef CNNBIG5_H_
#define CNNBIG5_H_
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
class CnnBig5 {
	public:
	    CnnBig5();
	    void obtenerPersonalidad(cv::Mat rostro, float* big5);
	private:
	    int __model_dims[3] = {1, 208, 208};
	    samplesCommon::BufferManager* buffers;
	    IExecutionContext* context;
};
#endif