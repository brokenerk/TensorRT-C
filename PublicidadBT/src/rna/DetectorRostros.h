#ifndef DETECTORROSTROS_H_
#define DETECTORROSTROS_H_
#include "./../util/buffers.h"
#include "./../util/common.h"
#include "./../util/logger.h"
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
class DetectorRostros {
	public:
	    DetectorRostros();
	    vector<cv::Mat> detectarRostros(cv::Mat image);
	private:
	    int __model_dims[3] = {3, 300, 300};
	    int __model_layout = 7;
	    samplesCommon::BufferManager* buffers;
	    IExecutionContext* context;
};
#endif