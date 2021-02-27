#ifndef DETECTORROSTROS_H_
#define DETECTORROSTROS_H_
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
const string gSampleName = "TensorRT.sample_SSD";
class DetectorRostros {
    template <typename T>
    using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;
	public:
	    DetectorRostros();
	    void detectarRostros(cv::Mat image);
	private:
	    shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
	    int __model_dims[3] = {3, 300, 300};
	    int __model_layout = 7;
};
#endif