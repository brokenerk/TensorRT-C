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

const string gSampleName = "TensorRT.CnnBig5";

//!
//! \brief  The CnnBig5 class implements the Big5 sample
//!
//! \details It creates the network using a Uff model
//!
class CnnBig5 {
    template <typename T>
    using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;

public:
    CnnBig5();

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(cv::Mat image);

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

private:
    //!
    //! \brief Parses a Uff model for Big5 and creates a TensorRT network
    //!
    void constructNetwork(SampleUniquePtr<nvuffparser::IUffParser>& parser,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(cv::Mat image, const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

CnnBig5::CnnBig5() {}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Big5 network by parsing the Uff model
//!          and builds the engine that will be used to run Big5 (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool CnnBig5::build() {
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
        return false;

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
        return false;

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return false;

    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
        return false;

    constructNetwork(parser, network);
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(3_GiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    //config->setFlag(BuilderFlag::kFP16);

    samplesCommon::enableDLA(builder.get(), config.get(), -1);

    mEngine = shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), 
    											samplesCommon::InferDeleter());

    if (!mEngine)
        return false;
    return true;
}

//!
//! \brief Uses a Uff parser to create the Big5 Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the Big5 network
//!
//! \param builder Pointer to the engine builder
//!
void CnnBig5::constructNetwork(SampleUniquePtr<nvuffparser::IUffParser>& parser,
    							SampleUniquePtr<nvinfer1::INetworkDefinition>& network) {
    // Register tensorflow input
    parser->registerInput("input_4",
                          nvinfer1::Dims3(1, 208, 208),
                          nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput("dense_12/BiasAdd");
    parser->parse("cnn4_train_2.uff", *network, nvinfer1::DataType::kFLOAT);
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer
//!
bool CnnBig5::processInput(cv::Mat image, const samplesCommon::BufferManager& buffers) {
    const int inputC = 1;
    const int inputH = 208;
    const int inputW = 208;
    const int batchSize = 1;

    cv::Mat resize_image;
	cv::resize(image, resize_image, cv::Size(inputH, inputW));
    cv::cvtColor(resize_image, resize_image, cv::COLOR_RGB2GRAY);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("input_4"));

	for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i) {
		for (int c = 0; c < inputC; ++c) {
			for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
				hostDataBuffer[i * volImg + c * volChl + j] = (1.0 / 255.0) * float(resize_image.data[j * inputC + c]);
		}
	}

    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool CnnBig5::verifyOutput(const samplesCommon::BufferManager& buffers) {
    const float* detection = static_cast<const float*>(buffers.getHostBuffer("dense_12/BiasAdd"));
    const float* big5 = &detection[0];
    gLogInfo << "Big5 [" << big5[0] << " " << big5[1] << " " << big5[2] << " " << big5[3] << " " << big5[4] << endl;

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample.
//!  It allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool CnnBig5::infer(cv::Mat image) {
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, 1);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
        return false;

    // Read the input data into the managed buffers
    if (!processInput(image, buffers))
        return false;

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(1, buffers.getDeviceBindings().data());
    if (!status)
        return false;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
        return false;

    return true;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool CnnBig5::teardown() {
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

int main(int argc, char** argv) {
	if (argc != 2) {
        printf("usage: CnnBig5 <Video_Path>\n");
        return -1;
    }
    cv::VideoCapture cap(argv[1]);
    cv::namedWindow("Video Demo", cv::WINDOW_AUTOSIZE);

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    CnnBig5 sample;
    gLogInfo << "Building and running a GPU inference engine for Big5" << endl;

    if (!sample.build())
        return gLogger.reportFail(sampleTest);

    while(1) {
        cv::Mat frame;

        if (!cap.read(frame))
            break;

        if (!sample.infer(frame))
        	return gLogger.reportFail(sampleTest);

	    if (!sample.teardown())
	        return gLogger.reportFail(sampleTest);

        cv::imshow("Video Demo", frame);

        char key = cv::waitKey(10);
        if (key == 27) // ESC
            break;

    }
    cap.release();
    return gLogger.reportPass(sampleTest);
}
