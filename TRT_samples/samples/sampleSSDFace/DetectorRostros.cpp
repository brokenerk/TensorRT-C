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

//!
//! \brief  The SampleSSD class implements the SSD sample
//!
//! \details It creates the network using a Uff model
//!
class SampleSSD {
    template <typename T>
    using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleSSD();

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
    //! \brief Parses a Uff model for SSD and creates a TensorRT network
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
    bool verifyOutput(cv::Mat image, const samplesCommon::BufferManager& buffers);

    shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

SampleSSD::SampleSSD() {}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the SSD network by parsing the Uff model
//!          and builds the engine that will be used to run SSD (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleSSD::build() {
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
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
//! \brief Uses a Uff parser to create the SSD Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the SSD network
//!
//! \param builder Pointer to the engine builder
//!
void SampleSSD::constructNetwork(SampleUniquePtr<nvuffparser::IUffParser>& parser,
    							SampleUniquePtr<nvinfer1::INetworkDefinition>& network) {
    // Register tensorflow input
    parser->registerInput("Input",
                          nvinfer1::Dims3(3, 300, 300),
                          nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput("NMS");
    parser->parse("cnn_SSD.uff", *network, nvinfer1::DataType::kFLOAT);
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer
//!
bool SampleSSD::processInput(cv::Mat image, const samplesCommon::BufferManager& buffers) {
    const int inputC = 3;
    const int inputH = 300;
    const int inputW = 300;
    const int batchSize = 1;

    cv::Mat input_image;
	cv::resize(image, input_image, cv::Size(inputH, inputW));
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("Input"));

	for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i) {
		for (int c = 0; c < inputC; ++c) {
			for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
				hostDataBuffer[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(input_image.data[j * inputC + c]) - 1.0;
		}
	}

    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleSSD::verifyOutput(cv::Mat image, const samplesCommon::BufferManager& buffers) {
    cv::namedWindow("Video Demo", cv::WINDOW_AUTOSIZE);

    const float* detection = static_cast<const float*>(buffers.getHostBuffer("NMS"));
    const float* output = &detection[0];

    int width = image.cols;
    int height = image.rows;

    int i = 0;
    while(1) {
        int prefix = i * 7;
        float score = output[prefix+2];

        if(score > 0.6) {
            float xmin  = output[prefix+3]*width;
            float ymin  = output[prefix+4]*height;
            float xmax  = output[prefix+5]*width;
            float ymax  = output[prefix+6]*height;
            cv::rectangle(image, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 255, 0), 5);
        }
        else
            break;

        i++;
    }
    cv::imshow("Video Demo", image);


    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample.
//!  It allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleSSD::infer(cv::Mat image) {
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
    if (!verifyOutput(image, buffers))
        return false;

    return true;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleSSD::teardown() {
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

int main(int argc, char** argv) {
	if (argc != 2) {
        printf("usage: sampleSSD <Video_Path>\n");
        return -1;
    }
    cv::VideoCapture cap(argv[1]);

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleSSD sample;
    gLogInfo << "Building and running a GPU inference engine for SSD" << endl;

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

        char key = cv::waitKey(10);
        if (key == 27) // ESC
            break;

    }
    cap.release();
    return gLogger.reportPass(sampleTest);
}
