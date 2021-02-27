#include "DetectorRostros.h"

DetectorRostros::DetectorRostros() {
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

    // Register tensorflow input
    parser->registerInput("Input",
                          nvinfer1::Dims3(__model_dims[0], __model_dims[1], __model_dims[2]),
                          nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput("NMS");
    parser->parse("cnn_SSD.uff", *network, nvinfer1::DataType::kFLOAT);

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(3_GiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    samplesCommon::enableDLA(builder.get(), config.get(), -1);

    mEngine = shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), 
                                                samplesCommon::InferDeleter()); 
}

void DetectorRostros::detectarRostros(cv::Mat image) {
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, 1);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    // Read the input data into the managed buffers
    const int inputC = __model_dims[0];
    const int inputH = __model_dims[1];
    const int inputW = __model_dims[2];
    const int batchSize = 1;

    //Preprocessing of input image
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

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    context->execute(1, buffers.getDeviceBindings().data());

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    //cv::namedWindow("Video Demo", cv::WINDOW_AUTOSIZE);

    const float* detection = static_cast<const float*>(buffers.getHostBuffer("NMS"));
    const float* output = &detection[0];

    int width = image.cols;
    int height = image.rows;

    int i = 0;
    while(1) {
        int prefix = i * __model_layout;
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
    //cv::imshow("Video Demo", image);

    nvuffparser::shutdownProtobufLibrary();
}
/*
int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: detectorRostros <Video_Path>\n");
        return -1;
    }
    cv::VideoCapture cap(argv[1]);

    gLogInfo << "Building and running a GPU inference engine for SSD" << endl;
    DetectorRostros ssd;

    while(1) {
        cv::Mat frame;

        if (!cap.read(frame))
            break;

        ssd.detectarRostros(frame);

        char key = cv::waitKey(10);
        if (key == 27) // ESC
            break;

    }
    cap.release();
    return 0;
}
*/