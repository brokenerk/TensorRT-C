#include "CnnEG.h"

CnnEG::CnnEG() {
    // Variables del modelo
    string model_TRTbin = "./../bin/models/EG/TRT_cnn_EG.bin";
    string __PATH_UFF_SAVED_MODEL = "./../bin/models/EG/frozen_graph_inference.uff";

    // Check if BIN exists
    struct stat s;
    if(stat(model_TRTbin.c_str(), &s) != 0) {
        // Builder, network, config and parser
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
        nvinfer1::INetworkDefinition* network = builder->createNetwork();
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        nvuffparser::IUffParser* parser = nvuffparser::createUffParser();

        // Parse UFF model
        parser->registerInput("input",
                              nvinfer1::Dims3(this->__model_dims[0], this->__model_dims[1], this->__model_dims[2]),
                              nvuffparser::UffInputOrder::kNCHW);
        parser->registerOutput("Identity");
        parser->registerOutput("Identity_1");
        parser->parse(__PATH_UFF_SAVED_MODEL.c_str(), *network, nvinfer1::DataType::kFLOAT);

        // Set config
        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(3_GiB);
        config->setFlag(BuilderFlag::kGPU_FALLBACK);

        // Build engine
        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Write engine in BIN file
        nvinfer1::IHostMemory* buf = engine->serialize();
        ofstream f(model_TRTbin, ios::out | ios::binary);
        f.write((char*)buf->data(), buf->size());
        f.close();
        buf->destroy();
    }

    // Read engine
    vector<char> buf;
    ifstream f(model_TRTbin, ios::binary);
    f.seekg(0, f.end);
    size_t size = f.tellg();
    f.seekg(0, f.beg);
    buf.resize(size);
    f.read(buf.data(), size);
    f.close();
    // Create engine
    nvinfer1::IRuntime* runtime = createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buf.data(), size, nullptr);

    // Create context and buffers
    this->context = engine->createExecutionContext();
    this->buffers = new samplesCommon::BufferManager(engine, 1);
}

void CnnEG::obtenerEG(cv::Mat rostro, float* e_g) {
    const int inputC = this->__model_dims[0];
    const int inputH = this->__model_dims[1];
    const int inputW = this->__model_dims[2];
    const int batchSize = 1;

    float* hostDataBuffer = static_cast<float*>(this->buffers->getHostBuffer("input"));
    // Normalize image
    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i) {
        for (int c = 0; c < inputC; ++c) {
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
                hostDataBuffer[i * volImg + c * volChl + j] = float(rostro.data[j * inputC + c]) / 255.0;
        }
    }

    // Memcpy from host input buffers to device input buffers
    this->buffers->copyInputToDevice();
    // Run inference
    this->context->execute(1, this->buffers->getDeviceBindings().data());
    // Memcpy from device output buffers to host output buffers
    this->buffers->copyOutputToHost();

    // Post-process detections and verify results
    const float* detectionAge = static_cast<const float*>(this->buffers->getHostBuffer("Identity"));
    const float* detectionGenre = static_cast<const float*>(this->buffers->getHostBuffer("Identity_1"));

    const float* g = &detectionGenre[0];
    const float* edad = &detectionAge[0];

    /* Funcion escalon para genero
       Revisar solo la primera posicion
       [1 0] Mujer -> 0
       [0 1] Hombre -> 1
    */
    int genero;
    if(g[0] > 0)
        genero = 0; //MUJER
    else
        genero = 1; //HOMBRE

    e_g[0] = (float)edad[0];
    e_g[1] = (float)genero;
    nvuffparser::shutdownProtobufLibrary();
}