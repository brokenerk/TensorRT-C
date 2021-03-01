#include "DetectorRostros.h"

DetectorRostros::DetectorRostros() {
	// Initialize TensorRT
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    // Variables del modelo
    string model_TRTbin = "./../bin/pretrained_model/TRT_ssd_mobilenet_v2.bin";
    string __PATH_UFF_SAVED_MODEL = "./../bin/pretrained_model/cnn_SSD.uff";

    // Check if BIN exists
    struct stat s;
    if(stat(model_TRTbin.c_str(), &s) != 0) {
    	// Builder, network, config and parser
	    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
	    nvinfer1::INetworkDefinition* network = builder->createNetwork();
	    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();

	    // Parse UFF model
	    parser->registerInput("Input",
	                          nvinfer1::Dims3(__model_dims[0], __model_dims[1], __model_dims[2]),
	                          nvuffparser::UffInputOrder::kNCHW);
	    parser->registerOutput("NMS");
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
	context = engine->createExecutionContext();
	buffers = new samplesCommon::BufferManager(engine, 1);
}

vector<cv::Mat> DetectorRostros::detectarRostros(cv::Mat image) {
    vector<cv::Mat> __rostros;
    
    const int inputC = __model_dims[0];
    const int inputH = __model_dims[1];
    const int inputW = __model_dims[2];
    const int batchSize = 1;

    // Pre-process image for inference
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(inputH, inputW));
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer("Input"));
    // Normalize image
    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i) {
        for (int c = 0; c < inputC; ++c) {
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
                hostDataBuffer[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(input_image.data[j * inputC + c]) - 1.0;
        }
    }

    // Memcpy from host input buffers to device input buffers
    buffers->copyInputToDevice();
    // Run inference
    context->execute(1, buffers->getDeviceBindings().data());
    // Memcpy from device output buffers to host output buffers
    buffers->copyOutputToHost();

    // Post-process detections and verify results
    const float* detection = static_cast<const float*>(buffers->getHostBuffer("NMS"));
    const float* output = &detection[0];
    int width = image.cols;
    int height = image.rows;
    

    // Check detections
    int i = 0;
    while(1) {
        int prefix = i * __model_layout;
        float score = output[prefix+2];

        // Cuando score>= 0.6 se considera como cara
        if(score >= 0.6) {
            float xmin  = output[prefix+3]*width;
            float ymin  = output[prefix+4]*height;
            float xmax  = output[prefix+5]*width;
            float ymax  = output[prefix+6]*height;
            //cv::rectangle(image, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 255, 0), 5);

            // border: lista con coordenadas del rostro en formato (yMin, xMin, yMax, xmax)
            int border[4] = {(int)ymin, (int)xmin, (int)ymax, (int)xmax};
            int ancho = border[3] - border[1];
            int alto = border[2] - border[0];

            /*
                Dado a que se nesecita un formato selfie en la red EG y big5 se agrega a la imagen:
                10% en yMin(Top)
                10% en xMin(Left)
                10% en xMax(Rigth)
                40% en yMax(Bottom)
                sidesMargins: tamanio que se agrega a los lados del rostro (10% del ancho del rostro).
                topMargin: tamanio que se agrega arriba del rostro (10% del alto del rostro).
                bottomMargin: tamanio que se agrega abajo del rostro (40% del alto del rostro).
                Las coordenadas se quedan tal cual para ocuparlos en el modulo de AR.
            */
            float sidesMargins = ancho * 0.1;
            float topMargin = alto * 0.1;
            float bottomMargin = alto * 0.4;

            // Para yMin (boder[0]) para formato selfie
            border[0] = ((border[0]-topMargin) > 0) ? (int)(border[0]-topMargin) : 0;

            // Para xMin (boder[1]) para formato selfie
            border[1] = ((border[1]-sidesMargins) > 0) ? (int)(border[1]-sidesMargins) : 0;

            // Para yMax (boder[2]) para formato selfie
            border[2] = ((border[2]+bottomMargin) < height) ? (int)(border[2]+bottomMargin) : height;

            // Para xMax (boder[3]) para formato selfie
            border[3] = ((border[3]+sidesMargins) < width) ? (int)(border[3]+sidesMargins) : width;

            // Tamanios de la selfie
            int selfieHeight = border[2]-border[0];
            int selfieWidth = border[3]-border[1];

            cv::Mat imageP = image(cv::Range(border[0], border[2]), cv::Range(border[1], border[3]));

            /*
                Redimension de 208x208
                aux: imagen totalmente en negro
                margin: posicion de los margenes
                Primero se redimenciona proporcinalemente la imagen en formato
                selfie para que el lado mas grande tenga un valor de 208 y el otro
                lado se mantenga proporcinalmente. Despues se suma la imagen en
                formato selfie reescalada sobre la imagen en negro (aux) respetando los margenes.
            */
            // Redimensiona 208x208
            cv::Mat aux = cv::Mat::zeros(208, 208, CV_8UC3);
            cv::Mat selfieImage;

            if (selfieHeight > selfieWidth) {
                int newWidth = (int)(selfieWidth * 208 / selfieHeight);
                cv::resize(imageP, selfieImage, cv::Size(newWidth, 208));
                int margin = (int)((208-newWidth)/2);
                selfieImage.copyTo(aux(cv::Range::all(), cv::Range(margin, margin+newWidth)));
            }
            else {
                int newHeight = (int)(selfieHeight * 208 / selfieWidth);
                cv::resize(imageP, selfieImage, cv::Size(208, newHeight));
                int margin = (int)((208-newHeight)/2);
                selfieImage.copyTo(aux(cv::Range(margin, margin+newHeight), cv::Range::all()));
            }
            // Normalizar
            aux.convertTo(aux, CV_32FC1);
            aux = aux / 255.0;
            __rostros.push_back(aux);
    
        }
        else
            break;

        i++;
    }
    nvuffparser::shutdownProtobufLibrary();
    return __rostros;
}