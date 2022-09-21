#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
//#define USE_FP16  // comment out this if want to use FP32

using namespace nvinfer1;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = 112;
static const int INPUT_W = 112;
static const int OUTPUT_SIZE = 512;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

class ArcFace
{
    // stuff we know about the network and the input/output blobs
    private:
        IExecutionContext* context;
        // ICudaEngine* engine;
    public:
        ArcFace(const std::string& engine_path)
        {
            cudaSetDevice(DEVICE);
            // create a model using the API directly and serialize it to a stream
            char *trtModelStream{nullptr};
            size_t size{0};

            std::ifstream file(engine_path, std::ios::binary);
            if (!file.good()) {
                std::cerr << "read " << engine_path << " error!" << std::endl;
                // return -1;
            }

            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();

            IRuntime* runtime = createInferRuntime(gLogger);
            assert(runtime != nullptr);
            ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
            engine = runtime->deserializeCudaEngine(trtModelStream, size);
            assert(engine != nullptr);
            this->context = engine->createExecutionContext();
            assert(context != nullptr);
            delete[] trtModelStream;
            // std::cout << "Init" << std::endl;
        }


    void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
        const ICudaEngine& engine = context.getEngine();

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
    }

    std::vector<std::vector<float>> extract(cv::Mat img) {
        const ICudaEngine& engine = this->context->getEngine();
        std::vector<std::vector<float>> batch_results;
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
            data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
            data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        }
        // Run inference
        doInference(*context, data, prob, BATCH_SIZE);
        // result = prob
        cv::Mat out(512, 1, CV_32FC1, prob);
        // std::vector
        cv::Mat out_norm;
        cv::normalize(out, out_norm);
        // std::cout<<prob<<std::endl;

        batch_results.push_back(out_norm);

        // Release stream and buffers

        return batch_results;
    }
};
