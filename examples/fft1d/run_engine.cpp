// Deserialize and execute the FFT1D engine. An impulse at index zero has a
// constant FFT, giving an exact end-to-end reference without a CPU FFT.
// Usage:
//   run_fft1d_engine [model.engine] [plugin-library.so]

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace
{
#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t const error{call};                                         \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n'; \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (false)

class Logger final : public nvinfer1::ILogger
{
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::cout << "[TensorRT] " << message << '\n';
        }
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* object) const
    {
        delete object;
    }
};

size_t volume(nvinfer1::Dims const& dimensions)
{
    size_t result{1U};
    for (int32_t i{0}; i < dimensions.nbDims; ++i)
    {
        if (dimensions.d[i] <= 0)
        {
            return 0U;
        }
        result *= static_cast<size_t>(dimensions.d[i]);
    }
    return result;
}
} // namespace

int main(int argc, char** argv)
{
    std::string const enginePath{argc > 1 ? argv[1]
                                         : "artifacts/fft1d.engine"};
    std::string const pluginPath{argc > 2 ? argv[2]
                                         : "build/plugins/libtrt_fft_plugins.so"};
    Logger logger{};
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime{
        nvinfer1::createInferRuntime(logger)};
    if (!runtime
        || runtime->getPluginRegistry().loadLibrary(pluginPath.c_str())
            == nullptr)
    {
        std::cerr << "Failed to initialize runtime or load " << pluginPath
                  << '\n';
        return EXIT_FAILURE;
    }

    std::ifstream input{enginePath, std::ios::binary | std::ios::ate};
    if (!input)
    {
        std::cerr << "Failed to open engine: " << enginePath << '\n';
        return EXIT_FAILURE;
    }
    size_t const engineSize{static_cast<size_t>(input.tellg())};
    input.seekg(0, std::ios::beg);
    std::vector<char> engineBytes(engineSize);
    input.read(engineBytes.data(), static_cast<std::streamsize>(engineSize));

    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine{
        runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size())};
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context{
        engine ? engine->createExecutionContext() : nullptr};
    if (!engine || !context)
    {
        std::cerr << "Failed to deserialize engine or create context\n";
        return EXIT_FAILURE;
    }

    char const* const names[]{"real_input", "imag_input", "real_output",
                              "imag_output"};
    std::unordered_map<std::string, std::vector<float>> hostBuffers;
    std::unordered_map<std::string, void*> deviceBuffers;
    for (char const* name : names)
    {
        nvinfer1::Dims const shape{engine->getTensorShape(name)};
        size_t const count{volume(shape)};
        if (count == 0U
            || engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        {
            std::cerr << "Missing, dynamic, or non-FP32 tensor: " << name
                      << '\n';
            return EXIT_FAILURE;
        }
        hostBuffers.emplace(name, std::vector<float>(count, 0.0F));
        void* pointer{};
        CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(float)));
        deviceBuffers.emplace(name, pointer);
        if (!context->setTensorAddress(name, pointer))
        {
            std::cerr << "Failed to bind tensor: " << name << '\n';
            return EXIT_FAILURE;
        }
    }

    nvinfer1::Dims const inputShape{engine->getTensorShape("real_input")};
    size_t const fftLength{
        static_cast<size_t>(inputShape.d[inputShape.nbDims - 1])};
    size_t const elementCount{hostBuffers.at("real_input").size()};
    // For every transform: x[0] = 2 - 0.5j and all other samples are zero.
    // Its unnormalized forward FFT is 2 - 0.5j at every output bin.
    for (size_t offset{0U}; offset < elementCount; offset += fftLength)
    {
        hostBuffers.at("real_input")[offset] = 2.0F;
        hostBuffers.at("imag_input")[offset] = -0.5F;
    }

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (char const* name : {names[0], names[1]})
    {
        CUDA_CHECK(cudaMemcpyAsync(deviceBuffers.at(name),
                                   hostBuffers.at(name).data(),
                                   elementCount * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (!context->enqueueV3(stream))
    {
        std::cerr << "TensorRT enqueueV3 failed\n";
        return EXIT_FAILURE;
    }
    for (char const* name : {names[2], names[3]})
    {
        CUDA_CHECK(cudaMemcpyAsync(hostBuffers.at(name).data(),
                                   deviceBuffers.at(name),
                                   elementCount * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float const tolerance{1.0e-5F};
    for (size_t i{0U}; i < elementCount; ++i)
    {
        if (std::abs(hostBuffers.at("real_output")[i] - 2.0F) > tolerance
            || std::abs(hostBuffers.at("imag_output")[i] + 0.5F)
                > tolerance)
        {
            std::cerr << "FFT verification failed at element " << i << ": "
                      << hostBuffers.at("real_output")[i] << " + j("
                      << hostBuffers.at("imag_output")[i] << ")\n";
            return EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto const& allocation : deviceBuffers)
    {
        CUDA_CHECK(cudaFree(allocation.second));
    }
    std::cout << "FFT1D end-to-end verification passed for "
              << elementCount / fftLength << " transforms of length "
              << fftLength << ".\n";
    return EXIT_SUCCESS;
}
