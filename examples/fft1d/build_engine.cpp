// Build a TensorRT 8.6 engine from an ONNX graph containing the FFT1D plugin
// node. Usage:
//   build_fft1d_engine [model.onnx] [model.engine] [plugin-library.so]

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace
{
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
} // namespace

int main(int argc, char** argv)
{
    std::string const onnxPath{argc > 1 ? argv[1]
                                       : "artifacts/fft1d.onnx"};
    std::string const enginePath{argc > 2 ? argv[2]
                                         : "artifacts/fft1d.engine"};
    std::string const pluginPath{argc > 3 ? argv[3]
                                         : "build/plugins/libtrt_fft_plugins.so"};
    Logger logger{};

    std::unique_ptr<nvinfer1::IBuilder, InferDeleter> builder{
        nvinfer1::createInferBuilder(logger)};
    if (!builder)
    {
        std::cerr << "Failed to create TensorRT builder\n";
        return EXIT_FAILURE;
    }

    // Register FFT1DPluginCreator before the parser sees the custom node.
    void* pluginHandle{builder->getPluginRegistry().loadLibrary(
        pluginPath.c_str())};
    if (pluginHandle == nullptr)
    {
        std::cerr << "Failed to load plugin library: " << pluginPath << '\n';
        return EXIT_FAILURE;
    }

    uint32_t const explicitBatch{
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> network{
        builder->createNetworkV2(explicitBatch)};
    if (!network)
    {
        std::cerr << "Failed to create TensorRT network\n";
        return EXIT_FAILURE;
    }
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser{
        nvonnxparser::createParser(*network, logger)};
    if (!parser
        || !parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "Failed to parse ONNX model: " << onnxPath << '\n';
        if (parser)
        {
            for (int32_t i{0}; i < parser->getNbErrors(); ++i)
            {
                std::cerr << parser->getError(i)->desc() << '\n';
            }
        }
        return EXIT_FAILURE;
    }

    if (network->getNbInputs() != 2 || network->getNbOutputs() != 2)
    {
        std::cerr << "FFT1D demo expects two FP32 inputs and outputs\n";
        return EXIT_FAILURE;
    }
    uint32_t const linearFormat{
        1U << static_cast<uint32_t>(nvinfer1::TensorFormat::kLINEAR)};
    for (int32_t i{0}; i < network->getNbInputs(); ++i)
    {
        network->getInput(i)->setType(nvinfer1::DataType::kFLOAT);
        network->getInput(i)->setAllowedFormats(linearFormat);
    }
    for (int32_t i{0}; i < network->getNbOutputs(); ++i)
    {
        network->getOutput(i)->setType(nvinfer1::DataType::kFLOAT);
        network->getOutput(i)->setAllowedFormats(linearFormat);
    }

    std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter> config{
        builder->createBuilderConfig()};
    if (!config)
    {
        std::cerr << "Failed to create TensorRT builder config\n";
        return EXIT_FAILURE;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               1ULL << 30U);

    std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> serializedEngine{
        builder->buildSerializedNetwork(*network, *config)};
    if (!serializedEngine)
    {
        std::cerr << "Failed to build FFT1D TensorRT engine\n";
        return EXIT_FAILURE;
    }

    std::ofstream output{enginePath, std::ios::binary};
    if (!output)
    {
        std::cerr << "Failed to open engine output: " << enginePath << '\n';
        return EXIT_FAILURE;
    }
    output.write(static_cast<char const*>(serializedEngine->data()),
                 static_cast<std::streamsize>(serializedEngine->size()));
    if (!output)
    {
        std::cerr << "Failed to write engine: " << enginePath << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "Serialized FFT1D engine to " << enginePath << '\n';
    return EXIT_SUCCESS;
}
