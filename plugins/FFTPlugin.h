#ifndef TENSORRT_FFT_PLUGIN_H
#define TENSORRT_FFT_PLUGIN_H

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cufft.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

constexpr char const* kFFT1D_PLUGIN_NAME{"FFT1D"};
constexpr char const* kFFT2D_PLUGIN_NAME{"FFT2D"};
constexpr char const* kFFT_PLUGIN_VERSION{"1"};

namespace nvinfer1
{
namespace plugin
{

// TensorRT never sees a complex tensor. Input 0/1 are the real/imaginary
// float tensors; output 0/1 are the transformed real/imaginary float tensors.
struct FFTPluginParameters
{
    int32_t axis{-1};
    int32_t inverse{0};
    int32_t normalize{0};
    // Number of trailing dimensions transformed by cuFFT: 1 or 2.
    int32_t rank{1};
};

class FFTPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    explicit FFTPlugin(FFTPluginParameters parameters);
    FFTPlugin(void const* serialData, size_t serialLength);
    ~FFTPlugin() override;

    int32_t getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    int32_t getTransformRank() const noexcept;

private:
    FFTPlugin(FFTPluginParameters parameters, int32_t fftDimension0,
              int32_t fftDimension1, int32_t batchCount);

    FFTPluginParameters mParameters{};
    // rank=1: dimension0=N, dimension1=1.
    // rank=2: dimension0=H, dimension1=W.
    int32_t mFFTDimension0{0};
    int32_t mFFTDimension1{1};
    int32_t mBatchCount{0};
    bool mConfigurationValid{false};
    cufftHandle mPlan{};
    bool mPlanCreated{false};
    size_t mCufftWorkspaceSize{0};
    std::string mNamespace{};
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_FFT_PLUGIN_H
