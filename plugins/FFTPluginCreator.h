#ifndef TENSORRT_FFT_PLUGIN_CREATOR_H
#define TENSORRT_FFT_PLUGIN_CREATOR_H

#include <cstdint>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace nvinfer1
{
namespace plugin
{

class FFTPluginCreator : public nvinfer1::IPluginCreator
{
public:
    FFTPluginCreator(char const* pluginName, int32_t transformRank);
    ~FFTPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fieldCollection) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::vector<nvinfer1::PluginField> mFields{};
    nvinfer1::PluginFieldCollection mFieldCollection{};
    std::string mPluginName{};
    std::string mNamespace{};
    int32_t mTransformRank{};
};

class FFT1DPluginCreator final : public FFTPluginCreator
{
public:
    FFT1DPluginCreator();
};

class FFT2DPluginCreator final : public FFTPluginCreator
{
public:
    FFT2DPluginCreator();
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_FFT_PLUGIN_CREATOR_H
