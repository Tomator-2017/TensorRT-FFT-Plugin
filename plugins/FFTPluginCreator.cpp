#include "FFTPluginCreator.h"

#include <cstring>
#include <new>

#include "FFTPlugin.h"

namespace nvinfer1
{
namespace plugin
{

REGISTER_TENSORRT_PLUGIN(FFT1DPluginCreator);
REGISTER_TENSORRT_PLUGIN(FFT2DPluginCreator);

FFTPluginCreator::FFTPluginCreator(char const* pluginName,
                                   int32_t transformRank)
    : mPluginName{pluginName}, mTransformRank{transformRank}
{
    // These names must match attributes emitted by the PyTorch ONNX symbolic.
    mFields.emplace_back("axis", nullptr,
                         nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("inverse", nullptr,
                         nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("normalize", nullptr,
                         nvinfer1::PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

char const* FFTPluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

char const* FFTPluginCreator::getPluginVersion() const noexcept
{
    return kFFT_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const*
FFTPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

nvinfer1::IPluginV2* FFTPluginCreator::createPlugin(
    char const*,
    nvinfer1::PluginFieldCollection const* fieldCollection) noexcept
{
    FFTPluginParameters parameters{};
    parameters.rank = mTransformRank;
    if (fieldCollection != nullptr)
    {
        if (fieldCollection->nbFields < 0
            || (fieldCollection->nbFields > 0
                && fieldCollection->fields == nullptr))
        {
            return nullptr;
        }
        for (int32_t i{0}; i < fieldCollection->nbFields; ++i)
        {
            nvinfer1::PluginField const& field{fieldCollection->fields[i]};
            if (field.name == nullptr || field.data == nullptr
                || field.length != 1
                || field.type != nvinfer1::PluginFieldType::kINT32)
            {
                return nullptr;
            }
            int32_t const value{*static_cast<int32_t const*>(field.data)};
            if (std::strcmp(field.name, "axis") == 0)
            {
                parameters.axis = value;
            }
            else if (std::strcmp(field.name, "inverse") == 0)
            {
                parameters.inverse = value;
            }
            else if (std::strcmp(field.name, "normalize") == 0)
            {
                parameters.normalize = value;
            }
            else
            {
                return nullptr;
            }
        }
    }

    if (parameters.axis != -1
        || (parameters.inverse != 0 && parameters.inverse != 1)
        || (parameters.normalize != 0 && parameters.normalize != 1))
    {
        return nullptr;
    }

    FFTPlugin* plugin{new (std::nothrow) FFTPlugin{parameters}};
    if (plugin != nullptr)
    {
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    return plugin;
}

nvinfer1::IPluginV2* FFTPluginCreator::deserializePlugin(
    char const*, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        FFTPlugin* plugin{new FFTPlugin{serialData, serialLength}};
        if (plugin->getTransformRank() != mTransformRank)
        {
            delete plugin;
            return nullptr;
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

void FFTPluginCreator::setPluginNamespace(
    char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* FFTPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FFT1DPluginCreator::FFT1DPluginCreator()
    : FFTPluginCreator{kFFT1D_PLUGIN_NAME, 1}
{
}

FFT2DPluginCreator::FFT2DPluginCreator()
    : FFTPluginCreator{kFFT2D_PLUGIN_NAME, 2}
{
}

} // namespace plugin
} // namespace nvinfer1
