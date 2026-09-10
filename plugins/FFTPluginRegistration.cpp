// Dynamic plugin-library entry points. TensorRT 8.6 looks for
// getPluginCreators; newer TensorRT releases use getCreators. Exporting both
// keeps the same library directly loadable in either environment.

#include <cstdint>
#include <mutex>

#include <NvInferRuntime.h>

#include "FFTPluginCreator.h"

namespace
{
class ThreadSafeLoggerFinder
{
public:
    void set(nvinfer1::ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lock{mMutex};
        if (mFinder == nullptr && finder != nullptr)
        {
            mFinder = finder;
        }
    }

private:
    nvinfer1::ILoggerFinder* mFinder{nullptr};
    std::mutex mMutex{};
};

ThreadSafeLoggerFinder gLoggerFinder{};

nvinfer1::IPluginCreator* const* creators(int32_t& count)
{
    count = 2;
    static nvinfer1::plugin::FFT1DPluginCreator fft1dCreator{};
    static nvinfer1::plugin::FFT2DPluginCreator fft2dCreator{};
    static nvinfer1::IPluginCreator* const list[]{&fft1dCreator,
                                                  &fft2dCreator};
    return list;
}
} // namespace

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    gLoggerFinder.set(finder);
}

extern "C" nvinfer1::IPluginCreator* const*
getPluginCreators(int32_t& count)
{
    return creators(count);
}

extern "C" nvinfer1::IPluginCreator* const* getCreators(int32_t& count)
{
    return creators(count);
}
