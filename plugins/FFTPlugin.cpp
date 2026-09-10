#include "FFTPlugin.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

#include "FFTPluginKernels.h"

namespace nvinfer1
{
namespace plugin
{
namespace
{
constexpr uint32_t kSerializationMagic{0x46544654U}; // "FTFT"
constexpr int32_t kSerializationVersion{2};
constexpr size_t kWorkspaceAlignment{256U};
constexpr size_t kLegacySerializationSize{sizeof(uint32_t)
                                          + 6U * sizeof(int32_t)};

int32_t reportRuntimeError(char const* stage, int32_t status) noexcept
{
    std::fprintf(stderr, "FFTPlugin failure at %s (status=%d)\n", stage,
                 status);
    return 1;
}

template <typename T>
void writeValue(char*& cursor, T const& value)
{
    std::memcpy(cursor, &value, sizeof(T));
    cursor += sizeof(T);
}

template <typename T>
T readValue(char const*& cursor, char const* end)
{
    if (static_cast<size_t>(end - cursor) < sizeof(T))
    {
        throw std::runtime_error("Truncated FFT plugin serialization data");
    }
    T value{};
    std::memcpy(&value, cursor, sizeof(T));
    cursor += sizeof(T);
    return value;
}

size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

bool getConcreteShape(nvinfer1::Dims const& dims, int32_t rank,
                      int32_t& fftDimension0, int32_t& fftDimension1,
                      int32_t& batchCount)
{
    if ((rank != 1 && rank != 2) || dims.nbDims < rank)
    {
        return false;
    }

    int64_t batches{1};
    for (int32_t i{0}; i < dims.nbDims - rank; ++i)
    {
        if (dims.d[i] <= 0)
        {
            return false;
        }
        batches *= dims.d[i];
        if (batches > std::numeric_limits<int32_t>::max())
        {
            return false;
        }
    }

    fftDimension0 = dims.d[dims.nbDims - rank];
    fftDimension1 = rank == 2 ? dims.d[dims.nbDims - 1] : 1;
    if (fftDimension0 <= 0 || fftDimension1 <= 0)
    {
        return false;
    }
    int64_t const transformElements{static_cast<int64_t>(fftDimension0)
                                    * fftDimension1};
    // cufftPlanMany uses an int transform distance.
    if (transformElements > std::numeric_limits<int32_t>::max())
    {
        return false;
    }
    if (static_cast<uint64_t>(batches)
        > std::numeric_limits<size_t>::max()
            / (static_cast<size_t>(transformElements) * sizeof(cufftComplex)))
    {
        return false;
    }
    batchCount = static_cast<int32_t>(batches);
    return true;
}

bool sameShape(nvinfer1::Dims const& lhs, nvinfer1::Dims const& rhs)
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t i{0}; i < lhs.nbDims; ++i)
    {
        if (lhs.d[i] != rhs.d[i])
        {
            return false;
        }
    }
    return true;
}
} // namespace

FFTPlugin::FFTPlugin(FFTPluginParameters parameters)
    : mParameters{parameters}
{
}

FFTPlugin::FFTPlugin(FFTPluginParameters parameters, int32_t fftDimension0,
                     int32_t fftDimension1, int32_t batchCount)
    : mParameters{parameters}, mFFTDimension0{fftDimension0},
      mFFTDimension1{fftDimension1}, mBatchCount{batchCount},
      mConfigurationValid{fftDimension0 > 0 && fftDimension1 > 0
                          && batchCount > 0}
{
}

FFTPlugin::FFTPlugin(void const* serialData, size_t serialLength)
{
    if (serialData == nullptr
        || (serialLength != getSerializationSize()
            && serialLength != kLegacySerializationSize))
    {
        throw std::runtime_error("Invalid FFT plugin serialization size");
    }

    char const* cursor{static_cast<char const*>(serialData)};
    char const* const end{cursor + serialLength};
    uint32_t const magic{readValue<uint32_t>(cursor, end)};
    int32_t const version{readValue<int32_t>(cursor, end)};
    if (magic != kSerializationMagic || (version != 1 && version != 2))
    {
        throw std::runtime_error("Unsupported FFT plugin serialization format");
    }

    mParameters.axis = readValue<int32_t>(cursor, end);
    mParameters.inverse = readValue<int32_t>(cursor, end);
    mParameters.normalize = readValue<int32_t>(cursor, end);
    if (version == 2)
    {
        mParameters.rank = readValue<int32_t>(cursor, end);
        mFFTDimension0 = readValue<int32_t>(cursor, end);
        mFFTDimension1 = readValue<int32_t>(cursor, end);
    }
    else
    {
        mParameters.rank = 1;
        mFFTDimension0 = readValue<int32_t>(cursor, end);
        mFFTDimension1 = 1;
    }
    mBatchCount = readValue<int32_t>(cursor, end);
    mConfigurationValid = mParameters.axis == -1
        && (mParameters.inverse == 0 || mParameters.inverse == 1)
        && (mParameters.normalize == 0 || mParameters.normalize == 1)
        && (mParameters.rank == 1 || mParameters.rank == 2)
        && mFFTDimension0 > 0 && mFFTDimension1 > 0 && mBatchCount > 0
        && cursor == end;
    if (!mConfigurationValid)
    {
        throw std::runtime_error("Invalid serialized FFT plugin parameters");
    }
}

FFTPlugin::~FFTPlugin()
{
    terminate();
}

int32_t FFTPlugin::getNbOutputs() const noexcept
{
    return 2;
}

nvinfer1::DimsExprs FFTPlugin::getOutputDimensions(
    int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder&) noexcept
{
    if (inputs == nullptr || nbInputs != 2 || outputIndex < 0
        || outputIndex >= 2)
    {
        return nvinfer1::DimsExprs{-1, {}};
    }
    // Both FFT outputs preserve the input shape.
    return inputs[outputIndex];
}

bool FFTPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != 2 || nbOutputs != 2 || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    // cuFFT is fed through an internal cufftComplex buffer, while every
    // TensorRT-visible tensor remains planar FP32 and LINEAR.
    return inOut[pos].type == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
        && (pos == 0
            || (inOut[pos].type == inOut[0].type
                && inOut[pos].format == inOut[0].format));
}

void FFTPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept
{
    mConfigurationValid = false;
    if (inputs == nullptr || outputs == nullptr || nbInputs != 2
        || nbOutputs != 2 || mParameters.axis != -1
        || (mParameters.rank != 1 && mParameters.rank != 2)
        || !sameShape(inputs[0].desc.dims, inputs[1].desc.dims)
        || !sameShape(inputs[0].desc.dims, outputs[0].desc.dims)
        || !sameShape(inputs[0].desc.dims, outputs[1].desc.dims))
    {
        return;
    }

    int32_t fftDimension0{};
    int32_t fftDimension1{};
    int32_t batchCount{};
    if (!getConcreteShape(inputs[0].desc.dims, mParameters.rank,
                          fftDimension0, fftDimension1, batchCount))
    {
        // A cuFFT plan has a fixed transform length and batch count. This v1
        // implementation intentionally rejects wildcard dimensions instead of
        // creating plans in enqueue(), which would add allocation/sync jitter.
        return;
    }

    mFFTDimension0 = fftDimension0;
    mFFTDimension1 = fftDimension1;
    mBatchCount = batchCount;
    mConfigurationValid = true;
}

size_t FFTPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const*, int32_t nbOutputs) const noexcept
{
    if (!mConfigurationValid || nbInputs != 2 || nbOutputs != 2)
    {
        return 0U;
    }
    size_t const transformElements{static_cast<size_t>(mFFTDimension0)
                                   * static_cast<size_t>(mFFTDimension1)};
    size_t const elementCount{transformElements
                              * static_cast<size_t>(mBatchCount)};
    size_t const complexBytes{elementCount * sizeof(cufftComplex)};
    return alignUp(complexBytes, kWorkspaceAlignment) + mCufftWorkspaceSize;
}

int32_t FFTPlugin::initialize() noexcept
{
    if (mPlanCreated)
    {
        return 0;
    }
    if (!mConfigurationValid)
    {
        return 1;
    }

    if (cufftCreate(&mPlan) != CUFFT_SUCCESS)
    {
        return 1;
    }
    mPlanCreated = true;

    // TensorRT owns the complete execution workspace. Disabling cuFFT's auto
    // allocation avoids hidden cudaMalloc/cudaFree calls and lets enqueue use
    // the workspace pointer supplied by TensorRT.
    if (cufftSetAutoAllocation(mPlan, 0) != CUFFT_SUCCESS)
    {
        terminate();
        return 1;
    }

    int32_t dimensions[2]{mFFTDimension0, mFFTDimension1};
    int32_t const transformElements{mFFTDimension0 * mFFTDimension1};
    if (cufftMakePlanMany(mPlan, mParameters.rank, dimensions, nullptr, 1,
                          transformElements, nullptr, 1, transformElements,
                          CUFFT_C2C, mBatchCount,
                          &mCufftWorkspaceSize)
        != CUFFT_SUCCESS)
    {
        terminate();
        return 1;
    }
    return 0;
}

void FFTPlugin::terminate() noexcept
{
    if (mPlanCreated)
    {
        cufftDestroy(mPlan);
        mPlanCreated = false;
        mPlan = cufftHandle{};
        mCufftWorkspaceSize = 0U;
    }
}

int32_t FFTPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (!mPlanCreated || inputDesc == nullptr || inputs == nullptr
        || outputs == nullptr || workspace == nullptr)
    {
        int32_t const invalidMask{(!mPlanCreated ? 1 : 0)
                                  | (inputDesc == nullptr ? 2 : 0)
                                  | (inputs == nullptr ? 4 : 0)
                                  | (outputs == nullptr ? 8 : 0)
                                  | (workspace == nullptr ? 16 : 0)};
        return reportRuntimeError("enqueue argument/plan validation",
                                  invalidMask);
    }

    int32_t fftDimension0{};
    int32_t fftDimension1{};
    int32_t batchCount{};
    if (!getConcreteShape(inputDesc[0].dims, mParameters.rank,
                          fftDimension0, fftDimension1, batchCount)
        || fftDimension0 != mFFTDimension0
        || fftDimension1 != mFFTDimension1 || batchCount != mBatchCount
        || !sameShape(inputDesc[0].dims, inputDesc[1].dims))
    {
        return reportRuntimeError("runtime shape validation", 0);
    }

    size_t const transformElements{static_cast<size_t>(mFFTDimension0)
                                   * static_cast<size_t>(mFFTDimension1)};
    size_t const elementCount{transformElements
                              * static_cast<size_t>(mBatchCount)};
    size_t const complexBytes{elementCount * sizeof(cufftComplex)};
    size_t const cufftOffset{alignUp(complexBytes, kWorkspaceAlignment)};
    auto* complexBuffer{static_cast<cufftComplex*>(workspace)};
    auto* cufftWorkspace{static_cast<char*>(workspace) + cufftOffset};

    auto const* realInput{static_cast<float const*>(inputs[0])};
    auto const* imagInput{static_cast<float const*>(inputs[1])};
    auto* realOutput{static_cast<float*>(outputs[0])};
    auto* imagOutput{static_cast<float*>(outputs[1])};

    cudaError_t const packStatus{launchPackComplex(
        realInput, imagInput, complexBuffer, elementCount, stream)};
    if (packStatus != cudaSuccess)
    {
        return reportRuntimeError("pack CUDA kernel",
                                  static_cast<int32_t>(packStatus));
    }
    cufftResult const streamStatus{cufftSetStream(mPlan, stream)};
    if (streamStatus != CUFFT_SUCCESS)
    {
        return reportRuntimeError("cufftSetStream",
                                  static_cast<int32_t>(streamStatus));
    }
    cufftResult const workAreaStatus{cufftSetWorkArea(mPlan, cufftWorkspace)};
    if (workAreaStatus != CUFFT_SUCCESS)
    {
        return reportRuntimeError("cufftSetWorkArea",
                                  static_cast<int32_t>(workAreaStatus));
    }

    int32_t const direction{mParameters.inverse ? CUFFT_INVERSE
                                                : CUFFT_FORWARD};
    cufftResult const executeStatus{
        cufftExecC2C(mPlan, complexBuffer, complexBuffer, direction)};
    if (executeStatus != CUFFT_SUCCESS)
    {
        return reportRuntimeError("cufftExecC2C",
                                  static_cast<int32_t>(executeStatus));
    }

    float const scale{mParameters.normalize
                          ? 1.0F / static_cast<float>(transformElements)
                          : 1.0F};
    cudaError_t const unpackStatus{launchUnpackComplex(
        complexBuffer, realOutput, imagOutput, elementCount, scale, stream)};
    return unpackStatus == cudaSuccess
        ? 0
        : reportRuntimeError("unpack CUDA kernel",
                             static_cast<int32_t>(unpackStatus));
}

size_t FFTPlugin::getSerializationSize() const noexcept
{
    return sizeof(uint32_t) + 8U * sizeof(int32_t);
}

void FFTPlugin::serialize(void* buffer) const noexcept
{
    if (buffer == nullptr)
    {
        return;
    }
    char* cursor{static_cast<char*>(buffer)};
    writeValue(cursor, kSerializationMagic);
    writeValue(cursor, kSerializationVersion);
    writeValue(cursor, mParameters.axis);
    writeValue(cursor, mParameters.inverse);
    writeValue(cursor, mParameters.normalize);
    writeValue(cursor, mParameters.rank);
    writeValue(cursor, mFFTDimension0);
    writeValue(cursor, mFFTDimension1);
    writeValue(cursor, mBatchCount);
}

char const* FFTPlugin::getPluginType() const noexcept
{
    return mParameters.rank == 2 ? kFFT2D_PLUGIN_NAME : kFFT1D_PLUGIN_NAME;
}

char const* FFTPlugin::getPluginVersion() const noexcept
{
    return kFFT_PLUGIN_VERSION;
}

void FFTPlugin::destroy() noexcept
{
    delete this;
}

nvinfer1::IPluginV2DynamicExt* FFTPlugin::clone() const noexcept
{
    FFTPlugin* plugin{new (std::nothrow)
                          FFTPlugin{mParameters, mFFTDimension0,
                                    mFFTDimension1, mBatchCount}};
    if (plugin != nullptr)
    {
        plugin->setPluginNamespace(mNamespace.c_str());
        // TensorRT 8.6 can clone an already initialized plugin for an
        // execution context without calling initialize() again on that clone.
        // Give every clone its own cuFFT handle: sharing one handle would make
        // cufftSetStream/cufftSetWorkArea unsafe across execution contexts.
        if (mPlanCreated && plugin->initialize() != 0)
        {
            delete plugin;
            return nullptr;
        }
    }
    return plugin;
}

nvinfer1::DataType FFTPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept
{
    if (inputTypes == nullptr || nbInputs != 2 || index < 0 || index >= 2)
    {
        return nvinfer1::DataType::kFLOAT;
    }
    return inputTypes[index];
}

void FFTPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* FFTPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t FFTPlugin::getTransformRank() const noexcept
{
    return mParameters.rank;
}

} // namespace plugin
} // namespace nvinfer1
