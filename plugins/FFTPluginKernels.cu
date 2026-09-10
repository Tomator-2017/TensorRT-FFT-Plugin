#include "FFTPluginKernels.h"

#include <algorithm>
#include <cstdint>

namespace
{
constexpr int32_t kThreadsPerBlock{256};
constexpr int32_t kMaxBlocks{65535};

__global__ void packComplexKernel(float const* realInput,
                                  float const* imagInput,
                                  cufftComplex* complexOutput, size_t count)
{
    size_t const stride{static_cast<size_t>(blockDim.x) * gridDim.x};
    for (size_t index{static_cast<size_t>(blockIdx.x) * blockDim.x
                      + threadIdx.x};
         index < count; index += stride)
    {
        complexOutput[index].x = realInput[index];
        complexOutput[index].y = imagInput[index];
    }
}

__global__ void unpackComplexKernel(cufftComplex const* complexInput,
                                    float* realOutput, float* imagOutput,
                                    size_t count, float scale)
{
    size_t const stride{static_cast<size_t>(blockDim.x) * gridDim.x};
    for (size_t index{static_cast<size_t>(blockIdx.x) * blockDim.x
                      + threadIdx.x};
         index < count; index += stride)
    {
        realOutput[index] = complexInput[index].x * scale;
        imagOutput[index] = complexInput[index].y * scale;
    }
}

int32_t getBlockCount(size_t count)
{
    size_t const blocks{(count + kThreadsPerBlock - 1U) / kThreadsPerBlock};
    return static_cast<int32_t>(std::min<size_t>(blocks, kMaxBlocks));
}
} // namespace

cudaError_t launchPackComplex(float const* realInput, float const* imagInput,
                              cufftComplex* complexOutput, size_t count,
                              cudaStream_t stream)
{
    if (count == 0U)
    {
        return cudaSuccess;
    }
    packComplexKernel<<<getBlockCount(count), kThreadsPerBlock, 0, stream>>>(
        realInput, imagInput, complexOutput, count);
    return cudaGetLastError();
}

cudaError_t launchUnpackComplex(cufftComplex const* complexInput,
                                float* realOutput, float* imagOutput,
                                size_t count, float scale,
                                cudaStream_t stream)
{
    if (count == 0U)
    {
        return cudaSuccess;
    }
    unpackComplexKernel<<<getBlockCount(count), kThreadsPerBlock, 0, stream>>>(
        complexInput, realOutput, imagOutput, count, scale);
    return cudaGetLastError();
}
