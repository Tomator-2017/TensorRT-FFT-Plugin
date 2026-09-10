#ifndef TENSORRT_FFT_PLUGIN_KERNELS_H
#define TENSORRT_FFT_PLUGIN_KERNELS_H

#include <cstddef>

#include <cuda_runtime.h>
#include <cufft.h>

// Convert between TensorRT's two planar FP32 tensors and cuFFT's interleaved
// cufftComplex representation. All pointers refer to device memory.
cudaError_t launchPackComplex(float const* realInput, float const* imagInput,
                              cufftComplex* complexOutput, size_t count,
                              cudaStream_t stream);

cudaError_t launchUnpackComplex(cufftComplex const* complexInput,
                                float* realOutput, float* imagOutput,
                                size_t count, float scale,
                                cudaStream_t stream);

#endif // TENSORRT_FFT_PLUGIN_KERNELS_H
