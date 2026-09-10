# One shared library exports two TensorRT creators: FFT1D and FFT2D. Both use
# planar FP32 TensorRT tensors and share the same CUDA/cuFFT implementation.
add_library(
    trt_fft_plugins
    SHARED
    FFTPlugin.cpp
    FFTPluginCreator.cpp
    FFTPluginKernels.cu
    FFTPluginRegistration.cpp
)
target_include_directories(
    trt_fft_plugins
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(
    trt_fft_plugins
    PRIVATE
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    CUDA::cudart
    CUDA::cufft
)
set_target_properties(
    trt_fft_plugins
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS trt_fft_plugins LIBRARY DESTINATION lib)
