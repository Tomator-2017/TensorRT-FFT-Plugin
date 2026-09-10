cmake_minimum_required(VERSION 3.20)

project(TensorRTFFTPlugins VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS ${TENSORRT_ROOT} /usr /usr/local
    PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu
    REQUIRED
)
set(TENSORRT_LIBRARY_HINTS
    ${TENSORRT_ROOT}
    /usr
    /usr/local
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
)
find_library(
    NVINFER_LIBRARY
    nvinfer
    HINTS ${TENSORRT_LIBRARY_HINTS}
    PATH_SUFFIXES lib lib64
    REQUIRED
)
find_library(
    NVINFER_PLUGIN_LIBRARY
    nvinfer_plugin
    HINTS ${TENSORRT_LIBRARY_HINTS}
    PATH_SUFFIXES lib lib64
    REQUIRED
)
find_library(
    NVONNXPARSER_LIBRARY
    nvonnxparser
    HINTS ${TENSORRT_LIBRARY_HINTS}
    PATH_SUFFIXES lib lib64
    REQUIRED
)

add_subdirectory(plugins)
add_subdirectory(examples)
