add_executable(build_fft1d_engine fft1d/build_engine.cpp)
target_include_directories(build_fft1d_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(
    build_fft1d_engine
    PRIVATE
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    CUDA::cudart
)

add_executable(run_fft1d_engine fft1d/run_engine.cpp)
target_include_directories(run_fft1d_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(
    run_fft1d_engine
    PRIVATE
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    CUDA::cudart
)

add_executable(build_fft2d_engine fft2d/build_engine.cpp)
target_include_directories(build_fft2d_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(
    build_fft2d_engine
    PRIVATE
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    CUDA::cudart
)

add_executable(run_fft2d_engine fft2d/run_engine.cpp)
target_include_directories(run_fft2d_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(
    run_fft2d_engine
    PRIVATE
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    CUDA::cudart
)

install(
    TARGETS
        build_fft1d_engine
        run_fft1d_engine
        build_fft2d_engine
        run_fft2d_engine
    RUNTIME DESTINATION bin
)
