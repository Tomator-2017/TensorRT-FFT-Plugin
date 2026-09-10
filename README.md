# TensorRT 8.6 FFT1D / FFT2D Plugins

基于 cuFFT 的 TensorRT 8.6 自定义傅里叶变换插件。项目提供两个独立的
TensorRT Plugin Creator：

- `FFT1D`：对输入张量最后一维执行批量一维 FFT/IFFT。
- `FFT2D`：对 BCHW 等张量最后两维执行批量二维 FFT/IFFT。

TensorRT 不支持复数张量，因此两个插件都使用两个 FP32 输入表示实部和虚部，
并输出两个 FP32 张量。pack、cuFFT 和 unpack 全部在同一 CUDA stream 上完成，
插件执行期间没有 CPU FFT 或 Host/Device 数据拷贝。

项目统一使用 `nvcr.io/nvidia/pytorch:24.02-py3`，其中包含本项目需要的
TensorRT 8.6、CUDA、cuFFT、PyTorch 和 ONNX 环境。

## 目录结构

```text
.
├── plugins/              # FFT1D/FFT2D TensorRT Creator 与共享 CUDA/cuFFT 实现
├── trt_fft/              # torch.ops 自定义算子及 ONNX symbolic
├── examples/
│   ├── fft1d/            # 一维 ONNX 导出、engine 构建和 C++ 推理验证
│   ├── fft2d/            # 二维 ONNX 导出、engine 构建和 C++ 推理验证
│   └── cnn_fft2d/        # 独立 CNN+FFT2D 模型及 PyTorch/TensorRT benchmark
├── tests/                # PyTorch eager 参考语义测试
├── artifacts/            # 运行时生成的 ONNX、checkpoint 和 engine，不提交 Git
├── docs/                 # 插件接口和实现说明
└── docker/Dockerfile     # 唯一支持的开发环境
```

## 快速开始

在仓库根目录启动容器：

```bash
docker run -it --rm --gpus device=0 \
  -v "$(pwd)":/mnt -w /mnt \
  nvcr.io/nvidia/pytorch:24.02-py3
```

容器内编译插件和 C++ 示例：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

验证 `FFT1D`：

```bash
python -m examples.fft1d.export_onnx
./build/examples/build_fft1d_engine
./build/examples/run_fft1d_engine
```

验证 `FFT2D`：

```bash
python -m examples.fft2d.export_onnx
./build/examples/build_fft2d_engine
./build/examples/run_fft2d_engine
```

独立导出 CNN+`FFT2D` 并比较 PyTorch 与 TensorRT：

```bash
python -m examples.cnn_fft2d.export_onnx
python -m examples.cnn_fft2d.benchmark
```

CPU 侧 PyTorch reference 单元测试：

```bash
python -m unittest discover -s tests -v
```

二维 benchmark 报告 FFT2D 插件误差、完整 CNN+FFT2D 误差、GPU 延迟、吞吐量
和 TensorRT 相对 PyTorch eager 的加速比。详细设计和参数说明见
[docs/FFT_PLUGIN.md](docs/FFT_PLUGIN.md)。

## 限制

- TensorRT 输入输出只支持 `FP32 + LINEAR`。
- 实部和虚部必须形状一致。
- 当前版本仅支持静态 shape，cuFFT plan 在插件初始化阶段创建。
- `FFT1D` 固定变换最后一维；`FFT2D` 固定变换最后两维。
- TensorRT engine 和插件动态库必须在上述容器中构建与运行。
