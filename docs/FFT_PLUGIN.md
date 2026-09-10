# TensorRT 8.6 FFT 插件说明

## 插件接口

动态库 `libtrt_fft_plugins.so` 注册两个独立的 TensorRT Plugin Creator：

| Plugin | 变换范围 | 批处理方式 |
|---|---|---|
| `FFT1D` | 最后一维 `N` | 其余维度乘积为 batch count |
| `FFT2D` | 最后两维 `H×W` | 其余维度乘积为 batch count |

两个插件的 TensorRT 接口完全采用平面 FP32 张量：

```text
real_input  (FP32) ---\                      /--> real_output (FP32)
                       CUDA pack + cuFFT C2C
imag_input  (FP32) ---/                      \--> imag_output (FP32)
```

- 输入 0：复数输入的实部。
- 输入 1：复数输入的虚部。
- 输出 0：变换结果的实部。
- 输出 1：变换结果的虚部。
- 四个张量必须具有相同的静态 shape，格式为 `FP32 + LINEAR`。
- TensorRT 网络中不会出现复数数据类型。

两个 Creator 都公开以下 `PluginField`：

- `axis=-1`：变换维度位于张量末尾，当前只接受 `-1`。
- `inverse=0/1`：选择 `CUFFT_FORWARD` 或 `CUFFT_INVERSE`。
- `normalize=0/1`：是否对输出乘 `1/N` 或 `1/(H*W)`。

变换 rank 由 Plugin 类型决定，不再作为公开字段：`FFT1D` 固定为 1，`FFT2D`
固定为 2。参数、固定 shape 和 batch count 会被序列化到 TensorRT engine，
反序列化时 Creator 还会校验序列化数据中的 rank 是否与自身匹配。

## GPU 执行路径

插件的 `enqueue()` 在 TensorRT 提供的 CUDA stream 上依次执行：

1. CUDA pack kernel 把实部、虚部两个 FP32 tensor 合并为 workspace 中的
   `cufftComplex`。
2. `cufftExecC2C` 原位执行批量一维或二维变换。
3. CUDA unpack kernel 将结果拆成两个 FP32 输出，并按配置完成归一化。

cuFFT auto-allocation 已关闭。中间复数 buffer 和 cuFFT work area 都来自
TensorRT workspace，因此插件执行路径不包含 `cudaMalloc`、CPU FFT 或
Host/Device copy。cuFFT plan 在 `initialize()` 中创建，并由每个 plugin clone
独立持有，避免不同 execution context 共享 plan 和 stream 状态。

## 固定环境

本项目只以以下 NGC 镜像为构建和运行环境：

```text
nvcr.io/nvidia/pytorch:24.02-py3
```

在宿主机仓库根目录执行：

```bash
docker run -it --rm --gpus device=0 \
  -v "$(pwd)":/mnt -w /mnt \
  nvcr.io/nvidia/pytorch:24.02-py3
```

容器内编译：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

主要输出为：

```text
build/plugins/libtrt_fft_plugins.so
build/examples/build_fft1d_engine
build/examples/run_fft1d_engine
build/examples/build_fft2d_engine
build/examples/run_fft2d_engine
```

也可以构建仓库提供的派生镜像：

```bash
docker build -f docker/Dockerfile -t tensorrt-fft:8.6 .
```

## FFT1D 导出与端到端验证

导出 ONNX：

```bash
python -m examples.fft1d.export_onnx
```

默认输入 shape 为 `[2,3,16]`，生成 `artifacts/fft1d.onnx`。ONNX 中的自定义
节点为 `trt.plugins::FFT1D`。可以修改静态 shape：

```bash
python -m examples.fft1d.export_onnx \
  --batch 4 --channels 8 --length 1024
```

构建和运行 engine：

```bash
./build/examples/build_fft1d_engine
./build/examples/run_fft1d_engine
```

C++ runtime 使用冲激序列验证输出：每个序列 `x[0]=2-0.5j`，其余为零，未归一化
正向 FFT 的所有频点应为 `2-0.5j`。

## FFT2D 导出与端到端验证

`examples/fft2d/` 是不包含 CNN 的纯二维插件示例。输入为独立的实部、虚部
FP32 tensor，默认 shape 为 `[2,3,16,16]`：

```bash
python -m examples.fft2d.export_onnx
./build/examples/build_fft2d_engine
./build/examples/run_fft2d_engine
```

对应生成物为：

```text
artifacts/fft2d.onnx
artifacts/fft2d.engine
```

导出其他静态二维 shape：

```bash
python -m examples.fft2d.export_onnx \
  --batch 4 --channels 8 --height 128 --width 128
```

C++ runtime 为每个 `H×W` 平面的 `(0,0)` 设置 `2-0.5j` 冲激，验证二维 FFT
所有频点均为 `2-0.5j`。

## CNN + FFT2D benchmark

CNN 性能示例与纯 FFT2D 验证完全分开，位于 `examples/cnn_fft2d/`。模型结构为：

```text
BCHW image -> Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> Conv1x1 -> FFT2D
```

CNN 输出作为 FFT 实部，虚部由 `zeros_like` 生成。模型额外输出 FFT 前的 CNN
特征，使 benchmark 可以将 CNN tactic 误差和 FFT2D 插件误差分开统计。

先独立导出 ONNX 和相同权重的 PyTorch checkpoint：

```bash
python -m examples.cnn_fft2d.export_onnx
```

默认生成：

```text
artifacts/fft2d_cnn.onnx
artifacts/fft2d_cnn.pth
```

再运行独立 benchmark：

```bash
python -m examples.cnn_fft2d.benchmark
```

默认输入为 `[1,3,128,128]`，CNN 输出 16 通道，预热 30 次并测量 100 次。
修改网络 shape 时要重新生成 ONNX 和 checkpoint：

```bash
python -m examples.cnn_fft2d.export_onnx \
  --batch 4 --height 256 --width 256 --features 32
python -m examples.cnn_fft2d.benchmark --warmup 50 --iterations 500
```

允许 TensorRT 为 CNN 选择 FP16 tactic：

```bash
python -m examples.cnn_fft2d.benchmark --fp16
```

benchmark 将 PyTorch CUDA tensor 地址直接绑定给 TensorRT context，计时区间不
包含 H2D/D2H，并报告：

- `FFT plugin-only accuracy`：对 TensorRT CNN 实际输出分别执行 FFT2D Plugin
  和 `torch.fft.fft2`，只衡量 pack/cuFFT/unpack 误差。
- `End-to-end accuracy`：比较完整 PyTorch CNN+FFT2D 与 TensorRT 网络。
- PyTorch eager 与 TensorRT 的 CUDA event 平均延迟、吞吐量和加速比。

结果会受到 GPU 型号、频率、TensorRT tactic、shape 和热状态影响，应在固定环境
下多次测量。这里的 PyTorch 基线是 eager mode，不代表 `torch.compile` 的性能。

## PyTorch 与 ONNX 映射

`trt_fft/ops.py` 注册两个 PyTorch 自定义算子：

```python
real_y, imag_y = torch.ops.trt_custom.fft1d(real_x, imag_x, 0, 0)
real_y, imag_y = torch.ops.trt_custom.fft2d(real_x, imag_x, 0, 0)
```

它们在 PyTorch eager 模式下调用 `torch.fft` 作为参考语义；ONNX 导出时分别映射
为 `trt.plugins::FFT1D` 和 `trt.plugins::FFT2D`。TensorRT ONNX parser 加载
`libtrt_fft_plugins.so` 后，根据节点类型选择对应 Creator。

## 已知限制

- 仅支持静态 shape、FP32、LINEAR 和 C2C 变换。
- `FFT1D` 只变换最后一维；`FFT2D` 只变换最后两维。
- 当前没有实现动态 shape 下的 plan cache。
- 插件只在 GPU 上运行，不能交给 DLA。
- `.engine` 包含设备相关 tactic，不应跨 GPU 或 TensorRT 版本复用。

## 源码索引

- `plugins/FFTPlugin.*`：公共 `IPluginV2DynamicExt`、cuFFT plan、workspace、
  enqueue 和序列化。
- `plugins/FFTPluginCreator.*`：`FFT1DPluginCreator`、`FFT2DPluginCreator` 与
  PluginField 解析。
- `plugins/FFTPluginRegistration.cpp`：动态库 Creator 导出入口。
- `plugins/FFTPluginKernels.*`：平面 FP32 与 `cufftComplex` 的 GPU 转换。
- `trt_fft/ops.py`：`torch.ops` eager 实现与 ONNX symbolic。
- `examples/fft1d/`：一维导出、engine 构建和 C++ 验证。
- `examples/fft2d/`：纯二维导出、engine 构建和 C++ 验证。
- `examples/cnn_fft2d/`：独立 CNN+FFT2D 模型和精度/性能 benchmark。
