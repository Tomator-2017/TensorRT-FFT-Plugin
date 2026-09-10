"""Benchmark a CNN + 2-D FFT TensorRT graph against native PyTorch.

The TensorRT graph uses the FFT2D plugin backed by a batched 2-D cuFFT plan.
All TensorRT-visible tensors are real FP32 tensors; no complex TensorRT
type is used. PyTorch CUDA tensor addresses are bound directly to TensorRT, so
latency measurements exclude host/device copies.

First export the dedicated 2-D model, then run the benchmark inside the
nvcr.io/nvidia/pytorch:24.02-py3 container from the repository root:
    python -m examples.cnn_fft2d.export_onnx
    python -m examples.cnn_fft2d.benchmark
"""

import argparse
import ctypes
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import tensorrt as trt
import torch

from examples.cnn_fft2d.model import CNNFFT2D


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TensorRT FFT plugin accuracy and latency with PyTorch"
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--input-seed",
        type=int,
        default=2027,
        help="seed used only to generate benchmark input data",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="allow FP16 for CNN layers; FFT plugin I/O remains FP32",
    )
    parser.add_argument(
        "--plugin",
        default="build/plugins/libtrt_fft_plugins.so",
        help="path to libtrt_fft_plugins.so",
    )
    parser.add_argument("--onnx", default="artifacts/fft2d_cnn.onnx")
    parser.add_argument("--checkpoint", default="artifacts/fft2d_cnn.pth")
    parser.add_argument("--engine", default="artifacts/fft2d_cnn.engine")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be nonnegative and --iterations positive")


def load_plugin(plugin_path: Path) -> None:
    if not plugin_path.is_file():
        raise FileNotFoundError(
            f"Plugin not found: {plugin_path}. Build trt_fft_plugins first."
        )
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    ctypes.CDLL(str(plugin_path.resolve()), mode=mode)


def load_model_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run "
            "python -m examples.cnn_fft2d.export_onnx first."
        )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("format_version") != 1:
        raise RuntimeError("Unsupported FFT2D checkpoint format")

    input_shape = tuple(checkpoint["input_shape"])
    if len(input_shape) != 4 or any(d <= 0 for d in input_shape):
        raise RuntimeError(f"Invalid BCHW input shape in checkpoint: {input_shape}")
    input_channels = int(checkpoint["input_channels"])
    features = int(checkpoint["features"])
    if input_shape[1] != input_channels or input_channels <= 0 or features <= 0:
        raise RuntimeError("Inconsistent CNN dimensions in checkpoint")

    model = CNNFFT2D(input_channels, features).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model, input_shape, features


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool) -> bytes:
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")
    engine_bytes = bytes(serialized)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    return engine_bytes


def deserialize_engine(engine_bytes: bytes):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the engine")
    # Keep runtime alive as long as the engine.
    return runtime, engine


def make_trt_runner(engine, image: torch.Tensor):
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    expected_outputs = {"real_output", "imag_output", "cnn_features"}
    tensor_names = {
        engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
    }
    if "image" not in tensor_names or not expected_outputs.issubset(tensor_names):
        raise RuntimeError(f"Unexpected TensorRT I/O names: {sorted(tensor_names)}")

    engine_input_shape = tuple(engine.get_tensor_shape("image"))
    if engine_input_shape != tuple(image.shape):
        raise RuntimeError(
            "ONNX and checkpoint input shapes do not match: "
            f"engine={engine_input_shape}, checkpoint={tuple(image.shape)}. "
            "Export both files again with examples.cnn_fft2d.export_onnx."
        )
    if engine.get_tensor_dtype("image") != trt.DataType.FLOAT:
        raise RuntimeError("The FFT2D benchmark expects an FP32 image input")

    output_shape = tuple(engine.get_tensor_shape("real_output"))
    if any(d <= 0 for d in output_shape):
        raise RuntimeError("This FFT plugin benchmark requires static shapes")
    for name in expected_outputs:
        if tuple(engine.get_tensor_shape(name)) != output_shape:
            raise RuntimeError(f"TensorRT output shape mismatch for {name}")
        if engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
            raise RuntimeError(f"TensorRT output {name} must be FP32")
    real_output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    imag_output = torch.empty_like(real_output)
    cnn_features = torch.empty_like(real_output)

    bindings = {
        "image": image,
        "real_output": real_output,
        "imag_output": imag_output,
        "cnn_features": cnn_features,
    }
    for name, tensor in bindings.items():
        if not tensor.is_contiguous():
            raise RuntimeError(f"TensorRT binding {name} is not contiguous")
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Failed to bind TensorRT tensor {name}")

    stream_handle = torch.cuda.current_stream().cuda_stream

    def run():
        if not context.execute_async_v3(stream_handle=stream_handle):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return real_output, imag_output, cnn_features

    return context, run


def benchmark_cuda(
    function: Callable[[], object], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def error_metrics(
    actual_real: torch.Tensor,
    actual_imag: torch.Tensor,
    expected: torch.Tensor,
) -> Tuple[float, float, float]:
    actual = torch.complex(actual_real, actual_imag)
    difference = actual - expected
    max_abs = difference.abs().max().item()
    rmse = difference.abs().square().mean().sqrt().item()
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    ).item()
    return max_abs, rmse, relative_l2


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    repo_root = Path(__file__).resolve().parents[2]
    plugin_path = repo_root / args.plugin
    onnx_path = repo_root / args.onnx
    checkpoint_path = repo_root / args.checkpoint
    engine_path = repo_root / args.engine
    load_plugin(plugin_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Run "
            "python -m examples.cnn_fft2d.export_onnx first."
        )
    model, image_shape, feature_channels = load_model_checkpoint(
        checkpoint_path
    )
    torch.manual_seed(args.input_seed)
    torch.cuda.manual_seed_all(args.input_seed)
    image = torch.randn(image_shape, dtype=torch.float32)

    engine_bytes = build_engine(onnx_path, engine_path, args.fp16)

    model = model.cuda().eval()
    image = image.cuda().contiguous()
    runtime, engine = deserialize_engine(engine_bytes)
    context, trt_run = make_trt_runner(engine, image)

    def torch_run():
        features = model.cnn_forward(image)
        return torch.fft.fft2(features, dim=(-2, -1), norm="backward")

    with torch.inference_mode():
        expected_features = model.cnn_forward(image)
        expected = torch.fft.fft2(
            expected_features, dim=(-2, -1), norm="backward"
        )
        actual_real, actual_imag, actual_features = trt_run()
        torch.cuda.synchronize()
        (
            end_to_end_max_abs,
            end_to_end_rmse,
            end_to_end_relative_l2,
        ) = error_metrics(actual_real, actual_imag, expected)
        # This reference uses TensorRT's own CNN output as FFT input, isolating
        # the numerical error introduced specifically by pack + cuFFT + unpack.
        plugin_reference = torch.fft.fft2(
            actual_features, dim=(-2, -1), norm="backward"
        )
        plugin_max_abs, plugin_rmse, plugin_relative_l2 = error_metrics(
            actual_real, actual_imag, plugin_reference
        )
        cnn_relative_l2 = (
            torch.linalg.vector_norm(actual_features - expected_features)
            / torch.linalg.vector_norm(expected_features).clamp_min(1.0e-12)
        ).item()

        torch_ms = benchmark_cuda(torch_run, args.warmup, args.iterations)
        trt_ms = benchmark_cuda(trt_run, args.warmup, args.iterations)

    speedup = torch_ms / trt_ms
    batch_size, _, height, width = image_shape
    print("\nCNN + FFT2D benchmark")
    print(f"  Tensor shape       : {image_shape}")
    print(
        f"  CNN feature shape  : "
        f"{(batch_size, feature_channels, height, width)}"
    )
    print(f"  TensorRT CNN FP16  : {args.fp16}")
    print("  FFT plugin-only accuracy (same TensorRT CNN activation):")
    print(f"    Max absolute error : {plugin_max_abs:.6e}")
    print(f"    RMSE               : {plugin_rmse:.6e}")
    print(f"    Relative L2 error  : {plugin_relative_l2:.6e}")
    print("  End-to-end accuracy (PyTorch CNN+FFT2D vs TensorRT):")
    print(f"    Max absolute error : {end_to_end_max_abs:.6e}")
    print(f"    RMSE               : {end_to_end_rmse:.6e}")
    print(f"    Relative L2 error  : {end_to_end_relative_l2:.6e}")
    print(f"    CNN Relative L2    : {cnn_relative_l2:.6e}")
    print(f"  PyTorch CUDA       : {torch_ms:.4f} ms")
    print(f"  TensorRT           : {trt_ms:.4f} ms")
    print(f"  PyTorch throughput : {batch_size * 1000.0 / torch_ms:.2f} images/s")
    print(f"  TensorRT throughput: {batch_size * 1000.0 / trt_ms:.2f} images/s")
    print(f"  TensorRT speedup   : {speedup:.3f}x")

    # Keep TensorRT-owned objects alive through the final GPU synchronization.
    _ = runtime, context
    metrics = [
        plugin_max_abs,
        plugin_rmse,
        plugin_relative_l2,
        end_to_end_max_abs,
        end_to_end_rmse,
        end_to_end_relative_l2,
        cnn_relative_l2,
    ]
    if not np.isfinite(metrics).all():
        raise RuntimeError("Non-finite accuracy metric detected")


if __name__ == "__main__":
    main()
