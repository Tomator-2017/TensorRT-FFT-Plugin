"""Export a static BCHW CNN + FFT2D model and its PyTorch weights."""

import argparse
from pathlib import Path

import torch

from examples.cnn_fft2d.model import CNNFFT2D
from trt_fft import register_fft2d_symbolic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CNN + FFT2D ONNX model")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--onnx", default="artifacts/fft2d_cnn.onnx")
    parser.add_argument("--checkpoint", default="artifacts/fft2d_cnn.pth")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch", "channels", "height", "width", "features"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")


def export_fft2d_model(args: argparse.Namespace) -> None:
    validate_args(args)
    torch.manual_seed(args.seed)
    model = CNNFFT2D(args.channels, args.features).eval()
    input_shape = (args.batch, args.channels, args.height, args.width)
    image = torch.randn(input_shape, dtype=torch.float32)

    # Verify FFT2D eager behavior before exporting the custom ONNX node.
    with torch.inference_mode():
        real_output, imag_output, features = model(image)
        expected = torch.fft.fft2(features, dim=(-2, -1), norm="backward")
        torch.testing.assert_close(real_output, expected.real)
        torch.testing.assert_close(imag_output, expected.imag)

    onnx_path = Path(args.onnx)
    checkpoint_path = Path(args.checkpoint)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    register_fft2d_symbolic(17)
    torch.onnx.export(
        model,
        (image,),
        str(onnx_path),
        input_names=["image"],
        output_names=["real_output", "imag_output", "cnn_features"],
        opset_version=17,
        custom_opsets={"trt.plugins": 1},
        do_constant_folding=True,
    )

    # Benchmark reconstructs exactly the same PyTorch network and weights.
    torch.save(
        {
            "format_version": 1,
            "input_shape": input_shape,
            "input_channels": args.channels,
            "features": args.features,
            "seed": args.seed,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Exported FFT2D ONNX model to {onnx_path}")
    print(f"Saved matching PyTorch weights to {checkpoint_path}")


def main() -> None:
    export_fft2d_model(parse_args())


if __name__ == "__main__":
    main()
