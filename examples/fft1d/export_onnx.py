"""Export a static two-input/two-output FFT1D custom ONNX node."""

import argparse
from pathlib import Path

import torch
from torch import nn

from trt_fft import fft1d
from trt_fft import register_fft1d_symbolic


class FFT1DModel(nn.Module):
    def __init__(self, inverse: bool = False, normalize: bool = False):
        super().__init__()
        self.inverse = int(inverse)
        self.normalize = int(normalize)

    def forward(self, real_input: torch.Tensor, imag_input: torch.Tensor):
        return fft1d(real_input, imag_input, self.inverse, self.normalize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an FFT1D ONNX model")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--onnx", default="artifacts/fft1d.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = (args.batch, args.channels, args.length)
    if any(d <= 0 for d in shape):
        raise ValueError("batch, channels, and length must be positive")

    torch.manual_seed(args.seed)
    model = FFT1DModel().eval()
    real_input = torch.randn(shape, dtype=torch.float32)
    imag_input = torch.randn(shape, dtype=torch.float32)
    with torch.inference_mode():
        actual_real, actual_imag = model(real_input, imag_input)
        expected = torch.fft.fft(torch.complex(real_input, imag_input), dim=-1)
        torch.testing.assert_close(actual_real, expected.real)
        torch.testing.assert_close(actual_imag, expected.imag)

    output_path = Path(args.onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    register_fft1d_symbolic(17)
    torch.onnx.export(
        model,
        (real_input, imag_input),
        str(output_path),
        input_names=["real_input", "imag_input"],
        output_names=["real_output", "imag_output"],
        opset_version=17,
        custom_opsets={"trt.plugins": 1},
        do_constant_folding=True,
    )
    print(f"Exported FFT1D ONNX model to {output_path}")


if __name__ == "__main__":
    main()
