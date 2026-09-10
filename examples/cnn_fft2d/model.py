"""PyTorch model shared by FFT2D export and benchmark scripts."""

from typing import Tuple

import torch
from torch import nn

from trt_fft import fft2d


class CNNFFT2D(nn.Module):
    """A small image CNN followed by the planar FP32 FFT2D operator."""

    def __init__(self, input_channels: int, features: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, features, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, 1, bias=True),
        )

    def cnn_forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.cnn(image)

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = self.cnn_forward(image)
        imag = torch.zeros_like(real)

        # FFT2D transforms trailing H/W in one cuFFT plan. TensorRT sees two
        # independent FP32 tensors, never a complex TensorRT value.
        real_output, imag_output = fft2d(real, imag)

        # This extra output lets the benchmark compare FFT accuracy using the
        # exact TensorRT CNN activation, independently of CNN tactic errors.
        return real_output, imag_output, real
