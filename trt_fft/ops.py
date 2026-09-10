"""PyTorch custom operators matching the TensorRT FFT1D/FFT2D plugins."""

from typing import Tuple

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _parse_arg


_OP_LIBRARY = torch.library.Library("trt_custom", "DEF")
_OP_LIBRARY.define(
    "fft1d(Tensor real, Tensor imag, int inverse=0, int normalize=0) "
    "-> (Tensor, Tensor)"
)
_OP_LIBRARY.define(
    "fft2d(Tensor real, Tensor imag, int inverse=0, int normalize=0) "
    "-> (Tensor, Tensor)"
)


def _validate_inputs(real: torch.Tensor, imag: torch.Tensor, rank: int) -> None:
    if real.dtype != torch.float32 or imag.dtype != torch.float32:
        raise TypeError("FFT plugins require two float32 tensors")
    if real.shape != imag.shape:
        raise ValueError("real and imaginary tensors must have identical shapes")
    if real.dim() < rank:
        raise ValueError(f"FFT{rank}D requires input rank >= {rank}")


def _eager_fft(
    real: torch.Tensor,
    imag: torch.Tensor,
    inverse: int,
    normalize: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(real, imag, rank)
    if inverse not in (0, 1) or normalize not in (0, 1):
        raise ValueError("inverse and normalize must be 0 or 1")

    dimensions = (-1,) if rank == 1 else (-2, -1)
    complex_input = torch.complex(real, imag)
    if inverse:
        # PyTorch norm="forward" and cuFFT inverse are both unnormalized.
        result = torch.fft.ifftn(complex_input, dim=dimensions, norm="forward")
    else:
        result = torch.fft.fftn(complex_input, dim=dimensions, norm="backward")
    if normalize:
        transform_size = real.shape[-1]
        if rank == 2:
            transform_size *= real.shape[-2]
        result = result / transform_size
    return result.real, result.imag


@torch.library.impl(_OP_LIBRARY, "fft1d", "CompositeExplicitAutograd")
def _fft1d_eager(
    real: torch.Tensor,
    imag: torch.Tensor,
    inverse: int = 0,
    normalize: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _eager_fft(real, imag, inverse, normalize, rank=1)


@torch.library.impl(_OP_LIBRARY, "fft2d", "CompositeExplicitAutograd")
def _fft2d_eager(
    real: torch.Tensor,
    imag: torch.Tensor,
    inverse: int = 0,
    normalize: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _eager_fft(real, imag, inverse, normalize, rank=2)


def fft1d(
    real: torch.Tensor,
    imag: torch.Tensor,
    inverse: int = 0,
    normalize: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the eager reference for the TensorRT FFT1D plugin."""
    return torch.ops.trt_custom.fft1d(real, imag, inverse, normalize)


def fft2d(
    real: torch.Tensor,
    imag: torch.Tensor,
    inverse: int = 0,
    normalize: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the eager reference for the TensorRT FFT2D plugin."""
    return torch.ops.trt_custom.fft2d(real, imag, inverse, normalize)


def _symbolic(plugin_name: str, graph, real, imag, inverse, normalize):
    inverse_value = _parse_arg(inverse, "i")
    normalize_value = _parse_arg(normalize, "i")
    real_output, imag_output = graph.op(
        f"trt.plugins::{plugin_name}",
        real,
        imag,
        axis_i=-1,
        inverse_i=inverse_value,
        normalize_i=normalize_value,
        outputs=2,
    )
    real_output.setType(real.type())
    imag_output.setType(imag.type())
    return real_output, imag_output


def _fft1d_symbolic(graph, real, imag, inverse, normalize):
    return _symbolic("FFT1D", graph, real, imag, inverse, normalize)


def _fft2d_symbolic(graph, real, imag, inverse, normalize):
    return _symbolic("FFT2D", graph, real, imag, inverse, normalize)


def register_fft1d_symbolic(opset_version: int = 17) -> None:
    register_custom_op_symbolic(
        "trt_custom::fft1d", _fft1d_symbolic, opset_version
    )


def register_fft2d_symbolic(opset_version: int = 17) -> None:
    register_custom_op_symbolic(
        "trt_custom::fft2d", _fft2d_symbolic, opset_version
    )
