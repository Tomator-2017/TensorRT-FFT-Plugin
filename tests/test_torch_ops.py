"""CPU tests for the PyTorch reference semantics of both FFT plugins."""

import unittest

import torch

from trt_fft import fft1d
from trt_fft import fft2d


class FFTReferenceOperatorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_fft1d_matches_torch(self) -> None:
        real = torch.randn(2, 3, 16, dtype=torch.float32)
        imag = torch.randn_like(real)
        actual_real, actual_imag = fft1d(real, imag)
        expected = torch.fft.fft(torch.complex(real, imag), dim=-1)
        torch.testing.assert_close(actual_real, expected.real)
        torch.testing.assert_close(actual_imag, expected.imag)

    def test_fft2d_matches_torch(self) -> None:
        real = torch.randn(2, 4, 8, 12, dtype=torch.float32)
        imag = torch.randn_like(real)
        actual_real, actual_imag = fft2d(real, imag)
        expected = torch.fft.fft2(torch.complex(real, imag), dim=(-2, -1))
        torch.testing.assert_close(actual_real, expected.real)
        torch.testing.assert_close(actual_imag, expected.imag)

    def test_normalized_inverse_fft2d_matches_torch(self) -> None:
        real = torch.randn(1, 2, 7, 9, dtype=torch.float32)
        imag = torch.randn_like(real)
        actual_real, actual_imag = fft2d(
            real, imag, inverse=1, normalize=1
        )
        expected = torch.fft.ifft2(torch.complex(real, imag), dim=(-2, -1))
        torch.testing.assert_close(actual_real, expected.real)
        torch.testing.assert_close(actual_imag, expected.imag)


if __name__ == "__main__":
    unittest.main()
