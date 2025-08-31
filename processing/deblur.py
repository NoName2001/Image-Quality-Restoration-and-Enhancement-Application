from typing import Literal
import numpy as np
import cv2
from scipy.signal import fftconvolve


def wiener_deconvolution(img_bgr: np.ndarray, kernel_size: int = 9, sigma: float = 2.0, k: float = 0.01) -> np.ndarray:
    """Simple Wiener deconvolution with Gaussian PSF per channel."""
    kernel_size = max(3, int(kernel_size) | 1)  # odd
    psf_1d = cv2.getGaussianKernel(kernel_size, sigma)
    psf = psf_1d @ psf_1d.T
    psf /= psf.sum()

    def wiener(channel: np.ndarray) -> np.ndarray:
        channel = channel.astype(np.float32) / 255.0
        H = np.fft.fft2(psf, s=channel.shape)
        G = np.fft.fft2(channel)
        H_conj = np.conj(H)
        F_hat = (H_conj / (H * H_conj + k)) * G
        f = np.fft.ifft2(F_hat)
        f = np.real(f)
        f = np.clip(f, 0, 1)
        return (f * 255.0).astype(np.uint8)

    b, g, r = cv2.split(img_bgr)
    b = wiener(b)
    g = wiener(g)
    r = wiener(r)
    return cv2.merge([b, g, r])


ess_kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)


def unsharp_mask(img_bgr: np.ndarray, amount: float = 1.0, radius: int = 1) -> np.ndarray:
    radius = max(1, int(radius))
    blurred = cv2.GaussianBlur(img_bgr, (radius * 2 + 1, radius * 2 + 1), 0)
    sharpened = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    return sharpened


def deblur(
    img_bgr: np.ndarray,
    method: Literal["wiener", "unsharp"] = "wiener",
    strength: float = 0.5,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if method == "wiener":
        ksize = 5 + int(12 * strength)  # 5..17
        sigma = 1.0 + 3.0 * strength    # 1..4
        k = 0.01 + 0.19 * (1.0 - strength)  # 0.2..0.01 inverse
        return wiener_deconvolution(img_bgr, kernel_size=ksize, sigma=sigma, k=k)
    else:
        amount = 0.5 + 1.5 * strength   # 0.5..2.0
        radius = 1 + int(4 * strength)  # 1..5
        return unsharp_mask(img_bgr, amount=amount, radius=radius)
