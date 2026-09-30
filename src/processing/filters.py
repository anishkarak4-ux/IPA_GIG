"""Spatial domain low-pass and edge-preserving filtering algorithms.

Academic Overview:
    - Grayscale Conversion:
      Luminance projection: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601).
    - Intensity Histogram:
      Discrete probability distribution of 8-bit pixel values across [0, 255].
    - Histogram Equalization:
      Non-linear monotonic mapping using the cumulative distribution function (CDF)
      to linearize the intensity distribution and enhance global dynamic range.
    - Gaussian Low-Pass Filter:
      2D isotropic Gaussian convolution kernel:
      G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2)).
      Suppresses high-frequency Gaussian / sensor noise.
    - Median Filter:
      Non-linear rank-order filter replacing each pixel with the statistical median
      of its spatial neighborhood. Highly effective against salt-and-pepper noise.
    - Bilateral Filter:
      Edge-preserving non-linear filter combining spatial domain Gaussian distance
      with photometric range Gaussian distance:
      BF[I]_p = (1 / W_p) * sum_{q in S} I_q * G_{sigma_s}(||p - q||) * G_{sigma_r}(|I_p - I_q|).
"""

from typing import Sequence
import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input image array to 8-bit single-channel grayscale.

    Args:
        image: Input array (2D grayscale, 3-channel BGR, or 4-channel BGRA).

    Returns:
        2D uint8 grayscale numpy array.

    Raises:
        ValueError: If input image is None, empty, or has unsupported channel count.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert empty or None image to grayscale.")

    if image.ndim == 2:
        return image.copy().astype(np.uint8)

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy().astype(np.uint8)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def compute_histogram(gray_image: np.ndarray) -> np.ndarray:
    """Compute 256-bin 1D intensity histogram for an 8-bit grayscale image.

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        1D float32 numpy array of shape (256,) containing frequency counts.
        The sum of elements equals the total number of pixels.

    Raises:
        ValueError: If input image is empty or not single-channel.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot compute histogram on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    hist = cv2.calcHist([gray_image.astype(np.uint8)], [0], None, [256], [0, 256])
    return hist.flatten().astype(np.float32)


def equalize_histogram(gray_image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to enhance image dynamic range.

    Transforms intensity values via normalized cumulative distribution function:
        s_k = T(r_k) = (L - 1) * sum_{j=0}^k (n_j / N)

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        Equalized 2D uint8 grayscale image with linearized cumulative histogram.

    Raises:
        ValueError: If input image is empty or not single-channel.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot equalize empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    return cv2.equalizeHist(gray_image.astype(np.uint8))


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply 2D Gaussian low-pass spatial smoothing filter.

    Args:
        image: Input 2D grayscale or 3D color image array.
        kernel_size: Size of square Gaussian kernel (must be positive odd integer).
        sigma: Standard deviation in X/Y (if 0.0, computed from kernel size as 0.3*((ksize-1)*0.5 - 1) + 0.8).

    Returns:
        Filtered image array with same shape and dtype as input.

    Raises:
        ValueError: If kernel_size is non-positive or even, or sigma is negative.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply Gaussian blur to empty or None image.")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Gaussian kernel_size must be a positive odd integer, got {kernel_size}")

    if sigma < 0.0:
        raise ValueError(f"Gaussian sigma must be non-negative, got {sigma}")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply non-linear median rank-order filter for impulsive noise suppression.

    Replaces each pixel with the statistical median of the kernel aperture.

    Args:
        image: Input 2D grayscale or 3D color image array.
        kernel_size: Aperture linear size (must be positive odd integer > 1).

    Returns:
        Median-filtered image array with same shape and dtype as input.

    Raises:
        ValueError: If kernel_size is non-positive, even, or less than 3.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply median filter to empty or None image.")

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(f"Median kernel_size must be an odd integer >= 3, got {kernel_size}")

    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral smoothing filter.

    Combines spatial closeness Gaussian weights with photometric radiometric similarity weights.

    Args:
        image: Input 2D grayscale or 3D color uint8 image array.
        d: Diameter of each pixel neighborhood (if <= 0, computed from sigma_space).
        sigma_color: Filter sigma in the color/intensity space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        Smoothed image preserving sharp gradient boundaries.

    Raises:
        ValueError: If sigma_color or sigma_space are non-positive.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply bilateral filter to empty or None image.")

    if sigma_color <= 0.0 or sigma_space <= 0.0:
        raise ValueError("Bilateral filter sigma_color and sigma_space must be positive floats.")

    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


class SpatialFilterSuite:
    """Class wrapper providing static methods for spatial and radiometric filtering."""

    to_gray = staticmethod(convert_to_grayscale)
    compute_histogram = staticmethod(compute_histogram)
    equalize_histogram = staticmethod(equalize_histogram)
    apply_gaussian = staticmethod(apply_gaussian_blur)
    apply_median = staticmethod(apply_median_filter)
    apply_bilateral = staticmethod(apply_bilateral_filter)
