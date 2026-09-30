"""Differential gradient and edge detection algorithms.

Academic Overview:
    - Sobel Operator:
      Discrete differentiation operator computing approximations of the gradient of image
      intensity function. Uses two 3x3 (or ksize x ksize) convolution kernels:
          Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] * I
          Gy = [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]] * I
      Analytical magnitude: G = sqrt(Gx^2 + Gy^2)
      Analytical orientation: theta = arctan2(Gy, Gx)
    - Laplacian Operator:
      Second-order isotropic derivative detecting zero-crossings corresponding to intensity inflection:
          del^2 I = (d^2 I / dx^2) + (d^2 I / dy^2)
    - Canny Edge Detector:
      Multi-stage optimal edge detection algorithm comprising:
      1. Gaussian smoothing to mitigate sensor noise.
      2. Intensity gradient magnitude and orientation calculation via Sobel.
      3. Non-Maximum Suppression (NMS) along gradient directions to thin edge ridges.
      4. Hysteresis dual-thresholding (low_thr, high_thr) to eliminate spurious weak edges.
"""

from typing import Sequence
import cv2
import numpy as np


def compute_sobel(
    gray_image: np.ndarray,
    ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mathematical Sobel horizontal/vertical gradients, magnitude, and direction.

    Args:
        gray_image: 2D uint8 or float grayscale image array.
        ksize: Aperture size of the Sobel kernel (must be 1, 3, 5, or 7).

    Returns:
        Tuple of (gx, gy, magnitude, direction):
            - gx: Horizontal gradient (float64, CV_64F).
            - gy: Vertical gradient (float64, CV_64F).
            - magnitude: Exact Euclidean gradient magnitude sqrt(gx^2 + gy^2) (float64).
            - direction: Gradient orientation angle in radians in range [-pi, pi] (float64).

    Raises:
        ValueError: If input image is empty or ksize is invalid.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot compute Sobel on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Sobel expects a 2D grayscale image, got shape {gray_image.shape}")

    if ksize not in (1, 3, 5, 7):
        raise ValueError(f"Sobel ksize must be 1, 3, 5, or 7; got {ksize}")

    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    return gx, gy, magnitude, direction


def normalize_magnitude_to_uint8(magnitude: np.ndarray) -> np.ndarray:
    """Normalize a continuous float gradient magnitude array to an 8-bit [0, 255] uint8 image.

    Args:
        magnitude: 2D floating-point gradient magnitude array.

    Returns:
        2D uint8 image scaled to dynamic range [0, 255].
    """
    if magnitude is None or magnitude.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    max_val = float(np.max(magnitude))
    if max_val <= 1e-6:
        return np.zeros_like(magnitude, dtype=np.uint8)

    norm = (magnitude / max_val * 255.0).clip(0, 255).astype(np.uint8)
    return norm


def apply_sobel(
    gray_image: np.ndarray,
    ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradients and return (gx, gy, normalized_magnitude_uint8).

    Args:
        gray_image: 2D grayscale image array.
        ksize: Aperture size (1, 3, 5, or 7).

    Returns:
        Tuple of (gx_float64, gy_float64, mag_uint8).
    """
    gx, gy, magnitude, _ = compute_sobel(gray_image, ksize=ksize)
    mag_uint8 = normalize_magnitude_to_uint8(magnitude)
    return gx, gy, mag_uint8


def apply_laplacian(
    gray_image: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """Apply second-order differential Laplacian operator.

    Args:
        gray_image: 2D grayscale image array.
        ksize: Aperture size (positive odd integer, typically 1, 3, or 5).

    Returns:
        2D uint8 absolute scaled Laplacian response image.

    Raises:
        ValueError: If input image is empty or ksize is non-positive / even.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply Laplacian on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Laplacian expects a 2D grayscale image, got shape {gray_image.shape}")

    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"Laplacian ksize must be a positive odd integer, got {ksize}")

    lap = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)


def apply_canny(
    image: np.ndarray,
    threshold1: float = 50.0,
    threshold2: float = 150.0,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """Apply Canny edge detection with non-maximum suppression and hysteresis thresholding.

    Args:
        image: 2D grayscale uint8 image array.
        threshold1: Lower hysteresis threshold for edge linking.
        threshold2: Upper hysteresis threshold for initial strong edge selection.
        aperture_size: Sobel aperture size (3, 5, or 7).
        l2_gradient: If True, uses exact Euclidean norm sqrt(dx^2 + dy^2).

    Returns:
        Binary 2D uint8 edge mask (pixels are 0 or 255).

    Raises:
        ValueError: If thresholds are negative or aperture size is invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply Canny on empty or None image.")

    if threshold1 < 0.0 or threshold2 < 0.0:
        raise ValueError("Canny thresholds must be non-negative floats.")

    if aperture_size not in (3, 5, 7):
        raise ValueError(f"Canny aperture_size must be 3, 5, or 7; got {aperture_size}")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(
        gray.astype(np.uint8),
        threshold1=threshold1,
        threshold2=threshold2,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )


class EdgeDetectorSuite:
    """Class wrapper providing static methods for gradient and differential edge operations."""

    compute_sobel = staticmethod(compute_sobel)
    normalize_magnitude = staticmethod(normalize_magnitude_to_uint8)
    apply_sobel = staticmethod(apply_sobel)
    apply_laplacian = staticmethod(apply_laplacian)
    apply_canny = staticmethod(apply_canny)
