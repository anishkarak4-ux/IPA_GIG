"""Differential gradient and edge detection algorithms.

Academic Overview (Syllabus Units II & VI):
    - Roberts Cross Operator (2x2):
      Diagonal difference approximations:
          Gx = [[ 1,  0], [ 0, -1]] * I
          Gy = [[ 0,  1], [-1,  0]] * I
          Magnitude: M = sqrt(Gx^2 + Gy^2)
    - Prewitt Operator (3x3):
      Directional first-order gradient kernels:
          Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] * I
          Gy = [[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]] * I
          Magnitude: M = sqrt(Gx^2 + Gy^2)
    - Sobel Operator (3x3):
      Discrete differentiation operator with center pixel weighting:
          Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] * I
          Gy = [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]] * I
          Analytical magnitude: G = sqrt(Gx^2 + Gy^2)
          Orientation: theta = arctan2(Gy, Gx)
    - Laplacian Operator:
      Second-order isotropic derivative detecting zero-crossings:
          del^2 I = (d^2 I / dx^2) + (d^2 I / dy^2)
    - Laplacian of Gaussian (LoG / Marr-Hildreth):
      Combines Gaussian smoothing G_sigma with Laplacian del^2:
          LoG(x, y) = -1/(pi*sigma^4) * [1 - (x^2 + y^2)/(2*sigma^2)] * exp(-(x^2 + y^2)/(2*sigma^2))
    - Canny Edge Detector:
      Multi-stage optimal edge detection comprising:
      1. Gaussian smoothing to mitigate noise.
      2. Gradient calculation via Sobel.
      3. Non-Maximum Suppression (NMS) along gradient directions.
      4. Hysteresis dual-thresholding to link strong and weak edges.
"""

from typing import Sequence
import cv2
import numpy as np


def compute_sobel(
    gray_image: np.ndarray,
    ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mathematical Sobel horizontal/vertical gradients, magnitude, and direction."""
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
    """Normalize a continuous float gradient magnitude array to an 8-bit [0, 255] uint8 image."""
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
    """Compute Sobel gradients and return (gx, gy, normalized_magnitude_uint8)."""
    gx, gy, magnitude, _ = compute_sobel(gray_image, ksize=ksize)
    mag_uint8 = normalize_magnitude_to_uint8(magnitude)
    return gx, gy, mag_uint8


def apply_roberts(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Roberts Cross gradient operators (2x2 diagonal kernels).

    Returns:
        Tuple of (gx, gy, normalized_magnitude_uint8).
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot compute Roberts on empty or None image.")

    kernel_x = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    kernel_y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    src = gray_image.astype(np.float32)
    gx = cv2.filter2D(src, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(src, cv2.CV_32F, kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_uint8 = normalize_magnitude_to_uint8(magnitude)

    return gx, gy, mag_uint8


def apply_prewitt(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Prewitt gradient operators (3x3 first-order kernels).

    Returns:
        Tuple of (gx, gy, normalized_magnitude_uint8).
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot compute Prewitt on empty or None image.")

    kernel_x = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    kernel_y = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    src = gray_image.astype(np.float32)
    gx = cv2.filter2D(src, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(src, cv2.CV_32F, kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_uint8 = normalize_magnitude_to_uint8(magnitude)

    return gx, gy, mag_uint8


def apply_laplacian(
    gray_image: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """Apply second-order differential Laplacian operator."""
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply Laplacian on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Laplacian expects a 2D grayscale image, got shape {gray_image.shape}")

    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"Laplacian ksize must be a positive odd integer, got {ksize}")

    lap = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)


def apply_log(gray_image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Laplacian of Gaussian (LoG / Marr-Hildreth) edge operator.

    Convolves with Gaussian low-pass filter, then computes Laplacian response.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply LoG on empty or None image.")

    blurred = cv2.GaussianBlur(gray_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    return cv2.convertScaleAbs(lap)


def apply_canny(
    image: np.ndarray,
    threshold1: float = 50.0,
    threshold2: float = 150.0,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """Apply Canny edge detection with non-maximum suppression and hysteresis thresholding."""
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
    apply_roberts = staticmethod(apply_roberts)
    apply_prewitt = staticmethod(apply_prewitt)
    apply_laplacian = staticmethod(apply_laplacian)
    apply_log = staticmethod(apply_log)
    apply_canny = staticmethod(apply_canny)
