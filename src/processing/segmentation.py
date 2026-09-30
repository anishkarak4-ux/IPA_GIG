"""Image segmentation and automated thresholding algorithms.

Academic Overview:
    - Otsu's Method (Global Optimal Thresholding):
      Statistical clustering algorithm that automatically calculates an optimal threshold t*
      by maximizing the between-class variance:
          sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2
      where omega_0, omega_1 are class probabilities and mu_0, mu_1 are class mean intensities.
      Optimal for bi-modal histogram distributions.
    - Adaptive Local Thresholding:
      Computes a dynamic threshold T(x, y) for each pixel based on a local neighborhood window:
          T(x, y) = Mean_Local(x, y) - C   (or Gaussian_Weighted_Local(x, y) - C)
      Effective under non-uniform illumination or intensity gradients.
"""

from typing import Literal
import cv2
import numpy as np


def apply_otsu(gray_image: np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate Otsu's optimal global threshold and generate a binary segmentation mask.

    Args:
        gray_image: 2D uint8 grayscale image array.

    Returns:
        Tuple of (optimal_threshold, binary_mask):
            - optimal_threshold: Float threshold intensity computed by Otsu's method.
            - binary_mask: 2D uint8 binary mask where foreground pixels are 255 and background are 0.

    Raises:
        ValueError: If input image is empty, None, or not single-channel.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply Otsu thresholding on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Otsu thresholding expects a 2D grayscale image, got shape {gray_image.shape}")

    thresh_val, mask = cv2.threshold(
        gray_image.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(thresh_val), mask


def apply_adaptive_threshold(
    gray_image: np.ndarray,
    block_size: int = 11,
    c_constant: float = 2.0,
    method: Literal["gaussian", "mean"] = "gaussian",
) -> np.ndarray:
    """Compute local adaptive thresholding over a spatial neighborhood.

    Args:
        gray_image: 2D uint8 grayscale image array.
        block_size: Spatial neighborhood size (must be an odd integer > 1).
        c_constant: Constant subtracted from local mean or weighted mean.
        method: Neighborhood weighting scheme ("gaussian" or "mean").

    Returns:
        2D uint8 binary segmentation mask (0 or 255).

    Raises:
        ValueError: If block_size is <= 1 or even, or method is unrecognized.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply adaptive thresholding on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Adaptive thresholding expects a 2D grayscale image, got shape {gray_image.shape}")

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(f"Adaptive threshold block_size must be an odd integer > 1, got {block_size}")

    method_lower = method.lower().strip()
    if method_lower == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method_lower == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError(f"Unsupported adaptive method '{method}'. Choose 'gaussian' or 'mean'.")

    return cv2.adaptiveThreshold(
        gray_image.astype(np.uint8),
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        c_constant,
    )


def apply_binary_threshold(
    gray_image: np.ndarray,
    threshold_val: float = 127.0,
) -> np.ndarray:
    """Apply manual hard thresholding.

    Args:
        gray_image: 2D uint8 grayscale image array.
        threshold_val: Fixed cutoff intensity threshold in [0, 255].

    Returns:
        2D uint8 binary mask (0 or 255).
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply binary thresholding on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Binary thresholding expects a 2D grayscale image, got shape {gray_image.shape}")

    _, mask = cv2.threshold(gray_image.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
    return mask


class SegmentationSuite:
    """Class wrapper providing static methods for image segmentation and thresholding."""

    apply_otsu = staticmethod(apply_otsu)
    apply_adaptive_threshold = staticmethod(apply_adaptive_threshold)
    apply_binary_threshold = staticmethod(apply_binary_threshold)
