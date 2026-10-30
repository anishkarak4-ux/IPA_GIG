"""Image segmentation, automated thresholding, clustering, and distance transforms.

Academic Overview (Syllabus Units I & VI):
    - Otsu's Method (Global Optimal Thresholding):
      Statistical clustering maximizing between-class variance sigma_B^2(t).
    - Adaptive Local Thresholding:
      Dynamic neighborhood thresholding T(x, y) = Local_Mean - C.
    - Distance Transforms (Unit I Pixel Relationships):
      Calculates the distance of each foreground pixel to the nearest background pixel:
      1. Euclidean (L2): sqrt((x1 - x2)^2 + (y1 - y2)^2)
      2. City Block (D4 / Manhattan L1): |x1 - x2| + |y1 - y2|
      3. Chessboard (D8 / Chebyshev L_inf): max(|x1 - x2|, |y1 - y2|)
    - K-Means Clustering Segmentation:
      Unsupervised vector quantization clustering image pixels into K clusters based on color/intensity.
    - Marker-Controlled Watershed Segmentation:
      Topographical flood basin simulation using distance transform markers to partition touching objects.
"""

from typing import Literal
import cv2
import numpy as np


def apply_otsu(gray_image: np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate Otsu's optimal global threshold and generate a binary segmentation mask."""
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
    """Compute local adaptive thresholding over a spatial neighborhood."""
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
    """Apply manual hard thresholding."""
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot apply binary thresholding on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Binary thresholding expects a 2D grayscale image, got shape {gray_image.shape}")

    _, mask = cv2.threshold(gray_image.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
    return mask


def compute_distance_transforms(binary_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Compute Euclidean (L2), City Block (D4), and Chessboard (D8) distance maps.

    Args:
        binary_mask: 2D uint8 binary mask (values 0 and 255).

    Returns:
        Dictionary mapping distance metric names to normalized uint8 display maps.
    """
    if binary_mask is None or binary_mask.size == 0:
        return {}

    mask_u8 = (binary_mask > 0).astype(np.uint8)

    # 1. Euclidean Distance (L2)
    dist_l2 = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    # 2. City Block Distance (D4 / L1)
    dist_l1 = cv2.distanceTransform(mask_u8, cv2.DIST_L1, 3)
    # 3. Chessboard Distance (D8 / C)
    dist_c = cv2.distanceTransform(mask_u8, cv2.DIST_C, 3)

    def _normalize_dist(d: np.ndarray) -> np.ndarray:
        max_d = float(np.max(d))
        if max_d > 1e-4:
            return (d / max_d * 255.0).clip(0, 255).astype(np.uint8)
        return np.zeros_like(d, dtype=np.uint8)

    return {
        "Euclidean_L2": _normalize_dist(dist_l2),
        "CityBlock_D4": _normalize_dist(dist_l1),
        "Chessboard_D8": _normalize_dist(dist_c),
    }


def apply_kmeans_segmentation(
    image: np.ndarray,
    k: int = 3,
    max_iter: int = 20,
    eps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment image using K-Means color/intensity clustering.

    Args:
        image: 2D grayscale or 3D BGR image.
        k: Number of semantic clusters (default: 3).
        max_iter: Maximum EM iterations.
        eps: Convergence threshold epsilon.

    Returns:
        Tuple of (segmented_quantized_image, labels_map).
    """
    if image is None or image.size == 0:
        return image, np.zeros((0, 0), dtype=np.int32)

    is_color = (image.ndim == 3)
    pixel_data = image.reshape((-1, 3 if is_color else 1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    _, labels, centers = cv2.kmeans(
        pixel_data,
        k,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    centers_u8 = np.uint8(centers)
    segmented = centers_u8[labels.flatten()].reshape(image.shape)
    labels_map = labels.reshape(image.shape[:2])

    return segmented, labels_map


def apply_watershed_segmentation(
    bgr_image: np.ndarray,
    binary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment objects using Marker-Controlled Watershed algorithm.

    Uses morphological opening and distance transform peaks to create watershed markers.

    Returns:
        Tuple of (watershed_boundary_overlay, marker_labels).
    """
    if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
        return bgr_image, np.zeros((0, 0), dtype=np.int32)

    if binary_mask is None:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal via opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area via distance transform threshold
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_val = float(np.max(dist_transform))
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * max_val, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region between foreground and background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    overlay = bgr_image.copy()
    markers_ws = cv2.watershed(overlay, markers)
    # Watershed boundaries are marked with -1
    overlay[markers_ws == -1] = [0, 0, 255]  # Red boundaries

    return overlay, markers_ws


class SegmentationSuite:
    """Class wrapper providing static methods for image segmentation and thresholding."""

    apply_otsu = staticmethod(apply_otsu)
    apply_adaptive_threshold = staticmethod(apply_adaptive_threshold)
    apply_binary_threshold = staticmethod(apply_binary_threshold)
    compute_distance_transforms = staticmethod(compute_distance_transforms)
    apply_kmeans = staticmethod(apply_kmeans_segmentation)
    apply_watershed = staticmethod(apply_watershed_segmentation)
