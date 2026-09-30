"""Mathematical morphology and geometric contour characterization.

Academic Overview:
    - Structuring Element (SE):
      Geometric probe B positioned across the image to evaluate spatial inclusion / intersection.
      Standard shapes: Rectangle, Cross, Ellipse.
    - Erosion (A (-) B):
      Shrinks foreground regions by retaining only pixels where the SE completely fits:
          A (-) B = { z | (B)_z is subset of A }
    - Dilation (A (+) B):
      Expands foreground regions by retaining pixels where the SE overlaps with foreground:
          A (+) B = { z | (B_hat)_z intersect A != empty }
    - Morphological Opening (A o B = (A (-) B) (+) B):
      Erosion followed by dilation. Eliminates small bright noise spikes, thins protrusions,
      and separates weakly connected components without noticeably altering area.
    - Morphological Closing (A * B = (A (+) B) (-) B):
      Dilation followed by erosion. Fills internal holes, bridges narrow background gaps,
      and smooths outer boundary concavities.
    - Morphological Gradient ((A (+) B) - (A (-) B)):
      Boundary extraction highlighting edge transitions.
    - Shape Geometry Metrics:
      - Area (A): Total enclosed pixel surface area.
      - Perimeter (P): Boundary arc length.
      - Aspect Ratio: Width / Height of the minimal bounding rectangle.
      - Circularity (Isoperimetric Quotient): 4*pi*A / (P^2). Values approach 1.0 for a perfect circle.
      - Solidity: A / Area(Convex_Hull). Measures boundary convexity and roughness.
      - Centroid: First-order spatial moment centroid (m10/m00, m01/m00).
"""

from dataclasses import dataclass
import math
from typing import Literal, Sequence
import cv2
import numpy as np


@dataclass
class ContourMetrics:
    """Extracted geometric features of the tracked object silhouette."""

    area: float = 0.0
    perimeter: float = 0.0
    aspect_ratio: float = 0.0
    circularity: float = 0.0
    solidity: float = 0.0
    centroid: tuple[int, int] | None = None
    bounding_rect: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    contour_count: int = 0


MORPH_SHAPES: dict[str, int] = {
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
    "ellipse": cv2.MORPH_ELLIPSE,
}


def get_structuring_element(
    kernel_size: int = 3,
    shape: str = "rect",
) -> np.ndarray:
    """Generate a 2D morphological structuring element.

    Args:
        kernel_size: Linear kernel size (must be a positive odd integer).
        shape: Kernel geometry ('rect', 'cross', or 'ellipse').

    Returns:
        2D uint8 structuring element array.

    Raises:
        ValueError: If kernel_size is non-positive or even, or shape is unknown.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Morphology kernel_size must be a positive odd integer, got {kernel_size}")

    shape_key = shape.lower().strip()
    if shape_key not in MORPH_SHAPES:
        raise ValueError(
            f"Unsupported structuring element shape '{shape}'. Choose from: {', '.join(sorted(MORPH_SHAPES.keys()))}"
        )

    return cv2.getStructuringElement(MORPH_SHAPES[shape_key], (kernel_size, kernel_size))


def apply_erosion(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Apply morphological erosion to shrink foreground regions.

    Args:
        mask: 2D uint8 binary mask array.
        kernel_size: Structuring element size (positive odd integer).
        iterations: Number of sequential erosion passes (>= 1).
        shape: Kernel geometry ('rect', 'cross', or 'ellipse').

    Returns:
        Eroded 2D uint8 binary mask.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Cannot erode empty or None mask.")

    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)


def apply_dilation(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Apply morphological dilation to expand foreground regions.

    Args:
        mask: 2D uint8 binary mask array.
        kernel_size: Structuring element size (positive odd integer).
        iterations: Number of sequential dilation passes (>= 1).
        shape: Kernel geometry ('rect', 'cross', or 'ellipse').

    Returns:
        Dilated 2D uint8 binary mask.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Cannot dilate empty or None mask.")

    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)


def apply_opening(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Apply morphological opening (erosion followed by dilation) for noise filtering.

    Args:
        mask: 2D uint8 binary mask array.
        kernel_size: Structuring element size (positive odd integer).
        iterations: Number of opening passes (>= 1).
        shape: Kernel geometry ('rect', 'cross', or 'ellipse').

    Returns:
        Opened 2D uint8 binary mask.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Cannot apply opening to empty or None mask.")

    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion) for hole bridging.

    Args:
        mask: 2D uint8 binary mask array.
        kernel_size: Structuring element size (positive odd integer).
        iterations: Number of closing passes (>= 1).
        shape: Kernel geometry ('rect', 'cross', or 'ellipse').

    Returns:
        Closed 2D uint8 binary mask.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Cannot apply closing to empty or None mask.")

    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_morphological_gradient(
    mask: np.ndarray,
    kernel_size: int = 3,
    shape: str = "rect",
) -> np.ndarray:
    """Apply morphological gradient (dilation minus erosion) to isolate object boundaries.

    Args:
        mask: 2D uint8 binary mask array.
        kernel_size: Structuring element size (positive odd integer).
        shape: Kernel geometry ('rect', 'cross', or 'ellipse').

    Returns:
        Gradient boundary 2D uint8 binary mask.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Cannot apply morphological gradient to empty or None mask.")

    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)


def extract_contour_metrics(binary_mask: np.ndarray) -> ContourMetrics:
    """Extract geometric shape descriptors and moments from a binary segmentation mask.

    Selects the largest connected component by contour area as the primary object.
    Safely handles empty masks, zero-area contours, and degenerate shapes without NaN/Inf.

    Args:
        binary_mask: 2D uint8 binary mask array.

    Returns:
        Populated ContourMetrics object with area, perimeter, aspect ratio,
        circularity, solidity, centroid, and bounding rectangle.
    """
    if binary_mask is None or binary_mask.size == 0:
        return ContourMetrics()

    # Ensure 8-bit single channel
    mask_8u = binary_mask.astype(np.uint8) if binary_mask.dtype != np.uint8 else binary_mask
    if mask_8u.ndim != 2:
        return ContourMetrics()

    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ContourMetrics(contour_count=0)

    # Deterministically select the largest contour by surface area
    primary_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(primary_contour))
    perimeter = float(cv2.arcLength(primary_contour, closed=True))

    bx, by, bw, bh = cv2.boundingRect(primary_contour)
    bounding_rect = (int(bx), int(by), int(bw), int(bh))

    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

    # Isoperimetric circularity: 4*pi*A / (P^2)
    circularity = (4.0 * math.pi * area) / (perimeter**2) if perimeter > 0.0 else 0.0
    if not math.isfinite(circularity):
        circularity = 0.0

    # Solidity: area / convex_hull_area
    hull = cv2.convexHull(primary_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0.0 else 0.0
    if not math.isfinite(solidity):
        solidity = 0.0

    # Spatial centroid via spatial moments
    moments = cv2.moments(primary_contour)
    m00 = moments["m00"]
    if abs(m00) > 1e-5:
        cx = int(round(moments["m10"] / m00))
        cy = int(round(moments["m01"] / m00))
        centroid = (cx, cy)
    else:
        centroid = (bx + bw // 2, by + bh // 2)

    return ContourMetrics(
        area=round(area, 2),
        perimeter=round(perimeter, 2),
        aspect_ratio=round(aspect_ratio, 4),
        circularity=round(circularity, 4),
        solidity=round(solidity, 4),
        centroid=centroid,
        bounding_rect=bounding_rect,
        contour_count=len(contours),
    )


class MorphologyAnalyzer:
    """Class wrapper providing static methods for mathematical morphology and shape analysis."""

    get_structuring_element = staticmethod(get_structuring_element)
    apply_erosion = staticmethod(apply_erosion)
    apply_dilation = staticmethod(apply_dilation)
    apply_opening = staticmethod(apply_opening)
    apply_closing = staticmethod(apply_closing)
    apply_morphological_gradient = staticmethod(apply_morphological_gradient)
    extract_contour_metrics = staticmethod(extract_contour_metrics)
