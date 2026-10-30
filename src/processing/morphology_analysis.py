"""Mathematical morphology and geometric contour characterization.

Academic Overview (Syllabus Units VI & VII):
    - Structuring Element (SE):
      Geometric probe B (Rectangle, Cross, Ellipse).
    - Fundamental Morphology:
      - Erosion: A (-) B = { z | (B)_z subset of A }
      - Dilation: A (+) B = { z | (B_hat)_z intersect A != empty }
      - Opening: A o B = (A (-) B) (+) B (removes bright noise, separates components)
      - Closing: A * B = (A (+) B) (-) B (fills dark holes, bridges gaps)
      - Morphological Gradient: (A (+) B) - (A (-) B) (boundary transition)
      - Top-Hat: A - (A o B) (isolates elements brighter than background)
      - Black-Hat: (A * B) - A (isolates elements darker than background)
      - Morphological Boundary: beta(A) = A - (A (-) B)
      - Hole-Filling: Flood-fill reconstruction eliminating interior voids
      - Skeletonization: Iterative morphological thinning extracting topological spine
    - Shape Geometry & Invariant Descriptors:
      - Area (A), Perimeter (P), Aspect Ratio, Circularity (4*pi*A / P^2), Solidity (A / A_hull)
      - Extent: A / A_bounding_box
      - Rotated Bounding Box & Orientation Angle theta
      - Polygon Approximation (Douglas-Peucker) & Vertex Count
      - Hu's 7 Invariant Moments (phi_1 - phi_7) invariant to translation, scale, rotation
      - 8-Directional Freeman Chain Code representation
      - GLCM Texture Features (Contrast, Dissimilarity, Homogeneity, Energy, Correlation)
"""

from dataclasses import dataclass, field
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
    # Enriched Academic Descriptors (Unit VII):
    extent: float = 0.0
    orientation_deg: float = 0.0
    rotated_bbox: tuple[tuple[float, float], tuple[float, float], float] | None = None
    polygon_vertices: int = 0
    hu_moments: list[float] = field(default_factory=list)
    chain_code: str = ""
    glcm_features: dict[str, float] = field(default_factory=dict)
    lbp_features: dict[str, float] = field(default_factory=dict)


MORPH_SHAPES: dict[str, int] = {
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
    "ellipse": cv2.MORPH_ELLIPSE,
}


def get_structuring_element(
    kernel_size: int = 3,
    shape: str = "rect",
) -> np.ndarray:
    """Generate a 2D morphological structuring element."""
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
    """Apply morphological erosion to shrink foreground regions."""
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
    """Apply morphological dilation to expand foreground regions."""
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
    """Apply morphological opening (erosion followed by dilation) for noise filtering."""
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
    """Apply morphological closing (dilation followed by erosion) for hole bridging."""
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
    """Apply morphological gradient (dilation minus erosion) to isolate object boundaries."""
    if mask is None or mask.size == 0:
        raise ValueError("Cannot apply morphological gradient to empty or None mask.")
    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)


def apply_tophat(mask: np.ndarray, kernel_size: int = 9, shape: str = "rect") -> np.ndarray:
    """Apply White Top-Hat filter: A - (A o B) to isolate bright elements."""
    if mask is None or mask.size == 0:
        return mask
    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_TOPHAT, kernel)


def apply_blackhat(mask: np.ndarray, kernel_size: int = 9, shape: str = "rect") -> np.ndarray:
    """Apply Black-Hat filter: (A * B) - A to isolate dark elements."""
    if mask is None or mask.size == 0:
        return mask
    kernel = get_structuring_element(kernel_size=kernel_size, shape=shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)


def apply_morphological_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Extract standard internal morphological boundary: beta(A) = A - (A (-) B)."""
    if mask is None or mask.size == 0:
        return mask
    eroded = apply_erosion(mask, kernel_size=kernel_size)
    return cv2.subtract(mask.astype(np.uint8), eroded)


def apply_hole_filling(binary_mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using flood-fill morphological reconstruction."""
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask

    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood_filled = mask_u8.copy()
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Floodfill background from origin (0, 0)
    cv2.floodFill(flood_filled, fill_mask, (0, 0), 255)
    # Invert floodfilled image
    inv_flood = cv2.bitwise_not(flood_filled)
    # Combine original foreground with filled holes
    return cv2.bitwise_or(mask_u8, inv_flood)


def apply_skeletonization(binary_mask: np.ndarray, max_iters: int = 50) -> np.ndarray:
    """Compute topological morphological skeleton of a binary silhouette via opening subtraction."""
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask

    img = (binary_mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(max_iters):
        eroded = cv2.erode(img, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def compute_glcm_features(gray_roi: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Calculate Gray-Level Co-occurrence Matrix (GLCM) statistical texture descriptors.

    Computes Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.
    """
    if gray_roi is None or gray_roi.size == 0:
        return {"contrast": 0.0, "dissimilarity": 0.0, "homogeneity": 0.0, "energy": 0.0, "correlation": 0.0}

    # Quantize to 16 gray levels for robust, fast GLCM computation
    q = (gray_roi.astype(np.float32) / 16.0).clip(0, 15).astype(np.int32)
    h, w = q.shape
    glcm = np.zeros((16, 16), dtype=np.float64)

    # Horizontal adjacency (dx=1, dy=0)
    for y in range(h):
        for x in range(w - 1):
            if mask is None or (mask[y, x] > 0 and mask[y, x + 1] > 0):
                i = q[y, x]
                j = q[y, x + 1]
                glcm[i, j] += 1.0

    total = np.sum(glcm)
    if total <= 0:
        return {"contrast": 0.0, "dissimilarity": 0.0, "homogeneity": 0.0, "energy": 0.0, "correlation": 0.0}

    p = glcm / total
    i_indices, j_indices = np.indices((16, 16))
    diff = np.abs(i_indices - j_indices)

    contrast = float(np.sum(p * (diff ** 2)))
    dissimilarity = float(np.sum(p * diff))
    homogeneity = float(np.sum(p / (1.0 + (diff ** 2))))
    energy = float(np.sum(p ** 2))

    mu_i = float(np.sum(i_indices * p))
    mu_j = float(np.sum(j_indices * p))
    sigma_i = float(np.sqrt(np.sum(((i_indices - mu_i) ** 2) * p)))
    sigma_j = float(np.sqrt(np.sum(((j_indices - mu_j) ** 2) * p)))

    if sigma_i * sigma_j > 1e-6:
        correlation = float(np.sum((i_indices - mu_i) * (j_indices - mu_j) * p) / (sigma_i * sigma_j))
    else:
        correlation = 1.0

    return {
        "contrast": round(contrast, 3),
        "dissimilarity": round(dissimilarity, 3),
        "homogeneity": round(homogeneity, 3),
        "energy": round(energy, 4),
        "correlation": round(correlation, 3),
    }


def compute_chain_code(contour: np.ndarray, max_length: int = 64) -> str:
    """Generate 8-directional Freeman Chain Code string representing contour boundary."""
    if contour is None or len(contour) < 2:
        return ""

    pts = contour.reshape(-1, 2)
    # 8-direction mapping (dx, dy)
    dir_map = {
        (1, 0): "0", (1, -1): "1", (0, -1): "2", (-1, -1): "3",
        (-1, 0): "4", (-1, 1): "5", (0, 1): "6", (1, 1): "7",
    }
    chain: list[str] = []

    for k in range(min(max_length, len(pts) - 1)):
        dx = int(np.sign(pts[k + 1, 0] - pts[k, 0]))
        dy = int(np.sign(pts[k + 1, 1] - pts[k, 1]))
        code = dir_map.get((dx, dy), "")
        if code:
            chain.append(code)

    return "".join(chain)


def extract_contour_metrics(
    binary_mask: np.ndarray,
    gray_roi: np.ndarray | None = None,
) -> ContourMetrics:
    """Extract comprehensive geometric shape descriptors and invariant moments."""
    if binary_mask is None or binary_mask.size == 0:
        return ContourMetrics()

    mask_8u = binary_mask.astype(np.uint8) if binary_mask.dtype != np.uint8 else binary_mask
    if mask_8u.ndim != 2:
        return ContourMetrics()

    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ContourMetrics(contour_count=0)

    # Primary contour: largest surface area
    primary_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(primary_contour))
    perimeter = float(cv2.arcLength(primary_contour, closed=True))

    bx, by, bw, bh = cv2.boundingRect(primary_contour)
    bounding_rect = (int(bx), int(by), int(bw), int(bh))

    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

    # Extent: Area / Bounding Box Area
    bbox_area = float(bw * bh)
    extent = (area / bbox_area) if bbox_area > 0.0 else 0.0

    # Circularity: 4*pi*A / (P^2)
    circularity = (4.0 * math.pi * area) / (perimeter**2) if perimeter > 0.0 else 0.0
    if not math.isfinite(circularity):
        circularity = 0.0

    # Solidity: area / convex_hull_area
    hull = cv2.convexHull(primary_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0.0 else 0.0
    if not math.isfinite(solidity):
        solidity = 0.0

    # Spatial moments and Centroid
    moments = cv2.moments(primary_contour)
    m00 = moments["m00"]
    if abs(m00) > 1e-5:
        cx = int(round(moments["m10"] / m00))
        cy = int(round(moments["m01"] / m00))
        centroid = (cx, cy)
    else:
        centroid = (bx + bw // 2, by + bh // 2)

    # Hu Invariant Moments (log-transformed for numerical readability)
    hu_raw = cv2.HuMoments(moments).flatten()
    hu_log: list[float] = []
    for h_val in hu_raw:
        if abs(h_val) > 1e-20:
            hu_log.append(round(-math.copysign(1.0, h_val) * math.log10(abs(h_val)), 4))
        else:
            hu_log.append(0.0)

    # Rotated Bounding Box & Principal Orientation
    if len(primary_contour) >= 5:
        rot_rect = cv2.minAreaRect(primary_contour)
        orientation_deg = round(rot_rect[2], 2)
    else:
        rot_rect = None
        orientation_deg = 0.0

    # Douglas-Peucker Polygon Approximation
    epsilon = 0.02 * perimeter if perimeter > 0.0 else 1.0
    approx_poly = cv2.approxPolyDP(primary_contour, epsilon, closed=True)
    poly_vertices = len(approx_poly)

    # Chain code
    chain_code = compute_chain_code(primary_contour, max_length=64)

    # GLCM Texture features on target ROI
    glcm_dict = compute_glcm_features(gray_roi, mask=mask_8u) if gray_roi is not None else {}

    # Local Binary Pattern texture metrics
    _, lbp_dict = compute_local_binary_pattern(gray_roi) if gray_roi is not None else (None, {})

    return ContourMetrics(
        area=round(area, 2),
        perimeter=round(perimeter, 2),
        aspect_ratio=round(aspect_ratio, 4),
        circularity=round(circularity, 4),
        solidity=round(solidity, 4),
        centroid=centroid,
        bounding_rect=bounding_rect,
        contour_count=len(contours),
        extent=round(extent, 4),
        orientation_deg=orientation_deg,
        rotated_bbox=rot_rect,
        polygon_vertices=poly_vertices,
        hu_moments=hu_log,
        chain_code=chain_code,
        glcm_features=glcm_dict,
        lbp_features=lbp_dict,
    )


def compute_local_binary_pattern(gray_image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute 8-neighborhood Local Binary Pattern (LBP) texture representation.

    Syllabus Unit VII: Texture Analysis & Feature Extraction.
    Evaluates 8 circular/square neighbors around each pixel:
        LBP(x_c, y_c) = sum_{p=0}^7 s(g_p - g_c) * 2^p
    where s(x) = 1 if x >= 0 else 0.

    Returns:
        Tuple of (lbp_image_uint8, lbp_features_dict).
    """
    if gray_image is None or gray_image.size == 0 or gray_image.ndim != 2:
        return np.zeros((0, 0), dtype=np.uint8), {"lbp_entropy": 0.0, "lbp_energy": 0.0, "lbp_uniformity": 0.0}

    h, w = gray_image.shape
    if h < 3 or w < 3:
        return gray_image.copy(), {"lbp_entropy": 0.0, "lbp_energy": 0.0, "lbp_uniformity": 0.0}

    center = gray_image[1:-1, 1:-1].astype(np.int16)

    n0 = (gray_image[1:-1, 2:].astype(np.int16) >= center).astype(np.uint8)    # ( 0,  1)
    n1 = (gray_image[:-2, 2:].astype(np.int16) >= center).astype(np.uint8)     # (-1,  1)
    n2 = (gray_image[:-2, 1:-1].astype(np.int16) >= center).astype(np.uint8)   # (-1,  0)
    n3 = (gray_image[:-2, :-2].astype(np.int16) >= center).astype(np.uint8)    # (-1, -1)
    n4 = (gray_image[1:-1, :-2].astype(np.int16) >= center).astype(np.uint8)   # ( 0, -1)
    n5 = (gray_image[2:, :-2].astype(np.int16) >= center).astype(np.uint8)     # ( 1, -1)
    n6 = (gray_image[2:, 1:-1].astype(np.int16) >= center).astype(np.uint8)    # ( 1,  0)
    n7 = (gray_image[2:, 2:].astype(np.int16) >= center).astype(np.uint8)      # ( 1,  1)

    lbp = (n0 * 1 + n1 * 2 + n2 * 4 + n3 * 8 + n4 * 16 + n5 * 32 + n6 * 64 + n7 * 128).astype(np.uint8)
    lbp_padded = cv2.copyMakeBorder(lbp, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # Statistical features on LBP histogram
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).ravel()
    prob = hist / float(lbp.size)
    prob_nz = prob[prob > 0]
    entropy = float(-np.sum(prob_nz * np.log2(prob_nz)))
    energy = float(np.sum(prob ** 2))
    uniformity = float(np.max(prob))

    features = {
        "lbp_entropy": round(entropy, 3),
        "lbp_energy": round(energy, 4),
        "lbp_uniformity": round(uniformity, 3),
    }

    return lbp_padded, features


class MorphologyAnalyzer:
    """Class wrapper providing static methods for mathematical morphology and shape analysis."""

    get_structuring_element = staticmethod(get_structuring_element)
    apply_erosion = staticmethod(apply_erosion)
    apply_dilation = staticmethod(apply_dilation)
    apply_opening = staticmethod(apply_opening)
    apply_closing = staticmethod(apply_closing)
    apply_morphological_gradient = staticmethod(apply_morphological_gradient)
    apply_tophat = staticmethod(apply_tophat)
    apply_blackhat = staticmethod(apply_blackhat)
    apply_morphological_boundary = staticmethod(apply_morphological_boundary)
    apply_hole_filling = staticmethod(apply_hole_filling)
    apply_skeletonization = staticmethod(apply_skeletonization)
    extract_contour_metrics = staticmethod(extract_contour_metrics)
    compute_local_binary_pattern = staticmethod(compute_local_binary_pattern)
