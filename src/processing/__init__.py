"""Digital Image Processing (DIP) package.

Provides modular spatial filtering, edge detection, segmentation,
morphological analysis, and end-to-end snapshot pipeline orchestration.
"""

from .dip_pipeline import (
    ContourMetrics,
    DIPConfig,
    DIPPipeline,
    DIPResult,
    SnapshotContext,
)
from .edge_detectors import (
    EdgeDetectorSuite,
    apply_canny,
    apply_laplacian,
    apply_sobel,
    compute_sobel,
)
from .filters import (
    SpatialFilterSuite,
    apply_bilateral_filter,
    apply_gaussian_blur,
    apply_median_filter,
    compute_histogram,
    convert_to_grayscale,
    equalize_histogram,
)
from .morphology_analysis import (
    MorphologyAnalyzer,
    apply_closing,
    apply_dilation,
    apply_erosion,
    apply_morphological_gradient,
    apply_opening,
    extract_contour_metrics,
    get_structuring_element,
)
from .segmentation import (
    SegmentationSuite,
    apply_adaptive_threshold,
    apply_binary_threshold,
    apply_otsu,
)

__all__ = [
    "SnapshotContext",
    "DIPConfig",
    "DIPResult",
    "DIPPipeline",
    "ContourMetrics",
    "SpatialFilterSuite",
    "convert_to_grayscale",
    "compute_histogram",
    "equalize_histogram",
    "apply_gaussian_blur",
    "apply_median_filter",
    "apply_bilateral_filter",
    "EdgeDetectorSuite",
    "compute_sobel",
    "apply_sobel",
    "apply_laplacian",
    "apply_canny",
    "SegmentationSuite",
    "apply_otsu",
    "apply_adaptive_threshold",
    "apply_binary_threshold",
    "MorphologyAnalyzer",
    "get_structuring_element",
    "apply_erosion",
    "apply_dilation",
    "apply_opening",
    "apply_closing",
    "apply_morphological_gradient",
    "extract_contour_metrics",
]
