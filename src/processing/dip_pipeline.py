"""Digital Image Processing (DIP) pipeline orchestrator.

Coordinates deterministic execution of classical DIP stages on a captured snapshot:
    1. Color to Grayscale conversion
    2. Intensity Histogram computation (raw & equalized)
    3. Global Histogram Equalization
    4. Spatial Low-Pass Smoothing (Gaussian & Median filtering)
    5. Differential Edge Detection (Sobel magnitude, Laplacian, Canny)
    6. Automatic Segmentation (Otsu optimal thresholding & Adaptive thresholding)
    7. Mathematical Morphology (Opening & Closing)
    8. Contour Extraction & Shape Geometry Metrics
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .filters import (
    convert_to_grayscale,
    compute_histogram,
    equalize_histogram,
    apply_gaussian_blur,
    apply_median_filter,
    apply_bilateral_filter,
)
from .edge_detectors import (
    apply_sobel,
    apply_laplacian,
    apply_canny,
)
from .segmentation import (
    apply_otsu,
    apply_adaptive_threshold,
)
from .morphology_analysis import (
    ContourMetrics,
    apply_opening,
    apply_closing,
    extract_contour_metrics,
)


@dataclass
class SnapshotContext:
    """Inputs passed from the tracking loop when SPACE is pressed."""

    frame_index: int
    timestamp: float
    full_frame: np.ndarray
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    roi: np.ndarray
    track_id: int | None = None
    tracker_name: str = ""
    telemetry: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DIPConfig:
    """Configuration parameters for each stage of the DIP pipeline."""

    gaussian_ksize: int = 5
    gaussian_sigma: float = 0.0
    median_ksize: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    sobel_ksize: int = 3
    canny_thresh1: float = 50.0
    canny_thresh2: float = 150.0
    canny_aperture: int = 3
    adaptive_block_size: int = 11
    adaptive_c: float = 2.0
    morph_ksize: int = 3
    morph_shape: str = "rect"
    morph_iterations: int = 1

    def validate(self) -> None:
        """Validate pipeline parameters, raising ValueError on improper values."""
        if self.gaussian_ksize <= 0 or self.gaussian_ksize % 2 == 0:
            raise ValueError(f"gaussian_ksize must be positive odd integer, got {self.gaussian_ksize}")
        if self.gaussian_sigma < 0.0:
            raise ValueError(f"gaussian_sigma must be non-negative, got {self.gaussian_sigma}")
        if self.median_ksize <= 1 or self.median_ksize % 2 == 0:
            raise ValueError(f"median_ksize must be odd integer >= 3, got {self.median_ksize}")
        if self.sobel_ksize not in (1, 3, 5, 7):
            raise ValueError(f"sobel_ksize must be 1, 3, 5, or 7, got {self.sobel_ksize}")
        if self.canny_thresh1 < 0.0 or self.canny_thresh2 < 0.0:
            raise ValueError("canny thresholds must be non-negative floats")
        if self.canny_aperture not in (3, 5, 7):
            raise ValueError(f"canny_aperture must be 3, 5, or 7, got {self.canny_aperture}")
        if self.adaptive_block_size <= 1 or self.adaptive_block_size % 2 == 0:
            raise ValueError(f"adaptive_block_size must be odd integer > 1, got {self.adaptive_block_size}")
        if self.morph_ksize <= 0 or self.morph_ksize % 2 == 0:
            raise ValueError(f"morph_ksize must be positive odd integer, got {self.morph_ksize}")
        if self.morph_iterations < 1:
            raise ValueError(f"morph_iterations must be >= 1, got {self.morph_iterations}")


@dataclass
class DIPResult:
    """Composite container holding outputs from all DIP processing stages."""

    context: SnapshotContext
    gray_roi: np.ndarray | None = None
    equalized_roi: np.ndarray | None = None
    raw_histogram: np.ndarray | None = None
    equalized_histogram: np.ndarray | None = None
    gaussian_roi: np.ndarray | None = None
    median_roi: np.ndarray | None = None
    sobel_magnitude: np.ndarray | None = None
    canny_edges: np.ndarray | None = None
    otsu_mask: np.ndarray | None = None
    morphology_mask: np.ndarray | None = None
    contour_metrics: ContourMetrics = field(default_factory=ContourMetrics)
    # Supplementary features for academic completeness:
    laplacian_edges: np.ndarray | None = None
    adaptive_mask: np.ndarray | None = None
    otsu_threshold: float = 0.0


class DIPPipeline:
    """Deterministic orchestrator for all snapshot digital image processing."""

    def __init__(self, config: DIPConfig | None = None) -> None:
        """Initialize DIP pipeline with optional configuration overrides."""
        self.config = config or DIPConfig()
        self.config.validate()

    def process(self, context: SnapshotContext) -> DIPResult:
        """Execute full digital image processing suite on the captured snapshot.

        Treats input context as immutable. The primary analysis operates on the tracked ROI,
        while preserving the full frame within context for dual-context inspection.

        Args:
            context: Immutable SnapshotContext holding full frame, bbox, and ROI.

        Returns:
            Fully populated DIPResult with intermediate and final representations.
        """
        if context is None or context.roi is None or context.roi.size == 0:
            return DIPResult(context=context)

        # Working copy to ensure input context remains unmutated
        roi = context.roi.copy()

        # 1. Grayscale Conversion
        gray_roi = convert_to_grayscale(roi)

        # 2. Raw Intensity Histogram
        raw_histogram = compute_histogram(gray_roi)

        # 3. Global Histogram Equalization
        equalized_roi = equalize_histogram(gray_roi)
        equalized_histogram = compute_histogram(equalized_roi)

        # 4. Spatial Denoising (performed on grayscale ROI)
        gaussian_roi = apply_gaussian_blur(
            gray_roi,
            kernel_size=self.config.gaussian_ksize,
            sigma=self.config.gaussian_sigma,
        )
        median_roi = apply_median_filter(
            gray_roi,
            kernel_size=self.config.median_ksize,
        )

        # 5. Differential Edge Detection (operating on smoothed image)
        _, _, sobel_magnitude = apply_sobel(
            gaussian_roi,
            ksize=self.config.sobel_ksize,
        )
        laplacian_edges = apply_laplacian(
            gaussian_roi,
            ksize=self.config.sobel_ksize,
        )
        canny_edges = apply_canny(
            gaussian_roi,
            threshold1=self.config.canny_thresh1,
            threshold2=self.config.canny_thresh2,
            aperture_size=self.config.canny_aperture,
        )

        # 6. Automatic and Adaptive Segmentation
        otsu_thresh, otsu_mask = apply_otsu(gaussian_roi)
        adaptive_mask = apply_adaptive_threshold(
            gaussian_roi,
            block_size=self.config.adaptive_block_size,
            c_constant=self.config.adaptive_c,
        )

        # 7. Mathematical Morphology (Opening followed by Closing to remove noise and fill voids)
        morph_opened = apply_opening(
            otsu_mask,
            kernel_size=self.config.morph_ksize,
            iterations=self.config.morph_iterations,
            shape=self.config.morph_shape,
        )
        morphology_mask = apply_closing(
            morph_opened,
            kernel_size=self.config.morph_ksize,
            iterations=self.config.morph_iterations,
            shape=self.config.morph_shape,
        )

        # 8. Contour and Shape Geometric Descriptors
        contour_metrics = extract_contour_metrics(morphology_mask)

        return DIPResult(
            context=context,
            gray_roi=gray_roi,
            equalized_roi=equalized_roi,
            raw_histogram=raw_histogram,
            equalized_histogram=equalized_histogram,
            gaussian_roi=gaussian_roi,
            median_roi=median_roi,
            sobel_magnitude=sobel_magnitude,
            canny_edges=canny_edges,
            otsu_mask=otsu_mask,
            morphology_mask=morphology_mask,
            contour_metrics=contour_metrics,
            laplacian_edges=laplacian_edges,
            adaptive_mask=adaptive_mask,
            otsu_threshold=otsu_thresh,
        )
