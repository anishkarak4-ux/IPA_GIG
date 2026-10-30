"""Digital Image Processing (DIP) pipeline orchestrator.

Coordinates deterministic execution of comprehensive classical DIP stages across
Syllabus Units I to VII on a captured snapshot:
    - Unit I: Dynamic quantization, distance transforms (L2, D4, D8).
    - Unit II: Histograms with statistical moments, Gamma/Log/Negative transforms,
      contrast stretching, Roberts, Prewitt, Sobel, Laplacian, LoG, Canny, unsharp masking.
    - Unit III: 2D FFT magnitude spectrum, Butterworth LPF, Gaussian HPF, Homomorphic filtering.
    - Unit IV: Synthetic noise modeling, Wiener deconvolution motion deblurring.
    - Unit V: Multi-color spaces (HSV, Lab), color equalization, pseudo-coloring,
      Shannon entropy, RLE compression.
    - Unit VI: Otsu, Adaptive, K-Means clustering, Marker Watershed, Top-Hat/Black-Hat,
      morphological boundaries, hole-filling, skeletonization.
    - Unit VII: Invariant Hu moments, rotated bounding box, polygon approximation,
      Freeman chain codes, GLCM texture statistics.
"""

from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np

from .filters import (
    convert_to_grayscale,
    compute_histogram,
    compute_histogram_statistics,
    equalize_histogram,
    apply_gamma_transform,
    apply_log_transform,
    apply_image_negative,
    apply_contrast_stretching,
    apply_clahe,
    extract_bit_planes,
    simulate_quantization,
    apply_gaussian_blur,
    apply_median_filter,
    apply_bilateral_filter,
    apply_unsharp_mask,
)
from .edge_detectors import (
    apply_sobel,
    apply_roberts,
    apply_prewitt,
    apply_laplacian,
    apply_log,
    apply_canny,
)
from .segmentation import (
    apply_otsu,
    apply_adaptive_threshold,
    compute_distance_transforms,
    apply_kmeans_segmentation,
    apply_watershed_segmentation,
)
from .morphology_analysis import (
    ContourMetrics,
    apply_opening,
    apply_closing,
    apply_morphological_gradient,
    apply_tophat,
    apply_blackhat,
    apply_morphological_boundary,
    apply_hole_filling,
    apply_skeletonization,
    extract_contour_metrics,
    compute_local_binary_pattern,
)
from .frequency import (
    compute_fft2_spectrum,
    compute_phase_spectrum,
    compute_block_dct,
    apply_frequency_filter,
    apply_homomorphic_filter,
)
from .color_restoration import (
    decompose_color_spaces,
    equalize_color_image,
    apply_pseudocolor,
    inject_noise,
    generate_all_noise_models,
    restore_motion_blur,
    compute_shannon_entropy,
    compute_rle_compression_ratio,
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
    gamma_val: float = 0.6
    fft_cutoff: float = 25.0
    kmeans_clusters: int = 3

    def validate(self) -> None:
        """Validate pipeline parameters."""
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
    laplacian_edges: np.ndarray | None = None
    adaptive_mask: np.ndarray | None = None
    otsu_threshold: float = 0.0

    # Extended Syllabus Features (Units I to VII):
    histogram_stats: dict[str, float] = field(default_factory=dict)
    gamma_roi: np.ndarray | None = None
    log_roi: np.ndarray | None = None
    negative_roi: np.ndarray | None = None
    stretched_roi: np.ndarray | None = None
    bit_planes: list[np.ndarray] = field(default_factory=list)
    quantized_roi: np.ndarray | None = None
    unsharp_roi: np.ndarray | None = None
    roberts_magnitude: np.ndarray | None = None
    prewitt_magnitude: np.ndarray | None = None
    log_edges: np.ndarray | None = None
    fft_spectrum: np.ndarray | None = None
    butterworth_lpf: np.ndarray | None = None
    gaussian_hpf: np.ndarray | None = None
    homomorphic_roi: np.ndarray | None = None
    noisy_roi: np.ndarray | None = None
    deblurred_roi: np.ndarray | None = None
    hsv_channels: dict[str, np.ndarray] = field(default_factory=dict)
    color_equalized_roi: np.ndarray | None = None
    pseudocolor_roi: np.ndarray | None = None
    shannon_entropy: float = 0.0
    rle_compression_ratio: float = 1.0
    distance_transforms: dict[str, np.ndarray] = field(default_factory=dict)
    kmeans_roi: np.ndarray | None = None
    watershed_roi: np.ndarray | None = None
    tophat_roi: np.ndarray | None = None
    blackhat_roi: np.ndarray | None = None
    morph_boundary: np.ndarray | None = None
    skeleton_roi: np.ndarray | None = None

    # Advanced Syllabus Additions (Units II, III, IV, V, VII):
    clahe_roi: np.ndarray | None = None
    lbp_roi: np.ndarray | None = None
    lbp_features: dict[str, float] = field(default_factory=dict)
    fft_phase: np.ndarray | None = None
    dct_reconstructed: np.ndarray | None = None
    dct_metrics: dict[str, float] = field(default_factory=dict)
    noise_models: dict[str, dict[str, Any]] = field(default_factory=dict)


class DIPPipeline:
    """Deterministic orchestrator for all snapshot digital image processing."""

    def __init__(self, config: DIPConfig | None = None) -> None:
        """Initialize DIP pipeline with optional configuration overrides."""
        self.config = config or DIPConfig()
        self.config.validate()

    def process(self, context: SnapshotContext) -> DIPResult:
        """Execute full digital image processing suite across Units I to VII on snapshot.

        Treats input context as immutable. Operates on the tracked ROI while preserving
        the full frame for context.
        """
        if context is None or context.roi is None or context.roi.size == 0:
            return DIPResult(context=context)

        roi = context.roi.copy()
        is_color = (roi.ndim == 3 and roi.shape[2] >= 3)

        # 1. Grayscale Conversion & Intensity Histograms (Units I & II)
        gray_roi = convert_to_grayscale(roi)
        raw_histogram = compute_histogram(gray_roi)
        equalized_roi = equalize_histogram(gray_roi)
        clahe_roi = apply_clahe(roi if is_color else gray_roi, clip_limit=2.5)
        equalized_histogram = compute_histogram(equalized_roi)
        hist_stats = compute_histogram_statistics(gray_roi)

        # 2. Point Transformations (Unit II)
        gamma_roi = apply_gamma_transform(gray_roi, gamma=self.config.gamma_val)
        log_roi = apply_log_transform(gray_roi)
        negative_roi = apply_image_negative(gray_roi)
        stretched_roi = apply_contrast_stretching(gray_roi)
        bit_planes = extract_bit_planes(gray_roi)
        quantized_roi = simulate_quantization(gray_roi, bits=4)

        # 3. Spatial Smoothing & Sharpening (Unit II)
        gaussian_roi = apply_gaussian_blur(
            gray_roi,
            kernel_size=self.config.gaussian_ksize,
            sigma=self.config.gaussian_sigma,
        )
        median_roi = apply_median_filter(
            gray_roi,
            kernel_size=self.config.median_ksize,
        )
        unsharp_roi = apply_unsharp_mask(gray_roi, kernel_size=5, strength=1.5)

        # 4. Differential Edge Detectors (Units II & VI)
        _, _, sobel_mag = apply_sobel(gaussian_roi, ksize=self.config.sobel_ksize)
        _, _, roberts_mag = apply_roberts(gaussian_roi)
        _, _, prewitt_mag = apply_prewitt(gaussian_roi)
        laplacian_edges = apply_laplacian(gaussian_roi, ksize=self.config.sobel_ksize)
        log_edges = apply_log(gray_roi, ksize=5, sigma=1.0)
        canny_edges = apply_canny(
            gaussian_roi,
            threshold1=self.config.canny_thresh1,
            threshold2=self.config.canny_thresh2,
            aperture_size=self.config.canny_aperture,
        )

        # 5. Frequency Domain Processing (Unit III)
        _, fft_spectrum = compute_fft2_spectrum(gray_roi)
        fft_phase = compute_phase_spectrum(gray_roi)
        butterworth_lpf = apply_frequency_filter(
            gray_roi, cutoff_radius=self.config.fft_cutoff, filter_type="butterworth", pass_type="low"
        )
        gaussian_hpf = apply_frequency_filter(
            gray_roi, cutoff_radius=self.config.fft_cutoff, filter_type="gaussian", pass_type="high"
        )
        homomorphic_roi = apply_homomorphic_filter(gray_roi, cutoff_radius=self.config.fft_cutoff)

        # 6. Transform Coding (JPEG 8x8 DCT) & Energy Compaction (Unit V)
        dct_reconstructed, dct_metrics = compute_block_dct(gray_roi, block_size=8, quality=50)

        # 7. Color Spaces, Degradation & Complete Noise Suite (Units IV & V)
        decomposed = decompose_color_spaces(roi) if is_color else {}
        hsv_channels = decomposed.get("HSV", {})
        color_eq = equalize_color_image(roi, method="hsv") if is_color else equalized_roi
        pseudocolor = apply_pseudocolor(gray_roi)
        noisy_roi = inject_noise(gray_roi, noise_type="gaussian", sigma=20.0)
        noise_models = generate_all_noise_models(roi if is_color else gray_roi)

        # Motion blur restoration using telemetry velocity vector if available
        vel_px = 12.0
        angle_deg = 0.0
        if context.telemetry is not None:
            vx = getattr(context.telemetry, "vx", 0.0)
            vy = getattr(context.telemetry, "vy", 0.0)
            speed = math.sqrt(vx * vx + vy * vy)
            if speed > 1.0:
                vel_px = max(5.0, min(25.0, speed * 0.1))
                angle_deg = math.degrees(math.atan2(vy, vx))
        deblurred_roi = restore_motion_blur(gray_roi, length=int(vel_px), angle_deg=angle_deg)

        entropy_val = compute_shannon_entropy(gray_roi)

        # 8. Segmentation & Thresholding (Unit VI)
        otsu_thresh, otsu_mask = apply_otsu(gaussian_roi)
        adaptive_mask = apply_adaptive_threshold(
            gaussian_roi,
            block_size=self.config.adaptive_block_size,
            c_constant=self.config.adaptive_c,
        )
        dist_maps = compute_distance_transforms(otsu_mask)
        kmeans_roi, _ = apply_kmeans_segmentation(roi if is_color else gray_roi, k=self.config.kmeans_clusters)
        watershed_roi, _ = apply_watershed_segmentation(
            roi if is_color else cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR), binary_mask=otsu_mask
        )

        # 9. Morphology & Topological Analysis (Unit VI)
        morph_opened = apply_opening(
            otsu_mask,
            kernel_size=self.config.morph_ksize,
            iterations=self.config.morph_iterations,
            shape=self.config.morph_shape,
        )
        morph_closed = apply_closing(
            morph_opened,
            kernel_size=self.config.morph_ksize,
            iterations=self.config.morph_iterations,
            shape=self.config.morph_shape,
        )
        morphology_mask = apply_hole_filling(morph_closed)
        tophat = apply_tophat(gray_roi, kernel_size=9)
        blackhat = apply_blackhat(gray_roi, kernel_size=9)
        morph_boundary = apply_morphological_boundary(morphology_mask)
        skeleton = apply_skeletonization(morphology_mask)

        # 10. Geometric Descriptors, LBP Texture & Invariant Features (Unit VII)
        contour_metrics = extract_contour_metrics(morphology_mask, gray_roi=gray_roi)
        rle_ratio, _ = compute_rle_compression_ratio(morphology_mask)
        lbp_roi, lbp_features = compute_local_binary_pattern(gray_roi)

        return DIPResult(
            context=context,
            gray_roi=gray_roi,
            equalized_roi=equalized_roi,
            raw_histogram=raw_histogram,
            equalized_histogram=equalized_histogram,
            gaussian_roi=gaussian_roi,
            median_roi=median_roi,
            sobel_magnitude=sobel_mag,
            canny_edges=canny_edges,
            otsu_mask=otsu_mask,
            morphology_mask=morphology_mask,
            contour_metrics=contour_metrics,
            laplacian_edges=laplacian_edges,
            adaptive_mask=adaptive_mask,
            otsu_threshold=otsu_thresh,
            histogram_stats=hist_stats,
            gamma_roi=gamma_roi,
            log_roi=log_roi,
            negative_roi=negative_roi,
            stretched_roi=stretched_roi,
            bit_planes=bit_planes,
            quantized_roi=quantized_roi,
            unsharp_roi=unsharp_roi,
            roberts_magnitude=roberts_mag,
            prewitt_magnitude=prewitt_mag,
            log_edges=log_edges,
            fft_spectrum=fft_spectrum,
            butterworth_lpf=butterworth_lpf,
            gaussian_hpf=gaussian_hpf,
            homomorphic_roi=homomorphic_roi,
            noisy_roi=noisy_roi,
            deblurred_roi=deblurred_roi,
            hsv_channels=hsv_channels,
            color_equalized_roi=color_eq,
            pseudocolor_roi=pseudocolor,
            shannon_entropy=round(entropy_val, 3),
            rle_compression_ratio=round(rle_ratio, 2),
            distance_transforms=dist_maps,
            kmeans_roi=kmeans_roi,
            watershed_roi=watershed_roi,
            tophat_roi=tophat,
            blackhat_roi=blackhat,
            morph_boundary=morph_boundary,
            skeleton_roi=skeleton,
            clahe_roi=clahe_roi,
            lbp_roi=lbp_roi,
            lbp_features=lbp_features,
            fft_phase=fft_phase,
            dct_reconstructed=dct_reconstructed,
            dct_metrics=dct_metrics,
            noise_models=noise_models,
        )
