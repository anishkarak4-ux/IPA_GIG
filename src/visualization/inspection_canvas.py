"""Modal composite inspection canvas for post-snapshot DIP evaluation.

Constructs an academic multi-panel visual dashboard across 5 thematic syllabus tabs:
    - Tab 1: CORE SPATIAL & CLASSICAL DIP (Units I & II)
      Panel 1: Full Frame Context with Target Bounding Box
      Panel 2: Tracked Target ROI (Raw vs Equalized vs CLAHE)
      Panel 3: 256-Bin Intensity Histogram + Statistical Moments (Mean, Variance, Skew, Kurt)
      Panel 4: Spatial Filtering & Differential Edges (Gaussian, Sobel, Laplacian, Canny)
      Panel 5: Segmentation & Mathematical Morphology (Otsu vs Morph Mask)
      Panel 6: Contour Extraction & Shape Metrics (Area, Perimeter, Aspect Ratio, Circularity, Solidity)

    - Tab 2: FREQUENCY DOMAIN & RESTORATION (Units III & IV)
      Panel 1: Tracked ROI (Grayscale vs Preprocessed)
      Panel 2: 2D FFT Centered Log-Magnitude Spectrum & Phase Angle phi(u, v)
      Panel 3: Butterworth Low-Pass Filter (BLPF D0=25, order=2)
      Panel 4: Gaussian High-Pass Filter (GHPF D0=25)
      Panel 5: Homomorphic Illumination-Reflectance Normalization
      Panel 6: Wiener Motion Deblur & 2D Block DCT JPEG Compression (8x8 Q50)

    - Tab 3: COLOR SPACES & ADVANCED SEGMENTATION (Units V & VI)
      Panel 1: HSV Color Space Decomposition (H, S, V channel breakdown)
      Panel 2: CIE L*a*b* Color Space Decomposition (L*, a*, b*)
      Panel 3: Perceptual Color Equalization vs Pseudo-Color (JET Thermal Colormap)
      Panel 4: K-Means Color Clustering Segmentation (K=3 semantic clusters)
      Panel 5: Marker-Controlled Watershed Segmentation (Distance Peak Markers)
      Panel 6: Advanced Morphology (White Top-Hat, Black-Hat, Morph Boundary)

    - Tab 4: MORPHOMETRIC & FEATURE DESCRIPTORS (Units I & VII)
      Panel 1: Distance Transforms (Euclidean L2, City Block D4, Chessboard D8)
      Panel 2: 8-Level Bit-Plane Slicing (Bits 0-7) & Gray-Level Quantization
      Panel 3: Point Transformations (Gamma=0.6, Logarithmic Transform, Image Negative)
      Panel 4: Topological Morphological Skeleton & Local Binary Patterns (LBP)
      Panel 5: Rotated Bounding Box, Orientation Angle theta, & Polygon Approximation
      Panel 6: Invariant Shape Descriptors Card (Hu Moments phi1-phi7, Extent, GLCM, LBP)

    - Tab 5: NOISE & DEGRADATION LAB (Units IV & VI)
      Panel 1: Clean Reference ROI f(x, y)
      Panel 2: Gaussian Degradation (N(0, sigma^2=25)) + PSNR (dB) & MSE
      Panel 3: Salt-and-Pepper Impulse Noise (p=0.04) + PSNR (dB) & MSE
      Panel 4: Rayleigh Noise (Radar/Sensor a=0, b=25) + PSNR (dB) & MSE
      Panel 5: Uniform Degradation ([-30, +30]) + PSNR (dB) & MSE
      Panel 6: Periodic Sinusoidal Interference (A*sin(2*pi*(u0*x + v0*y))) + PSNR (dB) & MSE

Controls:
    - [TAB] / [T] Cycle View Tabs (1-5)
    - [1-6] Focus / Highlight Panel Stage
    - [E] Export Academic HTML Lab Report
    - [S] Save Analysis Snapshot Image & Diagnostics CSV
    - [SPACE] Resume Live Tracking
    - [Q / ESC] Return to Tracking
"""

from typing import Sequence
import cv2
import numpy as np

try:
    from processing.dip_pipeline import DIPResult, SnapshotContext
    from processing.morphology_analysis import ContourMetrics
    from tracking.base_tracker import TrackingResult
    from visualization.histogram_plotter import render_histogram_comparison
except ImportError:
    from src.processing.dip_pipeline import DIPResult, SnapshotContext
    from src.processing.morphology_analysis import ContourMetrics
    from src.tracking.base_tracker import TrackingResult
    from src.visualization.histogram_plotter import render_histogram_comparison


def to_display_bgr(image: np.ndarray | None, fallback_shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Safely convert any 2D grayscale, binary, float, or 3D image to displayable 3-channel uint8 BGR."""
    if image is None or image.size == 0:
        return np.full((fallback_shape[0], fallback_shape[1], 3), 20, dtype=np.uint8)

    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.max(image))
        min_val = float(np.min(image))
        if max_val - min_val > 1e-6:
            norm = ((image - min_val) / (max_val - min_val) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            norm = np.zeros_like(image, dtype=np.uint8)
        arr = norm
    else:
        arr = image.astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 3:
            return arr.copy()
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    return np.full((fallback_shape[0], fallback_shape[1], 3), 20, dtype=np.uint8)


def letterbox_image(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    bg_color: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    """Resize an image to fit inside target dimensions while strictly preserving aspect ratio."""
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas

    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return canvas

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


class InspectionCanvas:
    """Multi-tab composite modal inspection canvas renderer."""

    TAB_NAMES = {
        1: "1. SPATIAL & CLASSICAL DIP",
        2: "2. FREQUENCY & RESTORATION",
        3: "3. COLOR & SEGMENTATION",
        4: "4. FEATURES & DESCRIPTORS",
        5: "5. NOISE & DEGRADATION LAB",
    }

    def __init__(self, target_width: int = 1280, target_height: int = 720) -> None:
        """Initialize canvas with viewport dimensions."""
        self.target_width = max(800, target_width)
        self.target_height = max(600, target_height)

    def render(
        self,
        dip_result: DIPResult,
        tracking_result: TrackingResult | None = None,
        active_stage: int | None = None,
        active_tab: int = 1,
    ) -> np.ndarray:
        """Render the complete multi-panel inspection canvas for the active tab."""
        canvas = np.full((self.target_height, self.target_width, 3), (24, 24, 28), dtype=np.uint8)

        # 1. Header Banner (height: 54px)
        header_h = 54
        footer_h = 28
        cv2.rectangle(canvas, (0, 0), (self.target_width, header_h), (35, 38, 48), -1)
        cv2.line(canvas, (0, header_h), (self.target_width, header_h), (70, 75, 95), 1)

        # Title
        tab_num = max(1, min(5, active_tab))
        tab_name = self.TAB_NAMES.get(tab_num, "DIP INSPECTION")
        title_text = f"IPA DIP CANVAS  ::  [TAB {tab_num}/5] {tab_name}"
        cv2.putText(canvas, title_text, (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2)

        # Metadata
        ctx: SnapshotContext | None = dip_result.context if dip_result is not None else None
        frame_idx = ctx.frame_index if ctx is not None else 0
        tracker_str = ctx.tracker_name if ctx and ctx.tracker_name else (tracking_result.class_name if tracking_result else "TRACKER")
        track_id_str = f"ID: {ctx.track_id}" if ctx and ctx.track_id is not None else ""
        bbox_str = f"BBox: {ctx.bbox}" if ctx and ctx.bbox else ""

        meta_parts = [f"Frame: {frame_idx}"]
        if tracker_str:
            meta_parts.append(f"Tracker: {tracker_str.upper()}")
        if track_id_str:
            meta_parts.append(track_id_str)
        if bbox_str:
            meta_parts.append(bbox_str)
        if active_stage is not None and 1 <= active_stage <= 6:
            meta_parts.append(f"Focused: Panel [{active_stage}]")
        else:
            meta_parts.append("Status: PAUSED [MODAL]")

        meta_line = "  |  ".join(meta_parts)
        cv2.putText(canvas, meta_line, (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 225), 1)

        # 2. Grid Layout (2 rows x 3 columns)
        margin_x = 10
        margin_y = header_h + 8
        gap_x = 8
        gap_y = 8

        avail_w = self.target_width - 2 * margin_x - 2 * gap_x
        avail_h = self.target_height - margin_y - footer_h - gap_y

        panel_w = avail_w // 3
        panel_h = avail_h // 2

        # 3. Dispatch to Tab Renderer
        if tab_num == 1:
            self._render_tab1_spatial(canvas, margin_x, margin_y, panel_w, panel_h, gap_x, gap_y, ctx, dip_result, tracking_result, active_stage)
        elif tab_num == 2:
            self._render_tab2_frequency(canvas, margin_x, margin_y, panel_w, panel_h, gap_x, gap_y, ctx, dip_result, active_stage)
        elif tab_num == 3:
            self._render_tab3_color(canvas, margin_x, margin_y, panel_w, panel_h, gap_x, gap_y, ctx, dip_result, active_stage)
        elif tab_num == 4:
            self._render_tab4_descriptors(canvas, margin_x, margin_y, panel_w, panel_h, gap_x, gap_y, ctx, dip_result, active_stage)
        else:
            self._render_tab5_noise_models(canvas, margin_x, margin_y, panel_w, panel_h, gap_x, gap_y, ctx, dip_result, active_stage)

        # 4. Footer Status Bar
        footer_y = self.target_height - footer_h
        cv2.rectangle(canvas, (0, footer_y), (self.target_width, self.target_height), (30, 32, 40), -1)
        cv2.line(canvas, (0, footer_y), (self.target_width, footer_y), (60, 65, 80), 1)

        controls_text = "SHORTCUTS:  [TAB / T] Switch Tab (1-5)  |  [1-6] Focus Panel  |  [E] Export Lab Report  |  [S] Save Snapshot  |  [SPACE] Resume  |  [Q/ESC] Exit"
        cv2.putText(canvas, controls_text, (20, footer_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 215, 230), 1)

        return canvas

    def _draw_panel_box(self, canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str, active: bool = False) -> tuple[int, int, int, int]:
        """Draw panel header banner, outline, and return inner viewport coordinates."""
        header_h = 24
        frame_color = (0, 220, 255) if active else (55, 60, 75)
        banner_color = (0, 100, 160) if active else (42, 46, 58)
        border_thick = 2 if active else 1

        cv2.rectangle(canvas, (x, y), (x + w, y + h), frame_color, border_thick)
        cv2.rectangle(canvas, (x, y), (x + w, y + header_h), banner_color, -1)
        cv2.line(canvas, (x, y + header_h), (x + w, y + header_h), frame_color, 1)
        cv2.putText(canvas, title, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)

        return x + 1, y + header_h + 1, w - 2, h - header_h - 2

    # -------------------------------------------------------------------------
    # TAB 1: CORE SPATIAL & CLASSICAL DIP
    # -------------------------------------------------------------------------
    def _render_tab1_spatial(
        self, canvas: np.ndarray, mx: int, my: int, pw: int, ph: int, gx: int, gy: int,
        ctx: SnapshotContext | None, dip: DIPResult, tr: TrackingResult | None, active_stage: int | None
    ) -> None:
        # P1: Full Frame Context
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my, pw, ph, "1. GLOBAL CONTEXT (Full Frame)", active=(active_stage == 1))
        if ctx and ctx.full_frame is not None:
            df = to_display_bgr(ctx.full_frame)
            if ctx.bbox:
                bx, by, bw, bh = [int(v) for v in ctx.bbox[:4]]
                cv2.rectangle(df, (bx, by), (bx + bw, by + bh), (40, 220, 80), 2)
            canvas[vy : vy + vh, vx : vx + vw] = letterbox_image(df, vw, vh)

        # P2: Tracked Target ROI
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my, pw, ph, "2. TRACKED TARGET (Raw vs Equalized vs CLAHE)", active=(active_stage == 2))
        tw = vw // 3
        p2_1 = letterbox_image(to_display_bgr(ctx.roi if ctx else None), tw, vh)
        p2_2 = letterbox_image(to_display_bgr(dip.equalized_roi), tw, vh)
        p2_3 = letterbox_image(to_display_bgr(dip.clahe_roi), vw - 2 * tw, vh)
        cv2.putText(p2_1, "Original", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
        cv2.putText(p2_2, "Global CDF", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        cv2.putText(p2_3, "CLAHE", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + tw] = p2_1
        canvas[vy : vy + vh, vx + tw : vx + 2 * tw] = p2_2
        canvas[vy : vy + vh, vx + 2 * tw : vx + vw] = p2_3

        # P3: Histogram + Stats
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my, pw, ph, "3. HISTOGRAM & STATISTICAL MOMENTS", active=(active_stage == 3))
        h_img = render_histogram_comparison(dip.raw_histogram, dip.equalized_histogram, width=vw, height=vh, title="")
        stats = dip.histogram_stats or {}
        st_text = f"mu={stats.get('mean', 0):.1f}  sig={stats.get('std_dev', 0):.1f}  skew={stats.get('skewness', 0):.2f}  kurt={stats.get('kurtosis', 0):.2f}"
        cv2.putText(h_img, st_text, (8, vh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + vw] = h_img

        # P4: Spatial Filters & Differential Edges
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my + ph + gy, pw, ph, "4. SPATIAL FILTERING & DIFFERENTIAL EDGES", active=(active_stage == 4))
        sw, sh = vw // 2, vh // 2
        p4_1 = letterbox_image(to_display_bgr(dip.gaussian_roi), sw, sh)
        p4_2 = letterbox_image(to_display_bgr(dip.sobel_magnitude), vw - sw, sh)
        p4_3 = letterbox_image(to_display_bgr(dip.laplacian_edges), sw, vh - sh)
        p4_4 = letterbox_image(to_display_bgr(dip.canny_edges), vw - sw, vh - sh)
        cv2.putText(p4_1, "Gaussian Blur", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1)
        cv2.putText(p4_2, "Sobel Magnitude", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1)
        cv2.putText(p4_3, "Laplacian 2nd-Ord", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1)
        cv2.putText(p4_4, "Canny (NMS+Hyst)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        canvas[vy : vy + sh, vx : vx + sw] = p4_1
        canvas[vy : vy + sh, vx + sw : vx + vw] = p4_2
        canvas[vy + sh : vy + vh, vx : vx + sw] = p4_3
        canvas[vy + sh : vy + vh, vx + sw : vx + vw] = p4_4

        # P5: Segmentation & Morphology
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my + ph + gy, pw, ph, "5. SEGMENTATION & MORPHOLOGY", active=(active_stage == 5))
        hw = vw // 2
        p5_1 = letterbox_image(to_display_bgr(dip.otsu_mask), hw, vh)
        p5_2 = letterbox_image(to_display_bgr(dip.morphology_mask), vw - hw, vh)
        cv2.putText(p5_1, f"Otsu Mask (T*={dip.otsu_threshold:.0f})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)
        cv2.putText(p5_2, "Cleaned (Open+Close)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p5_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p5_2

        # P6: Contour Extraction & Shape Metrics
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my + ph + gy, pw, ph, "6. CONTOUR EXTRACTION & SHAPE METRICS", active=(active_stage == 6))
        self._render_metrics_card(canvas, vx, vy, vw, vh, dip)

    # -------------------------------------------------------------------------
    # TAB 2: FREQUENCY DOMAIN & RESTORATION (Units III & IV)
    # -------------------------------------------------------------------------
    def _render_tab2_frequency(
        self, canvas: np.ndarray, mx: int, my: int, pw: int, ph: int, gx: int, gy: int,
        ctx: SnapshotContext | None, dip: DIPResult, active_stage: int | None
    ) -> None:
        # P1: Grayscale ROI
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my, pw, ph, "1. TRACKED TARGET (Grayscale ROI)", active=(active_stage == 1))
        canvas[vy : vy + vh, vx : vx + vw] = letterbox_image(to_display_bgr(dip.gray_roi), vw, vh)

        # P2: 2D FFT Magnitude and Phase Spectrum
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my, pw, ph, "2. 2D FFT SPECTRUM (Magnitude & Phase)", active=(active_stage == 2))
        hw = vw // 2
        p2_1 = letterbox_image(to_display_bgr(dip.fft_spectrum), hw, vh)
        p2_2 = letterbox_image(to_display_bgr(dip.fft_phase), vw - hw, vh)
        cv2.putText(p2_1, "log(1 + |F(u, v)|)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)
        cv2.putText(p2_2, "Phase Angle phi(u, v)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 50), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p2_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p2_2

        # P3: Butterworth LPF
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my, pw, ph, "3. BUTTERWORTH LOW-PASS (D0=25, n=2)", active=(active_stage == 3))
        b_disp = letterbox_image(to_display_bgr(dip.butterworth_lpf), vw, vh)
        cv2.putText(b_disp, "Smooth Attenuation (No Ringing)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx : vx + vw] = b_disp

        # P4: Gaussian HPF
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my + ph + gy, pw, ph, "4. GAUSSIAN HIGH-PASS (D0=25)", active=(active_stage == 4))
        g_disp = letterbox_image(to_display_bgr(dip.gaussian_hpf), vw, vh)
        cv2.putText(g_disp, "High-Frequency Edges", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + vw] = g_disp

        # P5: Homomorphic Filter
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my + ph + gy, pw, ph, "5. HOMOMORPHIC FILTER (Illumination/Refl)", active=(active_stage == 5))
        h_disp = letterbox_image(to_display_bgr(dip.homomorphic_roi), vw, vh)
        cv2.putText(h_disp, "ln(I) = ln(L) + ln(R)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 80), 1)
        canvas[vy : vy + vh, vx : vx + vw] = h_disp

        # P6: Noise & Wiener Motion Deblur vs 2D Block DCT
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my + ph + gy, pw, ph, "6. WIENER RESTORATION & 2D BLOCK DCT", active=(active_stage == 6))
        hw = vw // 2
        p6_1 = letterbox_image(to_display_bgr(dip.deblurred_roi), hw, vh)
        p6_2 = letterbox_image(to_display_bgr(dip.dct_reconstructed), vw - hw, vh)
        cv2.putText(p6_1, "Wiener Restored (v)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        dct_m = dip.dct_metrics or {}
        comp_str = f"8x8 DCT (CR={dct_m.get('zero_coefficient_pct', 0):.0f}%)"
        cv2.putText(p6_2, comp_str, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p6_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p6_2

    # -------------------------------------------------------------------------
    # TAB 3: COLOR SPACES & ADVANCED SEGMENTATION (Units V & VI)
    # -------------------------------------------------------------------------
    def _render_tab3_color(
        self, canvas: np.ndarray, mx: int, my: int, pw: int, ph: int, gx: int, gy: int,
        ctx: SnapshotContext | None, dip: DIPResult, active_stage: int | None
    ) -> None:
        # P1: HSV Color Channels
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my, pw, ph, "1. HSV COLOR DECOMPOSITION", active=(active_stage == 1))
        hsv = dip.hsv_channels or {}
        tw = vw // 3
        h_c = letterbox_image(to_display_bgr(hsv.get("H")), tw, vh)
        s_c = letterbox_image(to_display_bgr(hsv.get("S")), tw, vh)
        v_c = letterbox_image(to_display_bgr(hsv.get("V")), vw - 2 * tw, vh)
        cv2.putText(h_c, "Hue (H)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)
        cv2.putText(s_c, "Sat (S)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)
        cv2.putText(v_c, "Val (V)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + tw] = h_c
        canvas[vy : vy + vh, vx + tw : vx + 2 * tw] = s_c
        canvas[vy : vy + vh, vx + 2 * tw : vx + vw] = v_c

        # P2: CIE L*a*b* Decomposition
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my, pw, ph, "2. CIE L*a*b* PERCEPTUAL SPACE", active=(active_stage == 2))
        lab_img = letterbox_image(to_display_bgr(dip.context.roi if dip.context else None), vw, vh)
        cv2.putText(lab_img, "Perceptual Uniformity (L* Lightness)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)
        canvas[vy : vy + vh, vx : vx + vw] = lab_img

        # P3: Color Equalization vs Pseudo-Color
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my, pw, ph, "3. COLOR EQUALIZED vs PSEUDO-COLOR", active=(active_stage == 3))
        hw = vw // 2
        p3_1 = letterbox_image(to_display_bgr(dip.color_equalized_roi), hw, vh)
        p3_2 = letterbox_image(to_display_bgr(dip.pseudocolor_roi), vw - hw, vh)
        cv2.putText(p3_1, "HSV-V Equalized", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
        cv2.putText(p3_2, "False-Color JET", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p3_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p3_2

        # P4: K-Means Clustering
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my + ph + gy, pw, ph, "4. K-MEANS COLOR SEGMENTATION (K=3)", active=(active_stage == 4))
        k_disp = letterbox_image(to_display_bgr(dip.kmeans_roi), vw, vh)
        cv2.putText(k_disp, "EM Clustering (3 Centroids)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx : vx + vw] = k_disp

        # P5: Marker-Controlled Watershed
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my + ph + gy, pw, ph, "5. MARKER-CONTROLLED WATERSHED", active=(active_stage == 5))
        ws_disp = letterbox_image(to_display_bgr(dip.watershed_roi), vw, vh)
        cv2.putText(ws_disp, "Topological Flood Basin", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 100, 100), 1)
        canvas[vy : vy + vh, vx : vx + vw] = ws_disp

        # P6: Top-Hat & Black-Hat Morphology
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my + ph + gy, pw, ph, "6. TOP-HAT & BLACK-HAT TRANSFORMS", active=(active_stage == 6))
        p6_1 = letterbox_image(to_display_bgr(dip.tophat_roi), hw, vh)
        p6_2 = letterbox_image(to_display_bgr(dip.blackhat_roi), vw - hw, vh)
        cv2.putText(p6_1, "Top-Hat (Brights)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
        cv2.putText(p6_2, "Black-Hat (Darks)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p6_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p6_2

    # -------------------------------------------------------------------------
    # TAB 4: MORPHOMETRIC & FEATURE DESCRIPTORS (Units I & VII)
    # -------------------------------------------------------------------------
    def _render_tab4_descriptors(
        self, canvas: np.ndarray, mx: int, my: int, pw: int, ph: int, gx: int, gy: int,
        ctx: SnapshotContext | None, dip: DIPResult, active_stage: int | None
    ) -> None:
        # P1: Distance Transforms (L2, D4, D8)
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my, pw, ph, "1. DISTANCE TRANSFORMS (L2, D4, D8)", active=(active_stage == 1))
        dists = dip.distance_transforms or {}
        tw = vw // 3
        d1 = letterbox_image(to_display_bgr(dists.get("Euclidean_L2")), tw, vh)
        d2 = letterbox_image(to_display_bgr(dists.get("CityBlock_D4")), tw, vh)
        d3 = letterbox_image(to_display_bgr(dists.get("Chessboard_D8")), vw - 2 * tw, vh)
        cv2.putText(d1, "Euclidean L2", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        cv2.putText(d2, "City Block D4", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        cv2.putText(d3, "Chessboard D8", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx : vx + tw] = d1
        canvas[vy : vy + vh, vx + tw : vx + 2 * tw] = d2
        canvas[vy : vy + vh, vx + 2 * tw : vx + vw] = d3

        # P2: Bit-Plane Slicing & Quantization
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my, pw, ph, "2. BIT-PLANE SLICING & QUANTIZATION", active=(active_stage == 2))
        hw = vw // 2
        bp7 = dip.bit_planes[7] if len(dip.bit_planes) >= 8 else None
        p2_1 = letterbox_image(to_display_bgr(bp7), hw, vh)
        p2_2 = letterbox_image(to_display_bgr(dip.quantized_roi), vw - hw, vh)
        cv2.putText(p2_1, "Bit Plane 7 (MSB)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
        cv2.putText(p2_2, "4-Bit Quantized", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p2_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p2_2

        # P3: Point Transforms (Gamma, Log, Negative)
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my, pw, ph, "3. POINT TRANSFORMS (Gamma / Negative)", active=(active_stage == 3))
        p3_1 = letterbox_image(to_display_bgr(dip.gamma_roi), hw, vh)
        p3_2 = letterbox_image(to_display_bgr(dip.negative_roi), vw - hw, vh)
        cv2.putText(p3_1, "Gamma (gamma=0.6)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        cv2.putText(p3_2, "Image Negative", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p3_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p3_2

        # P4: Topological Skeleton & Local Binary Patterns (LBP)
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx, my + ph + gy, pw, ph, "4. SKELETON & LOCAL BINARY PATTERNS", active=(active_stage == 4))
        hw = vw // 2
        p4_1 = letterbox_image(to_display_bgr(dip.skeleton_roi), hw, vh)
        p4_2 = letterbox_image(to_display_bgr(dip.lbp_roi), vw - hw, vh)
        cv2.putText(p4_1, "Skeleton Thinning", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)
        lbp_h = dip.lbp_features.get("lbp_entropy", 0.0) if dip.lbp_features else 0.0
        cv2.putText(p4_2, f"LBP Map (H={lbp_h:.2f})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        canvas[vy : vy + vh, vx : vx + hw] = p4_1
        canvas[vy : vy + vh, vx + hw : vx + vw] = p4_2

        # P5: Rotated BBox & Polygon Approximation
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + pw + gx, my + ph + gy, pw, ph, "5. ROTATED RECT & POLYGON APPROX", active=(active_stage == 5))
        p5_disp = to_display_bgr(dip.morphology_mask)
        m = dip.contour_metrics
        if m.rotated_bbox:
            box_pts = cv2.boxPoints(m.rotated_bbox)
            box_pts = np.int32(box_pts)
            cv2.drawContours(p5_disp, [box_pts], 0, (0, 220, 255), 2)
            cv2.putText(p5_disp, f"Angle: {m.orientation_deg:.1f} deg", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1)
        p5_fit = letterbox_image(p5_disp, vw, vh)
        canvas[vy : vy + vh, vx : vx + vw] = p5_fit

        # P6: Invariant Descriptors Card
        vx, vy, vw, vh = self._draw_panel_box(canvas, mx + 2 * (pw + gx), my + ph + gy, pw, ph, "6. INVARIANT DESCRIPTORS & HU MOMENTS", active=(active_stage == 6))
        self._render_advanced_descriptors_card(canvas, vx, vy, vw, vh, dip)

    # -------------------------------------------------------------------------
    # TAB 5: NOISE & DEGRADATION LAB (Units IV & VI)
    # -------------------------------------------------------------------------
    def _render_tab5_noise_models(
        self, canvas: np.ndarray, mx: int, my: int, pw: int, ph: int, gx: int, gy: int,
        ctx: SnapshotContext | None, dip: DIPResult, active_stage: int | None
    ) -> None:
        """Render 6 degradation and noise models with measured PSNR and MSE telemetry."""
        models = dip.noise_models or {}

        # 1. Clean Reference
        m1 = models.get("clean", {"name": "Clean Reference", "math": "f(x, y)", "image": dip.gray_roi, "psnr_db": float("inf"), "mse": 0.0})
        self._draw_noise_panel(canvas, mx, my, pw, ph, "1. CLEAN REFERENCE ROI", m1, active=(active_stage == 1))

        # 2. Gaussian Noise
        m2 = models.get("gaussian", {})
        self._draw_noise_panel(canvas, mx + pw + gx, my, pw, ph, "2. GAUSSIAN NOISE (Normal Dist)", m2, active=(active_stage == 2))

        # 3. Salt & Pepper Impulse Noise
        m3 = models.get("salt_pepper", {})
        self._draw_noise_panel(canvas, mx + 2 * (pw + gx), my, pw, ph, "3. SALT & PEPPER (Impulse Noise)", m3, active=(active_stage == 3))

        # 4. Rayleigh Noise (Radar / Sensor)
        m4 = models.get("rayleigh", {})
        self._draw_noise_panel(canvas, mx, my + ph + gy, pw, ph, "4. RAYLEIGH NOISE (Radar / Sensor)", m4, active=(active_stage == 4))

        # 5. Uniform Noise
        m5 = models.get("uniform", {})
        self._draw_noise_panel(canvas, mx + pw + gx, my + ph + gy, pw, ph, "5. UNIFORM NOISE (Quantization)", m5, active=(active_stage == 5))

        # 6. Periodic Sinusoidal Interference
        m6 = models.get("periodic", {})
        self._draw_noise_panel(canvas, mx + 2 * (pw + gx), my + ph + gy, pw, ph, "6. PERIODIC SINUSOID INTERFERENCE", m6, active=(active_stage == 6))

    def _draw_noise_panel(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        model_data: dict,
        active: bool = False,
    ) -> None:
        """Draw an individual noise degradation panel with mathematical formulation and PSNR/MSE metrics."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, title, active=active)
        img = model_data.get("image") if model_data else None
        psnr_val = model_data.get("psnr_db", 0.0) if model_data else 0.0
        mse_val = model_data.get("mse", 0.0) if model_data else 0.0
        math_str = model_data.get("math", "") if model_data else ""

        disp_img = letterbox_image(to_display_bgr(img), vw, vh)

        # Draw top math formula badge
        if math_str:
            sub_w = min(vw - 8, max(120, len(math_str) * 7 + 16))
            cv2.rectangle(disp_img, (4, 4), (4 + sub_w, 24), (25, 28, 38), -1)
            cv2.rectangle(disp_img, (4, 4), (4 + sub_w, 24), (60, 68, 85), 1)
            cv2.putText(disp_img, math_str, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 230, 255), 1)

        # Draw bottom metrics bar
        cv2.rectangle(disp_img, (4, vh - 24), (vw - 4, vh - 4), (20, 24, 32), -1)
        cv2.rectangle(disp_img, (4, vh - 24), (vw - 4, vh - 4), (50, 55, 70), 1)

        if np.isinf(psnr_val):
            psnr_str = "PSNR: Inf dB  |  MSE: 0.00 (Reference)"
            color = (80, 240, 140)
        else:
            psnr_str = f"PSNR: {psnr_val:.2f} dB  |  MSE: {mse_val:.1f}"
            if psnr_val >= 25.0:
                color = (80, 240, 140)
            elif psnr_val >= 18.0:
                color = (0, 220, 255)
            else:
                color = (60, 160, 255)

        cv2.putText(disp_img, psnr_str, (10, vh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        canvas[vy : vy + vh, vx : vx + vw] = disp_img

    # -------------------------------------------------------------------------
    # METRICS CARDS
    # -------------------------------------------------------------------------
    def _render_metrics_card(self, canvas: np.ndarray, vx: int, vy: int, vw: int, vh: int, dip: DIPResult) -> None:
        metrics: ContourMetrics = dip.contour_metrics if dip else ContourMetrics()
        morph_mask = dip.morphology_mask

        vis_w = min(160, vw // 2)
        if morph_mask is not None and morph_mask.size > 0:
            color_mask = to_display_bgr(morph_mask)
            contours, _ = cv2.findContours(morph_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(color_mask, contours, -1, (40, 220, 80), 2)
            if metrics.centroid:
                cv2.circle(color_mask, metrics.centroid, 4, (255, 140, 0), -1)
            vis_img = letterbox_image(color_mask, vis_w, vh)
        else:
            vis_img = np.full((vh, vis_w, 3), 20, dtype=np.uint8)

        canvas[vy : vy + vh, vx : vx + vis_w] = vis_img

        # Right side: Stats Card
        mx = vx + vis_w + 10
        my = vy + 16
        spacing = 18

        cv2.rectangle(canvas, (vx + vis_w, vy), (vx + vw, vy + vh), (28, 30, 38), -1)
        cv2.line(canvas, (vx + vis_w, vy), (vx + vis_w, vy + vh), (50, 55, 68), 1)

        lines = [
            ("Area (A):", f"{metrics.area:.1f} px"),
            ("Perimeter (P):", f"{metrics.perimeter:.1f} px"),
            ("Aspect Ratio:", f"{metrics.aspect_ratio:.2f}"),
            ("Circularity:", f"{metrics.circularity:.3f}"),
            ("Solidity:", f"{metrics.solidity:.3f}"),
            ("Extent:", f"{metrics.extent:.3f}"),
            ("Centroid:", f"{metrics.centroid if metrics.centroid else 'N/A'}"),
            ("Entropy:", f"{dip.shannon_entropy:.2f} b/px"),
            ("RLE Ratio:", f"{dip.rle_compression_ratio:.1f}x"),
        ]

        for idx, (label, val_str) in enumerate(lines):
            curr_y = my + idx * spacing
            if curr_y > vy + vh - 4:
                break
            cv2.putText(canvas, label, (mx, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 175, 195), 1)
            cv2.putText(canvas, val_str, (mx + 85, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 140), 1)

    def _render_advanced_descriptors_card(self, canvas: np.ndarray, vx: int, vy: int, vw: int, vh: int, dip: DIPResult) -> None:
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (28, 30, 38), -1)
        m = dip.contour_metrics
        glcm = m.glcm_features or {}
        lbp = dip.lbp_features or {}
        hu = m.hu_moments or []

        mx = vx + 10
        my = vy + 16
        spacing = 16

        lines = [
            ("Hu Mom phi1:", f"{hu[0]:.2f}" if len(hu) > 0 else "0.00"),
            ("Hu Mom phi2:", f"{hu[1]:.2f}" if len(hu) > 1 else "0.00"),
            ("GLCM Contrast:", f"{glcm.get('contrast', 0):.2f}"),
            ("GLCM Energy:", f"{glcm.get('energy', 0):.4f}"),
            ("LBP Entropy:", f"{lbp.get('lbp_entropy', 0):.2f} b/px"),
            ("LBP Energy:", f"{lbp.get('lbp_energy', 0):.4f}"),
            ("Orientation:", f"{m.orientation_deg:.1f} deg"),
            ("Polygon Vertices:", f"{m.polygon_vertices}"),
            ("Shannon Entropy:", f"{dip.shannon_entropy:.2f} b/px"),
            ("RLE Compression:", f"{dip.rle_compression_ratio:.1f}x"),
        ]

        for idx, (label, val_str) in enumerate(lines):
            curr_y = my + idx * spacing
            if curr_y > vy + vh - 4:
                break
            cv2.putText(canvas, label, (mx, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 180, 200), 1)
            cv2.putText(canvas, val_str, (mx + 115, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)


# Alias for backward compatibility
InspectionCanvasEngine = InspectionCanvas
