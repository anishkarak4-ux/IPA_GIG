"""Modal composite inspection canvas for post-snapshot DIP evaluation.

Constructs an academic multi-panel visual dashboard displaying both
contextual reference (Full Frame) and granular digital image processing
stages on the tracked target ROI:
    - Panel 1: Global Full Frame with target tracking bounding box
    - Panel 2: Tracked Target ROI (Raw vs Equalized comparison)
    - Panel 3: 256-Bin Intensity Histogram (Raw vs Equalized curves)
    - Panel 4: Spatial Filtering & Differential Edges (Gaussian, Median, Sobel, Canny)
    - Panel 5: Automated Segmentation & Mathematical Morphology (Otsu vs Morph Mask)
    - Panel 6: Contour Extraction & Shape Geometry Descriptors (Area, Perimeter, Aspect Ratio,
               Circularity, Solidity, Centroid)

Architecture Constraints:
    - Strictly non-mutating on input snapshot and analytical arrays.
    - Pure rendering function returning an OpenCV BGR numpy array without calling cv2.imshow().
    - Adaptive layout designed for standard 1280x720 / 1366x768 display viewports.
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
    """Safely convert any 2D grayscale, binary, float, or 3D image to displayable 3-channel uint8 BGR.

    Args:
        image: Input array or None.
        fallback_shape: Shape (height, width) for placeholder if image is None or empty.

    Returns:
        3-channel uint8 BGR numpy array.
    """
    if image is None or image.size == 0:
        return np.full((fallback_shape[0], fallback_shape[1], 3), 20, dtype=np.uint8)

    # Normalize floating-point arrays to [0, 255]
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
    """Resize an image to fit inside target dimensions while strictly preserving aspect ratio.

    Args:
        image: Source BGR image.
        target_w: Desired viewport width.
        target_h: Desired viewport height.
        bg_color: Background color for empty letterbox margins.

    Returns:
        New image array of exact shape (target_h, target_w, 3).
    """
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
    """Multi-panel modal composite inspection canvas renderer."""

    def __init__(self, target_width: int = 1280, target_height: int = 720) -> None:
        """Initialize canvas with viewport dimensions.

        Args:
            target_width: Total canvas width in pixels (default 1280).
            target_height: Total canvas height in pixels (default 720).
        """
        self.target_width = max(800, target_width)
        self.target_height = max(600, target_height)

    def render(
        self,
        dip_result: DIPResult,
        tracking_result: TrackingResult | None = None,
        active_stage: int | None = None,
    ) -> np.ndarray:
        """Render the complete multi-panel inspection canvas.

        Args:
            dip_result: Output container from DIPPipeline.process().
            tracking_result: Optional live TrackingResult for additional metadata.
            active_stage: Optional integer (1-6) indicating selected/highlighted stage.

        Returns:
            Rendered composite BGR image array of exact shape (target_height, target_width, 3).
        """
        canvas = np.full((self.target_height, self.target_width, 3), (24, 24, 28), dtype=np.uint8)

        # -----------------------------------------------------------------
        # 1. Header Banner (height: 52px)
        # -----------------------------------------------------------------
        header_h = 52
        footer_h = 28
        cv2.rectangle(canvas, (0, 0), (self.target_width, header_h), (35, 38, 48), -1)
        cv2.line(canvas, (0, header_h), (self.target_width, header_h), (70, 75, 95), 1)

        # Header Title
        title_text = "IPA DIGITAL IMAGE PROCESSING (DIP) INSPECTION CANVAS"
        cv2.putText(canvas, title_text, (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)

        # Extract context metadata
        ctx: SnapshotContext | None = dip_result.context if dip_result is not None else None
        frame_idx = ctx.frame_index if ctx is not None else 0
        tracker_str = ctx.tracker_name if ctx and ctx.tracker_name else (tracking_result.class_name if tracking_result else "TRACKER")
        track_id_str = f"ID: {ctx.track_id}" if ctx and ctx.track_id is not None else (f"ID: {tracking_result.track_id}" if tracking_result and tracking_result.track_id is not None else "")
        bbox_str = f"BBox: {ctx.bbox}" if ctx and ctx.bbox else ""

        meta_parts = [f"Frame: {frame_idx}"]
        if tracker_str:
            meta_parts.append(f"Algorithm: {tracker_str.upper()}")
        if track_id_str:
            meta_parts.append(track_id_str)
        if bbox_str:
            meta_parts.append(bbox_str)
        if active_stage is not None and 1 <= active_stage <= 6:
            meta_parts.append(f"Stage: [{active_stage}/6 SELECTED]")
        else:
            meta_parts.append("Status: PAUSED [SNAPSHOT]")

        meta_line = "  |  ".join(meta_parts)
        cv2.putText(canvas, meta_line, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 220), 1)

        # -----------------------------------------------------------------
        # 2. Grid Layout Computation (2 rows x 3 columns)
        # -----------------------------------------------------------------
        margin_x = 10
        margin_y = header_h + 8
        gap_x = 8
        gap_y = 8

        avail_w = self.target_width - 2 * margin_x - 2 * gap_x
        avail_h = self.target_height - margin_y - footer_h - gap_y

        panel_w = avail_w // 3
        panel_h = avail_h // 2

        # -----------------------------------------------------------------
        # 3. Render 6 Specialized Panels
        # -----------------------------------------------------------------
        # Panel 1: Full Frame Context (Row 0, Col 0)
        p1_x = margin_x
        p1_y = margin_y
        self._render_panel_full_frame(canvas, p1_x, p1_y, panel_w, panel_h, ctx, tracking_result, active=(active_stage == 1))

        # Panel 2: Tracked Target ROI (Row 0, Col 1)
        p2_x = margin_x + panel_w + gap_x
        p2_y = margin_y
        self._render_panel_roi_comparison(canvas, p2_x, p2_y, panel_w, panel_h, ctx, dip_result, active=(active_stage == 2))

        # Panel 3: 256-Bin Intensity Histogram (Row 0, Col 2)
        p3_x = margin_x + 2 * (panel_w + gap_x)
        p3_y = margin_y
        self._render_panel_histogram(canvas, p3_x, p3_y, panel_w, panel_h, dip_result, active=(active_stage == 3))

        # Panel 4: Spatial Filtering & Differential Edges (Row 1, Col 0)
        p4_x = margin_x
        p4_y = margin_y + panel_h + gap_y
        self._render_panel_edges_filters(canvas, p4_x, p4_y, panel_w, panel_h, dip_result, active=(active_stage == 4))

        # Panel 5: Segmentation & Mathematical Morphology (Row 1, Col 1)
        p5_x = margin_x + panel_w + gap_x
        p5_y = margin_y + panel_h + gap_y
        self._render_panel_segmentation(canvas, p5_x, p5_y, panel_w, panel_h, dip_result, active=(active_stage == 5))

        # Panel 6: Contour Extraction & Shape Metrics (Row 1, Col 2)
        p6_x = margin_x + 2 * (panel_w + gap_x)
        p6_y = margin_y + panel_h + gap_y
        self._render_panel_contours(canvas, p6_x, p6_y, panel_w, panel_h, dip_result, active=(active_stage == 6))

        # -----------------------------------------------------------------
        # 4. Footer Status Bar
        # -----------------------------------------------------------------
        footer_y = self.target_height - footer_h
        cv2.rectangle(canvas, (0, footer_y), (self.target_width, self.target_height), (30, 32, 40), -1)
        cv2.line(canvas, (0, footer_y), (self.target_width, footer_y), (60, 65, 80), 1)

        controls_text = "KEYBOARD SHORTCUTS:  [1-6] Select Stage  |  [S] Save Analysis Report  |  [SPACE] Resume Tracking  |  [Q / ESC] Return"
        cv2.putText(canvas, controls_text, (20, footer_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1)

        return canvas

    def _draw_panel_box(self, canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str, active: bool = False) -> tuple[int, int, int, int]:
        """Draw panel header banner, outline, and return inner viewport coordinates."""
        header_h = 24
        frame_color = (0, 220, 255) if active else (55, 60, 75)
        banner_color = (0, 100, 160) if active else (42, 46, 58)
        border_thick = 2 if active else 1

        # Panel frame
        cv2.rectangle(canvas, (x, y), (x + w, y + h), frame_color, border_thick)
        # Header banner
        cv2.rectangle(canvas, (x, y), (x + w, y + header_h), banner_color, -1)
        cv2.line(canvas, (x, y + header_h), (x + w, y + header_h), frame_color, 1)
        # Title text
        cv2.putText(canvas, title, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)

        return x + 1, y + header_h + 1, w - 2, h - header_h - 2

    def _render_panel_full_frame(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        ctx: SnapshotContext | None,
        tracking_result: TrackingResult | None,
        active: bool = False,
    ) -> None:
        """Panel 1: Global full frame with target bounding box."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, "1. GLOBAL CONTEXT (Full Frame)", active=active)

        if ctx is None or ctx.full_frame is None or ctx.full_frame.size == 0:
            cv2.putText(canvas, "No Full Frame", (vx + 20, vy + vh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return

        display_frame = to_display_bgr(ctx.full_frame)

        # Draw bounding box on full frame
        if ctx.bbox is not None:
            bx, by, bw, bh = [int(v) for v in ctx.bbox[:4]]
            if bw > 0 and bh > 0:
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (40, 220, 80), 2)
                cx = bx + bw // 2
                cy = by + bh // 2
                cv2.circle(display_frame, (cx, cy), 5, (255, 140, 0), -1)
                label = f"Target {ctx.track_id}" if ctx.track_id is not None else "Target"
                cv2.putText(display_frame, label, (bx, max(20, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3)
                cv2.putText(display_frame, label, (bx, max(20, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

        # Fit into viewport
        content = letterbox_image(display_frame, vw, vh)
        canvas[vy : vy + vh, vx : vx + vw] = content

    def _render_panel_roi_comparison(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        ctx: SnapshotContext | None,
        dip_result: DIPResult | None,
        active: bool = False,
    ) -> None:
        """Panel 2: Tracked Target ROI (Raw vs Equalized side-by-side)."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, "2. TRACKED TARGET (Raw vs Equalized ROI)", active=active)

        raw_roi = ctx.roi if ctx is not None else None
        eq_roi = dip_result.equalized_roi if dip_result is not None else None

        half_w = vw // 2
        # Left half: Raw ROI
        if raw_roi is not None and raw_roi.size > 0:
            left_img = letterbox_image(to_display_bgr(raw_roi), half_w, vh)
            cv2.putText(left_img, "Original Color", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3)
            cv2.putText(left_img, "Original Color", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
        else:
            left_img = np.full((vh, half_w, 3), 25, dtype=np.uint8)
            cv2.putText(left_img, "No ROI", (20, vh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Right half: Equalized ROI
        if eq_roi is not None and eq_roi.size > 0:
            right_img = letterbox_image(to_display_bgr(eq_roi), vw - half_w, vh)
            cv2.putText(right_img, "Equalized (CDF)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3)
            cv2.putText(right_img, "Equalized (CDF)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 240, 140), 1)
        else:
            right_img = np.full((vh, vw - half_w, 3), 25, dtype=np.uint8)
            cv2.putText(right_img, "No Equalized", (20, vh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Composite side by side
        canvas[vy : vy + vh, vx : vx + half_w] = left_img
        canvas[vy : vy + vh, vx + half_w : vx + vw] = right_img
        # Center dividing line
        cv2.line(canvas, (vx + half_w, vy), (vx + half_w, vy + vh), (55, 60, 75), 1)

    def _render_panel_histogram(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        dip_result: DIPResult | None,
        active: bool = False,
    ) -> None:
        """Panel 3: Dual 256-bin intensity histogram comparison."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, "3. INTENSITY HISTOGRAM (Raw vs Equalized)", active=active)

        raw_hist = dip_result.raw_histogram if dip_result is not None else None
        eq_hist = dip_result.equalized_histogram if dip_result is not None else None

        hist_img = render_histogram_comparison(
            raw_hist=raw_hist,
            eq_hist=eq_hist,
            width=vw,
            height=vh,
            title="",  # Title handled by panel header
            bg_color=(20, 20, 22),
            grid_color=(45, 48, 58),
            text_color=(200, 205, 215),
        )
        canvas[vy : vy + vh, vx : vx + vw] = hist_img

    def _render_panel_edges_filters(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        dip_result: DIPResult | None,
        active: bool = False,
    ) -> None:
        """Panel 4: Spatial Filtering & Differential Edges (2x2 Quad Grid)."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, "4. SPATIAL FILTERING & DIFFERENTIAL EDGES", active=active)

        sub_w = vw // 2
        sub_h = vh // 2

        # Top-Left: Gaussian Blur
        g_img = letterbox_image(to_display_bgr(dip_result.gaussian_roi if dip_result else None), sub_w, sub_h)
        cv2.putText(g_img, "Gaussian Blur", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1)
        canvas[vy : vy + sub_h, vx : vx + sub_w] = g_img

        # Top-Right: Sobel Magnitude
        s_img = letterbox_image(to_display_bgr(dip_result.sobel_magnitude if dip_result else None), vw - sub_w, sub_h)
        cv2.putText(s_img, "Sobel Magnitude", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1)
        canvas[vy : vy + sub_h, vx + sub_w : vx + vw] = s_img

        # Bottom-Left: Laplacian
        l_img = letterbox_image(to_display_bgr(dip_result.laplacian_edges if dip_result else None), sub_w, vh - sub_h)
        cv2.putText(l_img, "Laplacian (2nd Order)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1)
        canvas[vy + sub_h : vy + vh, vx : vx + sub_w] = l_img

        # Bottom-Right: Canny Edges
        c_img = letterbox_image(to_display_bgr(dip_result.canny_edges if dip_result else None), vw - sub_w, vh - sub_h)
        cv2.putText(c_img, "Canny (NMS + Hyst)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 240, 140), 1)
        canvas[vy + sub_h : vy + vh, vx + sub_w : vx + vw] = c_img

        # Dividing grid lines
        cv2.line(canvas, (vx + sub_w, vy), (vx + sub_w, vy + vh), (50, 55, 68), 1)
        cv2.line(canvas, (vx, vy + sub_h), (vx + vw, vy + sub_h), (50, 55, 68), 1)

    def _render_panel_segmentation(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        dip_result: DIPResult | None,
        active: bool = False,
    ) -> None:
        """Panel 5: Segmentation & Mathematical Morphology (Otsu vs Morph Cleaned)."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, "5. SEGMENTATION & MORPHOLOGY", active=active)

        half_w = vw // 2
        otsu_mask = dip_result.otsu_mask if dip_result else None
        morph_mask = dip_result.morphology_mask if dip_result else None
        t_val = dip_result.otsu_threshold if dip_result else 0.0

        # Left: Otsu Binary Mask
        left_img = letterbox_image(to_display_bgr(otsu_mask), half_w, vh)
        cv2.putText(left_img, f"Otsu Mask (t*={t_val:.0f})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3)
        cv2.putText(left_img, f"Otsu Mask (t*={t_val:.0f})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
        canvas[vy : vy + vh, vx : vx + half_w] = left_img

        # Right: Morphology Cleaned Mask (Open + Close)
        right_img = letterbox_image(to_display_bgr(morph_mask), vw - half_w, vh)
        cv2.putText(right_img, "Morphology (Open+Close)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3)
        cv2.putText(right_img, "Morphology (Open+Close)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 240, 140), 1)
        canvas[vy : vy + vh, vx + half_w : vx + vw] = right_img

        # Center dividing line
        cv2.line(canvas, (vx + half_w, vy), (vx + half_w, vy + vh), (55, 60, 75), 1)

    def _render_panel_contours(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        dip_result: DIPResult | None,
        active: bool = False,
    ) -> None:
        """Panel 6: Contour Extraction & Shape Geometry Descriptors."""
        vx, vy, vw, vh = self._draw_panel_box(canvas, x, y, w, h, "6. CONTOUR EXTRACTION & SHAPE METRICS", active=active)

        metrics: ContourMetrics = dip_result.contour_metrics if dip_result else ContourMetrics()
        morph_mask = dip_result.morphology_mask if dip_result else None

        # Left side (160px): Render contour on mask
        vis_w = min(170, vw // 2)
        metrics_w = vw - vis_w

        # Draw contour on color mask
        if morph_mask is not None and morph_mask.size > 0:
            color_mask = to_display_bgr(morph_mask)
            # Find contours for drawing
            contours, _ = cv2.findContours(morph_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(color_mask, contours, -1, (40, 220, 80), 2)
            # Draw centroid and bbox if present
            if metrics.centroid is not None:
                cv2.circle(color_mask, metrics.centroid, 4, (255, 140, 0), -1)
            if metrics.bounding_rect is not None:
                bx, by, bw, bh = metrics.bounding_rect
                cv2.rectangle(color_mask, (bx, by), (bx + bw, by + bh), (0, 220, 255), 1)
            vis_img = letterbox_image(color_mask, vis_w, vh)
        else:
            vis_img = np.full((vh, vis_w, 3), 20, dtype=np.uint8)
            cv2.putText(vis_img, "No Contour", (15, vh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)

        canvas[vy : vy + vh, vx : vx + vis_w] = vis_img

        # Right side: Formatted Academic Shape Metrics Card
        mx = vx + vis_w + 10
        my = vy + 18
        spacing = 20

        # Background card for metrics
        cv2.rectangle(canvas, (vx + vis_w, vy), (vx + vw, vy + vh), (28, 30, 38), -1)
        cv2.line(canvas, (vx + vis_w, vy), (vx + vis_w, vy + vh), (50, 55, 68), 1)

        metrics_lines = [
            ("Area (A):", f"{metrics.area:.1f} px"),
            ("Perimeter (P):", f"{metrics.perimeter:.1f} px"),
            ("Aspect Ratio:", f"{metrics.aspect_ratio:.2f}"),
            ("Circularity:", f"{metrics.circularity:.3f}"),
            ("Solidity:", f"{metrics.solidity:.3f}"),
            ("Centroid:", f"{metrics.centroid if metrics.centroid else 'N/A'}"),
            ("Contours:", f"{metrics.contour_count}"),
        ]

        for idx, (label, val_str) in enumerate(metrics_lines):
            curr_y = my + idx * spacing
            if curr_y > vy + vh - 6:
                break
            cv2.putText(canvas, label, (mx, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 195), 1)
            # Value in bright color
            val_x = mx + 95
            val_color = (80, 240, 140) if "Circularity" in label or "Area" in label else (240, 240, 240)
            cv2.putText(canvas, val_str, (val_x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, val_color, 1)


# Skeleton alias for backward compatibility
InspectionCanvasEngine = InspectionCanvas
