"""OpenCV-native 256-bin histogram renderer for inline canvas visualization.

Renders discrete intensity histograms directly into an OpenCV BGR bitmap array
without requiring external plotting libraries like matplotlib.

Features:
    - Single histogram rendering (raw or equalized).
    - Dual histogram overlay comparison (Raw vs Equalized) with distinct color coding and legend.
    - Configurable canvas dimensions, background colors, and margin spacing.
    - Grid lines at quarter intervals (0, 64, 128, 192, 255).
    - Robust safety against empty arrays, all-zero histograms, single-value spikes, and extreme counts.
"""

from typing import Sequence
import cv2
import numpy as np


def render_histogram(
    hist: np.ndarray | None,
    width: int = 320,
    height: int = 200,
    color: tuple[int, int, int] = (80, 180, 240),
    title: str = "Intensity Histogram",
    bg_color: tuple[int, int, int] = (25, 25, 25),
    grid_color: tuple[int, int, int] = (50, 50, 50),
    text_color: tuple[int, int, int] = (220, 220, 220),
) -> np.ndarray:
    """Render a single 256-bin intensity histogram into an OpenCV BGR image.

    Args:
        hist: 1D or 2D array of 256 numerical frequency counts (float or int).
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        color: BGR line/fill color for the histogram curve.
        title: Title string displayed at the top.
        bg_color: BGR canvas background color.
        grid_color: BGR axis and grid line color.
        text_color: BGR label font color.

    Returns:
        Rendered 3-channel uint8 BGR image of shape (height, width, 3).
    """
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Empty / None guard
    if hist is None or hist.size == 0:
        cv2.putText(
            canvas, "No Histogram Data", (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1
        )
        return canvas

    hist_data = hist.flatten().astype(np.float64)
    if len(hist_data) != 256:
        cv2.putText(
            canvas, f"Invalid Bins: {len(hist_data)}", (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1
        )
        return canvas

    # Layout geometry
    margin_left = 40
    margin_right = 15
    margin_top = 30
    margin_bottom = 25

    plot_w = max(10, width - margin_left - margin_right)
    plot_h = max(10, height - margin_top - margin_bottom)

    # Title
    cv2.putText(canvas, title, (margin_left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1)

    # Draw grid & bounding axes
    cv2.rectangle(
        canvas,
        (margin_left, margin_top),
        (margin_left + plot_w, margin_top + plot_h),
        grid_color, 1
    )

    # Vertical grid markers for 64, 128, 192
    for tick in (64, 128, 192):
        x = int(margin_left + (tick / 255.0) * plot_w)
        cv2.line(canvas, (x, margin_top), (x, margin_top + plot_h), grid_color, 1)

    # X-axis tick labels
    cv2.putText(canvas, "0", (margin_left - 4, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
    cv2.putText(canvas, "128", (margin_left + plot_w // 2 - 12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
    cv2.putText(canvas, "255", (margin_left + plot_w - 18, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)

    # Max value scaling
    max_val = float(np.max(hist_data))
    if max_val <= 0:
        # All zeros
        cv2.line(canvas, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), color, 1)
        cv2.putText(canvas, "0", (6, margin_top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, text_color, 1)
        return canvas

    # Max Y-axis label
    max_str = f"{int(max_val)}" if max_val >= 10 else f"{max_val:.1f}"
    if len(max_str) > 5:
        max_str = f"{max_val:.1e}"
    cv2.putText(canvas, max_str, (4, margin_top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    # Generate polyline points
    points = []
    points.append([margin_left, margin_top + plot_h])
    for i in range(256):
        x = int(margin_left + (i / 255.0) * plot_w)
        norm_val = min(1.0, max(0.0, hist_data[i] / max_val))
        y = int(margin_top + plot_h - norm_val * plot_h)
        points.append([x, y])
    points.append([margin_left + plot_w, margin_top + plot_h])

    poly_pts = np.array(points, dtype=np.int32)
    # Semi-transparent polygon fill
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly_pts], color)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # Outline curve
    line_pts = poly_pts[1:-1]
    cv2.polylines(canvas, [line_pts], isClosed=False, color=color, thickness=2)

    return canvas


def render_histogram_comparison(
    raw_hist: np.ndarray | None,
    eq_hist: np.ndarray | None,
    width: int = 380,
    height: int = 210,
    raw_color: tuple[int, int, int] = (80, 180, 240),   # Light Blue / Orange
    eq_color: tuple[int, int, int] = (80, 240, 140),    # Light Green
    title: str = "Intensity Histogram (Raw vs Equalized)",
    bg_color: tuple[int, int, int] = (25, 25, 25),
    grid_color: tuple[int, int, int] = (50, 50, 50),
    text_color: tuple[int, int, int] = (220, 220, 220),
) -> np.ndarray:
    """Render dual overlay comparison of raw and equalized 256-bin histograms.

    Args:
        raw_hist: 256-bin array for the original grayscale image.
        eq_hist: 256-bin array for the equalized grayscale image.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        raw_color: BGR curve color for raw histogram.
        eq_color: BGR curve color for equalized histogram.
        title: Title banner string.
        bg_color: Canvas background color.
        grid_color: Axis and grid line color.
        text_color: Text font color.

    Returns:
        Rendered 3-channel uint8 BGR image of shape (height, width, 3).
    """
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Fallbacks if both or either is missing
    if (raw_hist is None or raw_hist.size == 0) and (eq_hist is None or eq_hist.size == 0):
        cv2.putText(
            canvas, "No Histogram Data Available", (width // 5, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1
        )
        return canvas

    if eq_hist is None or eq_hist.size == 0:
        return render_histogram(raw_hist, width, height, raw_color, "Raw Histogram", bg_color, grid_color, text_color)

    if raw_hist is None or raw_hist.size == 0:
        return render_histogram(eq_hist, width, height, eq_color, "Equalized Histogram", bg_color, grid_color, text_color)

    raw_data = raw_hist.flatten().astype(np.float64)
    eq_data = eq_hist.flatten().astype(np.float64)

    margin_left = 40
    margin_right = 15
    margin_top = 34
    margin_bottom = 25

    plot_w = max(10, width - margin_left - margin_right)
    plot_h = max(10, height - margin_top - margin_bottom)

    # Title
    cv2.putText(canvas, title, (margin_left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1)

    # Legend at top right
    legend_x = max(margin_left + 10, width - 150)
    # Raw legend
    cv2.rectangle(canvas, (legend_x, 8), (legend_x + 12, 18), raw_color, -1)
    cv2.putText(canvas, "Raw", (legend_x + 16, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
    # Equalized legend
    cv2.rectangle(canvas, (legend_x + 60, 8), (legend_x + 72, 18), eq_color, -1)
    cv2.putText(canvas, "Equalized", (legend_x + 76, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)

    # Grid box
    cv2.rectangle(
        canvas,
        (margin_left, margin_top),
        (margin_left + plot_w, margin_top + plot_h),
        grid_color, 1
    )

    # Vertical grid markers
    for tick in (64, 128, 192):
        x = int(margin_left + (tick / 255.0) * plot_w)
        cv2.line(canvas, (x, margin_top), (x, margin_top + plot_h), grid_color, 1)

    # X-axis tick labels
    cv2.putText(canvas, "0", (margin_left - 4, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
    cv2.putText(canvas, "128", (margin_left + plot_w // 2 - 12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
    cv2.putText(canvas, "255", (margin_left + plot_w - 18, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)

    # Unified scaling: scale by max of both distributions so comparative height is preserved
    max_val = max(float(np.max(raw_data)), float(np.max(eq_data)))
    if max_val <= 0:
        max_val = 1.0

    max_str = f"{int(max_val)}" if max_val >= 10 else f"{max_val:.1f}"
    if len(max_str) > 5:
        max_str = f"{max_val:.1e}"
    cv2.putText(canvas, max_str, (4, margin_top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    # Draw Raw Histogram
    raw_pts = []
    raw_pts.append([margin_left, margin_top + plot_h])
    for i in range(256):
        x = int(margin_left + (i / 255.0) * plot_w)
        norm_val = min(1.0, max(0.0, raw_data[i] / max_val))
        y = int(margin_top + plot_h - norm_val * plot_h)
        raw_pts.append([x, y])
    raw_pts.append([margin_left + plot_w, margin_top + plot_h])

    # Draw Equalized Histogram
    eq_pts = []
    eq_pts.append([margin_left, margin_top + plot_h])
    for i in range(256):
        x = int(margin_left + (i / 255.0) * plot_w)
        norm_val = min(1.0, max(0.0, eq_data[i] / max_val))
        y = int(margin_top + plot_h - norm_val * plot_h)
        eq_pts.append([x, y])
    eq_pts.append([margin_left + plot_w, margin_top + plot_h])

    # Fills with transparency
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [np.array(raw_pts, dtype=np.int32)], raw_color)
    cv2.fillPoly(overlay, [np.array(eq_pts, dtype=np.int32)], eq_color)
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

    # Outline curves
    cv2.polylines(canvas, [np.array(raw_pts[1:-1], dtype=np.int32)], isClosed=False, color=raw_color, thickness=2)
    cv2.polylines(canvas, [np.array(eq_pts[1:-1], dtype=np.int32)], isClosed=False, color=eq_color, thickness=2)

    return canvas


class HistogramPlotter:
    """Class wrapper providing static methods for OpenCV histogram generation."""

    render_histogram = staticmethod(render_histogram)
    render_comparison = staticmethod(render_histogram_comparison)
    render_histogram_comparison = staticmethod(render_histogram_comparison)
