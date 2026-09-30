"""Visualization module providing stream overlay HUD, modal inspection canvas, and histogram plotting."""

from .histogram_plotter import (
    HistogramPlotter,
    render_histogram,
    render_histogram_comparison,
)
from .inspection_canvas import (
    InspectionCanvas,
    InspectionCanvasEngine,
    letterbox_image,
    to_display_bgr,
)
from .overlay import (
    TrackingVisualizer,
    draw_bounding_box,
    draw_hud,
    draw_trajectory,
    render_overlay,
)

__all__ = [
    "HistogramPlotter",
    "render_histogram",
    "render_histogram_comparison",
    "TrackingVisualizer",
    "draw_bounding_box",
    "draw_trajectory",
    "draw_hud",
    "render_overlay",
    "InspectionCanvas",
    "InspectionCanvasEngine",
    "to_display_bgr",
    "letterbox_image",
]
