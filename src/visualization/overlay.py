"""Tracking HUD and annotation visual overlays.

Renders live target tracking graphics onto video frames:
    - Target bounding box with high-contrast outline
    - Object centroid marker
    - Chronological trajectory path breadcrumb
    - Multi-line HUD telemetry panel with dual-pass dark outline rendering
    - Differentiates between Classical trackers and YOLO deep trackers
    - Strict non-mutation of input coordinates, frames, and trajectory queues
"""

from typing import Sequence
import cv2
import numpy as np

try:
    from tracking.base_tracker import TrackingResult
except ImportError:
    from src.tracking.base_tracker import TrackingResult


def draw_bounding_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    center: tuple[int, int] | None = None,
    success: bool = True,
    track_id: int | None = None,
    class_name: str | None = None,
    confidence: float | None = None,
    path_length: float | None = None,
    unit: str = "px",
    color: tuple[int, int, int] | None = None,
) -> None:
    """Draw tracking bounding box, centroid, and text labels directly onto frame.

    Args:
        frame: Target BGR image array (drawn in-place).
        bbox: Bounding box tuple (x, y, w, h).
        center: Centroid coordinate (cx, cy).
        success: True if active track lock, False if lost.
        track_id: Optional tracking ID (e.g. from YOLO).
        class_name: Optional detected object class name (e.g. from YOLO).
        confidence: Optional detection confidence in [0.0, 1.0].
        path_length: Optional cumulative distance for this track.
        unit: Measurement unit string ('px' or 'm').
        color: Optional BGR color override.
    """
    if bbox is None:
        return

    x, y, w, h = [int(v) for v in bbox[:4]]
    if w <= 0 or h <= 0:
        return

    box_color = color if color is not None else ((40, 180, 80) if success else (0, 0, 240))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Centroid
    cx, cy = center if center is not None else (x + w // 2, y + h // 2)
    cv2.circle(frame, (cx, cy), 4, (255, 140, 0), -1)

    # Construct label parts
    label_parts: list[str] = []
    if track_id is not None:
        label_parts.append(f"ID {track_id}")
    if class_name is not None and class_name.strip():
        label_parts.append(class_name.strip())
    if confidence is not None and confidence > 0.0:
        label_parts.append(f"{confidence:.2f}")
    if path_length is not None and path_length > 0.0:
        label_parts.append(f"{path_length:.0f}{unit}")

    if label_parts:
        label = " ".join(label_parts)
        text_y = max(24, y - 8)
        # High-contrast background outline
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 4)
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)


def draw_trajectory(
    frame: np.ndarray,
    points: Sequence[tuple[int, int]],
    color: tuple[int, int, int] = (255, 140, 0),
) -> None:
    """Draw chronological trajectory breadcrumb path across sequential frames.

    Args:
        frame: Target BGR image array (drawn in-place).
        points: Ordered sequence of (x, y) centroid coordinates.
        color: BGR trail line color.
    """
    if not points or len(points) < 2:
        if points and len(points) == 1 and points[0] is not None:
            cv2.circle(frame, points[0], 3, color, -1)
        return

    # Draw lines with dynamic thickness
    for index in range(1, len(points)):
        pt1 = points[index - 1]
        pt2 = points[index]
        if pt1 is not None and pt2 is not None:
            thickness = max(1, min(4, int(np.sqrt(index + 1) / 2)))
            cv2.line(frame, pt1, pt2, color, thickness)


def draw_hud(
    frame: np.ndarray,
    tracker_name: str = "",
    success: bool = True,
    fps: float = 0.0,
    total_distance: float = 0.0,
    unit: str = "px",
    frame_index: int | None = None,
    velocity: float | None = None,
    mirror: bool = False,
    preprocess: bool = False,
    calibration_enabled: bool = False,
    extra_lines: list[str] | None = None,
) -> None:
    """Render comprehensive tracking telemetry HUD in the top/bottom of frame.

    Args:
        frame: Target BGR image array (drawn in-place).
        tracker_name: Name of active tracking algorithm.
        success: Whether tracker currently holds an active lock.
        fps: Rolling average frames per second.
        total_distance: Cumulative traveled distance.
        unit: Measurement unit string ('px' or 'm').
        frame_index: Optional sequential frame number.
        velocity: Optional instantaneous velocity.
        mirror: Mirror mode indicator flag.
        preprocess: DIP preprocessing indicator flag.
        calibration_enabled: Real-world metric calibration flag.
        extra_lines: Optional additional lines to display.
    """
    height, width = frame.shape[:2]

    # Top HUD information
    info_lines: list[str] = []

    # FPS line
    if fps > 0.0:
        info_lines.append(f"FPS: {fps:.1f}")

    # Tracker & Status
    status_text = "Tracking" if success else "Lost / Searching"
    name_str = f" [{tracker_name.upper()}]" if tracker_name else ""
    info_lines.append(f"Status: {status_text}{name_str}")

    # Trajectory path length
    info_lines.append(f"Path: {total_distance:.1f} {unit}")

    # Velocity
    if velocity is not None and velocity > 0.0:
        info_lines.append(f"Velocity: {velocity:.1f} {unit}/s")

    # Frame Index
    if frame_index is not None:
        info_lines.append(f"Frame: {frame_index}")

    # Modes
    info_lines.append(f"Mirror: {'ON' if mirror else 'OFF'}")
    info_lines.append(f"DIP Preprocess: {'ON' if preprocess else 'OFF'}")
    if calibration_enabled:
        info_lines.append("Calibration: ON (meters)")

    if extra_lines:
        info_lines.extend(extra_lines)

    # Render lines at top-left with dual-pass high-contrast outlines
    start_y = 28
    line_spacing = 24
    for idx, text in enumerate(info_lines):
        y_pos = start_y + idx * line_spacing
        if y_pos > height - 10:
            break
        cv2.putText(frame, text, (14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 4)
        cv2.putText(frame, text, (14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)


def render_overlay(
    frame: np.ndarray,
    result: TrackingResult | None,
    points: Sequence[tuple[int, int]] | None = None,
    tracker_name: str = "",
    fps: float = 0.0,
    total_distance: float = 0.0,
    unit: str = "px",
    frame_index: int | None = None,
    velocity: float | None = None,
    mirror: bool = False,
    preprocess: bool = False,
    calibration_enabled: bool = False,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    """Composite visualizer function returning an annotated frame copy.

    Args:
        frame: Input video frame.
        result: Standardized TrackingResult from active tracker.
        points: Sequence of trajectory centroid coordinates.
        tracker_name: Algorithm name string.
        fps: Smoothed FPS value.
        total_distance: Traveled path distance.
        unit: Distance unit ('px' or 'm').
        frame_index: Current frame counter.
        velocity: Current motion velocity.
        mirror: Mirror flag.
        preprocess: Preprocessing flag.
        calibration_enabled: Calibration flag.
        extra_lines: Additional HUD messages.

    Returns:
        New annotated BGR frame copy without mutating input frame.
    """
    annotated = frame.copy()

    # Draw trajectory path
    if points:
        draw_trajectory(annotated, points)

    # Draw bounding box & target marker
    if result is not None:
        draw_bounding_box(
            annotated,
            bbox=result.bbox,
            center=result.center,
            success=result.success,
            track_id=result.track_id,
            class_name=result.class_name,
            confidence=result.confidence if result.confidence > 0.0 else None,
            path_length=total_distance if total_distance > 0 else None,
            unit=unit,
        )

    # Draw HUD telemetry
    draw_hud(
        annotated,
        tracker_name=tracker_name,
        success=result.success if result is not None else False,
        fps=fps,
        total_distance=total_distance,
        unit=unit,
        frame_index=frame_index,
        velocity=velocity,
        mirror=mirror,
        preprocess=preprocess,
        calibration_enabled=calibration_enabled,
        extra_lines=extra_lines,
    )

    return annotated


class TrackingVisualizer:
    """Class wrapper providing static methods for tracking overlays and HUD rendering."""

    draw_bounding_box = staticmethod(draw_bounding_box)
    draw_trajectory = staticmethod(draw_trajectory)
    draw_hud = staticmethod(draw_hud)
    render_overlay = staticmethod(render_overlay)
