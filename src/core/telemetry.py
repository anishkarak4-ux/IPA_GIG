"""Kinematics, smoothed FPS, and spatial calibration telemetry.

Responsibilities:
    - Centroid and bounding box geometric transformations (bbox and xyxy formats).
    - Euclidean step displacement and trajectory length computation.
    - Preserves corrected single-sum cumulative trajectory distance (no double counting).
    - Rolling-average smoothed FPS calculation.
    - Linear projective pixel-to-meter physical calibration.
    - Safe handling of missing, None, or lost tracking points without crashes.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Sequence
import numpy as np


def bbox_center(bbox: Sequence[int | float]) -> tuple[int, int]:
    """Compute (center_x, center_y) from (x, y, width, height) bounding box.

    Formula:
        center_x = x + width // 2
        center_y = y + height // 2
    """
    x, y, w, h = [int(v) for v in bbox[:4]]
    return x + w // 2, y + h // 2


def xyxy_center(xyxy: Sequence[int | float] | np.ndarray) -> tuple[int, int]:
    """Compute (center_x, center_y) from (x1, y1, x2, y2) bounding box.

    Formula:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
    """
    x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
    return (x1 + x2) // 2, (y1 + y2) // 2


def distance_between(
    pt1: tuple[int | float, int | float],
    pt2: tuple[int | float, int | float],
) -> float:
    """Calculate 2D Euclidean distance between two coordinate points.

    Formula:
        d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    """
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])


@dataclass
class CalibrationModel:
    """Converts pixel measurements to real-world metric units (meters)."""

    enabled: bool = False
    reference_width_m: float = 0.0
    reference_width_px: int = 0
    camera_height_m: float = 0.0
    camera_angle_deg: float = 0.0

    def pixels_to_meters(self, pixels: float) -> float:
        """Convert pixel displacement to meters using reference scale ratio.

        Formula:
            d_meters = d_pixels * (reference_width_m / reference_width_px)
        """
        if not self.enabled or self.reference_width_px <= 0 or self.reference_width_m <= 0.0:
            return float(pixels)
        return float(pixels) * (self.reference_width_m / float(self.reference_width_px))


@dataclass
class TrackingTelemetry:
    """Standardized tracking kinematics snapshot for a single frame."""

    frame_index: int
    timestamp: float
    fps: float
    success: bool
    bbox: tuple[int, int, int, int] | None = None
    center: tuple[int, int] | None = None
    step_distance: float = 0.0
    cumulative_distance: float = 0.0
    unit: str = "px"


class TelemetryTracker:
    """Maintains trajectory point history and computes rolling kinematics.

    Guarantees:
        - Cumulative distance is the running sum of step displacements.
        - Cumulative distance is NOT double-counted.
        - Missing or lost tracking points (None) do not add displacement and do not crash.
        - Trail buffer is bounded to max_trail_points.
        - Smoothed FPS uses rolling average over fps_window frames.
    """

    def __init__(
        self,
        fps_window: int = 30,
        max_trail_points: int = 100,
        calibration: CalibrationModel | None = None,
    ) -> None:
        self.fps_window = max(1, fps_window)
        self.max_trail_points = max(1, max_trail_points)
        self.calibration = calibration or CalibrationModel()

        self.fps_history: deque[float] = deque(maxlen=self.fps_window)
        self.points: deque[tuple[int, int]] = deque(maxlen=self.max_trail_points)
        self.cumulative_distance: float = 0.0
        self.last_valid_center: tuple[int, int] | None = None
        self._last_timestamp: float | None = None

    def compute_smoothed_fps(self, current_fps: float) -> float:
        """Compute smoothed FPS using a rolling average window."""
        if current_fps > 0:
            self.fps_history.append(current_fps)
        if len(self.fps_history) > 1:
            return sum(self.fps_history) / len(self.fps_history)
        if len(self.fps_history) == 1:
            return self.fps_history[0]
        return 0.0

    def update_fps_from_time(self, current_time: float | None = None) -> float:
        """Compute smoothed FPS based on elapsed wall-clock time between frames."""
        now = current_time if current_time is not None else time.perf_counter()
        if self._last_timestamp is None:
            self._last_timestamp = now
            return 0.0

        elapsed = now - self._last_timestamp
        self._last_timestamp = now
        instant_fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        return self.compute_smoothed_fps(instant_fps)

    def set_initial_center(self, center: tuple[int, int]) -> None:
        """Set initial centroid coordinate before tracking begins (e.g. from selectROI)."""
        self.last_valid_center = center
        self.points.clear()
        self.points.append(center)

    def record_step(
        self,
        frame_index: int,
        success: bool,
        center: tuple[int, int] | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        fps: float = 0.0,
        timestamp: float | None = None,
    ) -> TrackingTelemetry:
        """Process a single tracking step and return complete telemetry.

        Args:
            frame_index: Current video frame index.
            success: Whether the tracker successfully located the target.
            center: Target centroid (x, y) if successfully tracked.
            bbox: Target bounding box (x, y, w, h).
            fps: Smoothed FPS for the current frame.
            timestamp: Monotonic timestamp of the frame.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        step_dist = 0.0

        if success and center is not None:
            # If we have a previous valid center, compute Euclidean displacement
            if self.last_valid_center is not None:
                step_px = distance_between(self.last_valid_center, center)
                step_dist = self.calibration.pixels_to_meters(step_px)
                self.cumulative_distance += step_dist

            self.last_valid_center = center
            self.points.append(center)

        unit = "m" if self.calibration.enabled else "px"

        return TrackingTelemetry(
            frame_index=frame_index,
            timestamp=now,
            fps=fps,
            success=success,
            bbox=bbox,
            center=center,
            step_distance=step_dist,
            cumulative_distance=self.cumulative_distance,
            unit=unit,
        )

    def get_trail_points(self) -> list[tuple[int, int]]:
        """Return list of historical centroid coordinates for trajectory rendering."""
        return list(self.points)

    def get_cumulative_distance(self) -> float:
        """Return total cumulative trajectory distance accumulated so far."""
        return self.cumulative_distance

    def reset(self) -> None:
        """Reset trajectory history and cumulative distance."""
        self.fps_history.clear()
        self.points.clear()
        self.cumulative_distance = 0.0
        self.last_valid_center = None
        self._last_timestamp = None
