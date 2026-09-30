"""Classical correlation-filter and appearance tracker implementations.

Migrated from:
    src/tracker_app.py

Features:
    - Supports OpenCV tracking algorithms: CSRT, KCF, MIL, and MOSSE (legacy).
    - Manual ROI bounding-box initialization.
    - Normalized cross-correlation template matching fallback for re-acquisition upon loss.
    - Optional DIP preprocessing (histogram equalization + Gaussian blur).
    - Conforms to BaseTracker and outputs standardized TrackingResult objects.
"""

from typing import Any
import cv2
import numpy as np

from .base_tracker import BaseTracker, TrackingResult

try:
    from core.telemetry import bbox_center
except ImportError:
    from src.core.telemetry import bbox_center


TRACKER_BUILDERS: dict[str, str] = {
    "csrt": "TrackerCSRT_create",
    "kcf": "TrackerKCF_create",
    "mil": "TrackerMIL_create",
    "mosse": "TrackerMOSSE_create",
}

DEFAULT_REDETECT_THRESHOLD: float = 0.62


def create_tracker(name: str) -> Any:
    """Instantiate an OpenCV tracker object by algorithm name.

    Supports:
        - csrt: Channel and Spatial Reliability Tracker (high accuracy)
        - kcf: Kernelized Correlation Filters (fast, standard)
        - mil: Multiple Instance Learning (handles scale changes)
        - mosse: Minimum Output Sum of Squared Error (ultra fast, legacy)

    Raises:
        ValueError: If tracker name is unsupported.
        RuntimeError: If tracker symbol is missing in current OpenCV build.
    """
    normalized_name = name.lower().strip()
    if normalized_name not in TRACKER_BUILDERS:
        raise ValueError(
            f"Unsupported tracker '{name}'. Choose from: {', '.join(sorted(TRACKER_BUILDERS.keys()))}"
        )

    factory_name = TRACKER_BUILDERS[normalized_name]

    # Standard cv2 namespace
    if hasattr(cv2, factory_name):
        return getattr(cv2, factory_name)()

    # Legacy namespace (e.g. MOSSE in OpenCV 4.x contrib)
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, factory_name):
        return getattr(cv2.legacy, factory_name)()

    raise RuntimeError(
        f"OpenCV build does not include {factory_name}. "
        "Ensure opencv-contrib-python>=4.8,<5 is installed."
    )


def make_template(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    preprocess: bool = False,
) -> np.ndarray:
    """Extract reference template for normalized cross-correlation redetection.

    Preserves single-pass histogram equalization and optional Gaussian blur
    without redundant post-smoothing equalization.
    """
    x, y, w, h = [int(v) for v in bbox[:4]]
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0 or w <= 0 or h <= 0:
        raise RuntimeError("Selected ROI is empty.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if preprocess:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


def try_redetect(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = DEFAULT_REDETECT_THRESHOLD,
    preprocess: bool = False,
) -> tuple[bool, tuple[int, int, int, int] | None, float]:
    """Attempt target reacquisition using normalized cross-correlation template matching.

    Returns:
        (True, (x, y, w, h), score) if best match score >= threshold.
        (False, None, score) if target not reacquired.
    """
    if template is None or template.size == 0:
        return False, None, 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if preprocess:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    template_h, template_w = template.shape[:2]

    if gray.shape[0] < template_h or gray.shape[1] < template_w:
        return False, None, 0.0

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)

    match_score = float(score)
    if match_score < threshold:
        return False, None, match_score

    x, y = location
    return True, (int(x), int(y), int(template_w), int(template_h)), match_score


class ClassicalTracker(BaseTracker):
    """Classical correlation-filter and appearance tracker implementing BaseTracker.

    Encapsulates:
        - OpenCV tracker initialization and frame updates.
        - Automated template-matching recovery when correlation tracking fails.
        - Standardized TrackingResult population with consistent xywh coordinates.
        - Tracking state, template memory, and redetection event counts.
    """

    def __init__(
        self,
        tracker_name: str = "csrt",
        redetect_threshold: float = DEFAULT_REDETECT_THRESHOLD,
        preprocess: bool = False,
        tracker_type: str | None = None,
        mirror: bool = False,
    ) -> None:
        chosen_name = tracker_type if tracker_type is not None else tracker_name
        self.tracker_name = chosen_name.lower().strip()
        self.redetect_threshold = redetect_threshold
        self.preprocess = preprocess
        self.mirror = mirror

        self._tracker: Any = None
        self._template: np.ndarray | None = None
        self._is_tracking: bool = False
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_center: tuple[int, int] | None = None
        self._redetect_count: int = 0

    @property
    def template(self) -> np.ndarray | None:
        """Return the current template bitmap used for redetection."""
        return self._template

    @property
    def last_bbox(self) -> tuple[int, int, int, int] | None:
        """Return the last successfully estimated bounding box (x, y, w, h)."""
        return self._last_bbox

    @property
    def redetect_count(self) -> int:
        """Return total number of successful re-detection events during tracking."""
        return self._redetect_count

    def initialize(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Initialize the tracker on a region of interest.

        Args:
            frame: Video frame (BGR numpy array).
            bbox: Bounding box tuple (x, y, width, height).

        Returns:
            True on successful initialization, False if bbox is invalid.
        """
        x, y, w, h = [int(v) for v in bbox[:4]]
        if w <= 0 or h <= 0 or frame is None or frame.size == 0:
            self._is_tracking = False
            return False

        # Constrain bbox within frame boundaries
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        valid_bbox = (x, y, w, h)

        try:
            self._tracker = create_tracker(self.tracker_name)
            self._tracker.init(frame, valid_bbox)
            self._template = make_template(frame, valid_bbox, preprocess=self.preprocess)
            self._is_tracking = True
            self._last_bbox = valid_bbox
            self._last_center = bbox_center(valid_bbox)
            return True
        except Exception:
            self._is_tracking = False
            return False

    init = initialize

    def update(self, frame: np.ndarray) -> TrackingResult:
        """Process a new frame and return standardized TrackingResult.

        If tracking fails, automatically attempts template-matching recovery.
        """
        if frame is None or frame.size == 0:
            return TrackingResult(success=False, bbox=None, center=None, confidence=0.0)

        # If not active or tracker handle missing, attempt recovery if template exists
        if not self._is_tracking or self._tracker is None:
            if self._template is not None:
                reacquired, new_bbox, score = try_redetect(
                    frame, self._template, self.redetect_threshold, preprocess=self.preprocess
                )
                if reacquired and new_bbox is not None:
                    self._redetect_count += 1
                    self._tracker = create_tracker(self.tracker_name)
                    self._tracker.init(frame, new_bbox)
                    self._template = make_template(frame, new_bbox, preprocess=self.preprocess)
                    self._is_tracking = True
                    self._last_bbox = new_bbox
                    self._last_center = bbox_center(new_bbox)
                    return TrackingResult(
                        success=True,
                        bbox=new_bbox,
                        center=self._last_center,
                        confidence=score,
                    )
            return TrackingResult(success=False, bbox=None, center=None, confidence=0.0)

        # Primary tracker update
        success = False
        raw_bbox = None
        try:
            success, raw_bbox = self._tracker.update(frame)
        except Exception:
            success = False

        if success and raw_bbox is not None:
            x, y, w, h = [int(v) for v in raw_bbox[:4]]
            bbox = (x, y, w, h)
            center = bbox_center(bbox)
            self._last_bbox = bbox
            self._last_center = center
            self._is_tracking = True
            # Classical OpenCV trackers do not report per-frame probability; standard 1.0 on success
            return TrackingResult(
                success=True,
                bbox=bbox,
                center=center,
                confidence=1.0,
            )

        # Primary tracker lost target -> Fallback to template matching recovery
        if self._template is not None:
            reacquired, new_bbox, score = try_redetect(
                frame, self._template, self.redetect_threshold, preprocess=self.preprocess
            )
            if reacquired and new_bbox is not None:
                self._redetect_count += 1
                try:
                    self._tracker = create_tracker(self.tracker_name)
                    self._tracker.init(frame, new_bbox)
                    self._template = make_template(frame, new_bbox, preprocess=self.preprocess)
                    self._is_tracking = True
                    self._last_bbox = new_bbox
                    self._last_center = bbox_center(new_bbox)
                    return TrackingResult(
                        success=True,
                        bbox=new_bbox,
                        center=self._last_center,
                        confidence=score,
                    )
                except Exception:
                    pass

        self._is_tracking = False
        return TrackingResult(
            success=False,
            bbox=None,
            center=None,
            confidence=0.0,
        )

    def is_tracking(self) -> bool:
        """Return True if tracker currently holds a lock on the target."""
        return self._is_tracking

    def get_tracker_name(self) -> str:
        """Return uppercase tracker name."""
        return self.tracker_name.upper()

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._tracker = None
        self._template = None
        self._is_tracking = False
        self._last_bbox = None
        self._last_center = None
        self._redetect_count = 0


# Skeleton alias for backward compatibility
ClassicalTrackerEngine = ClassicalTracker
