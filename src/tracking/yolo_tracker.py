"""Deep-learning detection and association tracking engine using YOLOv8.

Migrated from:
    src/yolo_tracker.py

Features:
    - Ultralytics YOLOv8 Nano model loading and streaming inference.
    - Motion-based multi-object tracking via BoT-SORT (default) or ByteTrack.
    - Standardized TrackingResult outputs conforming to BaseTracker.
    - Click-to-lock single-target isolation state machine.
    - Multi-object detection internal registry (current_detections).
    - Optional DIP preprocessing (histogram equalization + Gaussian blur).
    - Separation of concerns: pure tracking engine without telemetry,
      VideoCapture, or GUI drawing duplication.
"""

from pathlib import Path
from typing import Any, Sequence
import cv2
import numpy as np
from ultralytics import YOLO

from .base_tracker import BaseTracker, TrackingResult

try:
    from core.telemetry import bbox_center, xyxy_center
except ImportError:
    from src.core.telemetry import bbox_center, xyxy_center


DEFAULT_MODEL_PATH: str = "yolov8n.pt"
DEFAULT_TRACKER_TYPE: str = "botsort.yaml"
DEFAULT_CONF_THRESHOLD: float = 0.25


def point_in_box(point: tuple[int, int], xyxy: Sequence[int | float] | np.ndarray) -> bool:
    """Check whether a 2D (x, y) point lies within an [x1, y1, x2, y2] bounding box."""
    x, y = point
    x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
    return x1 <= x <= x2 and y1 <= y <= y2


def point_in_bbox_xywh(point: tuple[int, int], bbox: Sequence[int | float]) -> bool:
    """Check whether a 2D (x, y) point lies within an (x, y, w, h) bounding box."""
    px, py = point
    x, y, w, h = [int(v) for v in bbox[:4]]
    return x <= px <= x + w and y <= py <= y + h


def compute_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two (x, y, w, h) bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = max(0, w1) * max(0, h1)
    box2_area = max(0, w2) * max(0, h2)
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def apply_dip_preprocessing(frame: np.ndarray) -> np.ndarray:
    """Apply DIP preprocessing: histogram equalization + Gaussian blur.

    Preserves baseline behavior from src/yolo_tracker.py.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class YOLOTracker(BaseTracker):
    """Deep-learning detection and association tracking engine implementing BaseTracker.

    Encapsulates:
        - Ultralytics YOLO inference with BoT-SORT / ByteTrack tracking.
        - Click-to-lock target selection and persistent track tracking.
        - Multi-object detections cache accessible via current_detections.
        - Standardized single-target TrackingResult return from update().
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tracker_type: str = DEFAULT_TRACKER_TYPE,
        conf: float = DEFAULT_CONF_THRESHOLD,
        track_all: bool = False,
        classes: list[int] | None = None,
        preprocess: bool = False,
        tracker_yaml: str | None = None,
        mirror: bool = False,
    ) -> None:
        """Initialize YOLO tracking engine.

        Args:
            model_path: Path to YOLO weights (default: yolov8n.pt).
            tracker_type: Tracker config ('botsort.yaml' or 'bytetrack.yaml').
            conf: Confidence threshold for detections (default: 0.25).
            track_all: If True, tracks all objects simultaneously without locking.
            classes: Optional list of class IDs to filter.
            preprocess: If True, applies DIP preprocessing before inference.
            tracker_yaml: Alias for tracker_type.
            mirror: Handled centrally by VideoSource.
        """
        chosen_tracker = tracker_yaml if tracker_yaml is not None else tracker_type
        self.model_path = str(model_path)
        self.tracker_type = chosen_tracker
        self.conf = float(conf)
        self.track_all = bool(track_all)
        self.classes = classes
        self.preprocess = bool(preprocess)
        self.mirror = mirror

        # Lazy / initial model instantiation
        self._model: YOLO = YOLO(self.model_path)

        # Tracking state
        self._is_tracking: bool = False
        self.selected_track_id: int | None = None
        self.pending_click: tuple[int, int] | None = None
        self._current_detections: list[TrackingResult] = []
        self._last_result: TrackingResult | None = None

    @property
    def current_detections(self) -> list[TrackingResult]:
        """Return all tracked object detections in the latest processed frame."""
        return list(self._current_detections)

    @property
    def all_detections(self) -> list[TrackingResult]:
        """Convenience alias for current_detections."""
        return list(self._current_detections)

    def set_pending_click(self, x: int, y: int) -> None:
        """Queue a mouse click point for target acquisition in the next frame update.

        Preserves baseline click-to-lock interaction semantics.
        """
        self.pending_click = (int(x), int(y))

    def select_target_at(self, point: tuple[int, int]) -> int | None:
        """Attempt to lock onto the detected object containing the specified (x, y) point.

        Args:
            point: (x, y) pixel coordinates.

        Returns:
            The selected track_id if locked, or None if point does not fall within any detection.
        """
        for det in self._current_detections:
            if det.bbox is not None and point_in_bbox_xywh(point, det.bbox):
                self.selected_track_id = det.track_id
                self._is_tracking = True
                return det.track_id
        return None

    def select_target_by_id(self, track_id: int) -> bool:
        """Lock onto a specific track ID.

        Args:
            track_id: Ultralytics track ID integer.

        Returns:
            True if locked successfully.
        """
        self.selected_track_id = int(track_id)
        self._is_tracking = True
        return True

    def clear_selection(self) -> None:
        """Clear current target lock, returning to unlocked state."""
        self.selected_track_id = None
        self.pending_click = None
        self._is_tracking = False

    def initialize(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Initialize or lock tracking on a target matching the provided bounding box or point.

        If bbox represents a small region (e.g., width <= 2 and height <= 2), it is
        treated as a click point. Otherwise, the detection having the highest IoU with
        the ROI is selected.

        Args:
            frame: Video frame (BGR numpy array).
            bbox: (x, y, width, height) tuple.

        Returns:
            True if a target was successfully locked, False otherwise.
        """
        if frame is None or frame.size == 0:
            self._is_tracking = False
            return False

        # Run detection / update on current frame to discover candidates
        self.update(frame)

        if not self._current_detections:
            self._is_tracking = False
            return False

        x, y, w, h = [int(v) for v in bbox[:4]]

        # Point click handling
        if w <= 2 and h <= 2:
            locked_id = self.select_target_at((x, y))
            return locked_id is not None

        # ROI Bounding box matching via IoU and centroid containment
        best_det: TrackingResult | None = None
        best_iou: float = 0.0

        for det in self._current_detections:
            if det.bbox is None:
                continue
            iou = compute_iou((x, y, w, h), det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        # Fallback: check if detection center is within requested bbox
        if best_det is None or best_iou < 0.05:
            for det in self._current_detections:
                if det.center is not None and point_in_bbox_xywh(det.center, (x, y, w, h)):
                    best_det = det
                    break

        if best_det is not None and best_det.track_id is not None:
            self.selected_track_id = best_det.track_id
            self._is_tracking = True
            return True

        self._is_tracking = False
        return False

    def update(self, frame: np.ndarray) -> TrackingResult:
        """Process a frame with YOLO + BoT-SORT / ByteTrack and return TrackingResult.

        Extracts all detections into current_detections, resolves any pending click-to-lock,
        and returns the standardized TrackingResult for the active target.

        Args:
            frame: Input video frame (BGR numpy array).

        Returns:
            Standardized TrackingResult with actual YOLO confidence, bbox, center,
            track_id, and class_name.
        """
        if frame is None or frame.size == 0:
            self._current_detections = []
            self._is_tracking = False
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                track_id=None,
                class_name=None,
            )

        # Apply DIP preprocessing if enabled
        infer_frame = apply_dip_preprocessing(frame) if self.preprocess else frame

        # Run streaming tracker inference with state persistence
        track_kwargs: dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker_type,
            "conf": self.conf,
            "verbose": False,
        }
        if self.classes is not None:
            track_kwargs["classes"] = self.classes

        try:
            results = self._model.track(infer_frame, **track_kwargs)
        except Exception:
            self._current_detections = []
            self._is_tracking = False
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                track_id=None,
                class_name=None,
            )

        if not results:
            self._current_detections = []
            self._is_tracking = False
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                track_id=None,
                class_name=None,
            )

        result = results[0]
        boxes = result.boxes
        new_detections: list[TrackingResult] = []

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()

            for track_id, class_id, conf, xyxy in zip(ids, classes, confs, xyxys):
                x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                bbox = (x1, y1, w, h)
                center = xyxy_center(xyxy)
                class_name = result.names.get(int(class_id), str(class_id))

                det_res = TrackingResult(
                    success=True,
                    bbox=bbox,
                    center=center,
                    confidence=float(conf),
                    track_id=int(track_id),
                    class_name=class_name,
                )
                new_detections.append(det_res)

        self._current_detections = new_detections

        # Process pending click target acquisition
        if self.pending_click is not None and not self.track_all:
            click_pt = self.pending_click
            self.pending_click = None
            for det in self._current_detections:
                if det.bbox is not None and point_in_bbox_xywh(click_pt, det.bbox):
                    self.selected_track_id = det.track_id
                    break

        # Resolve primary tracking result for BaseTracker contract
        if self.selected_track_id is not None:
            # Find the selected track in current detections
            matched_det: TrackingResult | None = None
            for det in self._current_detections:
                if det.track_id == self.selected_track_id:
                    matched_det = det
                    break

            if matched_det is not None:
                self._is_tracking = True
                self._last_result = matched_det
                return matched_det

            # Selected target lost / occluded in this frame: safe failure return
            self._is_tracking = False
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                track_id=None,
                class_name=None,
            )

        # No target explicitly selected
        if self.track_all:
            if self._current_detections:
                self._is_tracking = True
                # Return highest confidence detection as representative
                best_det = max(self._current_detections, key=lambda d: d.confidence)
                self._last_result = best_det
                return best_det

            self._is_tracking = False
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                track_id=None,
                class_name=None,
            )

        # Not track_all and no target selected: awaiting click-to-lock
        self._is_tracking = False
        return TrackingResult(
            success=False,
            bbox=None,
            center=None,
            confidence=0.0,
            track_id=None,
            class_name=None,
        )

    def is_tracking(self) -> bool:
        """Return True if tracker currently holds a lock on a tracked object."""
        return self._is_tracking

    def get_tracker_name(self) -> str:
        """Return human-readable algorithm name."""
        backend = Path(self.tracker_type).stem.upper()
        return f"YOLOv8n + {backend}"

    def reset(self) -> None:
        """Reset internal tracker state, predictor caches, and selections."""
        self.selected_track_id = None
        self.pending_click = None
        self._is_tracking = False
        self._current_detections.clear()
        self._last_result = None

        # Reset Ultralytics predictor trackers if present
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            if hasattr(self._model.predictor, "trackers") and self._model.predictor.trackers:
                for trk in self._model.predictor.trackers:
                    if hasattr(trk, "reset"):
                        try:
                            trk.reset()
                        except Exception:
                            pass


# Aliases for architecture and skeleton backward compatibility
YOLOTrackerEngine = YOLOTracker
