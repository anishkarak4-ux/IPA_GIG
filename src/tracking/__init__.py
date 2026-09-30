"""Tracking module providing unified base interface, classical, and YOLO tracker engines."""

from .base_tracker import BaseTracker, TrackingResult
from .classical_tracker import (
    ClassicalTracker,
    ClassicalTrackerEngine,
    create_tracker,
    make_template,
    try_redetect,
)
from .yolo_tracker import YOLOTracker, YOLOTrackerEngine

__all__ = [
    "BaseTracker",
    "TrackingResult",
    "ClassicalTracker",
    "ClassicalTrackerEngine",
    "create_tracker",
    "make_template",
    "try_redetect",
    "YOLOTracker",
    "YOLOTrackerEngine",
]
