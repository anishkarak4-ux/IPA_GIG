"""Core module providing video acquisition abstraction and kinematics telemetry."""

from .input_manager import FramePacket, VideoSource, normalize_source
from .telemetry import (
    CalibrationModel,
    TelemetryTracker,
    TrackingTelemetry,
    bbox_center,
    distance_between,
    xyxy_center,
)

__all__ = [
    "FramePacket",
    "VideoSource",
    "normalize_source",
    "TrackingTelemetry",
    "CalibrationModel",
    "TelemetryTracker",
    "bbox_center",
    "xyxy_center",
    "distance_between",
]
