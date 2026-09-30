"""Base tracker abstract interface and standardized tracking result contract.

Future Responsibility:
    Define a common contract that both classical OpenCV correlation trackers
    (CSRT, KCF, MIL, MOSSE) and deep-learning trackers (YOLOv8 + BoT-SORT / ByteTrack)
    will implement so the main application loop treats them uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TrackingResult:
    """Standardized output produced by any tracker on each frame."""
    success: bool
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h)
    center: tuple[int, int] | None          # (center_x, center_y)
    confidence: float                       # 0.0 to 1.0
    track_id: int | None = None
    class_name: str | None = None


class BaseTracker(ABC):
    """Abstract base class that all tracking implementations must satisfy."""

    @abstractmethod
    def initialize(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Initialize tracking on a specific region of interest."""
        pass

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackingResult:
        """Update tracker state with a new frame and return tracking result."""
        pass

    @abstractmethod
    def is_tracking(self) -> bool:
        """Check whether tracker currently has an active lock on target."""
        pass

    @abstractmethod
    def get_tracker_name(self) -> str:
        """Return human-readable algorithm name."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker internal state."""
        pass
