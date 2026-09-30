"""Input manager abstraction for video streams and webcam devices.

Responsibilities:
    - Normalizes video sources (device index, local file path, stream URL).
    - Manages cv2.VideoCapture lifecycle with safe open, read, and release.
    - Context manager support (with VideoSource(...) as src:) to prevent resource leaks.
    - Frame acquisition, frame indexing, and monotonic timestamp generation.
    - Horizontal mirroring for webcam streams (preserving BGR color format).
    - Source metadata querying (native FPS, width, height, frame count).
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol, runtime_checkable
import cv2
import numpy as np


def normalize_source(source: int | str | Path | None) -> int | str:
    """Normalize source argument into a webcam index (int) or video path/URL (str).

    Examples:
        None -> 0
        "0" -> 0
        "1" -> 1
        "data/input/sample.mp4" -> "data/input/sample.mp4"
        Path("data/input/sample.mp4") -> "data/input/sample.mp4"
    """
    if source is None or source == "":
        return 0
    if isinstance(source, Path):
        return str(source)
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return source


@dataclass
class FramePacket:
    """Encapsulates an acquired video frame along with capture telemetry."""

    frame_index: int
    timestamp: float
    raw_frame: np.ndarray
    display_frame: np.ndarray
    is_mirrored: bool
    source_info: str = ""


@runtime_checkable
class VideoSourceProtocol(Protocol):
    """Protocol defining standard video capture contracts."""

    def read_frame(self) -> tuple[bool, FramePacket | None]:
        """Read the next video frame packet."""
        ...

    def release(self) -> None:
        """Release underlying camera or file resource."""
        ...

    def get_fps(self) -> float:
        """Query native FPS of source."""
        ...

    def get_frame_dimensions(self) -> tuple[int, int]:
        """Return (width, height) of stream."""
        ...


class VideoSource:
    """Reusable OpenCV video capture abstraction implementing VideoSourceProtocol.

    Supports:
        - Local video files (mp4, avi, mkv, mov)
        - Live webcam streams (integer device indices 0, 1, ...)
        - Optional runtime horizontal mirroring (cv2.flip(frame, 1))
        - Deterministic frame index counter and monotonic timestamps
        - Context manager support for automatic cleanup
    """

    def __init__(
        self,
        source: int | str | Path | None = 0,
        mirror: bool = False,
    ) -> None:
        self.raw_source = source
        self.normalized_source = normalize_source(source)
        self.mirror = mirror
        self._cap: cv2.VideoCapture | None = None
        self._frame_index: int = 0
        self._is_opened: bool = False
        self._source_fps: float = 25.0
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = -1

    @property
    def is_opened(self) -> bool:
        """Return True if capture handle is open and ready."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_index(self) -> int:
        """Return current frame index."""
        return self._frame_index

    def open(self) -> bool:
        """Open the video capture source.

        Raises:
            RuntimeError: If the source cannot be opened by OpenCV.
        """
        if self.is_opened:
            return True

        # If it's a file path, verify it exists before opening for clear error reporting
        if isinstance(self.normalized_source, str) and not self.normalized_source.startswith(
            ("http://", "https://", "rtsp://")
        ):
            path = Path(self.normalized_source)
            if not path.exists():
                raise RuntimeError(f"Video file does not exist: {self.normalized_source}")

        self._cap = cv2.VideoCapture(self.normalized_source)
        if not self._cap.isOpened():
            source_desc = f"device {self.normalized_source}" if isinstance(self.normalized_source, int) else str(self.normalized_source)
            raise RuntimeError(f"Could not open video source: {source_desc}")

        self._is_opened = True
        self._frame_index = 0

        # Query metadata
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = float(fps) if fps and fps > 1.0 else 25.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return True

    def set_mirror(self, mirror: bool) -> None:
        """Toggle mirroring at runtime."""
        self.mirror = mirror

    def read_frame(self) -> tuple[bool, FramePacket | None]:
        """Acquire the next frame from the capture stream.

        Returns:
            (True, FramePacket) on successful read.
            (False, None) when end of stream is reached or on read error.
        """
        if not self.is_opened:
            self.open()

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            return False, None

        timestamp = time.perf_counter()
        raw_frame = frame  # BGR format preserved

        if self.mirror:
            display_frame = cv2.flip(frame, 1)
        else:
            display_frame = frame.copy()

        packet = FramePacket(
            frame_index=self._frame_index,
            timestamp=timestamp,
            raw_frame=raw_frame,
            display_frame=display_frame,
            is_mirrored=self.mirror,
            source_info=str(self.normalized_source),
        )

        self._frame_index += 1
        return True, packet

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture device."""
        if self._cap is not None:
            if self._cap.isOpened():
                self._cap.release()
            self._cap = None
        self._is_opened = False

    def get_fps(self) -> float:
        """Return native or estimated stream FPS."""
        return self._source_fps

    def get_frame_dimensions(self) -> tuple[int, int]:
        """Return (width, height) of frames."""
        return self._width, self._height

    def get_frame_count(self) -> int:
        """Return total frame count (-1 if live stream or unavailable)."""
        return self._frame_count

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self):
        """Yield frame packets until stream ends."""
        while True:
            ok, packet = self.read_frame()
            if not ok or packet is None:
                break
            yield packet
