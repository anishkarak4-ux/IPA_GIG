"""Centralized metrics logging and data persistence for tracking and DIP evaluations.

Responsibilities:
    - Thread-safe CSV logging of per-frame tracking telemetry.
    - Structured persistence of DIP snapshot analysis evaluations across Units I-VII.
    - Unified schema handling both Classical and YOLO trackers without fabricated fields.
    - Explicit session identification to distinguish runs across time.
    - Strict data integrity without duplicating telemetry calculations.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any

try:
    from processing.dip_pipeline import DIPResult
    from tracking.base_tracker import TrackingResult
except ImportError:
    from src.processing.dip_pipeline import DIPResult
    from src.tracking.base_tracker import TrackingResult


TRACKING_CSV_HEADER: list[str] = [
    "frame",
    "timestamp",
    "session_id",
    "tracker",
    "success",
    "x",
    "y",
    "w",
    "h",
    "center_x",
    "center_y",
    "step_px",
    "path_px",
    "fps",
    "velocity_px_s",
    "confidence",
    "track_id",
    "class",
]

SNAPSHOT_CSV_HEADER: list[str] = [
    "frame",
    "timestamp",
    "session_id",
    "tracker",
    "track_id",
    "class",
    "confidence",
    "x",
    "y",
    "w",
    "h",
    "roi_width",
    "roi_height",
    "otsu_threshold",
    "area",
    "perimeter",
    "aspect_ratio",
    "circularity",
    "solidity",
    "centroid_x",
    "centroid_y",
    "contour_count",
    "extent",
    "orientation_deg",
    "polygon_vertices",
    "hu_phi1",
    "hu_phi2",
    "hu_phi3",
    "glcm_contrast",
    "glcm_homogeneity",
    "shannon_entropy",
    "rle_compression_ratio",
    "mean_intensity",
    "std_intensity",
    "skewness",
    "kurtosis",
]


class MetricsLogger:
    """Thread-safe persistence manager for tracking and snapshot analysis data."""

    def __init__(
        self,
        tracking_csv_path: str | Path = "data/output/tracking_metrics.csv",
        snapshot_csv_path: str | Path = "data/output/snapshot_analysis.csv",
        session_id: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize MetricsLogger."""
        self.tracking_csv_path = Path(tracking_csv_path)
        self.snapshot_csv_path = Path(snapshot_csv_path)
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.overwrite = overwrite

        self._lock = threading.Lock()
        self._tracking_header_written = False
        self._snapshot_header_written = False

        self._init_files()

    def _init_files(self) -> None:
        """Ensure parent directories exist and manage CSV headers."""
        self.tracking_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Setup tracking CSV
            if self.overwrite or not self.tracking_csv_path.exists() or self.tracking_csv_path.stat().st_size == 0:
                with self.tracking_csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(TRACKING_CSV_HEADER)
                self._tracking_header_written = True
            else:
                self._tracking_header_written = True

            # Setup snapshot CSV
            if self.overwrite or not self.snapshot_csv_path.exists() or self.snapshot_csv_path.stat().st_size == 0:
                with self.snapshot_csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(SNAPSHOT_CSV_HEADER)
                self._snapshot_header_written = True
            else:
                self._snapshot_header_written = True

    def log_tracking(
        self,
        frame_index: int,
        tracker_name: str = "",
        success: bool = True,
        bbox: tuple[int, int, int, int] | None = None,
        center: tuple[int, int] | None = None,
        step_distance: float = 0.0,
        cumulative_distance: float = 0.0,
        fps: float = 0.0,
        velocity: float | None = None,
        confidence: float | None = None,
        track_id: int | None = None,
        class_name: str | None = None,
        timestamp: float | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log a single frame of tracking telemetry."""
        ts = timestamp if timestamp is not None else time.time()
        sess = session_id or self.session_id

        if bbox is not None and len(bbox) >= 4:
            x, y, w, h = [int(v) for v in bbox[:4]]
            x_str, y_str, w_str, h_str = str(x), str(y), str(w), str(h)
        else:
            x_str, y_str, w_str, h_str = "", "", "", ""

        if center is not None and len(center) >= 2:
            cx_str = str(int(center[0]))
            cy_str = str(int(center[1]))
        else:
            cx_str, cy_str = "", ""

        vel_str = f"{velocity:.2f}" if velocity is not None else ""
        conf_str = f"{confidence:.4f}" if confidence is not None else ""
        track_id_str = str(track_id) if track_id is not None else ""
        class_str = class_name or ""

        row = [
            frame_index,
            f"{ts:.4f}",
            sess,
            tracker_name,
            1 if success else 0,
            x_str,
            y_str,
            w_str,
            h_str,
            cx_str,
            cy_str,
            f"{step_distance:.2f}",
            f"{cumulative_distance:.2f}",
            f"{fps:.2f}",
            vel_str,
            conf_str,
            track_id_str,
            class_str,
        ]

        with self._lock:
            with self.tracking_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def log_tracking_result(
        self,
        frame_index: int,
        result: TrackingResult,
        step_distance: float = 0.0,
        cumulative_distance: float = 0.0,
        fps: float = 0.0,
        tracker_name: str = "",
        velocity: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Convenience method logging directly from a TrackingResult object."""
        self.log_tracking(
            frame_index=frame_index,
            tracker_name=tracker_name,
            success=result.success,
            bbox=result.bbox,
            center=result.center,
            step_distance=step_distance,
            cumulative_distance=cumulative_distance,
            fps=fps,
            velocity=velocity,
            confidence=result.confidence if result.confidence > 0.0 else None,
            track_id=result.track_id,
            class_name=result.class_name,
            timestamp=timestamp,
        )

    def log_snapshot_dip(
        self,
        dip_result: DIPResult,
        tracking_result: TrackingResult | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log quantitative diagnostic measurements across Units I-VII from a DIPResult snapshot."""
        if dip_result is None or dip_result.context is None:
            return

        ctx = dip_result.context
        metrics = dip_result.contour_metrics
        sess = session_id or self.session_id

        # Target attributes
        track_id = ctx.track_id if ctx.track_id is not None else (tracking_result.track_id if tracking_result else None)
        class_name = tracking_result.class_name if tracking_result and tracking_result.class_name else ctx.metadata.get("class", "")
        conf = tracking_result.confidence if tracking_result and tracking_result.confidence > 0.0 else ctx.metadata.get("conf", None)

        # Bounding box & ROI size
        if ctx.bbox is not None and len(ctx.bbox) >= 4:
            x, y, w, h = [int(v) for v in ctx.bbox[:4]]
            x_str, y_str, w_str, h_str = str(x), str(y), str(w), str(h)
        else:
            x_str, y_str, w_str, h_str = "", "", "", ""

        roi_h, roi_w = ctx.roi.shape[:2] if ctx.roi is not None and ctx.roi.size > 0 else (0, 0)

        # Contour centroid
        cx_str = str(metrics.centroid[0]) if metrics.centroid is not None else ""
        cy_str = str(metrics.centroid[1]) if metrics.centroid is not None else ""

        # Enriched descriptors
        hu = metrics.hu_moments or []
        hu1 = f"{hu[0]:.4f}" if len(hu) > 0 else ""
        hu2 = f"{hu[1]:.4f}" if len(hu) > 1 else ""
        hu3 = f"{hu[2]:.4f}" if len(hu) > 2 else ""

        glcm = metrics.glcm_features or {}
        glcm_cont = f"{glcm.get('contrast', 0):.3f}"
        glcm_homo = f"{glcm.get('homogeneity', 0):.3f}"

        stats = dip_result.histogram_stats or {}
        mu_str = f"{stats.get('mean', 0):.2f}"
        std_str = f"{stats.get('std_dev', 0):.2f}"
        skew_str = f"{stats.get('skewness', 0):.3f}"
        kurt_str = f"{stats.get('kurtosis', 0):.3f}"

        row = [
            ctx.frame_index,
            f"{ctx.timestamp:.4f}",
            sess,
            ctx.tracker_name,
            str(track_id) if track_id is not None else "",
            class_name,
            f"{conf:.4f}" if conf is not None else "",
            x_str,
            y_str,
            w_str,
            h_str,
            roi_w,
            roi_h,
            f"{dip_result.otsu_threshold:.2f}",
            f"{metrics.area:.2f}",
            f"{metrics.perimeter:.2f}",
            f"{metrics.aspect_ratio:.4f}",
            f"{metrics.circularity:.4f}",
            f"{metrics.solidity:.4f}",
            cx_str,
            cy_str,
            metrics.contour_count,
            f"{metrics.extent:.4f}",
            f"{metrics.orientation_deg:.2f}",
            metrics.polygon_vertices,
            hu1,
            hu2,
            hu3,
            glcm_cont,
            glcm_homo,
            f"{dip_result.shannon_entropy:.3f}",
            f"{dip_result.rle_compression_ratio:.2f}",
            mu_str,
            std_str,
            skew_str,
            kurt_str,
        ]

        with self._lock:
            with self.snapshot_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    # Backward compatibility alias
    log_tracking_frame = log_tracking
    log_snapshot_analysis = log_snapshot_dip
