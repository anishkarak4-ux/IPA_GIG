"""Configuration loader for YAML config files."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class YOLOTrackerConfig:
    source: str = "0"
    model: str = "yolov8n.pt"
    tracker: str = "botsort.yaml"
    conf: float = 0.25
    output: str = "data/output/yolo_tracked_output.avi"
    metrics: str = "data/output/yolo_tracking_metrics.csv"
    no_save: bool = False
    track_all: bool = False
    mirror: bool = True
    preprocess: bool = False
    trail_length: int = 80
    fps_avg: int = 30


@dataclass
class ClassicalTrackerConfig:
    video: str | None = None
    tracker: str = "csrt"
    output: str = "data/output/tracked_output.avi"
    metrics: str = "data/output/tracking_metrics.csv"
    no_save: bool = False
    redetect_threshold: float = 0.62
    mirror: bool = True
    preprocess: bool = False
    trail_length: int = 100
    fps_avg: int = 30


@dataclass
class CalibrationConfig:
    enabled: bool = False
    reference_object_width_m: float = 0.0
    reference_object_width_px: int = 0
    camera_height_m: float = 0.0
    camera_angle_deg: float = 0.0

    def pixels_to_meters(self, pixels: float) -> float:
        """Convert pixel distance to meters using calibration."""
        if not self.enabled or self.reference_object_width_px == 0:
            return pixels
        return pixels * (self.reference_object_width_m / self.reference_object_width_px)


@dataclass
class VisualizationConfig:
    show_fps: bool = True
    show_trail: bool = True
    show_bbox: bool = True
    show_centroid: bool = True
    trail_thickness: int = 2
    bbox_thickness: int = 2
    font_scale: float = 0.6
    colors: dict[str, list[int]] = field(default_factory=lambda: {
        "bbox": [40, 180, 80],
        "centroid": [255, 140, 0],
        "trail": [255, 140, 0],
        "text_bg": [20, 20, 20],
        "text_fg": [245, 245, 245],
    })


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "data/output/tracking.log"


@dataclass
class AppConfig:
    yolo_tracker: YOLOTrackerConfig = field(default_factory=YOLOTrackerConfig)
    classical_tracker: ClassicalTrackerConfig = field(default_factory=ClassicalTrackerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str | Path | None) -> AppConfig:
    """Load configuration from YAML file."""
    config = AppConfig()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        logging.warning("Config file not found: %s, using defaults", config_path)
        return config

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return config

    if "yolo_tracker" in data:
        config.yolo_tracker = YOLOTrackerConfig(**data["yolo_tracker"])
    if "classical_tracker" in data:
        config.classical_tracker = ClassicalTrackerConfig(**data["classical_tracker"])
    if "calibration" in data:
        config.calibration = CalibrationConfig(**data["calibration"])
    if "visualization" in data:
        config.visualization = VisualizationConfig(**data["visualization"])
    if "logging" in data:
        config.logging = LoggingConfig(**data["logging"])

    return config


def merge_args_with_config(args: argparse.Namespace, config: AppConfig, tracker_type: str) -> argparse.Namespace:
    """Merge command-line arguments with config file values (CLI takes precedence)."""
    if tracker_type == "yolo":
        cfg = config.yolo_tracker
        for key, value in vars(cfg).items():
            arg_value = getattr(args, key, None)
            if arg_value is None or (isinstance(arg_value, str) and arg_value == ""):
                setattr(args, key, value)
    elif tracker_type == "classical":
        cfg = config.classical_tracker
        for key, value in vars(cfg).items():
            arg_value = getattr(args, key, None)
            if arg_value is None or (isinstance(arg_value, str) and arg_value == ""):
                setattr(args, key, value)

    return args


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=log_level,
        format=config.format,
        handlers=handlers,
    )