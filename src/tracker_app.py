import argparse
import csv
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config_loader import load_config, merge_args_with_config, setup_logging, AppConfig


TRACKER_BUILDERS = {
    "csrt": "TrackerCSRT_create",
    "kcf": "TrackerKCF_create",
    "mil": "TrackerMIL_create",
    "mosse": "TrackerMOSSE_create",
}

DEFAULT_REDETECT_THRESHOLD = 0.62
DEFAULT_MAX_PATH_POINTS = 100


def create_tracker(name: str) -> cv2.Tracker:
    name = name.lower()
    if name not in TRACKER_BUILDERS:
        raise ValueError(
            f"Unsupported tracker '{name}'. Choose from: {', '.join(TRACKER_BUILDERS)}"
        )

    factory_name = TRACKER_BUILDERS[name]
    if hasattr(cv2, factory_name):
        return getattr(cv2, factory_name)()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, factory_name):
        return getattr(cv2.legacy, factory_name)()

    raise RuntimeError(
        f"OpenCV build does not include {factory_name}. "
        "Install opencv-contrib-python instead of opencv-python."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Professional real-time object tracking and trajectory visualization using Digital Image Processing."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video. Omit to use webcam.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="csrt",
        choices=sorted(TRACKER_BUILDERS.keys()),
        help="Tracking algorithm to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/tracked_output.avi",
        help="Path for annotated output video.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="data/output/tracking_metrics.csv",
        help="Path for per-frame tracking metrics CSV.",
    )
    parser.add_argument("--no-save", action="store_true", help="Run without saving output files.")
    parser.add_argument(
        "--redetect-threshold",
        type=float,
        default=DEFAULT_REDETECT_THRESHOLD,
        help="Template-match score needed to reacquire a lost object. Lower is more sensitive.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror/webcam mode: flip image like a mirror (default: ON for webcam).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_false",
        dest="mirror",
        help="Disable mirror mode: show original image orientation.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply DIP preprocessing (histogram equalization + Gaussian blur).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=DEFAULT_MAX_PATH_POINTS,
        help="Maximum number of trail points to display.",
    )
    parser.add_argument(
        "--fps-avg",
        type=int,
        default=30,
        help="Frames per second average window for FPS calculation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def open_capture(video_path: str | None) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(video_path if video_path else 0)
    if not capture.isOpened():
        source = video_path if video_path else "webcam"
        raise RuntimeError(f"Could not open {source}.")
    return capture


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = [int(v) for v in bbox]
    return x + w // 2, y + h // 2


def distance_between(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def make_template(frame: np.ndarray, bbox: tuple[int, int, int, int], preprocess: bool = False) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        raise RuntimeError("Selected ROI is empty.")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if preprocess:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def try_redetect(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float,
    preprocess: bool = False,
) -> tuple[bool, tuple[int, int, int, int] | None, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if preprocess:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    template_h, template_w = template.shape[:2]

    if gray.shape[0] < template_h or gray.shape[1] < template_w:
        return False, None, 0.0

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)

    if score < threshold:
        return False, None, score

    x, y = location
    return True, (x, y, template_w, template_h), score


def draw_overlay(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    tracker_name: str,
    success: bool,
    fps: float,
    points: list[tuple[int, int]],
    total_distance: float,
    mirror: bool = False,
    preprocess: bool = False,
    info_lines: list[str] | None = None,
    calibration_enabled: bool = False,
) -> None:
    height = frame.shape[0]
    color = (0, 180, 0) if success else (0, 0, 255)

    if success and bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        center = (x + w // 2, y + h // 2)
        cv2.circle(frame, center, 4, (255, 120, 0), -1)

    for index in range(1, len(points)):
        thickness = max(1, int(np.sqrt(index + 1) / 2))
        cv2.line(frame, points[index - 1], points[index], (255, 120, 0), thickness)

    tracker_name_display = tracker_name.upper()
    status_text = "Tracking" if success else "Lost"

    unit = "m" if calibration_enabled else "px"
    info = [
        f"Tracker: {tracker_name_display}",
        f"Status: {status_text}",
        f"FPS: {fps:.2f}",
        f"Path: {total_distance:.1f}{unit}",
    ]

    if mirror:
        info.insert(0, "Mirror: ON")
    else:
        info.insert(0, "Mirror: OFF")

    if preprocess:
        info.insert(1, "DIP Preprocessing: ON")
    else:
        info.insert(1, "DIP Preprocessing: OFF")

    if calibration_enabled:
        info.insert(1, "Calibration: ON (meters)")

    if info_lines:
        info.extend(info_lines)

    for index, text in enumerate(info):
        y_pos = height - 20 - (len(info) - 1 - index) * 24
        cv2.putText(frame, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4)
        cv2.putText(frame, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)


def compute_fps(current_fps: float, fps_history: deque, avg_window: int) -> float:
    """Compute smoothed FPS using a rolling average."""
    fps_history.append(current_fps)
    if len(fps_history) > avg_window:
        fps_history.popleft()
    if len(fps_history) > 1:
        return sum(fps_history) / len(fps_history)
    return current_fps


def main() -> None:
    args = parse_args()

    # Load configuration
    config: AppConfig = load_config(args.config)
    setup_logging(config.logging)
    logger = logging.getLogger(__name__)

    # Merge CLI args with config (CLI takes precedence)
    args = merge_args_with_config(args, config, "classical")

    mirror = args.mirror
    capture = open_capture(args.video)

    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Could not read the first frame.")

    tracker = create_tracker(args.tracker)
    bbox = cv2.selectROI("Object Tracking", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Object Tracking")

    if bbox == (0, 0, 0, 0):
        raise RuntimeError("No region selected. Select an object ROI to begin tracking.")

    tracker.init(frame, bbox)
    template = make_template(frame, bbox, preprocess=args.preprocess)

    writer: cv2.VideoWriter | None = None
    metrics_file = None
    metrics_writer = None

    if not args.no_save:
        output_path = Path(args.output)
        metrics_path = Path(args.metrics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        output_fps = source_fps if source_fps and source_fps > 1 else 25
        frame_size = (frame.shape[1], frame.shape[0])
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), output_fps, frame_size)

        metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_file)
        unit = "m" if config.calibration.enabled else "px"
        metrics_writer.writerow([
            "frame", "success", "x", "y", "w", "h",
            "center_x", "center_y", "fps", f"path_{unit}", "mirror", "preprocess"
        ])

    points = [bbox_center(bbox)]
    total_distance = 0.0
    frame_index = 0
    last_time = time.perf_counter()
    fps_history = deque(maxlen=args.fps_avg)
    preprocessing = args.preprocess

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # Apply mirroring if enabled
        if mirror:
            frame = cv2.flip(frame, 1)

        current_time = time.perf_counter()
        elapsed = current_time - last_time
        last_time = current_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps = compute_fps(current_fps, fps_history, args.fps_avg)

        success, bbox = tracker.update(frame)

        if not success:
            reacquired, new_bbox, score = try_redetect(
                frame, template, args.redetect_threshold, preprocess=args.preprocess
            )
            if reacquired:
                bbox = new_bbox
                tracker = create_tracker(args.tracker)
                tracker.init(frame, bbox)
                template = make_template(frame, bbox, preprocess=args.preprocess)
                success = True
            else:
                success = False

        center: tuple[int, int] | tuple[str, str] = ("", "")
        row_bbox: tuple[int, int, int, int] | tuple[str, str, str, str] = ("", "", "", "")

        if success:
            center = bbox_center(bbox)
            row_bbox = tuple(int(v) for v in bbox)
            if len(points) >= args.trail_length:
                points = points[-args.trail_length + 1:]
            pixel_dist = distance_between(points[-1], center) if points else 0
            real_dist = config.calibration.pixels_to_meters(pixel_dist)
            total_distance += real_dist if config.calibration.enabled else pixel_dist
            points.append(center)

        draw_overlay(
            frame, bbox, args.tracker, success, fps, points,
            total_distance, mirror=mirror, preprocess=preprocessing,
            info_lines=[f"Frame: {frame_index}"],
            calibration_enabled=config.calibration.enabled,
        )

        if writer is not None:
            writer.write(frame)
        if metrics_writer is not None:
            metrics_writer.writerow([
                frame_index, int(success), *row_bbox,
                *center, f"{fps:.2f}", f"{total_distance:.2f}", mirror, preprocessing
            ])

        cv2.imshow("Object Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            bbox = cv2.selectROI("Object Tracking", frame, fromCenter=False, showCrosshair=True)
            if bbox != (0, 0, 0, 0):
                tracker = create_tracker(args.tracker)
                tracker.init(frame, bbox)
                template = make_template(frame, bbox, preprocess=args.preprocess)
                points = [bbox_center(bbox)]
                total_distance = 0.0
        if key == ord("c"):
            mirror = not mirror
            logger.info("Mirror %s", "enabled" if mirror else "disabled")
        if key == ord("p"):
            preprocessing = not preprocessing
            logger.info("DIP Preprocessing %s", "enabled" if preprocessing else "disabled")

        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    if metrics_file is not None:
        metrics_file.close()
    cv2.destroyAllWindows()

    if not args.no_save:
        logger.info("Saved annotated video: %s", args.output)
        logger.info("Saved metrics CSV: %s", args.metrics)


if __name__ == "__main__":
    main()