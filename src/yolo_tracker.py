import argparse
import csv
import math
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config_loader import load_config, merge_args_with_config, setup_logging, AppConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Professional real-time object tracking using YOLO detection plus BoT-SORT/ByteTrack."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video path, image/video URL, or webcam index. Use 0 for default webcam.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path/name. yolov8n.pt is small and downloads automatically on first run.",
    )
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="BoT-SORT is more stable for occlusion; ByteTrack is faster.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--output", default="data/output/yolo_tracked_output.avi", help="Annotated output video.")
    parser.add_argument("--metrics", default="data/output/yolo_tracking_metrics.csv", help="Tracking metrics CSV.")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video or CSV.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Track and draw all detected objects instead of selecting one.",
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
        default=80,
        help="Maximum number of trail points to display per track.",
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


def normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def center_of_xyxy(xyxy: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_color(color: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(int(c) for c in color)
    return color


def draw_track(
    frame: np.ndarray,
    track_id: int,
    class_name: str | None,
    conf: float,
    xyxy: np.ndarray,
    trail: deque,
    path_length: float,
    color: tuple[int, int, int] = (40, 180, 80),
    show_conf: bool = True,
    show_path: bool = True,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    center = center_of_xyxy(xyxy)

    cv2.rectangle(frame, (x1, y1), (x2, y2), normalize_color(color), 2)
    cv2.circle(frame, center, 4, normalize_color((255, 140, 0)), -1)

    label_parts = [f"ID {track_id}"]
    if class_name is not None:
        label_parts.append(f"{class_name}")
    if show_conf:
        label_parts.append(f"{conf:.2f}")
    label_parts.append(f"path {path_length:.0f}px")
    label = " ".join(label_parts)

    cv2.putText(
        frame, label, (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, normalize_color((20, 20, 20)), 4
    )
    cv2.putText(
        frame, label, (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, normalize_color((245, 245, 245)), 2
    )

    points = list(trail)
    for index in range(1, len(points)):
        if points[index - 1] is not None and points[index] is not None:
            cv2.line(frame, points[index - 1], points[index], normalize_color((255, 140, 0)), 2)


def point_in_box(point: tuple[int, int], xyxy: np.ndarray) -> bool:
    x, y = point
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    return x1 <= x <= x2 and y1 <= y <= y2


def apply_dip_preprocessing(frame: np.ndarray) -> np.ndarray:
    """Apply DIP preprocessing: histogram equalization + Gaussian blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def compute_fps(fps_avg_window: int, current_fps: float, fps_history: deque) -> float:
    """Compute smoothed FPS using a rolling average."""
    fps_history.append(current_fps)
    if len(fps_history) > fps_avg_window:
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
    args = merge_args_with_config(args, config, "yolo")

    source = normalize_source(args.source)
    model = YOLO(args.model)
    mirror = args.mirror

    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    writer: cv2.VideoWriter | None = None
    metrics_file = None
    metrics_writer = None

    if not args.no_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            30.0,
            (640, 480),
        )
        metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([
            "frame", "track_id", "class", "confidence",
            "x1", "y1", "x2", "y2", "center_x", "center_y", "path_px"
        ])

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.trail_length))
    path_lengths: dict[int, float] = defaultdict(float)
    last_centers: dict[int, tuple[int, int]] = {}
    selected_track_id: int | None = None
    pending_click: tuple[int, int] | None = None
    fps_history: deque = deque(maxlen=args.fps_avg)

    def on_mouse(event: int, x: int, y: int, flags: int, userdata) -> None:
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = (x, y)

    window_name = "Professional YOLO Object Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        verbose=False,
    )

    frame_index = 0
    tick_start = cv2.getTickCount()
    for result in results:
        frame = result.orig_img.copy()

        # Initialize writer with actual frame size on first frame
        if writer is not None and writer.get(cv2.VIDEOWRITER_PROP_FRAMEBYTES) == 0:
            h, w = frame.shape[:2]
            writer.release()
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                30.0,
                (w, h),
            )

        # Apply mirroring if enabled (webcam default)
        if mirror:
            frame = cv2.flip(frame, 1)

        # Apply DIP preprocessing if requested
        display_frame = frame
        if args.preprocess:
            display_frame = apply_dip_preprocessing(frame)

        boxes = result.boxes
        detections = []

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()

            for track_id, class_id, conf, xyxy in zip(ids, classes, confs, xyxys):
                detections.append((track_id, class_id, conf, xyxy))

            if pending_click is not None and not args.all:
                for track_id, _, _, xyxy in detections:
                    if point_in_box(pending_click, xyxy):
                        selected_track_id = track_id
                        trails.clear()
                        path_lengths.clear()
                        last_centers.clear()
                        break
                pending_click = None

            for track_id, class_id, conf, xyxy in detections:
                if selected_track_id is not None and track_id != selected_track_id and not args.all:
                    continue

                center = center_of_xyxy(xyxy)
                if track_id in last_centers:
                    pixel_distance = distance(last_centers[track_id], center)
                    real_distance = config.calibration.pixels_to_meters(pixel_distance)
                    path_lengths[track_id] += real_distance if config.calibration.enabled else pixel_distance
                last_centers[track_id] = center
                trails[track_id].append(center)

                class_name = result.names.get(class_id, str(class_id))
                color = (40, 180, 80) if selected_track_id is not None else (180, 140, 40)
                draw_track(
                    frame, track_id, class_name, conf, xyxy,
                    trails[track_id], path_lengths[track_id], color
                )

                if metrics_writer is not None:
                    metrics_writer.writerow([
                        frame_index,
                        track_id,
                        class_name,
                        f"{conf:.4f}",
                        *[int(v) for v in xyxy],
                        *center,
                        f"{path_lengths[track_id]:.2f}",
                    ])

        # Compute FPS
        tick_end = cv2.getTickCount()
        current_fps = cv2.getTickFrequency() / max(1, tick_end - tick_start)
        tick_start = tick_end
        fps = compute_fps(args.fps_avg, current_fps, fps_history)

        # Draw UI overlay
        info_y = 30
        cv2.putText(frame, f"FPS: {fps:.2f}", (12, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4)
        cv2.putText(frame, f"FPS: {fps:.2f}", (12, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)

        status_text = (
            f"Selected ID: {selected_track_id}"
            if selected_track_id is not None
            else "Click an object to lock tracking"
        )
        cv2.putText(frame, status_text, (12, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4)
        cv2.putText(frame, status_text, (12, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)

        mirror_text = "Mirror: ON" if mirror else "Mirror: OFF"
        cv2.putText(frame, mirror_text, (12, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4)
        cv2.putText(frame, mirror_text, (12, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)

        prep_text = "DIP Preprocessing: ON" if args.preprocess else "DIP Preprocessing: OFF"
        cv2.putText(frame, prep_text, (12, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4)
        cv2.putText(frame, prep_text, (12, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)

        if config.calibration.enabled:
            cal_text = "Calibration: ON (meters)"
            cv2.putText(frame, cal_text, (12, info_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
            cv2.putText(frame, cal_text, (12, info_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
            control_y = info_y + 150
        else:
            control_y = info_y + 120

        control_text = "q=quit | r=reset | c=toggle mirror | p=toggle preprocessing"
        cv2.putText(frame, control_text, (12, control_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        cv2.putText(frame, control_text, (12, control_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)

        # Write frame
        write_frame = display_frame if args.preprocess else frame
        if writer is not None:
            writer.write(write_frame)

        # Show frame
        if args.preprocess:
            show_frame = apply_dip_preprocessing(frame)
            cv2.imshow(window_name, show_frame)
        else:
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selected_track_id = None
            trails.clear()
            path_lengths.clear()
            last_centers.clear()
        if key == ord("c"):
            mirror = not mirror
            logger.info("Mirror %s", "enabled" if mirror else "disabled")
        if key == ord("p"):
            args.preprocess = not args.preprocess
            logger.info("DIP Preprocessing %s", "enabled" if args.preprocess else "disabled")

        frame_index += 1

    if writer is not None:
        writer.release()
    if metrics_file is not None:
        metrics_file.close()
    cv2.destroyAllWindows()

    if not args.no_save:
        logger.info("Saved annotated video: %s", output_path)
        logger.info("Saved metrics CSV: %s", metrics_path)


if __name__ == "__main__":
    main()