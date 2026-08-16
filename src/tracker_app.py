import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np


TRACKER_BUILDERS = {
    "csrt": "TrackerCSRT_create",
    "kcf": "TrackerKCF_create",
    "mil": "TrackerMIL_create",
    "mosse": "TrackerMOSSE_create",
}

DEFAULT_REDETECT_THRESHOLD = 0.62


def create_tracker(name):
    name = name.lower()
    if name not in TRACKER_BUILDERS:
        raise ValueError(f"Unsupported tracker '{name}'. Choose from: {', '.join(TRACKER_BUILDERS)}")

    factory_name = TRACKER_BUILDERS[name]
    if hasattr(cv2, factory_name):
        return getattr(cv2, factory_name)()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, factory_name):
        return getattr(cv2.legacy, factory_name)()

    raise RuntimeError(
        f"OpenCV build does not include {factory_name}. "
        "Install opencv-contrib-python instead of opencv-python."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object tracking and trajectory visualization using OpenCV."
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input video. Omit to use webcam.")
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
    return parser.parse_args()


def open_capture(video_path):
    capture = cv2.VideoCapture(video_path if video_path else 0)
    if not capture.isOpened():
        source = video_path if video_path else "webcam"
        raise RuntimeError(f"Could not open {source}.")
    return capture


def bbox_center(bbox):
    x, y, w, h = [int(v) for v in bbox]
    return x + w // 2, y + h // 2


def distance_between(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def make_template(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        raise RuntimeError("Selected ROI is empty.")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def try_redetect(frame, template, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    template_h, template_w = template.shape[:2]

    if gray.shape[0] < template_h or gray.shape[1] < template_w:
        return False, None, 0.0

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)

    if score < threshold:
        return False, None, score

    x, y = location
    return True, (x, y, template_w, template_h), score


def draw_overlay(frame, bbox, tracker_name, success, fps, points, distance):
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

    info = [
        f"Tracker: {tracker_name.upper()}",
        f"Status: {'Tracking' if success else 'Lost'}",
        f"FPS: {fps:.2f}",
        f"Path: {distance:.1f}px",
        "Keys: s=select/reset, q=quit",
    ]

    for index, text in enumerate(info):
        y = height - 20 - (len(info) - 1 - index) * 24
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)


def main():
    args = parse_args()
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
    template = make_template(frame, bbox)

    writer = None
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
        metrics_writer.writerow(["frame", "success", "x", "y", "w", "h", "center_x", "center_y", "fps", "path_px"])

    points = [bbox_center(bbox)]
    total_distance = 0.0
    frame_index = 0
    last_time = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_time = time.perf_counter()
        elapsed = current_time - last_time
        last_time = current_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        success, bbox = tracker.update(frame)
        if not success:
            reacquired, new_bbox, _ = try_redetect(frame, template, args.redetect_threshold)
            if reacquired:
                bbox = new_bbox
                tracker = create_tracker(args.tracker)
                tracker.init(frame, bbox)
                success = True
        center = ("", "")
        row_bbox = ("", "", "", "")

        if success:
            center = bbox_center(bbox)
            row_bbox = tuple(int(v) for v in bbox)
            total_distance += distance_between(points[-1], center)
            points.append(center)

        draw_overlay(frame, bbox, args.tracker, success, fps, points, total_distance)

        if writer is not None:
            writer.write(frame)
        if metrics_writer is not None:
            metrics_writer.writerow([frame_index, int(success), *row_bbox, *center, f"{fps:.2f}", f"{total_distance:.2f}"])

        cv2.imshow("Object Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            bbox = cv2.selectROI("Object Tracking", frame, fromCenter=False, showCrosshair=True)
            if bbox != (0, 0, 0, 0):
                tracker = create_tracker(args.tracker)
                tracker.init(frame, bbox)
                template = make_template(frame, bbox)
                points = [bbox_center(bbox)]
                total_distance = 0.0

        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    if metrics_file is not None:
        metrics_file.close()
    cv2.destroyAllWindows()

    if not args.no_save:
        print(f"Saved annotated video: {args.output}")
        print(f"Saved metrics CSV: {args.metrics}")


if __name__ == "__main__":
    main()
