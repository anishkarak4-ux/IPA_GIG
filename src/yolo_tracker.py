import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Robust object tracking using YOLO detection plus ByteTrack or BoT-SORT."
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
    return parser.parse_args()


def normalize_source(source):
    return int(source) if source.isdigit() else source


def center_of_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def draw_track(frame, track_id, class_name, conf, xyxy, trail, path_length):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    center = center_of_xyxy(xyxy)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 80), 2)
    cv2.circle(frame, center, 4, (255, 140, 0), -1)

    label = f"ID {track_id} {class_name} {conf:.2f} path {path_length:.0f}px"
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 4)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)

    points = list(trail)
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], (255, 140, 0), 2)


def main():
    args = parse_args()
    source = normalize_source(args.source)
    model = YOLO(args.model)

    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    writer = None
    metrics_file = None
    metrics_writer = None

    trails = defaultdict(lambda: deque(maxlen=80))
    path_lengths = defaultdict(float)
    last_centers = {}

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        verbose=False,
    )

    frame_index = 0
    for result in results:
        frame = result.orig_img.copy()

        if writer is None and not args.no_save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), 25, (width, height))
            metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y", "path_px"])

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()

            for track_id, class_id, conf, xyxy in zip(ids, classes, confs, xyxys):
                center = center_of_xyxy(xyxy)
                if track_id in last_centers:
                    path_lengths[track_id] += distance(last_centers[track_id], center)
                last_centers[track_id] = center
                trails[track_id].append(center)

                class_name = result.names.get(class_id, str(class_id))
                draw_track(frame, track_id, class_name, conf, xyxy, trails[track_id], path_lengths[track_id])

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

        cv2.putText(frame, f"YOLO + {args.tracker}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 4)
        cv2.putText(frame, f"YOLO + {args.tracker}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
        cv2.imshow("YOLO Object Tracking", frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    if writer is not None:
        writer.release()
    if metrics_file is not None:
        metrics_file.close()
    cv2.destroyAllWindows()

    if not args.no_save:
        print(f"Saved annotated video: {output_path}")
        print(f"Saved metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
