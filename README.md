# Real-Time Object Tracking and Trajectory Visualization

Digital Image Processing project using Python and OpenCV.

The application lets a user select an object in a video or webcam stream, tracks it frame by frame, draws the bounding box, plots the movement path, estimates traveled pixel distance, displays FPS, and saves an annotated output video plus CSV metrics.

The project includes two trackers:

- `src/yolo_tracker.py`: recommended robust tracker using YOLO + BoT-SORT/ByteTrack.
- `src/tracker_app.py`: classical OpenCV ROI tracker for DIP algorithm comparison.

## Features

- Manual region-of-interest selection
- Object tracking using CSRT, KCF, MIL, or MOSSE
- Bounding box and centroid visualization
- Motion trail / trajectory plotting
- Automatic template-based re-detection after tracker loss
- FPS display
- Per-frame tracking metrics export
- Annotated video export

## DIP Concepts Used

- Video frame acquisition
- Region of interest selection
- Feature/correlation-based tracking
- Bounding box localization
- Centroid calculation
- Euclidean distance for trajectory length
- Frame-by-frame visualization

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use `opencv-contrib-python`, not plain `opencv-python`, because OpenCV trackers are part of the contrib build on many installations.

## Recommended Run

```bash
python src/yolo_tracker.py --source 0 --tracker botsort.yaml
```

This version detects common object classes automatically. Click one detected object in the window to lock tracking to that single object.

For faster tracking:

```bash
python src/yolo_tracker.py --source 0 --tracker bytetrack.yaml
```

To draw every detected object instead:

```bash
python src/yolo_tracker.py --source 0 --tracker botsort.yaml --all
```

## Run With A Video

Place a video in `data/input`, then run:

```bash
python src/yolo_tracker.py --source data/input/sample.mp4 --tracker botsort.yaml
```

The first YOLO run may download `yolov8n.pt`.

## Classical OpenCV Run

Use this version when you need to show traditional DIP tracking with manual ROI selection:

```bash
python src/tracker_app.py --tracker csrt
```

## Controls

For `yolo_tracker.py`:

- Click an object to track only that object.
- Press `r` to reset and choose another object.
- Press `q` to quit.

For `tracker_app.py`:

- Draw a box around the object and press Enter or Space to start.
- Press `s` to select/reset the tracked object.
- Press `q` to quit.

## Tracker Options

```bash
python src/tracker_app.py --tracker csrt
python src/tracker_app.py --tracker kcf
python src/tracker_app.py --tracker mil
python src/tracker_app.py --tracker mosse
```

Suggested comparison:

| Tracker | Strength | Weakness |
| --- | --- | --- |
| CSRT | Better accuracy | Slower |
| KCF | Good speed | Less robust than CSRT |
| MIL | Handles some appearance changes | Can drift |
| MOSSE | Very fast | Lower accuracy |

## Output

By default, the app writes:

- `data/output/tracked_output.avi`
- `data/output/tracking_metrics.csv`
- `data/output/yolo_tracked_output.avi`
- `data/output/yolo_tracking_metrics.csv`

Disable saving:

```bash
python src/tracker_app.py --no-save
```

## Project Aim

To design and implement a digital image processing system that tracks a selected object in real time and analyzes its motion trajectory using classical computer vision methods.
