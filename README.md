# Real-Time Object Tracking and Trajectory Visualization

Digital Image Processing project using Python and OpenCV.

The application lets a user select an object in a video or webcam stream, tracks it frame by frame, draws the bounding box, plots the movement path, estimates traveled pixel distance, displays FPS, and saves an annotated output video plus CSV metrics.

## Features

- Manual region-of-interest selection
- Object tracking using CSRT, KCF, MIL, or MOSSE
- Bounding box and centroid visualization
- Motion trail / trajectory plotting
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

## Run With Webcam

```bash
python src/tracker_app.py --tracker csrt
```

## Run With A Video

Place a video in `data/input`, then run:

```bash
python src/tracker_app.py --video data/input/sample.mp4 --tracker csrt
```

## Controls

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

Disable saving:

```bash
python src/tracker_app.py --no-save
```

## Project Aim

To design and implement a digital image processing system that tracks a selected object in real time and analyzes its motion trajectory using classical computer vision methods.

