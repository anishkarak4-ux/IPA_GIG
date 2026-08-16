# Project Report

## Title

Real-Time Object Tracking and Trajectory Visualization using Digital Image Processing

## Problem Statement

Object tracking is a common computer vision task where the goal is to locate the same object across consecutive video frames. This project implements a robust detector-based tracking system using YOLO with BoT-SORT/ByteTrack and also includes a classical OpenCV ROI tracker for comparison.

## Objectives

- Capture video frames from webcam or video file.
- Detect objects in each frame using YOLO.
- Track detected objects across frames using BoT-SORT or ByteTrack.
- Re-detect the object if tracking temporarily fails.
- Draw bounding boxes and centroid path.
- Measure frame rate and approximate pixel displacement.
- Export annotated video and per-frame metrics.

## Methodology

1. Read frames from webcam or video file.
2. Use YOLO to detect objects in the frame.
3. Send detections to BoT-SORT or ByteTrack.
4. Assign persistent IDs to tracked objects.
5. Keep recent centroid points for each ID.
6. Calculate the centroid of the bounding box.
7. Draw the bounding box, centroid, and trajectory line.
8. Save output video and CSV metrics.

## Algorithms

The recommended implementation supports:

- YOLO: object detection in each frame.
- BoT-SORT: tracking with motion association and stronger handling of occlusion.
- ByteTrack: fast tracking using high-confidence and low-confidence detections.

The comparison implementation supports four OpenCV trackers:

- CSRT: accurate correlation-filter tracker suitable for object appearance changes.
- KCF: faster correlation-filter tracker with good general performance.
- MIL: online learning tracker that can handle some object variation.
- MOSSE: very fast tracker, useful when speed matters more than accuracy.

## Mathematical Basis

Centroid of bounding box:

```text
center_x = x + w / 2
center_y = y + h / 2
```

Distance between consecutive centroids:

```text
d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

The total path length is the sum of distances between consecutive centroid positions.

Template matching score:

```text
score = normalized correlation between current frame region and selected object template
```

## Expected Result

The system should track a manually selected object and display a live annotated video stream. The output should include bounding box position, object path, tracking status, FPS, and exported metrics.

## Limitations

- Tracking can fail when the object is fully occluded.
- Re-detection can confuse similar-looking objects.
- Fast motion may reduce accuracy.
- Similar-looking background objects can cause tracker drift.
- Pixel distance is not real-world distance unless camera calibration is added.

## Future Scope

- Add automatic object detection before tracking.
- Add multi-object tracking.
- Convert pixel displacement to real-world distance using calibration.
- Add graph plotting from the exported CSV file.
