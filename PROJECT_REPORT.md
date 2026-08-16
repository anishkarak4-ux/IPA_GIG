# Project Report

## Title

Real-Time Object Tracking and Trajectory Visualization using Digital Image Processing

## Problem Statement

Object tracking is a common computer vision task where the goal is to locate the same object across consecutive video frames. This project implements an interactive tracking system using OpenCV trackers and visualizes object movement through bounding boxes, centroids, and trajectory paths.

## Objectives

- Capture video frames from webcam or video file.
- Allow the user to select the object region.
- Track the selected object across frames.
- Re-detect the object if tracking temporarily fails.
- Draw bounding boxes and centroid path.
- Measure frame rate and approximate pixel displacement.
- Export annotated video and per-frame metrics.

## Methodology

1. Read the first frame from the selected video source.
2. Select a region of interest around the target object.
3. Initialize an OpenCV tracker with the selected bounding box.
4. For each new frame, update the tracker and get the new bounding box.
5. If the tracker loses the object, use template matching with the original ROI to try reacquisition.
6. Calculate the centroid of the bounding box.
7. Draw the bounding box, centroid, and trajectory line.
8. Save output video and CSV metrics.

## Algorithms

The implementation supports four OpenCV trackers:

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
