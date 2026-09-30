# Real-Time Object Tracking & Modal Digital Image Processing (DIP) Inspection System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV Contrib](https://img.shields.io/badge/OpenCV--Contrib-4.14.0-green.svg)](https://opencv.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![Course](https://img.shields.io/badge/Course-Image%20Processing%20%26%20Applications-orange.svg)]()

A comprehensive, academic-grade application integrating real-time visual object tracking with an interactive, on-demand modal **Digital Image Processing (DIP)** inspection canvas.

---

## 1. Project Overview & Objective

The primary objective is to evaluate and bridge two tracking paradigms:
1. **Classical Correlation-Filter & Discriminative Tracking:** OpenCV CSRT, KCF, MIL, and MOSSE augmented with normalized cross-correlation (NCC) template redetection.
2. **Deep Learning Detection & Spatio-Temporal Association:** Ultralytics YOLOv8 nano paired with BoT-SORT (camera motion compensation + ReID) and ByteTrack (two-stage association).

Crucially, the system introduces a **Unified Tracking + Modal DIP Inspection Workflow**:
At any moment during live tracking, pressing the `SPACE` key freezes execution and captures a dual-context snapshot (**Full Frame Context + Tracked Target ROI**). The system then feeds this region of interest through a complete 8-stage image processing suite, rendering an interactive 1280×720 multi-panel diagnostic dashboard. Once analysis is inspected or saved, pressing `SPACE` immediately resumes live tracking with uninterrupted kinematic and trajectory continuity.

```
VIDEO STREAM (Webcam / File)
      ↓
OBJECT TRACKING (Classical / YOLOv8)
      ↓
LIVE TELEMETRY & TRAJECTORY HUD (Breadcrumbs, Speed, Single-Sum Distance)
      ↓
[SPACE KEY PRESSED]
      ↓
FREEZE + CAPTURE DUAL SNAPSHOT (Full Frame + Tracked Target ROI)
      ↓
MODAL DIP INSPECTION CANVAS (6 Specialized Diagnostic Viewports)
      ├── 1. Full Frame Context
      ├── 2. Target ROI (Original vs Equalized)
      ├── 3. 256-Bin Intensity Histogram & CDF Curves
      ├── 4. Spatial Denoising (Gaussian/Median) & Differential Edges (Sobel/Canny)
      ├── 5. Optimal Global Segmentation (Otsu) & Morphology (Open/Close)
      └── 6. Contour Extraction & Academic Shape Descriptors (Area, Circularity, Solidity)
      ↓
[S KEY] → SAVE COMPOSITE CANVAS PNG + LOG DIAGNOSTICS TO CSV
      ↓
[SPACE KEY] → CLOSE CANVAS & RESUME LIVE TRACKING
```

---

## 2. Academic Relevance to Image Processing & Applications

This project demonstrates core classical and modern DIP theoretical principles:

- **Point Operations & Radiometric Enhancement:** Probability density estimation via 256-bin intensity histograms; contrast enhancement via Cumulative Distribution Function (CDF) linearization (`cv2.equalizeHist`).
- **Spatial Domain Noise Suppression:** Linear Gaussian low-pass spatial convolution ($5\times 5$, $\sigma=1.0$) and nonlinear rank-order Median filtering ($5\times 5$) for impulse and sensor noise reduction.
- **Differential Edge Detection:** First-order spatial gradient operators (Sobel kernels $G_x$, $G_y$) with gradient magnitude $M(x,y) = \sqrt{G_x^2 + G_y^2}$ and directional orientation $\theta$; Canny optimal edge detection incorporating non-maximum suppression (NMS) and hysteresis thresholding.
- **Global Thresholding:** Otsu's discriminant criterion maximizing inter-class variance $\sigma_B^2(t)$ between foreground and background intensity distributions.
- **Mathematical Morphology:** Set-theoretic morphological Opening ($A \circ B = (A \ominus B) \oplus B$) for background noise decoupling, and Closing ($A \bullet B = (A \oplus B) \ominus B$) for interior hole filling and boundary bridging.
- **Geometric & Morphometric Shape Descriptors:** Green's theorem contour area ($A$), perimeter arc length ($P$), aspect ratio, circularity ($C = 4\pi A / P^2$), and convex hull solidity ($S = A / A_{\text{hull}}$).
- **Kinematic Telemetry:** Euclidean displacement calculation with strict single-sum cumulative trajectory integration $D = \sum d_i$ (eliminating double-counting errors).

---

## 3. System Architecture & Repository Structure

The codebase is organized into cleanly decoupled, single-responsibility modules:

```
IPA_GIG/
├── config.yaml                    # Declarative configuration file
├── requirements.txt               # Pinned package dependencies
├── README.md                      # Comprehensive user manual
├── PROJECT_REPORT.md              # Academic technical report
├── models/
│   └── yolov8n.pt                 # YOLOv8 nano model weights
├── data/
│   ├── input/                     # Source video directory
│   └── output/                    # Persistent tracking results & logs
│       ├── tracking_metrics.csv   # Per-frame kinematics telemetry CSV
│       ├── snapshot_analysis.csv  # Quantitative DIP diagnostic CSV
│       ├── tracked_output.avi     # Annotated video output
│       ├── analytics/             # Publication-quality post-run plots
│       └── snapshots/             # Saved 1280x720 modal canvas PNGs
└── src/
    ├── main.py                    # UNIFIED CENTRAL ORCHESTRATION ENTRY POINT
    ├── config_loader.py           # Dataclass YAML configuration parser
    ├── tracker_app.py             # Baseline legacy Classical tracker script
    ├── yolo_tracker.py            # Baseline legacy YOLO tracker script
    ├── plot_metrics.py            # Baseline legacy metrics analyzer script
    ├── core/
    │   ├── input_manager.py       # VideoSource abstraction, FramePacket, mirroring
    │   └── telemetry.py           # TelemetryTracker, single-sum kinematics, calibration
    ├── tracking/
    │   ├── base_tracker.py        # BaseTracker protocol & TrackingResult contract
    │   ├── classical_tracker.py   # Modular CSRT, KCF, MIL, MOSSE with NCC redetection
    │   └── yolo_tracker.py        # Modular YOLOv8 + BoT-SORT / ByteTrack engine
    ├── processing/
    │   ├── dip_pipeline.py        # DIPPipeline orchestrator & SnapshotContext
    │   ├── filters.py             # Grayscale, Histograms, Equalization, Gaussian, Median
    │   ├── edge_detectors.py      # Sobel gradients, Laplacian, Canny hysteresis
    │   ├── segmentation.py        # Otsu thresholding, adaptive thresholding
    │   └── morphology_analysis.py # Morphological Open/Close, Contour metrics
    ├── visualization/
    │   ├── overlay.py             # Live HUD, bounding boxes, trajectory breadcrumbs
    │   ├── histogram_plotter.py   # Pure OpenCV 256-bin histogram rendering
    │   └── inspection_canvas.py   # 6-panel composite 1280x720 modal GUI canvas
    └── analytics/
        ├── metrics_logger.py      # Thread-safe telemetry & snapshot CSV persistence
        └── plot_metrics.py        # PerformancePlotter publication dashboards
```

---

## 4. Installation & Environment Setup

The system is tested and verified on Python 3.12 (Windows / Linux / macOS).

```bash
# 1. Clone repository
git clone https://github.com/anishkarak4-ux/dipa.git
cd IPA_GIG

# 2. Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **IMPORTANT:** Ensure `opencv-contrib-python>=4.8,<5` is installed. Do NOT install regular `opencv-python`, as it shadows tracking algorithms (`TrackerCSRT`, `TrackerKCF`, `TrackerMOSSE`).

---

## 5. Usage & Execution Guide

### Central Unified Entry Point (`src/main.py`)

All features are unified through `src/main.py`.

#### A. Classical Tracking Modes
```bash
# 1. CSRT Tracker on Webcam (Default, with horizontal mirroring)
python src/main.py --mode classical --tracker csrt

# 2. Fast KCF Tracker on Video File with DIP Preprocessing
python src/main.py --mode classical --tracker kcf --source data/output/tracked_output.avi --preprocess

# 3. Headless Batch Execution with Predefined ROI (Automated / Non-Interactive)
python src/main.py --mode classical --tracker csrt --source data/output/tracked_output.avi --init-roi 284,245,36,38 --max-frames 200 --headless
```

#### B. YOLO Deep Learning Modes
```bash
# 1. YOLOv8n with BoT-SORT (Occlusion-Resistant, Click-to-Lock Target Selection)
python src/main.py --mode yolo --tracker botsort.yaml --model models/yolov8n.pt

# 2. Fast YOLOv8n with ByteTrack and Post-Run Performance Plots
python src/main.py --mode yolo --tracker bytetrack.yaml --plot --plot-dir data/output/analytics

# 3. Track All Objects Simultaneously
python src/main.py --mode yolo --track-all --source data/output/tracked_output.avi
```

#### C. Legacy Baseline Executables (Backward Compatibility)
```bash
# Classical tracker baseline
python src/tracker_app.py --tracker csrt --video data/output/tracked_output.avi

# YOLO tracker baseline
python src/yolo_tracker.py --tracker botsort.yaml --source 0

# Baseline metrics analyzer
python src/plot_metrics.py --metrics data/output/tracking_metrics.csv
```

---

## 6. Interactive Keyboard & Mouse Controls

### During Live Tracking:
- **`SPACE`:** **Pause tracking and open the Modal DIP Inspection Canvas.**
- **`R`:** Reset tracker (prompts new ROI selection in Classical mode; clears target lock in YOLO mode).
- **`C`:** Toggle horizontal webcam mirroring (ON by default for natural webcam interaction).
- **`P`:** Toggle DIP contrast enhancement & Gaussian preprocessing.
- **`Mouse Left Click`:**
  - *Classical mode:* Select initial bounding box via crosshair selection.
  - *YOLO mode:* Click any detected object on screen to lock persistent tracking onto its ID.
- **`Q` / `ESC`:** Quit application and write output files.

### Inside Modal DIP Inspection Canvas:
- **`1` - `6`:** Highlight corresponding DIP inspection panel with a cyan outline banner:
  - `1`: Full Frame Context
  - `2`: Target ROI (Original vs Equalized)
  - `3`: 256-Bin Intensity Histogram & CDF Curves
  - `4`: Denoising Filters & Differential Edges
  - `5`: Otsu Segmentation & Morphology Mask
  - `6`: Contour Extraction & Academic Shape Descriptors
- **`0`:** Clear active panel selection.
- **`S`:** Save 1280×720 canvas image to `data/output/snapshots/` and log quantitative row to `snapshot_analysis.csv`.
- **`SPACE`:** Close modal canvas and resume live tracking immediately.
- **`Q` / `ESC`:** Exit modal canvas and resume tracking.

---

## 7. Objective Benchmark Results

Measurements taken deterministically on the verified test video (`tracked_output.avi`, $640\times 480$, 30 FPS, initial ROI `[284, 245, 36, 38]`):

### Classical Tracker Benchmark (200 Frames)

| Algorithm | Frames | Success Rate | Avg FPS | Min FPS | Max FPS | Trajectory Path | Avg Step | Redetections | Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSRT** | 200 | **100.0%** | **35.88** | 22.96 | 39.88 | 652.03 px | 3.95 px | 0 | 0 |
| **KCF** | 200 | **100.0%** | **309.18** | 48.44 | 495.22 | 294.82 px | 3.43 px | 4 | 0 |
| **MIL** | 200 | **100.0%** | **19.25** | 12.98 | 23.88 | 374.18 px | 2.69 px | 0 | 0 |
| **MOSSE**| 200 | **84.5%** | **3125.46** | 79.44 | 8051.53 | 652.02 px | 5.82 px | 66 | 31 |

*Note: FPS and robustness represent local CPU hardware measurements and are not universal bounds.*

### YOLO Tracker Benchmark (60 Frames, CPU)

| Configuration | Total Detections | Unique Tracks | Avg FPS | Avg Confidence | Min / Max Conf | Detected Classes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n + BoT-SORT** | 315 | 12 | **12.96** | 0.6904 | 0.2513 / 0.9398 | Person, Chair, Cell Phone |
| **YOLOv8n + ByteTrack** | 317 | 12 | **17.23** | 0.6879 | 0.2513 / 0.9398 | Person, Chair, Cell Phone |

---

## 8. Output Data Formats

### A. Tracking Metrics CSV (`tracking_metrics.csv`)
Logs per-frame kinematics and state information:
```csv
frame,timestamp,session_id,tracker,success,x,y,w,h,center_x,center_y,step_px,path_px,fps,velocity_px_s,confidence,track_id,class
0,1727694120.1234,20260930_130000,CSRT,1,284,245,36,38,302,264,0.00,0.00,30.00,0.00,1.0000,,
1,1727694120.1567,20260930_130000,CSRT,1,283,244,39,41,302,264,0.00,0.00,30.25,0.00,1.0000,,
```

### B. Snapshot Analysis CSV (`snapshot_analysis.csv`)
Logs quantitative image processing metrics calculated on paused targets:
```csv
frame,timestamp,session_id,tracker,track_id,class,confidence,x,y,w,h,roi_width,roi_height,otsu_threshold,area,perimeter,aspect_ratio,circularity,solidity,centroid_x,centroid_y,contour_count
5,1727694125.4321,20260930_130000,CSRT,,,1.0000,284,245,36,38,36,38,135.00,841.00,116.00,0.9474,0.7854,1.0000,18,19,1
```

---

## 9. Known Limitations & Future Improvements

1. **Hardware Dependent Inference:** On low-power CPUs without AVX2 or GPU acceleration, YOLO inference rate is typically 12–20 FPS; real-time 30+ FPS operation requires GPU/DirectML acceleration or model quantization (ONNX / OpenVINO).
2. **Extreme Occlusion:** Classical trackers can drift when an object is fully occluded for extended durations unless the template cross-correlation score recovers the object upon reappearance.
3. **Future Work:**
   - Multi-camera spatial re-identification and tracking handoff.
   - Deep learning semantic segmentation (e.g. YOLOv8-seg) integrated into Panel 5 alongside Otsu thresholding.
   - Stereo vision or depth camera integration (e.g. Intel RealSense) for true 3D spatial coordinate tracking in meters.