# Real-Time Object Tracking & Modal Digital Image Processing (DIP) Inspection System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV Contrib](https://img.shields.io/badge/OpenCV--Contrib-4.14.0-green.svg)](https://opencv.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![Course](https://img.shields.io/badge/Course-Image%20Processing%20%26%20Applications-orange.svg)]()
[![Repository](https://img.shields.io/badge/GitHub-IPA__GIG-181717.svg?logo=github)](https://github.com/anishkarak4-ux/IPA_GIG)

A comprehensive, academic-grade cyber-physical vision system integrating real-time visual object tracking with an interactive, on-demand modal **Digital Image Processing (DIP)** inspection workbench.

---

## 📑 Formal Project Documentation & Reports

The project documentation has been fully formalized according to rigorous IEEE academic standards:

| Document Format | Direct Link | Description |
| :--- | :--- | :--- |
| 📕 **Academic PDF Report (29 Pages)** | [**`reports/Unified_Tracking_and_Modal_DIP_Inspection_IPA.pdf`**](reports/Unified_Tracking_and_Modal_DIP_Inspection_IPA.pdf) | Formal IEEE-typeset project report including full mathematical formulations, SOTA comparative analysis, figures, and team contributions sign-off. |
| 📘 **Editable Word Document (.docx)** | [**`reports/Unified_Tracking_and_Modal_DIP_Inspection_IPA.docx`**](reports/Unified_Tracking_and_Modal_DIP_Inspection_IPA.docx) | Full Microsoft Word source file with high-resolution figures, tables, and academic formatting. |
| 📄 **Repository Markdown Report** | [**`PROJECT_REPORT.md`**](PROJECT_REPORT.md) | In-repo searchable academic report with complete LaTeX equations, DFDs, and state diagrams. |

---

## 1. Project Overview & Architecture

The system bridges two computer vision tracking paradigms with an in-situ image inspection laboratory:
1. **Classical Correlation-Filter & Discriminative Tracking:** OpenCV CSRT, KCF, MIL, and MOSSE augmented with multi-scale Normalized Cross-Correlation (NCC) template re-acquisition.
2. **Deep Learning Detection & Spatio-Temporal Association:** Ultralytics YOLOv8 nano paired with BoT-SORT (camera motion compensation + ReID) and ByteTrack (two-stage association).

### Unified Tracking + Modal DIP Inspection Workflow
At any moment during live tracking, pressing the `SPACE` key freezes execution and captures a dual-context snapshot (**Full Frame Context + Tracked Target ROI**). The system routes this ROI into a 5-tab, multi-domain DIP inspection workbench. Once inspection is complete, pressing `SPACE` immediately resumes live tracking with uninterrupted kinematic and trajectory continuity.

```
VIDEO STREAM (Webcam / File)
      ↓
OBJECT TRACKING (Classical CSRT/KCF/MIL/MOSSE or YOLOv8 BoT-SORT/ByteTrack)
      ↓
LIVE TELEMETRY & TRAJECTORY HUD (Breadcrumbs, Speed, Single-Sum Distance)
      ↓
[SPACE KEY PRESSED]
      ↓
FREEZE + CAPTURE DUAL SNAPSHOT (Full Frame Context + Tracked Target ROI)
      ↓
MODAL DIP INSPECTION WORKBENCH (5 Thematic Tabs & 6 Panels per Tab)
      ├── TAB 1: Core Spatial & Classical DIP (Raw vs CDF vs CLAHE, Hist Stats, Edges, Otsu, Contours)
      ├── TAB 2: Frequency Domain & Restoration (2D FFT Mag/Phase, BLPF, GHPF, Homomorphic, Wiener vs 2D Block DCT)
      ├── TAB 3: Color Spaces & Advanced Segmentation (HSV/Lab/YCbCr, JET Pseudo-Color, K-Means, Watershed, Top-Hat)
      ├── TAB 4: Morphometric & Feature Descriptors (Distances L2/D4/D8, Bit-Planes, Skeleton vs LBP, Hu Moments φ1-φ7)
      └── TAB 5: Noise & Degradation Lab (6 Models: Clean, Gaussian, Salt/Pepper, Rayleigh, Uniform, Periodic + PSNR/MSE)
      ↓
[E KEY] → EXPORT SINGLE-CLICK ACADEMIC HTML LAB REPORT (KaTeX Math + Embedded Canvases)
[S KEY] → SAVE COMPOSITE CANVAS PNG + LOG DIAGNOSTICS TO CSV
[SPACE KEY] → CLOSE CANVAS & RESUME LIVE TRACKING
```

---

## 2. Complete Step-by-Step Guide: How to Run the Application

### Step 1: Clone the Repository
Clone the repository from GitHub and navigate to the project directory:
```bash
git clone https://github.com/anishkarak4-ux/IPA_GIG.git
cd IPA_GIG
```

### Step 2: Create and Activate Virtual Environment
Use Python 3.10, 3.11, or 3.12:

* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Required Packages
Install all project dependencies with one command:
```bash
pip install -r requirements.txt
```

> **IMPORTANT:** Ensure `opencv-contrib-python>=4.8,<5` is installed (included in `requirements.txt`). Do NOT install standard `opencv-python`, as it lacks the tracking algorithms (`TrackerCSRT`, `TrackerKCF`, `TrackerMOSSE`).

---

### Step 4: Quick Start — 1-Command Launch

To start tracking immediately with your default webcam:
```bash
python src/main.py
```
**What happens on launch:**
1. A live webcam window opens with horizontal mirroring enabled (for natural webcam movement).
2. The system prompts you to select a target object:
   - Click and drag a bounding box over any object with your mouse.
   - Press **`ENTER`** or **`SPACE`** to confirm the selection.
3. The system locks onto the object with the CSRT tracker and displays live trajectory breadcrumbs, speed, and single-sum cumulative distance!

---

### Step 5: Full Execution Guide & Common Modes

All features are accessible through the unified entry point [`src/main.py`](src/main.py):

#### A. Classical Tracking Modes
```bash
# 1. CSRT Tracker on Webcam (Default, most robust bounding-box tracking)
python src/main.py --mode classical --tracker csrt

# 2. Ultra-fast KCF Tracker (300+ FPS)
python src/main.py --mode classical --tracker kcf

# 3. High-throughput MOSSE Tracker
python src/main.py --mode classical --tracker mosse

# 4. Classical Tracker on a Recorded Video File
python src/main.py --mode classical --tracker csrt --source path/to/video.mp4

# 5. Enable Real-Time DIP Contrast Equalization & Gaussian Smoothing Preprocessing
python src/main.py --mode classical --tracker csrt --preprocess
```

#### B. YOLO Deep Learning Modes
```bash
# 1. YOLOv8n with BoT-SORT (Camera motion compensation + Re-ID, Click-to-Lock Target)
python src/main.py --mode yolo --tracker botsort.yaml

# 2. YOLOv8n with ByteTrack (High-speed two-stage association)
python src/main.py --mode yolo --tracker bytetrack.yaml

# 3. Track All Detected Objects Simultaneously (Multi-Object Tracking)
python src/main.py --mode yolo --track-all

# 4. YOLO on Video File with Custom Confidence Threshold
python src/main.py --mode yolo --tracker botsort.yaml --source path/to/video.mp4 --conf 0.35
```

#### C. Automated / Headless Benchmarking
```bash
# Run headless batch benchmark over 200 frames with predefined ROI (no GUI popups)
python src/main.py --mode classical --tracker csrt --source data/output/tracked_output.avi --init-roi 284,245,36,38 --max-frames 200 --headless
```

#### D. Generating Post-Run Performance Dashboards
```bash
# Automatically generate publication-quality analytics plots upon exit
python src/main.py --mode classical --tracker csrt --plot --plot-dir data/output/analytics

# Or generate plots anytime from an existing metrics CSV log:
python src/plot_metrics.py --metrics data/output/tracking_metrics.csv
```

---

## 3. Interactive Keyboard & Mouse Controls

### During Live Tracking:

| Key / Input | Action Description |
| :---: | :--- |
| **`SPACE`** | **Pause tracking & open the Modal DIP Inspection Workbench** on current target ROI. |
| **`S`** | **Select new Target ROI** via interactive mouse crosshair (draw box, press `ENTER`). |
| **`Mouse Click`** | **In YOLO mode:** Click on any detected object bounding box to lock tracking onto that specific target ID. |
| **`C`** | **Toggle horizontal camera mirroring** (ON by default for natural webcam interaction). |
| **`P`** | **Toggle DIP preprocessing** (real-time CLAHE / histogram equalization and Gaussian smoothing). |
| **`R`** | **Reset tracker & clear trajectory history**. |
| **`E`** | **Export Academic HTML Lab Report** instantly for the active frame. |
| **`Q` / `ESC`** | **Quit application** and write output telemetry CSV and video files. |

---

### Inside Modal DIP Inspection Workbench:

When you press `SPACE`, the system freezes and displays the 6-panel diagnostic canvas:

| Key / Input | Action Description |
| :---: | :--- |
| **`T` or `TAB`** | **Cycle between all 5 Thematic DIP Tabs:**<br>• **Tab 1:** Spatial & Classical DIP (CDF vs CLAHE, Histograms, Sobel/Canny Edges, Otsu, Contours)<br>• **Tab 2:** Frequency Domain & Restoration (2D FFT Magnitude/Phase, BLPF, GHPF, Homomorphic, Wiener Filter, 2D Block DCT)<br>• **Tab 3:** Color Spaces & Advanced Segmentation (HSV, Lab, YCbCr, JET Pseudo-Color, K-Means, Watershed, Top-Hat)<br>• **Tab 4:** Morphometric & Feature Descriptors (Distances L2/D4/D8, Bit-Planes, Skeleton vs LBP, Hu Moments φ1-φ7)<br>• **Tab 5:** Noise & Degradation Lab (6 Models: Clean, Gaussian, Salt/Pepper, Rayleigh, Uniform, Periodic + PSNR/MSE) |
| **`1` – `6`** | **Highlight specific panel (1 to 6)** with an active cyan diagnostic outline. |
| **`0`** | **Clear active panel highlight**. |
| **`S`** | **Save 1280×720 composite canvas image** to `data/output/snapshots/` and append quantitative diagnostic row to `snapshot_analysis.csv`. |
| **`E`** | **Export Comprehensive Academic HTML Lab Report** with KaTeX math equations, tables, and embedded high-res canvases. |
| **`SPACE` / `Q` / `ESC`** | **Close inspection canvas and resume live tracking immediately** with full state continuity. |

---

## 4. System Architecture & Repository Structure

The codebase is organized into cleanly decoupled, single-responsibility modules:

```
IPA_GIG/
├── config.yaml                    # Declarative YAML configuration
├── requirements.txt               # Pinned package dependencies
├── README.md                      # Comprehensive execution & user manual
├── PROJECT_REPORT.md              # Academic technical report (IEEE format)
├── kiit_logo.png                  # University insignia asset
├── reports/                       # Formal Academic Reports
│   ├── Unified_Tracking_and_Modal_DIP_Inspection_IPA.pdf   # 29-page formal academic report
│   ├── Unified_Tracking_and_Modal_DIP_Inspection_IPA.docx  # Editable Word document
│   └── Unified_Tracking_and_Modal_DIP_Inspection_IPA.md    # Source Markdown report
├── models/
│   ├── .gitkeep                   # Models directory marker
│   └── yolov8n.pt                 # YOLOv8 nano model weights
├── data/
│   ├── input/                     # Source input video clips
│   └── output/                    # Persistent tracking results & logs
│       ├── tracking_metrics.csv   # Per-frame kinematics telemetry CSV
│       ├── snapshot_analysis.csv  # Quantitative DIP diagnostic CSV
│       ├── tracked_output.avi     # Annotated video output
│       ├── analytics/             # Publication-quality post-run plots
│       └── snapshots/             # Saved 1280x720 modal canvas PNGs & HTML reports
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
    │   ├── filters.py             # Grayscale, Histograms, Equalization, CLAHE, Point Transforms
    │   ├── edge_detectors.py      # Sobel, Prewitt, Roberts, Laplacian, LoG, Canny hysteresis
    │   ├── frequency.py           # 2D FFT Mag/Phase, BLPF, GHPF, Homomorphic, 2D Block DCT
    │   ├── color_restoration.py   # Color planes (HSV/Lab/YCbCr), Noise Lab, Wiener Deblur
    │   ├── segmentation.py        # Otsu thresholding, K-Means clustering, Watershed
    │   └── morphology_analysis.py # Morphology (Open/Close/TopHat/Skeleton), Hu Moments, GLCM, LBP
    ├── visualization/
    │   ├── overlay.py             # Live HUD, bounding boxes, trajectory breadcrumbs
    │   ├── histogram_plotter.py   # Pure OpenCV 256-bin histogram rendering
    │   └── inspection_canvas.py   # 5-tab composite 1280x720 modal GUI canvas
    └── analytics/
        ├── metrics_logger.py      # Thread-safe telemetry & snapshot CSV persistence
        ├── plot_metrics.py        # PerformancePlotter publication dashboards
        └── report_generator.py    # Automated standalone academic HTML lab report generator
```

---

## 5. Objective Benchmark Results

Measurements taken deterministically on test video sequence ($640\times 480$, 30 FPS, initial ROI `[284, 245, 36, 38]`):

### Classical Tracker Benchmark (200 Frames)

| Algorithm | Frames | Success Rate | Avg FPS | Min FPS | Max FPS | Trajectory Path | Avg Step | Redetections | Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSRT** | 200 | **100.0%** | **35.88** | 22.96 | 39.88 | 652.03 px | 3.95 px | 0 | 0 |
| **KCF** | 200 | **100.0%** | **309.18** | 48.44 | 495.22 | 294.82 px | 3.43 px | 4 | 0 |
| **MIL** | 200 | **100.0%** | **19.25** | 12.98 | 23.88 | 374.18 px | 2.69 px | 0 | 0 |
| **MOSSE**| 200 | **84.5%** | **3125.46** | 79.44 | 8051.53 | 652.02 px | 5.82 px | 66 | 31 |

### YOLO Tracker Benchmark (60 Frames, CPU)

| Configuration | Total Detections | Unique Tracks | Avg FPS | Avg Confidence | Min / Max Conf | Detected Classes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n + BoT-SORT** | 315 | 12 | **12.96** | 0.6904 | 0.2513 / 0.9398 | Person, Chair, Cell Phone |
| **YOLOv8n + ByteTrack** | 317 | 12 | **17.23** | 0.6879 | 0.2513 / 0.9398 | Person, Chair, Cell Phone |

---

## 6. Output Data Formats

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

## 7. Authors & Academic Affiliation

**Course:** Image Processing and Applications (IPA) Laboratory  
**Institution:** School of Computer Engineering, Kalinga Institute of Industrial Technology (KIIT) Deemed to be University, Bhubaneswar, Odisha, India  
**Supervisor:** Prof. Sricheta Parui  

| Student Name | Roll No. | Role & Contribution |
| :--- | :---: | :--- |
| **Anish Karak** | 24155833 | **Lead Developer** (Spatial transforms, contour geometry, Haralick texture) |
| **Yash Agrawal** | 241551089 | **Core Developer** (2D FFT frequency filtering, 8×8 block-DCT compression) |
| **Awesh Khan** | 24155837 | **Core Developer** (Color spaces, pseudo-color, stochastic noise degradation lab) |
| **Priyaranjan Sahu** | 241551055 | **Core Developer** (Segmentation, morphology, comparative tracking evaluation) |
