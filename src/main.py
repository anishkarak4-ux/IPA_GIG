"""IPA_GIG Central Orchestration & Unified Main Application.

Integrates real-time object tracking (Classical CSRT/KCF/MIL/MOSSE or YOLOv8 BoT-SORT/ByteTrack)
with live telemetry, trajectory visualization, metrics logging, and a modal Digital Image
Processing (DIP) inspection canvas accessible on demand (SPACE key).

Architecture:
    VideoSource (input_manager)
        ↓ FramePacket
    BaseTracker (classical_tracker | yolo_tracker)
        ↓ TrackingResult
    TelemetryTracker (telemetry)
        ↓ TrackingTelemetry
    OverlayRenderer (overlay) + MetricsLogger (metrics_logger)
        ↓
    [SPACE Key] → SnapshotContext → DIPPipeline → DIPResult → InspectionCanvas
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

# Ensure project root and src/ are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config_loader import AppConfig, load_config, merge_args_with_config, setup_logging
    from src.core.input_manager import FramePacket, VideoSource, normalize_source
    from src.core.telemetry import CalibrationModel, TelemetryTracker, TrackingTelemetry
    from src.tracking.base_tracker import BaseTracker, TrackingResult
    from src.tracking.classical_tracker import ClassicalTracker
    from src.tracking.yolo_tracker import YOLOTracker
    from src.processing.dip_pipeline import DIPPipeline, SnapshotContext, DIPResult
    from src.visualization.overlay import TrackingVisualizer, render_overlay
    from src.visualization.inspection_canvas import InspectionCanvas
    from src.analytics.metrics_logger import MetricsLogger
    from src.analytics.plot_metrics import PerformancePlotter
except ImportError:
    from config_loader import AppConfig, load_config, merge_args_with_config, setup_logging
    from core.input_manager import FramePacket, VideoSource, normalize_source
    from core.telemetry import CalibrationModel, TelemetryTracker, TrackingTelemetry
    from tracking.base_tracker import BaseTracker, TrackingResult
    from tracking.classical_tracker import ClassicalTracker
    from tracking.yolo_tracker import YOLOTracker
    from processing.dip_pipeline import DIPPipeline, SnapshotContext, DIPResult
    from visualization.overlay import TrackingVisualizer, render_overlay
    from visualization.inspection_canvas import InspectionCanvas
    from analytics.metrics_logger import MetricsLogger
    from analytics.plot_metrics import PerformancePlotter

logger = logging.getLogger("ipa_main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build unified command line interface for IPA tracking and DIP application."""
    parser = argparse.ArgumentParser(
        description="IPA_GIG: Unified Real-Time Object Tracking & Modal DIP Inspection Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Engine mode
    parser.add_argument(
        "--mode",
        choices=["classical", "yolo"],
        default="classical",
        help="Tracking engine mode: 'classical' (correlation filter) or 'yolo' (deep detector + tracker).",
    )

    # 2. Tracking algorithms
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Algorithm identifier: 'csrt', 'kcf', 'mil', 'mosse' for classical; 'botsort.yaml' or 'bytetrack.yaml' for YOLO.",
    )
    parser.add_argument(
        "--redetect-threshold",
        type=float,
        default=0.62,
        help="Template matching NCC confidence threshold for classical redetection.",
    )
    parser.add_argument(
        "--init-roi",
        type=str,
        default=None,
        help="Optional initial bounding box for classical tracker as 'x,y,w,h'. Bypasses manual ROI selection.",
    )

    # 3. YOLO specific options
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Path or name of YOLO model weights file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO object detection confidence threshold.",
    )
    parser.add_argument(
        "--track-all",
        action="store_true",
        default=False,
        help="In YOLO mode, render annotations for all detected targets simultaneously.",
    )

    # 4. Input & preprocessing
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input video file path or integer webcam device index (default: 0).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Synonym for --source (path to video file).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        default=None,
        help="Horizontally flip webcam frames for natural interaction.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        default=False,
        help="Disable horizontal frame mirroring.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Apply DIP contrast equalization and Gaussian smoothing before tracking.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=None,
        help="Maximum number of historical centroid coordinates in trajectory breadcrumb.",
    )
    parser.add_argument(
        "--fps-avg",
        type=int,
        default=30,
        help="Rolling average window size for FPS telemetry calculation.",
    )

    # 5. Output & persistence
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to save annotated output video (AVI format).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="File path to save per-frame tracking telemetry CSV.",
    )
    parser.add_argument(
        "--snapshot-metrics",
        type=str,
        default="data/output/snapshot_analysis.csv",
        help="File path to save quantitative snapshot DIP analysis CSV.",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="data/output/snapshots",
        help="Directory to save paused snapshot DIP inspection images.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Disable disk writing for output video and telemetry CSV files.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Automatically generate matplotlib performance plots upon session exit.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="data/output/analytics",
        help="Target directory for generated post-run analytics plots.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )

    # 6. Test / Automation options
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without displaying OpenCV GUI windows (for automated test environments).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Terminate session after processing a specified maximum number of frames.",
    )
    parser.add_argument(
        "--test-snapshot-at-frame",
        type=int,
        default=None,
        help="Automatically trigger a snapshot and DIP inspection at this frame index.",
    )

    return parser.parse_args(argv)


def build_tracker(mode: str, args: argparse.Namespace, config: AppConfig) -> BaseTracker:
    """Instantiate appropriate tracker backend conforming to BaseTracker protocol."""
    if mode == "classical":
        tracker_name = args.tracker or config.classical_tracker.tracker or "csrt"
        logger.info("Initializing ClassicalTracker [algorithm=%s, redetect_threshold=%.2f]", tracker_name, args.redetect_threshold)
        return ClassicalTracker(
            tracker_type=tracker_name,
            redetect_threshold=args.redetect_threshold,
            mirror=False,  # Mirroring is handled centrally by VideoSource
            preprocess=args.preprocess,
        )
    elif mode == "yolo":
        # Resolve tracker config: botsort.yaml or bytetrack.yaml
        yolo_tracker_cfg = args.tracker if (args.tracker and args.tracker.endswith(".yaml")) else config.yolo_tracker.tracker
        logger.info("Initializing YOLOTracker [model=%s, tracker=%s, conf=%.2f]", args.model, yolo_tracker_cfg, args.conf)
        return YOLOTracker(
            model_path=args.model,
            tracker_yaml=yolo_tracker_cfg,
            conf=args.conf,
            track_all=args.track_all,
            mirror=False,  # Centralized in VideoSource
            preprocess=args.preprocess,
        )
    else:
        raise ValueError(f"Unsupported tracking mode: {mode}. Must be 'classical' or 'yolo'.")


def build_snapshot_context(
    packet: FramePacket,
    tracking_result: TrackingResult | None,
    telemetry: TrackingTelemetry | None,
    tracker_name: str,
) -> SnapshotContext:
    """Construct dual-context snapshot payload (Full Frame + Tracked Target ROI)."""
    full_frame = packet.display_frame.copy()
    h, w = full_frame.shape[:2]

    roi: np.ndarray | None = None
    bbox: tuple[int, int, int, int] | None = None
    track_id: int | None = None

    if tracking_result is not None and tracking_result.bbox is not None and tracking_result.success:
        bx, by, bw, bh = [int(v) for v in tracking_result.bbox[:4]]
        # Safe coordinate clamping within frame boundaries
        x1 = max(0, min(bx, w - 1))
        y1 = max(0, min(by, h - 1))
        x2 = max(x1 + 1, min(bx + bw, w))
        y2 = max(y1 + 1, min(by + bh, h))

        if x2 > x1 and y2 > y1:
            roi = full_frame[y1:y2, x1:x2].copy()
            bbox = (x1, y1, x2 - x1, y2 - y1)
            track_id = tracking_result.track_id

    # Fallback ROI if target was lost or uninitialized: center crop
    if roi is None or roi.size == 0:
        crop_w, crop_h = min(w, 200), min(h, 200)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - crop_w // 2), max(0, cy - crop_h // 2)
        roi = full_frame[y1 : y1 + crop_h, x1 : x1 + crop_w].copy()
        bbox = (x1, y1, crop_w, crop_h)

    meta = {
        "class": tracking_result.class_name if tracking_result else "",
        "conf": tracking_result.confidence if tracking_result else 0.0,
    }

    return SnapshotContext(
        full_frame=full_frame,
        roi=roi,
        bbox=bbox,
        frame_index=packet.frame_index,
        timestamp=packet.timestamp,
        tracker_name=tracker_name,
        track_id=track_id,
        telemetry=telemetry,
        metadata=meta,
    )


def save_snapshot_artifacts(
    context: SnapshotContext,
    dip_result: DIPResult,
    canvas: np.ndarray,
    metrics_logger: MetricsLogger | None,
    snapshots_dir: Path,
    tracking_result: TrackingResult | None,
) -> Path:
    """Persist composite canvas image and append diagnostic row to snapshot CSV."""
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = int(context.timestamp * 1000)
    canvas_path = snapshots_dir / f"snapshot_frame_{context.frame_index}_{ts_tag}.png"

    cv2.imwrite(str(canvas_path), canvas)
    logger.info("Saved DIP inspection canvas image: %s", canvas_path)

    if metrics_logger is not None:
        metrics_logger.log_snapshot_dip(dip_result, tracking_result=tracking_result)
        logger.info("Logged snapshot diagnostic telemetry to CSV.")

    return canvas_path


def run_inspection_modal(
    context: SnapshotContext,
    tracking_result: TrackingResult | None,
    dip_pipeline: DIPPipeline,
    canvas_engine: InspectionCanvas,
    metrics_logger: MetricsLogger | None,
    snapshots_dir: Path,
    headless: bool = False,
) -> None:
    """Execute modal Digital Image Processing inspection and interactive user canvas."""
    logger.info("=== ENTERING MODAL DIP INSPECTION [Frame %d] ===", context.frame_index)

    # Run complete 8-stage DIP pipeline on snapshot context
    t_start = time.perf_counter()
    dip_result = dip_pipeline.process(context)
    dt_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info("DIP pipeline execution completed in %.2f ms", dt_ms)

    window_name = "IPA DIP Inspection Canvas"
    active_stage: int | None = None
    save_notification: str | None = None

    if headless:
        # In headless automated mode, generate canvas, test save, and return
        canvas = canvas_engine.render(dip_result, tracking_result, active_stage=active_stage)
        save_snapshot_artifacts(
            context, dip_result, canvas, metrics_logger, snapshots_dir, tracking_result
        )
        logger.info("=== HEADLESS MODAL DIP INSPECTION FINISHED ===")
        return

    # Interactive GUI loop
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Render composite 6-panel canvas
            canvas = canvas_engine.render(dip_result, tracking_result, active_stage=active_stage)

            if save_notification:
                # Flash save confirmation on footer
                h, w = canvas.shape[:2]
                cv2.rectangle(canvas, (18, h - 26), (w - 18, h - 4), (20, 100, 20), -1)
                cv2.putText(
                    canvas,
                    save_notification,
                    (28, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (240, 255, 240),
                    1,
                )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 32:  # SPACE: Resume tracking
                logger.info("Resuming live tracking from modal inspection.")
                break
            elif key in (27, ord("q"), ord("Q")):  # ESC / Q: Exit modal inspection
                logger.info("Closing modal inspection canvas.")
                break
            elif key in (ord("s"), ord("S")):  # S: Save analysis
                saved_path = save_snapshot_artifacts(
                    context, dip_result, canvas, metrics_logger, snapshots_dir, tracking_result
                )
                save_notification = f"ANALYSIS SAVED: {saved_path.name}"
            elif key in [ord(str(i)) for i in range(1, 7)]:
                active_stage = int(chr(key))
                logger.info("Highlighted DIP stage %d in inspection canvas.", active_stage)
            elif key == ord("0"):
                active_stage = None
    finally:
        cv2.destroyWindow(window_name)

    logger.info("=== EXITED MODAL DIP INSPECTION ===")


def run_tracking_app(args: argparse.Namespace) -> int:
    """Execute integrated real-time tracking, telemetry, and inspection orchestration."""
    # 1. Load configuration and setup logging
    config: AppConfig = load_config(args.config)
    setup_logging(config.logging)
    logger.info("Loaded configuration from %s", args.config)

    # 2. Merge CLI arguments with config defaults
    args = merge_args_with_config(args, config, args.mode)

    # Resolve mirror setting
    mirror = True
    if args.no_mirror:
        mirror = False
    elif args.mirror is not None:
        mirror = args.mirror
    else:
        mirror = config.classical_tracker.mirror if args.mode == "classical" else config.yolo_tracker.mirror

    # Resolve source
    raw_source = args.source or args.video
    if raw_source is None:
        raw_source = config.classical_tracker.video if args.mode == "classical" else config.yolo_tracker.source
    source = normalize_source(raw_source)

    # Resolve trail length and output paths
    default_trail = 100 if args.mode == "classical" else 80
    trail_len = args.trail_length or (
        config.classical_tracker.trail_length if args.mode == "classical" else config.yolo_tracker.trail_length
    ) or default_trail

    output_video_path = Path(args.output or (
        config.classical_tracker.output if args.mode == "classical" else config.yolo_tracker.output
    ))
    tracking_csv_path = Path(args.metrics or (
        config.classical_tracker.metrics if args.mode == "classical" else config.yolo_tracker.metrics
    ))
    snapshot_csv_path = Path(args.snapshot_metrics)
    snapshots_dir = Path(args.snapshots_dir)

    # 3. Instantiate Subsystems
    logger.info("Connecting VideoSource: %s (mirror=%s)", source, mirror)
    video_source = VideoSource(source=source, mirror=mirror)

    tracker = build_tracker(args.mode, args, config)

    calib_cfg = config.calibration
    calibration = CalibrationModel(
        enabled=calib_cfg.enabled,
        reference_width_m=calib_cfg.reference_object_width_m,
        reference_width_px=calib_cfg.reference_object_width_px,
        camera_height_m=calib_cfg.camera_height_m,
        camera_angle_deg=calib_cfg.camera_angle_deg,
    )
    telemetry_tracker = TelemetryTracker(
        fps_window=args.fps_avg,
        max_trail_points=trail_len,
        calibration=calibration,
    )

    metrics_logger: MetricsLogger | None = None
    if not args.no_save:
        metrics_logger = MetricsLogger(
            tracking_csv_path=tracking_csv_path,
            snapshot_csv_path=snapshot_csv_path,
            overwrite=True,
        )
        logger.info("Metrics logging enabled: %s & %s", tracking_csv_path, snapshot_csv_path)

    dip_pipeline = DIPPipeline()
    canvas_engine = InspectionCanvas(target_width=1280, target_height=720)

    # 4. Handle Classical Tracker ROI Initialization
    window_name = f"IPA Unified Tracker [{args.mode.upper()}]"
    pending_click: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = (x, y)

    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)

    # Read first frame for tracker setup
    ok, first_packet = video_source.read_frame()
    if not ok or first_packet is None:
        logger.error("Failed to read initial frame from source: %s", source)
        video_source.release()
        return 1

    first_frame = first_packet.display_frame
    frame_h, frame_w = first_frame.shape[:2]

    # Setup VideoWriter if saving
    video_writer: cv2.VideoWriter | None = None
    if not args.no_save:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fps_out = video_source.get_fps()
        if fps_out <= 1.0 or fps_out > 120.0:
            fps_out = 30.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps_out, (frame_w, frame_h))
        logger.info("Initialized VideoWriter: %s (%.1f FPS, %dx%d)", output_video_path, fps_out, frame_w, frame_h)

    # Initialize tracker if Classical
    if args.mode == "classical":
        init_bbox: tuple[int, int, int, int] | None = None
        if args.init_roi:
            try:
                parts = [int(v.strip()) for v in args.init_roi.split(",")]
                if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                    init_bbox = (parts[0], parts[1], parts[2], parts[3])
                    logger.info("Using CLI provided initial ROI: %s", init_bbox)
            except Exception as e:
                logger.warning("Invalid --init-roi '%s': %s", args.init_roi, e)

        if init_bbox is None and not args.headless:
            logger.info("Prompting user for ROI selection via OpenCV selectROI...")
            selected = cv2.selectROI(window_name, first_frame, fromCenter=False, showCrosshair=True)
            if selected != (0, 0, 0, 0):
                init_bbox = (int(selected[0]), int(selected[1]), int(selected[2]), int(selected[3]))
            else:
                logger.warning("No ROI selected by user. Tracking will be uninitialized.")

        if init_bbox is not None:
            tracker.init(first_frame, init_bbox)
            cx = init_bbox[0] + init_bbox[2] // 2
            cy = init_bbox[1] + init_bbox[3] // 2
            telemetry_tracker.set_initial_center((cx, cy))
            logger.info("Classical tracker successfully initialized on bbox: %s", init_bbox)

    # 5. Live Tracking Loop
    logger.info("Starting live tracking loop. Controls: [SPACE] Snapshot DIP, [R] Reset, [Q] Quit")
    current_packet: FramePacket | None = first_packet
    frame_count = 0
    t_loop_start = time.perf_counter()

    try:
        while current_packet is not None:
            frame = current_packet.display_frame
            f_idx = current_packet.frame_index
            t_stamp = current_packet.timestamp

            # Check maximum frames constraint (for testing / batch jobs)
            if args.max_frames is not None and frame_count >= args.max_frames:
                logger.info("Reached specified maximum frames limit (%d). Terminating.", args.max_frames)
                break

            # Handle mouse clicks for YOLO target selection
            if pending_click is not None:
                if args.mode == "yolo" and isinstance(tracker, YOLOTracker):
                    locked_id = tracker.select_target_by_point(pending_click, frame)
                    if locked_id is not None:
                        logger.info("Mouse clicked %s: Locked YOLO track ID %d", pending_click, locked_id)
                        telemetry_tracker.reset()
                pending_click = None

            # A. Update Tracker
            tracking_result = tracker.update(frame)

            # B. Update Telemetry
            fps_val = telemetry_tracker.update_fps_from_time()
            telemetry = telemetry_tracker.record_step(
                frame_index=f_idx,
                success=tracking_result.success,
                center=tracking_result.center,
                bbox=tracking_result.bbox,
                fps=fps_val,
                timestamp=t_stamp,
            )

            # C. Render Live Tracking HUD Overlay
            trail_pts = telemetry_tracker.get_trail_points()
            tracker_name_str = args.tracker or (args.mode if args.mode != "classical" else "CSRT")
            annotated_frame = render_overlay(
                frame=frame,
                result=tracking_result,
                points=trail_pts,
                tracker_name=tracker_name_str,
                fps=telemetry.fps,
                total_distance=telemetry.cumulative_distance,
                unit=telemetry.unit,
                frame_index=f_idx,
                mirror=video_source.mirror,
                preprocess=args.preprocess,
                calibration_enabled=config.calibration.enabled,
                extra_lines=["Controls: [SPACE] DIP Inspect  |  [R] Reset  |  [Q] Quit"],
            )

            # D. Log per-frame telemetry to CSV
            if metrics_logger is not None:
                metrics_logger.log_tracking(
                    frame_index=f_idx,
                    tracker_name=tracker_name_str,
                    success=tracking_result.success,
                    bbox=tracking_result.bbox,
                    center=tracking_result.center,
                    step_distance=telemetry.step_distance,
                    cumulative_distance=telemetry.cumulative_distance,
                    fps=telemetry.fps,
                    confidence=tracking_result.confidence,
                    track_id=tracking_result.track_id,
                    class_name=tracking_result.class_name,
                    timestamp=t_stamp,
                )

            # E. Write frame to output video
            if video_writer is not None:
                video_writer.write(annotated_frame)

            # F. Automated test snapshot trigger
            if args.test_snapshot_at_frame is not None and f_idx == args.test_snapshot_at_frame:
                logger.info("Triggering automated snapshot at frame %d", f_idx)
                snapshot_ctx = build_snapshot_context(
                    current_packet, tracking_result, telemetry, tracker_name_str
                )
                run_inspection_modal(
                    context=snapshot_ctx,
                    tracking_result=tracking_result,
                    dip_pipeline=dip_pipeline,
                    canvas_engine=canvas_engine,
                    metrics_logger=metrics_logger,
                    snapshots_dir=snapshots_dir,
                    headless=args.headless,
                )

            # G. Display live frame and handle user input
            if not args.headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q / ESC: Quit
                    logger.info("User requested exit via keyboard.")
                    break
                elif key == 32:  # SPACE: Modal DIP Inspection!
                    logger.info("SPACE pressed: Pausing tracking and entering DIP inspection modal.")
                    snapshot_ctx = build_snapshot_context(
                        current_packet, tracking_result, telemetry, tracker_name_str
                    )
                    run_inspection_modal(
                        context=snapshot_ctx,
                        tracking_result=tracking_result,
                        dip_pipeline=dip_pipeline,
                        canvas_engine=canvas_engine,
                        metrics_logger=metrics_logger,
                        snapshots_dir=snapshots_dir,
                        headless=False,
                    )
                elif key in (ord("r"), ord("R")):  # R: Reset
                    logger.info("Resetting tracker and telemetry.")
                    tracker.reset()
                    telemetry_tracker.reset()
                    if args.mode == "classical":
                        sel = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
                        if sel != (0, 0, 0, 0):
                            new_box = (int(sel[0]), int(sel[1]), int(sel[2]), int(sel[3]))
                            tracker.init(frame, new_box)
                            telemetry_tracker.set_initial_center((new_box[0] + new_box[2] // 2, new_box[1] + new_box[3] // 2))
                elif key in (ord("c"), ord("C")):  # C: Toggle mirror
                    video_source.mirror = not video_source.mirror
                    logger.info("Toggled mirror mode: %s", "ENABLED" if video_source.mirror else "DISABLED")
                elif key in (ord("p"), ord("P")):  # P: Toggle preprocessing
                    args.preprocess = not args.preprocess
                    logger.info("Toggled DIP preprocessing: %s", "ENABLED" if args.preprocess else "DISABLED")

            frame_count += 1
            ok, current_packet = video_source.read_frame()
            if not ok or current_packet is None:
                logger.info("End of video stream reached.")
                break

    finally:
        # 6. Teardown and Resource Cleanup
        logger.info("Cleaning up session resources...")
        video_source.release()
        if video_writer is not None:
            video_writer.release()
            logger.info("Saved annotated tracking video to: %s", output_video_path)
        if not args.headless:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_loop_start
    avg_session_fps = (frame_count / total_time) if total_time > 0 else 0.0
    final_dist = telemetry_tracker.get_cumulative_distance()
    unit_str = "m" if calib_cfg.enabled else "px"

    logger.info("Session complete: %d frames in %.2f s (%.1f avg FPS), path=%.2f %s",
                frame_count, total_time, avg_session_fps, final_dist, unit_str)

    # 7. Post-Run Analytics Plots
    if args.plot and not args.no_save:
        logger.info("Generating publication-quality post-run analytics plots in: %s", args.plot_dir)
        try:
            plotter = PerformancePlotter(output_dir=args.plot_dir)
            t_rows = plotter.load_csv(tracking_csv_path)
            s_rows = plotter.load_csv(snapshot_csv_path) if snapshot_csv_path.exists() else []

            stats = plotter.compute_statistics(t_rows, s_rows)
            if stats:
                plotter.print_summary(stats)
            plot_files = plotter.generate_all_plots(t_rows, s_rows)
            logger.info("Generated %d analytics plots successfully.", len(plot_files))
        except Exception as err:
            logger.error("Failed to generate analytics plots: %s", err, exc_info=True)

    return 0


def main() -> None:
    """Central entry point."""
    args = parse_args()
    sys.exit(run_tracking_app(args))


if __name__ == "__main__":
    main()
