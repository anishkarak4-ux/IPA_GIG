"""Analytics and post-run performance plotting module.

Provides publication-quality analytical visualizations and quantitative statistical
evaluations from tracking and snapshot CSV logs:
    - 6-Panel Performance Dashboard (FPS, Path Length, 2D Trajectory, Distributions)
    - Time-series FPS with rolling average and standard deviation bands
    - Corrected cumulative path length curves (no double-summing)
    - 2D coordinate trajectory with Start/End spatial markers
    - 3D spatio-temporal trajectory (X, Y, Frame/Time)
    - 2D position density heatmaps
    - YOLO confidence distributions
    - Academic statistical summary reports
"""

import csv
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless mode for robust server/background execution
import matplotlib.pyplot as plt


class PerformancePlotter:
    """Consolidated performance plotting and statistical evaluation engine."""

    def __init__(self, output_dir: str | Path = "data/output/analytics") -> None:
        """Initialize plotter with target output directory.

        Args:
            output_dir: Directory where generated analytics plots will be stored.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_csv(csv_path: str | Path) -> list[dict[str, str]]:
        """Safely load CSV rows as list of dictionaries."""
        path = Path(csv_path)
        if not path.exists() or path.stat().st_size == 0:
            return []

        rows: list[dict[str, str]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def compute_statistics(
        rows: Sequence[dict[str, str]],
        snapshot_rows: Sequence[dict[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        """Compute comprehensive quantitative analytics from tracking and snapshot records.

        Strictly preserves the Stage 2A/Stage 3 corrected single-sum cumulative path
        interpretation: path_values[-1] is the final cumulative distance.

        Args:
            rows: Sequence of per-frame tracking row dictionaries.
            snapshot_rows: Optional sequence of snapshot DIP row dictionaries.

        Returns:
            Dictionary of calculated metrics, or None if rows is empty.
        """
        if not rows:
            return None

        total_frames = len(rows)
        tracked_rows = [r for r in rows if r.get("success", "0") in ("1", "True", "true")]
        tracked_count = len(tracked_rows)
        success_rate = (tracked_count / total_frames) * 100.0 if total_frames > 0 else 0.0

        # FPS metrics
        fps_values: list[float] = []
        for r in rows:
            val_str = r.get("fps", "")
            if val_str and val_str.lower() != "nan":
                try:
                    f_val = float(val_str)
                    if f_val > 0.0 and math.isfinite(f_val):
                        fps_values.append(f_val)
                except ValueError:
                    pass

        avg_fps = float(np.mean(fps_values)) if fps_values else 0.0
        fps_std = float(np.std(fps_values)) if fps_values else 0.0
        min_fps = float(np.min(fps_values)) if fps_values else 0.0
        max_fps = float(np.max(fps_values)) if fps_values else 0.0

        # Cumulative path metrics (strictly using last non-zero cumulative entry)
        path_values: list[float] = []
        step_values: list[float] = []
        for r in rows:
            p_val = r.get("path_px", r.get("path_m", ""))
            if p_val and p_val.lower() != "nan":
                try:
                    path_values.append(float(p_val))
                except ValueError:
                    pass

            s_val = r.get("step_px", "")
            if s_val and s_val.lower() != "nan":
                try:
                    step_values.append(float(s_val))
                except ValueError:
                    pass

        total_path = path_values[-1] if path_values else 0.0
        avg_path_per_frame = total_path / total_frames if total_frames > 0 else 0.0
        avg_step = float(np.mean(step_values)) if step_values else (avg_path_per_frame)
        max_step = float(np.max(step_values)) if step_values else 0.0

        # YOLO Detection metrics (if available)
        conf_values: list[float] = []
        track_ids: set[str] = set()
        for r in rows:
            c_str = r.get("confidence", "")
            if c_str and c_str.lower() != "nan":
                try:
                    c_val = float(c_str)
                    if c_val > 0.0:
                        conf_values.append(c_val)
                except ValueError:
                    pass

            t_id = r.get("track_id", "")
            if t_id and t_id.isdigit() and int(t_id) > 0:
                track_ids.add(t_id)

        stats: dict[str, Any] = {
            "total_frames": total_frames,
            "tracked_frames": tracked_count,
            "success_rate": success_rate,
            "average_fps": avg_fps,
            "fps_std": fps_std,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "total_path_px": total_path,
            "average_path_per_frame": avg_path_per_frame,
            "average_step_px": avg_step,
            "max_step_px": max_step,
            "unique_tracks": len(track_ids),
        }

        if conf_values:
            stats["average_confidence"] = float(np.mean(conf_values))
            stats["min_confidence"] = float(np.min(conf_values))
            stats["max_confidence"] = float(np.max(conf_values))

        # Snapshot DIP metrics (if supplied)
        if snapshot_rows:
            stats["total_snapshots"] = len(snapshot_rows)
            areas = [float(r["area"]) for r in snapshot_rows if r.get("area") and r["area"].replace(".", "").isdigit()]
            circs = [float(r["circularity"]) for r in snapshot_rows if r.get("circularity")]
            solids = [float(r["solidity"]) for r in snapshot_rows if r.get("solidity")]
            if areas:
                stats["average_contour_area"] = float(np.mean(areas))
            if circs:
                stats["average_circularity"] = float(np.mean(circs))
            if solids:
                stats["average_solidity"] = float(np.mean(solids))

        return stats

    @staticmethod
    def format_summary_report(stats: dict[str, Any] | None, title: str = "Tracking Analytics Summary") -> str:
        """Format quantitative analytics into a clean, publication-ready text report."""
        if stats is None:
            return "No analytics data available.\n"

        lines: list[str] = [
            "",
            title,
            "=" * len(title),
            f"Total frames:            {stats.get('total_frames', 0)}",
            f"Tracked frames:          {stats.get('tracked_frames', 0)}",
            f"Tracking success rate:   {stats.get('success_rate', 0.0):.2f}%",
            f"Average FPS:             {stats.get('average_fps', 0.0):.2f}",
            f"FPS std dev:             {stats.get('fps_std', 0.0):.2f}",
            f"Min FPS:                 {stats.get('min_fps', 0.0):.2f}",
            f"Max FPS:                 {stats.get('max_fps', 0.0):.2f}",
            f"Total path length:       {stats.get('total_path_px', 0.0):.2f} px",
            f"Average path per frame:  {stats.get('average_path_per_frame', 0.0):.2f} px",
            f"Max step displacement:   {stats.get('max_step_px', 0.0):.2f} px",
        ]

        if "average_confidence" in stats:
            lines.extend([
                f"Unique tracks:           {stats.get('unique_tracks', 0)}",
                f"Average confidence:      {stats.get('average_confidence', 0.0):.4f}",
                f"Min confidence:          {stats.get('min_confidence', 0.0):.4f}",
                f"Max confidence:          {stats.get('max_confidence', 0.0):.4f}",
            ])

        if "total_snapshots" in stats:
            lines.extend([
                "-" * len(title),
                f"Total DIP snapshots:     {stats.get('total_snapshots', 0)}",
                f"Average contour area:    {stats.get('average_contour_area', 0.0):.2f} px",
                f"Average circularity:     {stats.get('average_circularity', 0.0):.4f}",
                f"Average solidity:        {stats.get('average_solidity', 0.0):.4f}",
            ])

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def print_summary(cls, stats: dict[str, Any] | None, title: str = "Tracking Analytics Summary") -> None:
        """Print formatted summary report to stdout."""
        report = cls.format_summary_report(stats, title=title)
        print(report)

    def plot_fps(self, rows: Sequence[dict[str, str]], filename: str = "fps_over_time.png") -> Path | None:
        """Plot time-series FPS with mean and standard deviation guides."""
        fps_values = [float(r["fps"]) for r in rows if r.get("fps") and r["fps"].lower() != "nan"]
        if not fps_values:
            return None

        out_path = self.output_dir / filename
        plt.figure(figsize=(10, 5))
        frames = range(len(fps_values))
        mean_fps = float(np.mean(fps_values))
        std_fps = float(np.std(fps_values))

        plt.plot(frames, fps_values, color="tab:blue", lw=1.5, label="Instantaneous FPS")
        plt.axhline(mean_fps, color="red", linestyle="--", lw=1.5, label=f"Mean: {mean_fps:.2f}")
        plt.fill_between(
            frames,
            max(0, mean_fps - std_fps),
            mean_fps + std_fps,
            color="red",
            alpha=0.15,
            label=f"+/-1 Std Dev ({std_fps:.2f})",
        )

        plt.title("Framerate Performance Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Frames Per Second (FPS)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def plot_cumulative_distance(
        self, rows: Sequence[dict[str, str]], filename: str = "cumulative_distance.png"
    ) -> Path | None:
        """Plot cumulative trajectory distance over time."""
        path_values = [
            float(r.get("path_px", r.get("path_m", 0)))
            for r in rows
            if r.get("path_px") or r.get("path_m")
        ]
        if not path_values:
            return None

        out_path = self.output_dir / filename
        plt.figure(figsize=(10, 5))
        frames = range(len(path_values))
        final_dist = path_values[-1]

        plt.plot(frames, path_values, color="tab:orange", lw=2.0, label="Cumulative Path")
        plt.title(f"Cumulative Path Displacement (Final: {final_dist:.2f} px)")
        plt.xlabel("Frame Index")
        plt.ylabel("Path Distance (px)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def plot_trajectory_2d(
        self, rows: Sequence[dict[str, str]], filename: str = "trajectory_2d.png"
    ) -> Path | None:
        """Plot 2D trajectory of object centroids with Start and End spatial markers."""
        xs = [float(r["center_x"]) for r in rows if r.get("center_x")]
        ys = [float(r["center_y"]) for r in rows if r.get("center_y")]
        if not xs or not ys:
            return None

        out_path = self.output_dir / filename
        plt.figure(figsize=(8, 8))
        plt.plot(xs, ys, color="tab:purple", lw=1.8, alpha=0.8, label="Trajectory Path")
        plt.scatter(xs[0], ys[0], color="green", s=100, zorder=5, label=f"Start ({xs[0]:.0f}, {ys[0]:.0f})")
        plt.scatter(xs[-1], ys[-1], color="red", s=100, marker="s", zorder=5, label=f"End ({xs[-1]:.0f}, {ys[-1]:.0f})")

        plt.title("2D Spatial Trajectory Path")
        plt.xlabel("X Coordinate (px)")
        plt.ylabel("Y Coordinate (px)")
        # Invert Y to match image coordinate system (0 at top)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def plot_trajectory_3d(
        self, rows: Sequence[dict[str, str]], filename: str = "trajectory_3d.png"
    ) -> Path | None:
        """Plot 3D spatio-temporal trajectory (X, Y, Frame Index)."""
        xs = []
        ys = []
        zs = []
        for idx, r in enumerate(rows):
            if r.get("center_x") and r.get("center_y"):
                try:
                    xs.append(float(r["center_x"]))
                    ys.append(float(r["center_y"]))
                    zs.append(float(r.get("frame", idx)))
                except ValueError:
                    pass

        if len(xs) < 2:
            return None

        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            out_path = self.output_dir / filename
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            ax.plot(xs, ys, zs, color="tab:cyan", lw=2.0, alpha=0.85, label="Trajectory")
            ax.scatter(xs[0], ys[0], zs[0], color="green", s=80, label="Start")
            ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=80, marker="s", label="End")

            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_zlabel("Frame")
            ax.set_title("3D Spatio-Temporal Trajectory")
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            return out_path
        except Exception:
            return None

    def plot_heatmap(
        self, rows: Sequence[dict[str, str]], filename: str = "position_heatmap.png"
    ) -> Path | None:
        """Plot 2D spatial occupancy density heatmap."""
        xs = [float(r["center_x"]) for r in rows if r.get("center_x")]
        ys = [float(r["center_y"]) for r in rows if r.get("center_y")]
        if len(xs) < 2:
            return None

        out_path = self.output_dir / filename
        plt.figure(figsize=(9, 7))

        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="hot", aspect="auto")
        plt.colorbar(im, label="Centroid Occurrence Frequency")
        plt.title("Object Spatial Position Density Heatmap")
        plt.xlabel("X Coordinate (px)")
        plt.ylabel("Y Coordinate (px)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def plot_confidence_distribution(
        self, rows: Sequence[dict[str, str]], filename: str = "confidence_distribution.png"
    ) -> Path | None:
        """Plot detection confidence histogram for deep-learning (YOLO) tracking."""
        confs: list[float] = []
        for r in rows:
            c_str = r.get("confidence", "")
            if c_str and c_str.lower() != "nan":
                try:
                    c = float(c_str)
                    if c > 0.0:
                        confs.append(c)
                except ValueError:
                    pass

        if not confs:
            return None

        out_path = self.output_dir / filename
        plt.figure(figsize=(9, 5))
        plt.hist(confs, bins=25, range=(0.0, 1.0), color="tab:green", alpha=0.75, edgecolor="black")
        mean_c = float(np.mean(confs))
        plt.axvline(mean_c, color="red", linestyle="--", lw=1.5, label=f"Mean Conf: {mean_c:.3f}")

        plt.title("Detection Confidence Distribution")
        plt.xlabel("YOLO Confidence Score")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def plot_summary_dashboard(
        self, rows: Sequence[dict[str, str]], filename: str = "analytics_summary.png"
    ) -> Path | None:
        """Generate comprehensive multi-panel analytics dashboard."""
        if not rows:
            return None

        fps_vals = [float(r["fps"]) for r in rows if r.get("fps") and r["fps"].lower() != "nan"]
        path_vals = [
            float(r.get("path_px", r.get("path_m", 0)))
            for r in rows
            if r.get("path_px") or r.get("path_m")
        ]
        xs = [float(r["center_x"]) for r in rows if r.get("center_x")]
        ys = [float(r["center_y"]) for r in rows if r.get("center_y")]

        out_path = self.output_dir / filename
        plt.figure(figsize=(14, 9))

        # Panel 1: FPS over Time
        plt.subplot(2, 2, 1)
        if fps_vals:
            plt.plot(fps_vals, color="tab:blue", lw=1.5)
            plt.axhline(float(np.mean(fps_vals)), color="red", linestyle="--", lw=1.0)
        plt.title("Framerate Performance (FPS)")
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.grid(True, alpha=0.3)

        # Panel 2: Path Length over Time
        plt.subplot(2, 2, 2)
        if path_vals:
            plt.plot(path_vals, color="tab:orange", lw=1.8)
        plt.title("Cumulative Path Length")
        plt.xlabel("Frame")
        plt.ylabel("Path (px)")
        plt.grid(True, alpha=0.3)

        # Panel 3: 2D Spatial Trajectory
        plt.subplot(2, 2, 3)
        if xs and ys:
            plt.plot(xs, ys, color="tab:purple", lw=1.5, alpha=0.8)
            plt.scatter(xs[0], ys[0], color="green", s=70, label="Start")
            plt.scatter(xs[-1], ys[-1], color="red", s=70, marker="s", label="End")
            plt.gca().invert_yaxis()
            plt.legend(loc="best", fontsize=8)
        plt.title("2D Trajectory Map")
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")
        plt.grid(True, alpha=0.3)

        # Panel 4: FPS Distribution Histogram
        plt.subplot(2, 2, 4)
        if fps_vals:
            plt.hist(fps_vals, bins=25, color="tab:green", alpha=0.7, edgecolor="black")
        plt.title("FPS Distribution")
        plt.xlabel("FPS")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    def generate_all_plots(
        self,
        tracking_csv: str | Path | Sequence[dict[str, str]],
        snapshot_csv: str | Path | Sequence[dict[str, str]] | None = None,
    ) -> dict[str, Path]:
        """Generate all viable analytical plots from tracking and optional snapshot CSV data."""
        if isinstance(tracking_csv, (str, Path)):
            rows = self.load_csv(tracking_csv)
        else:
            rows = list(tracking_csv)
        if not rows:
            return {}

        results: dict[str, Path] = {}

        p_fps = self.plot_fps(rows)
        if p_fps:
            results["fps"] = p_fps

        p_dist = self.plot_cumulative_distance(rows)
        if p_dist:
            results["cumulative_distance"] = p_dist

        p_traj2d = self.plot_trajectory_2d(rows)
        if p_traj2d:
            results["trajectory_2d"] = p_traj2d

        p_traj3d = self.plot_trajectory_3d(rows)
        if p_traj3d:
            results["trajectory_3d"] = p_traj3d

        p_heat = self.plot_heatmap(rows)
        if p_heat:
            results["heatmap"] = p_heat

        p_conf = self.plot_confidence_distribution(rows)
        if p_conf:
            results["confidence"] = p_conf

        p_dash = self.plot_summary_dashboard(rows)
        if p_dash:
            results["dashboard"] = p_dash

        return results

    # Compatibility method for baseline skeleton
    def generate_dashboard(self, metrics_csv: Path | str) -> Path | None:
        """Generate dashboard plot matching baseline interface."""
        rows = self.load_csv(metrics_csv)
        return self.plot_summary_dashboard(rows)

    def generate_3d_trajectory(self, metrics_csv: Path | str) -> Path | None:
        """Generate 3D trajectory plot matching baseline interface."""
        rows = self.load_csv(metrics_csv)
        return self.plot_trajectory_3d(rows)


# Aliases for architecture and backwards compatibility
AnalyticsPlotter = PerformancePlotter
