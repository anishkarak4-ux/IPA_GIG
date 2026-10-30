"""Analytics module providing telemetry persistence, report generation, and performance plotting."""

from .metrics_logger import (
    SNAPSHOT_CSV_HEADER,
    TRACKING_CSV_HEADER,
    MetricsLogger,
)
from .plot_metrics import (
    AnalyticsPlotter,
    PerformancePlotter,
)
from .report_generator import (
    HTMLReportGenerator,
    image_to_base64_data_uri,
)

__all__ = [
    "MetricsLogger",
    "TRACKING_CSV_HEADER",
    "SNAPSHOT_CSV_HEADER",
    "PerformancePlotter",
    "AnalyticsPlotter",
    "HTMLReportGenerator",
    "image_to_base64_data_uri",
]
