"""Local batch speech emotion analysis application."""

from .analytics import DistanceMetric, PairingMode, compare_runs
from .core import AnalysisConfig, BatchAnalyzer
from .reporting import read_run_report, write_comparison_report

__all__ = [
    "AnalysisConfig",
    "BatchAnalyzer",
    "DistanceMetric",
    "PairingMode",
    "compare_runs",
    "read_run_report",
    "write_comparison_report",
]
