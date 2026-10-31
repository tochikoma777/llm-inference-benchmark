"""Reporting: analysis, HTML dashboard export and Markdown summaries."""

from __future__ import annotations

from .analysis import Analysis
from .html_report import build_html_report, report_filename
from .markdown import build_markdown_report, dataframe_to_markdown

__all__ = [
    "Analysis",
    "build_html_report",
    "report_filename",
    "build_markdown_report",
    "dataframe_to_markdown",
]
