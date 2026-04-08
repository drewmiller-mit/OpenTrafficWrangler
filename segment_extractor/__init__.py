"""
Segment extraction toolkit.

This package exposes reusable components extracted from the legacy
`unified_get_segment.py` script so other entry points (CLI, API, notebooks)
can orchestrate the same workflow without copy/pasting helpers.
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
