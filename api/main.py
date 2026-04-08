"""
Uvicorn entrypoint: ``uvicorn api.main:app --reload``.
"""
from __future__ import annotations

from .app import create_app

app = create_app()
