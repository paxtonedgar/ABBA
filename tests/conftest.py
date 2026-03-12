"""Shared pytest bootstrap for the repository test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Allow `pytest` to run from the repository root without requiring callers to
# set PYTHONPATH manually.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
