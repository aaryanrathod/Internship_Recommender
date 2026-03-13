"""
Root conftest for the Internship Recommendation Engine test suite.

Ensures the project root is on ``sys.path`` so that ``engine.*``,
``api.*``, and ``config`` imports resolve correctly regardless of
how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
