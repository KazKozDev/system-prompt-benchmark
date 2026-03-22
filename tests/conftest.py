"""Shared pytest configuration."""

from __future__ import annotations

# Ensure the project root is on sys.path so src.* imports work without installing the package.
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
