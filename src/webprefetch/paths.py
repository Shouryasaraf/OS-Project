"""Default locations, relative to the repository root.

The disk study owns ``outputs/`` and ``models/`` at the root (``main.py`` also
writes ``outputs/eda.md``), so the web study keeps its own subdirectories and
the two never overwrite each other.
"""

from __future__ import annotations

from pathlib import Path

RAW = Path("data/raw/events.csv")
PROCESSED = Path("data/processed")
MODELS = Path("models/web")
OUTPUTS = Path("outputs/web")
# The committed outputs of the original pivot run, kept verbatim.
PIVOT_ORIGINAL = Path("results/pivot_original")
