"""Shared matplotlib and plotting helpers reused across scan scripts.

Provides a single headless matplotlib loader and shared reference-line drawing.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def print_console_text(text):
    """Print text after replacing characters unsupported by the console."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = str(text).encode(encoding, errors="replace").decode(encoding)
    print(safe_text, end="")


def require_matplotlib():
    """Import matplotlib in headless mode with a writable cache directory."""
    cache_dir = Path(tempfile.gettempdir()) / "dvcs_helicity_amp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    return plt, PdfPages


def add_pi_over_two_reference_lines(ax):
    """Draw pi/2 reference lines on scan maps."""
    ax.axvline(0.5 * np.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)
    ax.axhline(0.5 * np.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)
