"""Shared matplotlib and plotting helpers reused across scan scripts.

Provides a single ``_require_matplotlib`` implementation, common 2D-binning
utilities, and reference-line drawing so that every scan module does not
need its own copy.
"""

import os
import tempfile
from pathlib import Path

import numpy as np


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


def bin_edges_from_values(values, max_bins=96):
    """Return plotting bin edges adapted to discrete or continuous values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0, 1.0])

    unique = np.unique(values)
    if unique.size == 1:
        width = max(1.0e-6, abs(unique[0]) * 1.0e-6)
        return np.asarray([unique[0] - width, unique[0] + width])
    if unique.size <= max_bins:
        midpoints = 0.5 * (unique[:-1] + unique[1:])
        first = unique[0] - 0.5 * (unique[1] - unique[0])
        last = unique[-1] + 0.5 * (unique[-1] - unique[-2])
        return np.concatenate([[first], midpoints, [last]])
    return np.linspace(values.min(), values.max(), max_bins + 1)


def binned_mean_2d(x_values, y_values, z_values, x_edges, y_edges):
    """Return a masked 2D binned mean ``z`` on ``x``/``y`` bins."""
    finite = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & np.isfinite(z_values)
    )
    counts, _x_edges, _y_edges = np.histogram2d(
        x_values[finite], y_values[finite], bins=(x_edges, y_edges),
    )
    sums, _x_edges, _y_edges = np.histogram2d(
        x_values[finite], y_values[finite], bins=(x_edges, y_edges),
        weights=z_values[finite],
    )
    mean = np.full_like(sums, np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return np.ma.masked_invalid(mean.T)


def add_pi_over_two_reference_lines(ax):
    """Draw pi/2 reference lines on scan maps."""
    ax.axvline(0.5 * np.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)
    ax.axhline(0.5 * np.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)
