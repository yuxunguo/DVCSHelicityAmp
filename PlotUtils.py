"""Shared matplotlib and plotting helpers reused across scan scripts.

Provides a single headless matplotlib loader and shared reference-line drawing.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


POLAR_ANGLE_TICKS = np.asarray(
    (0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi)
)
POLAR_ANGLE_TICK_LABELS = (
    r"$0$",
    r"$\pi/4$",
    r"$\pi/2$",
    r"$3\pi/4$",
    r"$\pi$",
)
AZIMUTHAL_ANGLE_TICKS = np.asarray(
    (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi)
)
AZIMUTHAL_ANGLE_TICK_LABELS = (
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
    r"$3\pi/2$",
    r"$2\pi$",
)
ANGLE_EDGE_PADDING_FRACTION = 0.025


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


def _configure_angle_axis(ax, axis, limits, ticks, labels):
    """Apply angular ticks and grid with a small marker-safe edge margin."""
    span = limits[1] - limits[0]
    padding = ANGLE_EDGE_PADDING_FRACTION * span
    padded_limits = (limits[0] - padding, limits[1] + padding)
    if axis == "x":
        ax.set_xlim(*padded_limits)
        ax.set_xticks(ticks, labels)
    elif axis == "y":
        ax.set_ylim(*padded_limits)
        ax.set_yticks(ticks, labels)
    else:
        raise ValueError("axis must be 'x' or 'y'.")
    ax.grid(
        visible=True,
        which="major",
        axis=axis,
        color="0.55",
        linestyle="--",
        linewidth=0.6,
        alpha=0.65,
    )


def configure_polar_angle_axis(ax, axis="x"):
    """Format a polar-angle axis from zero through pi."""
    _configure_angle_axis(
        ax,
        axis,
        (0.0, np.pi),
        POLAR_ANGLE_TICKS,
        POLAR_ANGLE_TICK_LABELS,
    )


def configure_azimuthal_angle_axis(ax, axis="x"):
    """Format an azimuthal-angle axis from zero through 2 pi."""
    _configure_angle_axis(
        ax,
        axis,
        (0.0, 2.0 * np.pi),
        AZIMUTHAL_ANGLE_TICKS,
        AZIMUTHAL_ANGLE_TICK_LABELS,
    )


def configure_named_angle_axes(ax, x_name=None, y_name=None):
    """Format polar, azimuthal, and spin-mixing angle axes."""
    for axis, name in (("x", x_name), ("y", y_name)):
        normalized = str(name or "").lower()
        if (
            normalized.startswith("theta")
            or "_theta" in normalized
            or normalized.startswith("alpha")
            or "_alpha" in normalized
        ):
            configure_polar_angle_axis(ax, axis)
        elif normalized.startswith("phi") or "_phi" in normalized:
            configure_azimuthal_angle_axis(ax, axis)
