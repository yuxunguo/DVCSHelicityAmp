"""Shared matplotlib and plotting helpers reused across scan scripts.

Provides a single headless matplotlib loader and shared reference-line drawing.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from config import PROTON_MASS_GEV


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
MAJOR_GRID_STYLE = {
    "color": "0.52",
    "linestyle": "-",
    "linewidth": 0.7,
    "alpha": 0.72,
}
MINOR_GRID_STYLE = {
    "color": "0.62",
    "linestyle": ":",
    "linewidth": 0.65,
    "alpha": 0.72,
}
ANGLE_GRID_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 0.75,
    "alpha": 0.62,
}
PHOTON_CEILING_STYLE = {
    "color": "black",
    "linestyle": "-",
    "linewidth": 1.6,
    "alpha": 0.95,
    "zorder": 12,
}


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
    configure_visible_grid(ax)
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
        **ANGLE_GRID_STYLE,
    )


def configure_visible_grid(ax):
    """Draw consistent major and clearly visible minor grid lines."""
    ax.minorticks_on()
    ax.grid(visible=True, which="major", axis="both", **MAJOR_GRID_STYLE)
    ax.grid(visible=True, which="minor", axis="both", **MINOR_GRID_STYLE)


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
    configure_visible_grid(ax)
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


def photon_energy_maximum(sqrt_s, lepton_mass):
    """Return the three-body CM ceiling for the real-photon energy.

    The maximum occurs when the final proton--lepton system has its minimum
    invariant mass, ``m_p + m_l``.
    """
    sqrt_s = np.asarray(sqrt_s, dtype=float)
    lepton_mass = float(lepton_mass)
    threshold_mass = PROTON_MASS_GEV + lepton_mass
    with np.errstate(divide="ignore", invalid="ignore"):
        maximum = (sqrt_s**2 - threshold_mass**2) / (2.0 * sqrt_s)
    return np.where(sqrt_s >= threshold_mass, maximum, np.nan)


def draw_photon_energy_ceiling(ax, x_name, y_name, lepton_mass):
    """Overlay the physical ``E_gamma^max(sqrt(s))`` boundary when relevant."""
    if {x_name, y_name} != {"sqrt_s", "qOut"}:
        return None

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    sqrt_limits = x_limits if x_name == "sqrt_s" else y_limits
    lower = max(float(min(sqrt_limits)), PROTON_MASS_GEV + float(lepton_mass))
    upper = float(max(sqrt_limits))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None

    sqrt_s = np.linspace(lower, upper, 256)
    e_gamma_max = photon_energy_maximum(sqrt_s, lepton_mass)
    if x_name == "sqrt_s":
        line, = ax.plot(
            sqrt_s,
            e_gamma_max,
            label=r"$E_\gamma^{\max}(\sqrt{s})$",
            **PHOTON_CEILING_STYLE,
        )
    else:
        line, = ax.plot(
            e_gamma_max,
            sqrt_s,
            label=r"$E_\gamma^{\max}(\sqrt{s})$",
            **PHOTON_CEILING_STYLE,
        )
    # The guide must describe the already displayed phase space, not expand it.
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.legend(
        handles=[line],
        loc="upper left",
        borderaxespad=0.35,
        fontsize=8,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.55",
        borderpad=0.25,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    return line


def configure_phase_space_axes(
    ax,
    x_name=None,
    y_name=None,
    *,
    lepton_mass=None,
):
    """Apply shared grids, angle guides, and the photon-energy boundary."""
    configure_named_angle_axes(ax, x_name, y_name)
    if lepton_mass is not None:
        draw_photon_energy_ceiling(ax, x_name, y_name, lepton_mass)
