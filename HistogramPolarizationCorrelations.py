"""2D-histogram versions of the unclustered, without-contours
polarization-correlation panels, built from retained local minima.

This is a standalone reference tool: it reads the already-saved
``Data/<state>/cluster/clustered_minima.csv`` for each requested lepton and
state, pools the rows that pass ``within_polarization_cluster_cut`` (the same
"unclustered" selection used by the scatter-based correlation panels), and
renders the same nine axis-pair panels plus a 3x3 summary page as 2D
histograms instead of scatter points.  It also writes an aligned lower-
triangular 8x8 summary containing all 28 unordered pairs of the eight smearing
coordinates and their eight one-dimensional marginal distributions on the
diagonal.  No contours and no per-cluster coloring are used.

Two versions are written side by side:

``minima_only``
    Histogram of the retained local minima alone.
``smeared``
    Histogram of the retained minima pooled with their K-per-ray verified
    interior smear samples from ``GenerateSmearSamples.py``
    (``Data/<state>/cluster/smear_points_k<K>.csv``), showing the real width
    of each minimum's near-optimal region rather than just its point
    location.
Output is written under
``Output/GradientPhaseSpaceScan/<lepton>/Plots/<state>/
polarization_correlations_histogram/{minima_only,smeared}/``.

Run with

    python HistogramPolarizationCorrelations.py
"""

import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from PlotUtils import configure_phase_space_axes, require_matplotlib

# Mirrors GradientPhaseSpaceScanTool.POLARIZATION_CORRELATION_PANELS.
POLARIZATION_CORRELATION_PANELS = (
    ("sqrt_s", "qOut", r"$\sqrt{s}$ [GeV]", r"$E_\gamma$ [GeV]"),
    ("sqrt_s", "theta_gamma_out", r"$\sqrt{s}$ [GeV]", r"$\theta_\gamma$"),
    ("theta_p_out", "theta_gamma_out", r"$\theta_{p'}$", r"$\theta_\gamma$"),
    ("theta_p_out", "qOut", r"$\theta_{p'}$", r"$E_\gamma$ [GeV]"),
    ("theta_gamma_out", "qOut", r"$\theta_\gamma$", r"$E_\gamma$ [GeV]"),
    ("phi_p_out", "phi_gamma_out", r"$\phi_{p'}$", r"$\phi_\gamma$"),
    ("alpha_e", "alpha_p", r"$\alpha_e$", r"$\alpha_p$"),
    ("sqrt_s", "alpha_e", r"$\sqrt{s}$ [GeV]", r"$\alpha_e$"),
    ("sqrt_s", "alpha_p", r"$\sqrt{s}$ [GeV]", r"$\alpha_p$"),
)

# The eight Gaussian-smearing coordinates, in the same order used to build the
# covariance matrix.  Taking every unordered pair gives all C(8, 2) = 28
# two-dimensional projections.
PHASE_SPACE_COORDINATES = (
    ("sqrt_s", r"$\sqrt{s}$ [GeV]"),
    ("theta_p_out", r"$\theta_{p'}$"),
    ("theta_gamma_out", r"$\theta_\gamma$"),
    ("qOut", r"$E_\gamma$ [GeV]"),
    ("phi_p_out", r"$\phi_{p'}$"),
    ("phi_gamma_out", r"$\phi_\gamma$"),
    ("alpha_e", r"$\alpha_e$"),
    ("alpha_p", r"$\alpha_p$"),
)
FULL_CORRELATION_PANELS = tuple(
    (x_name, y_name, x_label, y_label)
    for (x_name, x_label), (y_name, y_label)
    in combinations(PHASE_SPACE_COORDINATES, 2)
)
assert len(FULL_CORRELATION_PANELS) == 28

LEPTON_MASSES_GEV = {
    "electron": 0.00051099895,
    "muon": 0.1056583755,
}
STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTONS = ("electron",)
SMEAR_K = 2
MIN_BINS = 20
MAX_BINS = 50
GAUSSIAN_BINS = 120
GAUSSIAN_COLORS = (
    "#fffde7",  # smallest nonzero probability: pale cream
    "#f6e65b",
    "#f89540",
    "#e84a5f",
    "#b12a90",
    "#6a00a8",
    "#240046",  # highest probability: deep violet
)
GAUSSIAN_CMAP = LinearSegmentedColormap.from_list(
    "softened_plasma_r", GAUSSIAN_COLORS, N=256,
)
GAUSSIAN_CMAP.set_bad("#ffffff")

SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _histogram_bin_count(n_points):
    """Return a point-count-scaled bin count, clamped to a sane range."""
    return int(np.clip(round(n_points**0.5), MIN_BINS, MAX_BINS))


def _draw_histogram_panel(
    ax, plot_values, x_name, y_name, x_label, y_label, lepton_mass,
    bin_count=None, log_color=False, color_vmin=None, color_vmax=None,
    cmap_name="viridis", summary_panel=False, weights=None,
):
    """Draw one 2D histogram panel and return its QuadMesh image."""
    x_values = plot_values[x_name]
    y_values = plot_values[y_name]
    bins = (
        bin_count
        if bin_count is not None
        else _histogram_bin_count(len(x_values))
    )
    norm = (
        LogNorm(vmin=color_vmin, vmax=color_vmax)
        if log_color
        else None
    )
    _counts, _xedges, _yedges, image = ax.hist2d(
        x_values,
        y_values,
        bins=bins,
        weights=weights,
        cmap=cmap_name,
        cmin=0.0 if weights is not None else 1,
        norm=norm,
    )
    label_size = 11 if summary_panel else 13
    ax.set_xlabel(x_label, fontsize=label_size, labelpad=4.0)
    ax.set_ylabel(y_label, fontsize=label_size, labelpad=4.0)
    ax.margins(0.06)
    configure_phase_space_axes(ax, x_name, y_name, lepton_mass=lepton_mass)
    ax.tick_params(
        labelsize=10 if summary_panel else 11,
        pad=2.0,
    )
    return image


def _shared_histogram_limits(
    plot_values, bin_count, weights=None, panels=POLARIZATION_CORRELATION_PANELS,
):
    """Return common occupied-bin limits across the requested projections."""
    positive_minima = []
    maxima = []
    for x_name, y_name, _x_label, _y_label in panels:
        counts, _x_edges, _y_edges = np.histogram2d(
            plot_values[x_name],
            plot_values[y_name],
            bins=bin_count,
            weights=weights,
        )
        occupied = counts[counts > 0.0]
        if occupied.size:
            positive_minima.append(float(np.min(occupied)))
        maxima.append(float(np.max(counts)))
    if not positive_minima:
        raise ValueError("Every histogram projection is empty.")
    return min(positive_minima), max(maxima)


def _write_summary(
    plt,
    output_dir,
    state_key,
    plot_values,
    panels,
    lepton_mass,
    *,
    nrows,
    ncols,
    figsize,
    filename_stem,
    bin_count,
    log_color,
    cmap_name,
    weights,
    color_vmin,
    color_vmax,
    margins,
    colorbar_rect,
):
    """Render one aligned multi-panel summary with a shared color scale."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(**margins)
    images = []
    for ax, (x_name, y_name, x_label, y_label) in zip(axes.ravel(), panels):
        images.append(
            _draw_histogram_panel(
                ax,
                plot_values,
                x_name,
                y_name,
                x_label,
                y_label,
                lepton_mass,
                bin_count=bin_count,
                log_color=log_color,
                color_vmin=color_vmin,
                color_vmax=color_vmax,
                cmap_name=cmap_name,
                summary_panel=True,
                weights=weights,
            )
        )
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    if log_color:
        colorbar_ax = fig.add_axes(colorbar_rect)
        fig.colorbar(
            images[0],
            cax=colorbar_ax,
            label="probability per bin (shared log scale)",
        )
    pdf_path = output_dir / f"{filename_stem}_{state_key}.pdf"
    png_path = output_dir / f"{filename_stem}_{state_key}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {pdf_path}")


def _write_full_correlation_matrix(
    plt,
    output_dir,
    state_key,
    plot_values,
    lepton_mass,
    *,
    bin_count,
    log_color,
    cmap_name,
    weights,
    color_vmin,
    color_vmax,
):
    """Write an 8x8 lower-triangular correlation and marginal matrix."""
    coordinate_count = len(PHASE_SPACE_COORDINATES)
    fig, axes = plt.subplots(
        coordinate_count,
        coordinate_count,
        figsize=(19.1 if log_color else 17.7, 18.6),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.91 if log_color else 0.988,
        bottom=0.060,
        top=0.988,
        wspace=0.055,
        hspace=0.055,
    )

    marginal_bins = (
        bin_count
        if bin_count is not None
        else _histogram_bin_count(len(next(iter(plot_values.values()))))
    )
    marginal_probabilities = {}
    marginal_edges = {}
    for field_name, _label in PHASE_SPACE_COORDINATES:
        probabilities, edges = np.histogram(
            plot_values[field_name],
            bins=marginal_bins,
            weights=weights,
        )
        marginal_probabilities[field_name] = probabilities
        marginal_edges[field_name] = edges

    images = []
    for row_index, (y_name, y_label) in enumerate(PHASE_SPACE_COORDINATES):
        for column_index, (x_name, x_label) in enumerate(PHASE_SPACE_COORDINATES):
            ax = axes[row_index, column_index]
            if column_index > row_index:
                ax.set_visible(False)
                continue

            if column_index == row_index:
                probabilities = marginal_probabilities[x_name]
                edges = marginal_edges[x_name]
                ax.stairs(
                    probabilities,
                    edges,
                    fill=True,
                    color="#b12a90",
                    edgecolor="#240046",
                    linewidth=0.8,
                    alpha=0.90,
                )
                configure_phase_space_axes(ax, x_name, None)
                ax.set_ylim(0.0, 1.06 * float(np.max(probabilities)))
            else:
                images.append(
                    _draw_histogram_panel(
                        ax,
                        plot_values,
                        x_name,
                        y_name,
                        "",
                        "",
                        lepton_mass,
                        bin_count=bin_count,
                        log_color=log_color,
                        color_vmin=color_vmin,
                        color_vmax=color_vmax,
                        cmap_name=cmap_name,
                        summary_panel=True,
                        weights=weights,
                    )
                )

            is_bottom_row = row_index == coordinate_count - 1
            is_left_column = column_index == 0
            ax.set_xlabel(x_label if is_bottom_row else "", fontsize=11, labelpad=4.0)
            if is_left_column:
                diagonal_label = (
                    "probability per bin" if weights is not None else "count"
                )
                ax.set_ylabel(
                    diagonal_label if row_index == 0 else y_label,
                    fontsize=11,
                    labelpad=4.0,
                )
            else:
                ax.set_ylabel("")
            ax.tick_params(
                labelbottom=is_bottom_row,
                labelleft=is_left_column,
                labelsize=9,
                pad=1.5,
            )

    if log_color:
        colorbar_ax = fig.add_axes((0.935, 0.085, 0.014, 0.83))
        fig.colorbar(
            images[0],
            cax=colorbar_ax,
            label="probability per bin (shared log scale)",
        )

    pdf_path = output_dir / f"00_full_correlations_{state_key}.pdf"
    png_path = output_dir / f"00_full_correlations_{state_key}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {pdf_path}")


def _load_rows(mode, lepton_name, state_key):
    """Return the pooled rows to histogram for one mode, or None if unready."""
    minima_path = (
        SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
        / "clustered_minima.csv"
    )
    if not minima_path.exists():
        print(f"skip {lepton_name}/{state_key}/{mode}: missing {minima_path}")
        return None
    rows = _read_csv_rows(minima_path)
    selected_rows = [
        row for row in rows if row["within_polarization_cluster_cut"] == "True"
    ]
    if not selected_rows:
        print(f"skip {lepton_name}/{state_key}/{mode}: no retained minima")
        return None
    if mode == "minima_only":
        return selected_rows
    if mode == "smeared":
        smear_path = (
            SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
            / f"smear_points_k{SMEAR_K}.csv"
        )
        if not smear_path.exists():
            print(
                f"skip {lepton_name}/{state_key}/smeared: missing {smear_path} "
                "(run GenerateSmearSamples.py first)"
            )
            return None
        smear_rows = _read_csv_rows(smear_path)
        return selected_rows + smear_rows
    if mode == "gaussian" or mode.startswith("gaussian_cut"):
        suffix = "" if mode == "gaussian" else mode[len("gaussian"):]
        gaussian_path = (
            SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
            / f"gaussian_points{suffix}.csv"
        )
        if not gaussian_path.exists():
            print(
                f"skip {lepton_name}/{state_key}/{mode}: missing {gaussian_path} "
                "(run GenerateGaussianSamples.py first)"
            )
            return None
        gaussian_rows = _read_csv_rows(gaussian_path)
        if mode.startswith("gaussian_cut"):
            target_cut = float(mode[len("gaussian_cut"):])
            selected_rows, gaussian_rows = _filter_by_threshold(
                mode, lepton_name, state_key, rows, selected_rows, gaussian_rows,
                target_cut,
            )
        return selected_rows + gaussian_rows
    if mode.startswith("gaussian_thres"):
        thres_part, _, width_part = mode[len("gaussian_thres"):].partition("_width")
        thres = float(thres_part)
        width = float(width_part)
        width_suffix = "" if width == 0.05 else f"_cut{width:g}"
        gaussian_path = (
            SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
            / f"gaussian_points{width_suffix}.csv"
        )
        if not gaussian_path.exists():
            print(
                f"skip {lepton_name}/{state_key}/{mode}: missing {gaussian_path} "
                "(generate that width's Gaussian samples first)"
            )
            return None
        gaussian_rows = _read_csv_rows(gaussian_path)
        selected_rows, gaussian_rows = _filter_by_threshold(
            mode, lepton_name, state_key, rows, selected_rows, gaussian_rows, thres,
        )
        return selected_rows + gaussian_rows
    if mode.startswith("gaussian_widthbudget"):
        budget = mode[len("gaussian_widthbudget"):]
        gaussian_path = (
            SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
            / f"gaussian_points_widthbudget{budget}.csv"
        )
        if not gaussian_path.exists():
            print(
                f"skip {lepton_name}/{state_key}/{mode}: missing {gaussian_path} "
                "(run GenerateGaussianSamples.py with width_budget first)"
            )
            return None
        gaussian_rows = _read_csv_rows(gaussian_path)
        # Minima with no remaining budget got no Gaussian points at all;
        # keep only the discrete points that still have a matching cloud.
        kept_ids = {row["local_minimum_id"] for row in gaussian_rows}
        before = len(selected_rows)
        selected_rows = [row for row in selected_rows if row["local_minimum_id"] in kept_ids]
        print(
            f"{lepton_name}/{state_key}/{mode}: {len(selected_rows)}/{before} minima "
            f"had remaining width budget"
        )
        return selected_rows + gaussian_rows
    raise ValueError(f"Unknown mode {mode!r}.")


def _filter_by_threshold(
    mode, lepton_name, state_key, all_rows, selected_rows, gaussian_rows, threshold,
):
    """Re-filter minima and matching smear points to absolute D <= threshold."""
    above_col = next(
        name for name in all_rows[0] if name.endswith("_above_global_minimum")
    )
    objective_key = above_col.removesuffix("_above_global_minimum")
    final_col = f"final_{objective_key}"
    if final_col in all_rows[0]:
        objective_col = final_col
    elif objective_key in all_rows[0]:
        objective_col = objective_key
    else:
        objective_candidates = [
            name for name in all_rows[0]
            if name.startswith("lepton_") and name.endswith(f"_{objective_key}")
        ]
        if len(objective_candidates) != 1:
            raise KeyError(
                f"Could not identify the absolute {objective_key} column: "
                f"{objective_candidates}"
            )
        objective_col = objective_candidates[0]
    before = len(selected_rows)
    selected_rows = [
        row for row in selected_rows if float(row[objective_col]) <= threshold
    ]
    kept_ids = {row["local_minimum_id"] for row in selected_rows}
    gaussian_before = len(gaussian_rows)
    gaussian_rows = [
        row for row in gaussian_rows if row["local_minimum_id"] in kept_ids
    ]
    print(
        f"{lepton_name}/{state_key}/{mode}: re-filtered minima to "
        f"absolute {objective_col} <= {threshold} "
        f"({len(selected_rows)}/{before} minima kept, "
        f"{len(gaussian_rows)}/{gaussian_before} gaussian points kept)"
    )
    return selected_rows, gaussian_rows


def _equal_minimum_probability_weights(rows):
    """Give each local minimum unit total weight, then normalize to one."""
    row_counts = Counter(row["local_minimum_id"] for row in rows)
    minimum_count = len(row_counts)
    if not minimum_count:
        raise ValueError("Cannot weight an empty Gaussian sample.")
    weights = np.fromiter(
        (
            1.0 / (minimum_count * row_counts[row["local_minimum_id"]])
            for row in rows
        ),
        dtype=float,
        count=len(rows),
    )
    return weights, row_counts


def build_state_histograms(
    mode,
    lepton_name,
    state_key,
    lepton_mass,
    *,
    summary_only=False,
):
    """Write the summary page and nine individual histogram panels."""
    plt, _PdfPages = require_matplotlib()
    rows = _load_rows(mode, lepton_name, state_key)
    if rows is None:
        return

    output_dir = (
        SCAN_ROOT / lepton_name / "Plots" / state_key
        / "polarization_correlations_histogram" / mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_count = GAUSSIAN_BINS if mode.startswith("gaussian") else None
    log_color = mode.startswith("gaussian")
    cmap_name = GAUSSIAN_CMAP if mode.startswith("gaussian") else "viridis"
    weights = None
    if mode.startswith("gaussian"):
        weights, row_counts = _equal_minimum_probability_weights(rows)
        counts = np.fromiter(row_counts.values(), dtype=int)
        print(
            f"{lepton_name}/{state_key}/{mode}: probability-per-bin "
            f"normalization over {len(row_counts)} equally weighted minima; "
            f"rows/minimum={counts.min()}..{counts.max()}, "
            f"sum(weights)={weights.sum():.12g}"
        )
    plotted_fields = {
        field_name
        for x_name, y_name, _x_label, _y_label in FULL_CORRELATION_PANELS
        for field_name in (x_name, y_name)
    }
    plot_values = {
        field_name: np.fromiter(
            (float(row[field_name]) for row in rows),
            dtype=float,
            count=len(rows),
        )
        for field_name in plotted_fields
    }
    if log_color:
        summary_vmin, summary_vmax = _shared_histogram_limits(
            plot_values, bin_count, weights=weights,
        )
    else:
        summary_vmin, summary_vmax = None, None
    # The correlation summary is 13.3 inches wide without a colorbar.  A
    # Gaussian histogram summary is wider so its 3x3 grid retains the same
    # physical width while the shared colorbar occupies a dedicated column.
    summary_width = 14.7 if log_color else 13.3
    summary_right = 0.90 if log_color else 0.984
    _write_summary(
        plt,
        output_dir,
        state_key,
        plot_values,
        POLARIZATION_CORRELATION_PANELS,
        lepton_mass,
        nrows=3,
        ncols=3,
        figsize=(summary_width, 11.9),
        filename_stem="00_summary",
        bin_count=bin_count,
        log_color=log_color,
        cmap_name=cmap_name,
        weights=weights,
        color_vmin=summary_vmin,
        color_vmax=summary_vmax,
        margins={
            "left": 0.079,
            "right": summary_right,
            "bottom": 0.073,
            "top": 0.985,
            "wspace": 0.18,
            "hspace": 0.18,
        },
        colorbar_rect=(0.925, 0.10, 0.018, 0.80),
    )

    if log_color:
        full_vmin, full_vmax = _shared_histogram_limits(
            plot_values,
            bin_count,
            weights=weights,
            panels=FULL_CORRELATION_PANELS,
        )
    else:
        full_vmin, full_vmax = None, None
    _write_full_correlation_matrix(
        plt,
        output_dir,
        state_key,
        plot_values,
        lepton_mass,
        bin_count=bin_count,
        log_color=log_color,
        cmap_name=cmap_name,
        weights=weights,
        color_vmin=full_vmin,
        color_vmax=full_vmax,
    )

    if summary_only:
        return

    for panel_index, (x_name, y_name, x_label, y_label) in enumerate(
        POLARIZATION_CORRELATION_PANELS, start=1,
    ):
        # Keep the main axes at the correlation panel's physical 4.05-inch
        # width and add a dedicated right column for the histogram colorbar.
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        fig.subplots_adjust(
            left=0.129,
            right=0.782,
            bottom=0.150,
            top=0.962,
        )
        image = _draw_histogram_panel(
            ax, plot_values, x_name, y_name, x_label, y_label, lepton_mass,
            bin_count=bin_count,
            log_color=log_color,
            color_vmin=summary_vmin,
            color_vmax=summary_vmax,
            cmap_name=cmap_name,
            weights=weights,
        )
        colorbar_ax = fig.add_axes((0.819, 0.150, 0.029, 0.812))
        fig.colorbar(
            image,
            cax=colorbar_ax,
            label="probability per bin (log scale)" if log_color else "count",
        )
        filename = (
            f"{panel_index:02d}_{y_name}_vs_{x_name}_histogram_{state_key}.pdf"
        )
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}")


def _parse_args(argv=None):
    """Parse optional selectors while preserving the historical default run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lepton", action="append", choices=LEPTON_MASSES_GEV,
        help="Lepton to render; repeat for multiple leptons.",
    )
    parser.add_argument(
        "--state", action="append", choices=STATES,
        help="State to render; repeat for multiple states.",
    )
    parser.add_argument(
        "--mode", action="append",
        help="Histogram mode; repeat for multiple modes.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Regenerate only the summary PDF and PNG for each selection.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Build reference histograms for every configured lepton, state, mode."""
    args = _parse_args(argv)
    leptons = tuple(args.lepton or LEPTONS)
    states = tuple(args.state or STATES)
    modes = tuple(args.mode or ("minima_only", "smeared"))
    for lepton_name in leptons:
        lepton_mass = LEPTON_MASSES_GEV[lepton_name]
        for state_key in states:
            for mode in modes:
                build_state_histograms(
                    mode,
                    lepton_name,
                    state_key,
                    lepton_mass,
                    summary_only=args.summary_only,
                )


if __name__ == "__main__":
    raise SystemExit(main())
