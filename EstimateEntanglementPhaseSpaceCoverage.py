"""Estimate overlap-corrected entanglement coverage in normalized 8D space.

Each saved local contour is approximated by an eight-dimensional ellipsoid.
Its covariance is rescaled from the saved contour increment to the remaining
absolute objective budget ``threshold - D_i`` for minimum ``i``.  A mixture
importance sampler then estimates the union volume, correcting both overlap
and truncation by the four bounded normalized coordinates.  The final four
coordinates are treated as a unit torus.

This is a geometric proxy based on discovered minima and local quadratic
contours.  It is not an unbiased physical phase-space probability.

The summary also reports coordinate-wise mean correlation semi-lengths and
decomposes the union volume into a product of those lengths, the unit 8-ball
volume, and separate geometry/width, boundary, and overlap corrections.  A
second table maps the ellipsoid shapes through the local scan Jacobian and
reports the same mean semi-lengths in GeV or radians.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool
from GaussianSmearWorker import full_covariance_from_contour
from PlotUtils import require_matplotlib


STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTON_NAME = "electron"
ABSOLUTE_THRESHOLD = 0.01
DIRECTION_COUNT = 1536
SCAN_DIMENSION = 8
DEFAULT_MONTE_CARLO_SAMPLES = 20_000
RANDOM_SEED = 20260806

OBJECTIVE_COLUMNS = {
    "W": "final_D_W",
    "GHZ": "lepton_electron_alpha_e_mix_proton_alpha_p_mix_dGHZ",
    "CEP": "final_one_minus_C_e_p",
    "CEGAMMA": "final_one_minus_C_e_gamma",
    "CPGAMMA": "final_one_minus_C_p_gamma",
}

SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"
OUTPUT_DIR = (
    SCAN_ROOT / LEPTON_NAME / "Data"
    / "phase_space_coverage_threshold0.01"
)
SUMMARY_PATH = OUTPUT_DIR / "coverage_summary.csv"
PHYSICAL_LENGTH_TABLE_PATH = OUTPUT_DIR / "physical_correlation_lengths.csv"
PLOT_PATH = OUTPUT_DIR / "coverage_summary.pdf"
PLOT_PREVIEW_PATH = OUTPUT_DIR / "coverage_summary_preview.png"

COORDINATE_NAMES = (
    "sqrt_s",
    "theta_p_out",
    "theta_gamma_out",
    "qOut",
    "phi_p_out",
    "phi_gamma_out",
    "alpha_e",
    "alpha_p",
)
COORDINATE_LABELS = (
    r"$\sqrt{s}$",
    r"$\theta_{p'}$",
    r"$\theta_\gamma$",
    r"$E_\gamma$",
    r"$\phi_{p'}$",
    r"$\phi_\gamma$",
    r"$\alpha_e$",
    r"$\alpha_p$",
)
PHYSICAL_LENGTH_UNITS = (
    "GeV",
    "rad",
    "rad",
    "GeV",
    "rad",
    "rad",
    "rad",
    "rad",
)


@dataclass
class Ellipsoid:
    """One local contour ellipsoid in normalized phase-space coordinates."""

    minimum_id: str
    objective: float
    remaining_budget: float
    center: np.ndarray
    inverse_shape: np.ndarray
    shape_sqrt: np.ndarray
    bounding_radii: np.ndarray
    volume: float
    contour_success_fraction: float


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer fields for empty CSV {path}.")
        fieldnames = tuple(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_contour(path):
    metadata = None
    center = None
    points = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] == "minimum":
                metadata = row
                center = np.asarray(
                    [float(row[f"center_u{index}"]) for index in range(8)],
                    dtype=float,
                )
            elif row["record_type"] == "sample":
                points.append(
                    [float(row[f"u{index}"]) for index in range(8)]
                )
    if metadata is None or center is None:
        raise ValueError(f"Contour has no minimum metadata row: {path}")
    return metadata, center, np.asarray(points, dtype=float).reshape((-1, 8))


def _selected_minima(state_key):
    cluster_path = (
        SCAN_ROOT / LEPTON_NAME / "Data" / state_key
        / "cluster" / "clustered_minima.csv"
    )
    rows = _read_csv(cluster_path)
    objective_column = OBJECTIVE_COLUMNS[state_key]
    return [
        row for row in rows
        if row["within_polarization_cluster_cut"] == "True"
        and float(row[objective_column]) <= ABSOLUTE_THRESHOLD
    ]


def _ellipsoids_for_state(state_key):
    definition = definitions.selected_definitions((state_key,))[0]
    gradient_tool.configure_scan(
        definition,
        leptons_to_process=(LEPTON_NAME,),
        gradient_workers=1,
    )
    rows = _selected_minima(state_key)
    objective_column = OBJECTIVE_COLUMNS[state_key]
    contour_dir = (
        SCAN_ROOT / LEPTON_NAME / "Data" / state_key
        / "contour" / "local_minima"
    )
    unit_ball_volume = math.pi ** (SCAN_DIMENSION / 2) / math.gamma(
        SCAN_DIMENSION / 2 + 1
    )
    ellipsoids = []
    per_minimum_rows = []
    missing = 0
    for row in rows:
        minimum_id = row["local_minimum_id"]
        contour_path = (
            contour_dir
            / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
        )
        if not contour_path.exists():
            missing += 1
            continue
        metadata, center, points = _load_contour(contour_path)
        expected_center = gradient_tool._unit_point_from_minimum_row(row)
        if not np.allclose(center, expected_center, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Stale contour center: {contour_path}")
        configured_directions = int(metadata["configured_direction_count"])
        if configured_directions != DIRECTION_COUNT:
            raise ValueError(f"Unexpected direction count: {contour_path}")
        if len(points) < SCAN_DIMENSION:
            missing += 1
            continue
        contour_delta = float(metadata["contour_delta"])
        objective = float(row[objective_column])
        remaining_budget = ABSOLUTE_THRESHOLD - objective
        if remaining_budget <= 0.0:
            continue
        boundary_covariance = full_covariance_from_contour(center, points)
        covariance = boundary_covariance * (remaining_budget / contour_delta)
        # For uniformly sampled directions on an ellipsoid surface,
        # Cov(boundary points) = shape / dimension.
        shape = SCAN_DIMENSION * covariance
        shape = 0.5 * (shape + shape.T)
        eigenvalues, eigenvectors = np.linalg.eigh(shape)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
            missing += 1
            continue
        inverse_shape = (
            eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
        )
        shape_sqrt = (
            eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        )
        log_determinant = float(np.sum(np.log(eigenvalues)))
        volume = unit_ball_volume * math.exp(0.5 * log_determinant)
        bounding_radii = np.sqrt(np.maximum(np.diag(shape), 0.0))
        success_fraction = len(points) / configured_directions
        ellipsoids.append(
            Ellipsoid(
                minimum_id=minimum_id,
                objective=objective,
                remaining_budget=remaining_budget,
                center=center,
                inverse_shape=inverse_shape,
                shape_sqrt=shape_sqrt,
                bounding_radii=bounding_radii,
                volume=volume,
                contour_success_fraction=success_fraction,
            )
        )
        per_minimum_rows.append({
            "state": state_key,
            "local_minimum_id": minimum_id,
            "objective": objective,
            "remaining_budget": remaining_budget,
            "saved_contour_delta": contour_delta,
            "contour_success_fraction": success_fraction,
            "ellipsoid_volume_before_bounds": volume,
            "maximum_bounding_radius": float(np.max(bounding_radii)),
            **{
                f"bounding_radius_u{index}": float(bounding_radii[index])
                for index in range(SCAN_DIMENSION)
            },
        })
    if missing:
        print(
            f"{state_key}: skipped {missing} minima with missing or unusable contours",
            flush=True,
        )
    if not ellipsoids:
        raise ValueError(f"No usable ellipsoids for {state_key}.")
    _write_csv(
        OUTPUT_DIR / f"{state_key}_local_ellipsoids.csv",
        per_minimum_rows,
    )
    return rows, ellipsoids


def _sample_ellipsoid_mixture(ellipsoids, sample_count, rng):
    volumes = np.asarray([item.volume for item in ellipsoids], dtype=float)
    volume_sum = float(np.sum(volumes))
    source_indices = rng.choice(
        len(ellipsoids), size=sample_count, p=volumes / volume_sum,
    )
    directions = rng.normal(size=(sample_count, SCAN_DIMENSION))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.random(sample_count) ** (1.0 / SCAN_DIMENSION)
    unit_ball_points = directions * radii[:, None]
    centers = np.asarray([item.center for item in ellipsoids])
    square_roots = np.asarray([item.shape_sqrt for item in ellipsoids])
    displacements = np.einsum(
        "nij,nj->ni", square_roots[source_indices], unit_ball_points,
    )
    points = centers[source_indices] + displacements
    points[:, 4:] %= 1.0
    inside_domain = np.all(
        (points[:, :4] >= 0.0) & (points[:, :4] <= 1.0), axis=1,
    )
    return points, inside_domain, volume_sum


def _membership_counts(points, inside_domain, ellipsoids, batch_size=64):
    centers = np.asarray([item.center for item in ellipsoids])
    inverse_shapes = np.asarray([item.inverse_shape for item in ellipsoids])
    bounding_radii = np.asarray([item.bounding_radii for item in ellipsoids])
    counts = np.zeros(len(points), dtype=np.int32)
    valid_indices = np.flatnonzero(inside_domain)
    for batch_start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[batch_start:batch_start + batch_size]
        differences = points[batch_indices, None, :] - centers[None, :, :]
        differences[:, :, 4:] = (
            differences[:, :, 4:] + 0.5
        ) % 1.0 - 0.5
        box_candidates = np.all(
            np.abs(differences) <= bounding_radii[None, :, :] + 1.0e-12,
            axis=2,
        )
        for local_index, point_index in enumerate(batch_indices):
            candidate_indices = np.flatnonzero(box_candidates[local_index])
            if not len(candidate_indices):
                continue
            candidate_differences = differences[
                local_index, candidate_indices, :
            ]
            distances = np.einsum(
                "ni,nij,nj->n",
                candidate_differences,
                inverse_shapes[candidate_indices],
                candidate_differences,
            )
            counts[point_index] = np.count_nonzero(distances <= 1.0 + 1.0e-10)
    if np.any(counts[inside_domain] < 1):
        missing = np.count_nonzero(counts[inside_domain] < 1)
        raise RuntimeError(
            f"Mixture membership lost its source ellipsoid for {missing} samples."
        )
    return counts


def _physical_coordinate_jacobian(center, step=1.0e-6):
    """Return the local physical-coordinate Jacobian with respect to unit u."""
    jacobian = np.empty((SCAN_DIMENSION, SCAN_DIMENSION), dtype=float)
    for direction in range(SCAN_DIMENSION):
        plus = np.asarray(center, dtype=float).copy()
        minus = np.asarray(center, dtype=float).copy()
        plus[direction] += step
        minus[direction] -= step
        plus_values = gradient_tool._plot_coordinate_values(plus)
        minus_values = gradient_tool._plot_coordinate_values(minus)
        jacobian[:, direction] = np.asarray(
            [
                (plus_values[name] - minus_values[name]) / (2.0 * step)
                for name in COORDINATE_NAMES
            ],
            dtype=float,
        )
    return jacobian


def _estimate_state(state_key, sample_count):
    selected_rows, ellipsoids = _ellipsoids_for_state(state_key)
    rng = np.random.default_rng(RANDOM_SEED + STATES.index(state_key))
    points, inside_domain, raw_volume_sum = _sample_ellipsoid_mixture(
        ellipsoids, sample_count, rng,
    )
    counts = _membership_counts(points, inside_domain, ellipsoids)
    contributions = np.zeros(sample_count, dtype=float)
    contributions[inside_domain] = 1.0 / counts[inside_domain]
    union_fraction = raw_volume_sum * float(np.mean(contributions))
    union_standard_error = raw_volume_sum * float(
        np.std(contributions, ddof=1) / np.sqrt(sample_count)
    )
    clipped_volume_sum = raw_volume_sum * float(np.mean(inside_domain))
    overlap_fraction = (
        1.0 - union_fraction / clipped_volume_sum
        if clipped_volume_sum > 0.0 else np.nan
    )
    inside_counts = counts[inside_domain]
    local_volumes = np.asarray(
        [item.volume for item in ellipsoids], dtype=float,
    )
    correlation_lengths = np.asarray(
        [item.bounding_radii for item in ellipsoids], dtype=float,
    )
    mean_correlation_lengths = np.mean(correlation_lengths, axis=0)
    physical_correlation_lengths = []
    for item in ellipsoids:
        normalized_shape = item.shape_sqrt @ item.shape_sqrt.T
        jacobian = _physical_coordinate_jacobian(item.center)
        physical_shape = jacobian @ normalized_shape @ jacobian.T
        physical_correlation_lengths.append(
            np.sqrt(np.maximum(np.diag(physical_shape), 0.0))
        )
    mean_physical_correlation_lengths = np.mean(
        np.asarray(physical_correlation_lengths), axis=0,
    )
    unit_ball_volume = math.pi ** (SCAN_DIMENSION / 2) / math.gamma(
        SCAN_DIMENSION / 2 + 1
    )
    factorized_mean_length_volume = (
        len(ellipsoids)
        * unit_ball_volume
        * float(np.prod(mean_correlation_lengths))
    )
    geometry_width_correction = (
        raw_volume_sum / factorized_mean_length_volume
        if factorized_mean_length_volume > 0.0 else np.nan
    )
    boundary_retention_factor = (
        clipped_volume_sum / raw_volume_sum
        if raw_volume_sum > 0.0 else np.nan
    )
    overlap_retention_factor = (
        union_fraction / clipped_volume_sum
        if clipped_volume_sum > 0.0 else np.nan
    )
    coverage_retention_factor = (
        boundary_retention_factor * overlap_retention_factor
    )
    total_volume_correction = (
        union_fraction / factorized_mean_length_volume
        if factorized_mean_length_volume > 0.0 else np.nan
    )
    correction_product = (
        geometry_width_correction
        * boundary_retention_factor
        * overlap_retention_factor
    )
    if not np.isclose(
        correction_product,
        total_volume_correction,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise RuntimeError(
            f"Factorized volume corrections do not close for {state_key}."
        )
    descending_volumes = np.sort(local_volumes)[::-1]
    effective_ellipsoid_count = float(
        np.sum(local_volumes) ** 2 / np.sum(local_volumes ** 2)
    )
    result = {
        "state": state_key,
        "absolute_threshold": ABSOLUTE_THRESHOLD,
        "selected_minimum_count": len(selected_rows),
        "usable_ellipsoid_count": len(ellipsoids),
        "configured_directions_per_minimum": DIRECTION_COUNT,
        "monte_carlo_samples": sample_count,
        "raw_summed_ellipsoid_volume": raw_volume_sum,
        "boundary_clipped_summed_volume": clipped_volume_sum,
        "union_coverage_fraction": union_fraction,
        "union_standard_error": union_standard_error,
        "union_relative_standard_error": (
            union_standard_error / union_fraction
            if union_fraction > 0.0 else np.nan
        ),
        "unit_8_ball_volume": unit_ball_volume,
        "factorized_mean_length_volume": factorized_mean_length_volume,
        "geometry_width_correction": geometry_width_correction,
        "boundary_retention_factor": boundary_retention_factor,
        "overlap_retention_factor": overlap_retention_factor,
        "coverage_retention_factor": coverage_retention_factor,
        "total_volume_correction": total_volume_correction,
        "correction_factor_product": correction_product,
        "factorized_reconstructed_union_volume": (
            factorized_mean_length_volume * total_volume_correction
        ),
        **{
            f"mean_correlation_length_{name}": float(
                mean_correlation_lengths[index]
            )
            for index, name in enumerate(COORDINATE_NAMES)
        },
        **{
            f"mean_physical_correlation_length_{name}": float(
                mean_physical_correlation_lengths[index]
            )
            for index, name in enumerate(COORDINATE_NAMES)
        },
        "overlap_fraction_of_clipped_sum": overlap_fraction,
        "bounded_domain_acceptance": float(np.mean(inside_domain)),
        "mean_overlap_multiplicity_inside_domain": float(
            np.mean(inside_counts)
        ),
        "median_overlap_multiplicity_inside_domain": float(
            np.median(inside_counts)
        ),
        "maximum_overlap_multiplicity_inside_domain": int(
            np.max(inside_counts)
        ),
        "median_local_volume": float(np.median(local_volumes)),
        "effective_ellipsoid_count_by_raw_volume": effective_ellipsoid_count,
        "largest_ellipsoid_raw_volume_share": float(
            descending_volumes[0] / np.sum(local_volumes)
        ),
        "top_10_ellipsoid_raw_volume_share": float(
            np.sum(descending_volumes[:10]) / np.sum(local_volumes)
        ),
        "mean_contour_success_fraction": float(
            np.mean([item.contour_success_fraction for item in ellipsoids])
        ),
        "maximum_bounding_radius": float(
            max(np.max(item.bounding_radii) for item in ellipsoids)
        ),
    }
    print(
        f"{state_key}: coverage={union_fraction:.6e} +/- "
        f"{union_standard_error:.2e}; overlap={overlap_fraction:.1%}; "
        f"bounded acceptance={np.mean(inside_domain):.1%}; "
        f"retention C={coverage_retention_factor:.3e}",
        flush=True,
    )
    return result


def _plot_summary(rows):
    plt, _PdfPages = require_matplotlib()
    states = [row["state"] for row in rows]
    state_colors = {
        "W": "#D55E00",
        "GHZ": "#CC79A7",
        "CEP": "#009E73",
        "CEGAMMA": "#0072B2",
        "CPGAMMA": "#E69F00",
    }
    state_markers = {
        "W": "o",
        "GHZ": "s",
        "CEP": "^",
        "CEGAMMA": "D",
        "CPGAMMA": "P",
    }
    coverage = np.asarray(
        [float(row["union_coverage_fraction"]) for row in rows]
    )
    errors = np.asarray([float(row["union_standard_error"]) for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    positions = np.arange(len(states))

    axes[0, 0].bar(
        positions, coverage, color=[state_colors[state] for state in states],
    )
    axes[0, 0].errorbar(
        positions, coverage, yerr=errors, fmt="none", color="black", capsize=3,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(positions, states)
    axes[0, 0].set_ylabel("normalized 8D union coverage")
    axes[0, 0].grid(axis="y", alpha=0.25)

    coordinate_positions = np.arange(SCAN_DIMENSION)
    for row in rows:
        lengths = [
            float(row[f"mean_correlation_length_{name}"])
            for name in COORDINATE_NAMES
        ]
        axes[0, 1].plot(
            coordinate_positions,
            lengths,
            marker=state_markers[row["state"]],
            linewidth=1.8,
            markersize=4.5,
            color=state_colors[row["state"]],
            label=row["state"],
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(coordinate_positions, COORDINATE_LABELS)
    axes[0, 1].set_ylabel(
        "mean normalized semi-length $\\bar v_d$ (dimensionless)"
    )
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(ncol=2, fontsize=8, frameon=True)

    correction_fields = (
        ("geometry_width_correction", r"$C_{\rm geom}$", "#6A00A8"),
        ("boundary_retention_factor", r"$C_{\rm bounds}$", "#E84A5F"),
        ("overlap_retention_factor", r"$C_{\rm overlap}$", "#F89540"),
        ("coverage_retention_factor", r"$C$", "#0072B2"),
    )
    bar_width = 0.19
    for factor_index, (field, label, color) in enumerate(correction_fields):
        offsets = positions + (factor_index - 1.5) * bar_width
        axes[1, 0].bar(
            offsets,
            [float(row[field]) for row in rows],
            width=bar_width,
            color=color,
            label=label,
        )
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8, alpha=0.65)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(positions, states)
    axes[1, 0].set_ylabel("multiplicative correction factor")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(ncol=2, fontsize=8, frameon=True)

    volume_stages = (
        ("factorized_mean_length_volume", "mean-length product", "#6A00A8"),
        ("raw_summed_ellipsoid_volume", "exact local ellipsoids", "#E84A5F"),
        ("boundary_clipped_summed_volume", "after bounds", "#F89540"),
        ("union_coverage_fraction", "after overlap", "#0072B2"),
    )
    for stage_index, (field, label, color) in enumerate(volume_stages):
        offsets = positions + (stage_index - 1.5) * bar_width
        axes[1, 1].bar(
            offsets,
            [float(row[field]) for row in rows],
            width=bar_width,
            color=color,
            label=label,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(positions, states)
    axes[1, 1].set_ylabel("normalized 8D volume")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(ncol=2, fontsize=8, frameon=True)

    figure.text(
        0.5,
        0.975,
        r"$V_{\cup}=N_{\rm ell}V_8\prod_{d=1}^{8}"
        r"(\Delta d_d\,\bar v_d)\,C_{\rm geom}C$,  "
        r"$V_8=\pi^4/24$,  "
        r"$C=C_{\rm bounds}C_{\rm overlap}$,  "
        r"$\Delta d_d=1$ (normalized coordinates)",
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.925,
        wspace=0.25,
        hspace=0.28,
    )
    figure.savefig(PLOT_PATH)
    figure.savefig(PLOT_PREVIEW_PATH, dpi=150)
    plt.close(figure)


def _physical_length_table_rows(rows):
    """Return one physical correlation-length matrix row per state."""
    return [
        {
            "state": row["state"],
            **{
                f"{coordinate_name}_{unit}": row[
                    f"mean_physical_correlation_length_{coordinate_name}"
                ]
                for coordinate_name, unit in zip(
                    COORDINATE_NAMES, PHYSICAL_LENGTH_UNITS,
                )
            },
        }
        for row in rows
    ]


def run(states=STATES, sample_count=DEFAULT_MONTE_CARLO_SAMPLES):
    if sample_count < 100:
        raise ValueError("Use at least 100 Monte Carlo samples per state.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_estimate_state(state, sample_count) for state in states]
    _write_csv(SUMMARY_PATH, rows)
    physical_length_rows = _physical_length_table_rows(rows)
    _write_csv(PHYSICAL_LENGTH_TABLE_PATH, physical_length_rows)
    _plot_summary(rows)
    print(f"wrote {SUMMARY_PATH}", flush=True)
    print(f"wrote {PHYSICAL_LENGTH_TABLE_PATH}", flush=True)
    print(f"wrote {PLOT_PATH}", flush=True)
    return rows


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", action="append", choices=STATES,
        help="State to estimate; repeat for multiple states.",
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_MONTE_CARLO_SAMPLES,
        help="Mixture importance samples per state.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(tuple(args.state or STATES), sample_count=args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
