"""Benchmark direct delta-D contours against covariance rescaling.

The production Gaussian histograms approximate a width of ``delta_D=0.02``
by multiplying covariance matrices measured on ``delta_D=0.05`` contours by
``0.02 / 0.05``.  This tool performs a non-destructive, resumable comparison
for a deterministic efficiency-stratified subset of CEGAMMA minima.

Run a one-minimum smoke test followed by the complete benchmark with

    python CompareGaussianContourRescaling.py --limit 1
    python CompareGaussianContourRescaling.py
"""

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool
from GaussianSmearWorker import (
    draw_gaussian_samples,
    full_covariance_from_contour,
    rescale_covariance_for_cut,
)
from HistogramPolarizationCorrelations import POLARIZATION_CORRELATION_PANELS
from PlotUtils import require_matplotlib


STATE_KEY = "CEGAMMA"
LEPTON_NAME = "electron"
ABSOLUTE_THRESHOLD = 0.01
EXISTING_DELTA = 0.05
DIRECT_DELTA = 0.02
SAMPLE_SIZE = 50
DIRECTION_COUNT = 1536

SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"
CLUSTER_DIR = SCAN_ROOT / LEPTON_NAME / "Data" / STATE_KEY / "cluster"
EXISTING_CONTOUR_DIR = (
    SCAN_ROOT / LEPTON_NAME / "Data" / STATE_KEY / "contour" / "local_minima"
)
OUTPUT_DIR = (
    SCAN_ROOT / LEPTON_NAME / "Data" / STATE_KEY
    / "contour_width_comparison" / "delta0.02_n50"
)
DIRECT_CONTOUR_DIR = OUTPUT_DIR / "local_minima"
PROGRESS_PATH = OUTPUT_DIR / "per_minimum_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "aggregate_summary.csv"
PLOT_PATH = OUTPUT_DIR / "covariance_comparison_summary.pdf"
PROJECTION_SUMMARY_PATH = OUTPUT_DIR / "projection_histogram_comparison.csv"
PROJECTION_PLOT_PATH = OUTPUT_DIR / "projection_histogram_comparison.pdf"

COORDINATE_NAMES = (
    "sqrt_s_unit",
    "theta_p_out_unit",
    "theta_gamma_out_unit",
    "qOut_fraction_unit",
    "phi_p_out_unit",
    "phi_gamma_out_unit",
    "alpha_e_unit",
    "alpha_p_unit",
)


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
    center = None
    points = []
    metadata = None
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


def _eligible_rows():
    minima_path = CLUSTER_DIR / "clustered_minima.csv"
    rows = _read_csv(minima_path)
    return [
        row for row in rows
        if row["within_polarization_cluster_cut"] == "True"
        and float(row["final_one_minus_C_e_gamma"]) <= ABSOLUTE_THRESHOLD
    ]


def _representative_rows(sample_size):
    """Select deterministic quantiles of the existing success distribution."""
    rows = _eligible_rows()
    diagnostics = {
        row["local_minimum_id"]: row
        for row in _read_csv(
            CLUSTER_DIR / "gaussian_sampling_diagnostics_cut0.02.csv"
        )
    }
    available = [
        row for row in rows
        if row["local_minimum_id"] in diagnostics
        and (
            EXISTING_CONTOUR_DIR
            / f"local_minimum_{int(row['local_minimum_id']):04d}_contour_samples.csv"
        ).exists()
    ]
    available.sort(
        key=lambda row: (
            int(diagnostics[row["local_minimum_id"]]["contour_point_count"]),
            int(row["local_minimum_id"]),
        )
    )
    if sample_size > len(available):
        raise ValueError(
            f"Requested {sample_size} minima but only {len(available)} are available."
        )
    indices = np.rint(
        np.linspace(0, len(available) - 1, sample_size)
    ).astype(int)
    if len(set(indices)) != sample_size:
        raise RuntimeError("Efficiency-stratified selection repeated an index.")
    selected = [available[index] for index in indices]
    median_efficiency = np.median(
        [
            int(diagnostics[row["local_minimum_id"]]["contour_point_count"])
            / DIRECTION_COUNT
            for row in selected
        ]
    )
    # Put a typical case first so --limit 1 is a representative smoke test.
    selected.sort(
        key=lambda row: (
            abs(
                int(diagnostics[row["local_minimum_id"]]["contour_point_count"])
                / DIRECTION_COUNT
                - median_efficiency
            ),
            int(row["local_minimum_id"]),
        )
    )
    return selected


def _correlation_matrix(covariance):
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviations, standard_deviations)
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )


def _covariance_metrics(scaled_covariance, direct_covariance):
    difference = scaled_covariance - direct_covariance
    direct_norm = np.linalg.norm(direct_covariance, ord="fro")
    relative_frobenius = (
        np.linalg.norm(difference, ord="fro") / direct_norm
        if direct_norm > 0.0 else np.nan
    )
    assumed_scale = DIRECT_DELTA / EXISTING_DELTA
    unscaled_covariance = scaled_covariance / assumed_scale
    best_scale = float(
        np.vdot(unscaled_covariance, direct_covariance).real
        / np.vdot(unscaled_covariance, unscaled_covariance).real
    )
    best_scaled_error = (
        np.linalg.norm(
            best_scale * unscaled_covariance - direct_covariance,
            ord="fro",
        )
        / direct_norm
        if direct_norm > 0.0 else np.nan
    )
    scaled_std = np.sqrt(np.maximum(np.diag(scaled_covariance), 0.0))
    direct_std = np.sqrt(np.maximum(np.diag(direct_covariance), 0.0))
    std_ratios = np.divide(
        scaled_std,
        direct_std,
        out=np.full_like(scaled_std, np.nan),
        where=direct_std > 0.0,
    )
    correlation_difference = (
        _correlation_matrix(scaled_covariance)
        - _correlation_matrix(direct_covariance)
    )
    off_diagonal = ~np.eye(len(correlation_difference), dtype=bool)

    scale = max(float(np.trace(direct_covariance)) / 8.0, 1.0e-12)
    regularization = 1.0e-10 * scale
    direct_regularized = direct_covariance + regularization * np.eye(8)
    scaled_regularized = scaled_covariance + regularization * np.eye(8)
    direct_eigenvalues, direct_eigenvectors = np.linalg.eigh(direct_regularized)
    inverse_root = (
        direct_eigenvectors
        @ np.diag(1.0 / np.sqrt(direct_eigenvalues))
        @ direct_eigenvectors.T
    )
    generalized = np.linalg.eigvalsh(
        inverse_root @ scaled_regularized @ inverse_root
    )
    log_volume_ratio = 0.5 * (
        np.linalg.slogdet(scaled_regularized)[1]
        - np.linalg.slogdet(direct_regularized)[1]
    )
    metrics = {
        "relative_frobenius_error": relative_frobenius,
        "best_covariance_scale": best_scale,
        "best_scale_over_assumed_scale": best_scale / assumed_scale,
        "best_scaled_relative_frobenius_error": best_scaled_error,
        "trace_ratio_scaled_over_direct": float(
            np.trace(scaled_covariance) / np.trace(direct_covariance)
        ),
        "std_ratio_min": float(np.nanmin(std_ratios)),
        "std_ratio_median": float(np.nanmedian(std_ratios)),
        "std_ratio_max": float(np.nanmax(std_ratios)),
        "correlation_rms_difference": float(
            np.sqrt(np.mean(correlation_difference[off_diagonal] ** 2))
        ),
        "correlation_max_abs_difference": float(
            np.max(np.abs(correlation_difference[off_diagonal]))
        ),
        "generalized_variance_ratio_min": float(generalized[0]),
        "generalized_variance_ratio_median": float(np.median(generalized)),
        "generalized_variance_ratio_max": float(generalized[-1]),
        "log_ellipsoid_volume_ratio": float(log_volume_ratio),
    }
    metrics.update(
        {
            f"std_ratio_u{index}_{name}": float(std_ratios[index])
            for index, name in enumerate(COORDINATE_NAMES)
        }
    )
    return metrics


def _direct_contour_path(minimum_id):
    return (
        DIRECT_CONTOUR_DIR
        / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
    )


def _load_valid_direct_contour(path, expected_center):
    if not path.exists():
        return None
    metadata, center, points = _load_contour(path)
    if (
        metadata["objective_name"] != "one_minus_C_e_gamma"
        or not np.isclose(float(metadata["contour_delta"]), DIRECT_DELTA)
        or int(metadata["configured_direction_count"]) != DIRECTION_COUNT
        or not np.allclose(center, expected_center, rtol=0.0, atol=1.0e-12)
    ):
        return None
    return points


def _comparison_row(row, prior=None):
    minimum_id = row["local_minimum_id"]
    expected_center = gradient_tool._unit_point_from_minimum_row(row)
    existing_path = (
        EXISTING_CONTOUR_DIR
        / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
    )
    metadata, center, existing_points = _load_contour(existing_path)
    if not np.isclose(float(metadata["contour_delta"]), EXISTING_DELTA):
        raise ValueError(f"Unexpected existing contour delta in {existing_path}")
    if int(metadata["configured_direction_count"]) != DIRECTION_COUNT:
        raise ValueError(f"Unexpected direction count in {existing_path}")
    if not np.allclose(center, expected_center, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Existing contour center is stale in {existing_path}")

    direct_path = _direct_contour_path(minimum_id)
    direct_points = _load_valid_direct_contour(direct_path, expected_center)
    if direct_points is None:
        started = perf_counter()
        contours = gradient_tool._configuration_contours([row], LEPTON_NAME)
        direct_seconds = perf_counter() - started
        direct_points = contours[0]
        csv_rows = gradient_tool._contour_csv_rows([row], contours)
        _write_csv(direct_path, csv_rows, fieldnames=tuple(csv_rows[0]))
        action = "generated"
    else:
        direct_seconds = (
            float(prior["direct_contour_seconds"])
            if prior and prior.get("direct_contour_seconds")
            else np.nan
        )
        action = "reused"

    existing_covariance = full_covariance_from_contour(
        center, existing_points,
    )
    scaled_covariance = rescale_covariance_for_cut(
        existing_covariance, EXISTING_DELTA, DIRECT_DELTA,
    )
    direct_covariance = full_covariance_from_contour(
        expected_center, direct_points,
    )
    result = {
        "local_minimum_id": minimum_id,
        "absolute_objective": float(row["final_one_minus_C_e_gamma"]),
        "existing_contour_delta": EXISTING_DELTA,
        "direct_contour_delta": DIRECT_DELTA,
        "configured_direction_count": DIRECTION_COUNT,
        "existing_success_count": len(existing_points),
        "direct_success_count": len(direct_points),
        "existing_success_fraction": len(existing_points) / DIRECTION_COUNT,
        "direct_success_fraction": len(direct_points) / DIRECTION_COUNT,
        "success_fraction_change": (
            len(direct_points) - len(existing_points)
        ) / DIRECTION_COUNT,
        "direct_contour_seconds": direct_seconds,
        **_covariance_metrics(scaled_covariance, direct_covariance),
    }
    return result, action


def _aggregate_rows(rows):
    metric_names = (
        "existing_success_fraction",
        "direct_success_fraction",
        "success_fraction_change",
        "direct_contour_seconds",
        "relative_frobenius_error",
        "best_covariance_scale",
        "best_scale_over_assumed_scale",
        "best_scaled_relative_frobenius_error",
        "trace_ratio_scaled_over_direct",
        "std_ratio_min",
        "std_ratio_median",
        "std_ratio_max",
        "correlation_rms_difference",
        "correlation_max_abs_difference",
        "generalized_variance_ratio_min",
        "generalized_variance_ratio_median",
        "generalized_variance_ratio_max",
        "log_ellipsoid_volume_ratio",
    )
    output = []
    for name in metric_names:
        values = np.asarray(
            [float(row[name]) for row in rows if row.get(name) not in (None, "")],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        output.append({
            "metric": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(np.max(values)),
        })
    return output


def _plot_summary(rows):
    plt, _PdfPages = require_matplotlib()
    existing = np.asarray(
        [float(row["existing_success_fraction"]) for row in rows]
    )
    direct = np.asarray([float(row["direct_success_fraction"]) for row in rows])
    runtime = np.asarray([float(row["direct_contour_seconds"]) for row in rows])
    relative_error = np.asarray(
        [float(row["relative_frobenius_error"]) for row in rows]
    )
    correlation_error = np.asarray(
        [float(row["correlation_rms_difference"]) for row in rows]
    )
    std_ratios = np.asarray(
        [
            [float(row[f"std_ratio_u{index}_{name}"]) for index, name in enumerate(COORDINATE_NAMES)]
            for row in rows
        ]
    )

    figure, axes = plt.subplots(2, 3, figsize=(13.3, 8.2))
    axes[0, 0].scatter(existing, direct, s=24, color="#7e03a8", alpha=0.8)
    axes[0, 0].plot((0, 1), (0, 1), color="black", linewidth=1)
    axes[0, 0].set(xlabel="success at delta D=0.05", ylabel="success at delta D=0.02")
    axes[0, 0].grid(alpha=0.25)

    finite_runtime = runtime[np.isfinite(runtime)]
    axes[0, 1].hist(finite_runtime, bins=12, color="#e84a5f", edgecolor="white")
    axes[0, 1].set(xlabel="direct contour time per minimum [s]", ylabel="minima")

    axes[0, 2].hist(relative_error, bins=12, color="#f89540", edgecolor="white")
    axes[0, 2].set(xlabel="relative covariance Frobenius error", ylabel="minima")

    axes[1, 0].boxplot(std_ratios, tick_labels=[f"u{i}" for i in range(8)])
    axes[1, 0].axhline(1.0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="normalized coordinate", ylabel="scaled/direct std. dev.")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].hist(correlation_error, bins=12, color="#b12a90", edgecolor="white")
    axes[1, 1].set(xlabel="correlation-matrix RMS difference", ylabel="minima")

    axes[1, 2].scatter(
        direct, relative_error, s=24, color="#240046", alpha=0.8,
    )
    axes[1, 2].set(
        xlabel="direct delta-D=0.02 success fraction",
        ylabel="relative covariance error",
    )
    axes[1, 2].grid(alpha=0.25)

    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.09, top=0.98,
        wspace=0.26, hspace=0.28,
    )
    figure.savefig(PLOT_PATH)
    plt.close(figure)


def _jensen_shannon_divergence(first, second):
    midpoint = 0.5 * (first + second)
    first_mask = first > 0.0
    second_mask = second > 0.0
    divergence = 0.5 * np.sum(
        first[first_mask] * np.log2(first[first_mask] / midpoint[first_mask])
    )
    divergence += 0.5 * np.sum(
        second[second_mask]
        * np.log2(second[second_mask] / midpoint[second_mask])
    )
    return float(divergence)


def _projection_histogram_comparison(comparison_rows, minima_by_id):
    """Compare pooled 120-bin projections against sampling-noise replicas."""
    methods = ("rescaled_a", "rescaled_b", "direct")
    fields = {
        field
        for x_name, y_name, _x_label, _y_label
        in POLARIZATION_CORRELATION_PANELS
        for field in (x_name, y_name)
    }
    chunks = {
        method: {field: [] for field in fields}
        for method in methods
    }
    rejected = {method: 0 for method in methods}
    attempted = {method: 0 for method in methods}

    for comparison in comparison_rows:
        minimum_id = comparison["local_minimum_id"]
        row = minima_by_id[minimum_id]
        center = gradient_tool._unit_point_from_minimum_row(row)
        _metadata, _saved_center, existing_points = _load_contour(
            EXISTING_CONTOUR_DIR
            / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
        )
        _metadata, _saved_center, direct_points = _load_contour(
            _direct_contour_path(minimum_id)
        )
        rescaled_covariance = rescale_covariance_for_cut(
            full_covariance_from_contour(center, existing_points),
            EXISTING_DELTA,
            DIRECT_DELTA,
        )
        direct_covariance = full_covariance_from_contour(
            center, direct_points,
        )
        seed = 20260806 + 10 * int(minimum_id)
        method_settings = (
            ("rescaled_a", rescaled_covariance, seed),
            ("rescaled_b", rescaled_covariance, seed + 1),
            ("direct", direct_covariance, seed),
        )
        for method, covariance, method_seed in method_settings:
            physical, diagnostics = draw_gaussian_samples(
                center,
                covariance,
                2000,
                np.random.default_rng(method_seed),
                return_diagnostics=True,
            )
            rejected[method] += diagnostics["rejected_samples"]
            attempted[method] += diagnostics["attempted_samples"]
            for field_index, field in enumerate(
                (
                    "sqrt_s", "theta_p_out", "theta_gamma_out", "qOut",
                    "phi_p_out", "phi_gamma_out", "alpha_e", "alpha_p",
                )
            ):
                if field in fields:
                    chunks[method][field].append(physical[:, field_index])
            for field in fields:
                chunks[method][field].append(
                    np.asarray([float(row[field])], dtype=float)
                )

    values = {
        method: {
            field: np.concatenate(field_chunks)
            for field, field_chunks in method_fields.items()
        }
        for method, method_fields in chunks.items()
    }
    output = []
    for x_name, y_name, x_label, y_label in POLARIZATION_CORRELATION_PANELS:
        x_all = np.concatenate([values[method][x_name] for method in methods])
        y_all = np.concatenate([values[method][y_name] for method in methods])
        histogram_range = (
            (float(np.min(x_all)), float(np.max(x_all))),
            (float(np.min(y_all)), float(np.max(y_all))),
        )
        probabilities = {}
        for method in methods:
            counts, _x_edges, _y_edges = np.histogram2d(
                values[method][x_name],
                values[method][y_name],
                bins=120,
                range=histogram_range,
            )
            probabilities[method] = counts / np.sum(counts)
        direct_tv = 0.5 * np.sum(
            np.abs(probabilities["direct"] - probabilities["rescaled_a"])
        )
        baseline_tv = 0.5 * np.sum(
            np.abs(probabilities["rescaled_b"] - probabilities["rescaled_a"])
        )
        direct_js = _jensen_shannon_divergence(
            probabilities["direct"], probabilities["rescaled_a"],
        )
        baseline_js = _jensen_shannon_divergence(
            probabilities["rescaled_b"], probabilities["rescaled_a"],
        )
        output.append({
            "x_name": x_name,
            "y_name": y_name,
            "x_label": x_label,
            "y_label": y_label,
            "direct_vs_rescaled_total_variation": float(direct_tv),
            "sampling_baseline_total_variation": float(baseline_tv),
            "excess_total_variation": float(direct_tv - baseline_tv),
            "direct_vs_rescaled_js_divergence_bits": direct_js,
            "sampling_baseline_js_divergence_bits": baseline_js,
            "excess_js_divergence_bits": direct_js - baseline_js,
        })
    for row in output:
        for method in methods:
            row[f"{method}_rejection_fraction"] = (
                rejected[method] / attempted[method]
            )
    _write_csv(PROJECTION_SUMMARY_PATH, output)

    plt, _PdfPages = require_matplotlib()
    labels = [f"{row['y_name']} vs {row['x_name']}" for row in output]
    positions = np.arange(len(output))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(13.3, 5.0))
    for ax, metric, baseline, ylabel in (
        (
            axes[0],
            "direct_vs_rescaled_total_variation",
            "sampling_baseline_total_variation",
            "total-variation distance",
        ),
        (
            axes[1],
            "direct_vs_rescaled_js_divergence_bits",
            "sampling_baseline_js_divergence_bits",
            "Jensen-Shannon divergence [bits]",
        ),
    ):
        ax.bar(
            positions - width / 2,
            [float(row[metric]) for row in output],
            width,
            color="#b12a90",
            label="direct 0.02 vs rescaled 0.05",
        )
        ax.bar(
            positions + width / 2,
            [float(row[baseline]) for row in output],
            width,
            color="#f6e65b",
            edgecolor="#6a00a8",
            label="sampling-noise baseline",
        )
        ax.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)
    figure.subplots_adjust(
        left=0.07, right=0.985, bottom=0.36, top=0.97, wspace=0.23,
    )
    figure.savefig(PROJECTION_PLOT_PATH)
    plt.close(figure)


def run(sample_size=SAMPLE_SIZE, limit=None, workers=None):
    if sample_size < 1:
        raise ValueError("sample_size must be positive.")
    if limit is not None and (limit < 1 or limit > sample_size):
        raise ValueError("limit must lie between 1 and sample_size.")
    if DIRECTION_COUNT != gradient_tool.PHASE_SPACE_CONFIG_CONTOUR_SAMPLES:
        raise RuntimeError(
            "This comparison requires the configured 1536-direction setup."
        )
    worker_count = max(1, int(workers or os.cpu_count() or 1))
    definition = definitions.selected_definitions((STATE_KEY,))[0]
    gradient_tool.configure_scan(
        definition,
        leptons_to_process=(LEPTON_NAME,),
        gradient_workers=worker_count,
    )
    gradient_tool.PHASE_SPACE_CONFIG_CONTOUR_DELTA = DIRECT_DELTA
    representative_rows = _representative_rows(sample_size)
    rows = representative_rows
    if limit is not None:
        rows = rows[:limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIRECT_CONTOUR_DIR.mkdir(parents=True, exist_ok=True)
    prior_rows = (
        {row["local_minimum_id"]: row for row in _read_csv(PROGRESS_PATH)}
        if PROGRESS_PATH.exists() else {}
    )
    results = dict(prior_rows)
    started = perf_counter()
    for index, row in enumerate(rows, start=1):
        minimum_id = row["local_minimum_id"]
        result, action = _comparison_row(row, prior=prior_rows.get(minimum_id))
        results[minimum_id] = result
        ordered_results = [
            results[item["local_minimum_id"]]
            for item in representative_rows
            if item["local_minimum_id"] in results
        ]
        _write_csv(PROGRESS_PATH, ordered_results)
        elapsed = perf_counter() - started
        remaining = elapsed / index * (len(rows) - index)
        print(
            f"[{index}/{len(rows)}] {action} minimum {minimum_id}: "
            f"success {result['existing_success_fraction']:.1%} -> "
            f"{result['direct_success_fraction']:.1%}; "
            f"covariance error={result['relative_frobenius_error']:.3f}; "
            f"time={result['direct_contour_seconds']:.2f} s; "
            f"ETA={remaining / 60.0:.1f} min",
            flush=True,
        )

    selected_ids = {row["local_minimum_id"] for row in rows}
    final_rows = [
        results[row["local_minimum_id"]]
        for row in representative_rows
        if row["local_minimum_id"] in selected_ids
    ]
    _write_csv(SUMMARY_PATH, _aggregate_rows(final_rows))
    _plot_summary(final_rows)
    minima_by_id = {row["local_minimum_id"]: row for row in representative_rows}
    _projection_histogram_comparison(final_rows, minima_by_id)
    print(f"wrote {PROGRESS_PATH}", flush=True)
    print(f"wrote {SUMMARY_PATH}", flush=True)
    print(f"wrote {PLOT_PATH}", flush=True)
    print(f"wrote {PROJECTION_SUMMARY_PATH}", flush=True)
    print(f"wrote {PROJECTION_PLOT_PATH}", flush=True)
    return final_rows


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N members of the fixed representative sample.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(args.sample_size, limit=args.limit, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
