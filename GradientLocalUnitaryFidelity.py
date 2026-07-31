"""Check canonical W/GHZ fidelity after final-particle local rotations.

This is a post-processing stage for ``GradientPhaseSpaceScan.py``.  It reads
saved ``scan/local_minima.csv`` files, reconstructs the normalized coherent
outgoing state in project order ``(e_out, p_out, gamma_out)``, and maximizes

    |<target| (U_e tensor U_p tensor U_gamma) |psi>|^2

over three independent SU(2) rotations in the project ``(-,+)`` basis.
Per-minimum rotation details and per-scan fidelity summaries are saved without
rerunning any phase-space optimization.

Edit the explicit controls below, then run

    python GradientLocalUnitaryFidelity.py
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import csv
from pathlib import Path
import sys
import zlib

import numpy as np
from scipy.optimize import minimize

from config import SCAN_WORKERS
from GradientPhaseSpaceCluster import POLARIZATION_CLUSTER_CUT
import GradientPhaseSpaceDefinitions as definitions
from GradientObjective import source_observable_name
from PhaseSpaceConfigScan import (
    _local_eigenbasis_rotations,
    _normalized_mixed_final_state,
    _su2_zyz,
    _su2_zyz_angles,
)
from PlotUtils import print_console_text, require_matplotlib
from SpinDensityMat import entanglement_measures_from_state, outgoing_spin_states


# Explicit script controls.
SCANS_TO_CHECK = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTONS_TO_CHECK = ("electron", "muon")
TARGETS_TO_CHECK = ("W", "GHZ")
# SciPy is imported in every Windows worker; cap the default to avoid paging
# pressure while retaining process-level parallelism.
FIDELITY_WORKERS = max(1, min(SCAN_WORKERS, 8))
LOCAL_UNITARY_RESTARTS = 12
LOCAL_UNITARY_MAX_ITERATIONS = 1000
# Each target screens minima by its matching LU-invariant distance: D_W for W
# and dGHZ for GHZ.  Their union is checked against both canonical targets.
# Set this to None for an exhaustive (and potentially expensive) check.
CANDIDATES_PER_TARGET_PER_SCAN = None
SKIP_MISSING_SCAN_INPUTS = True
# True rebuilds only the combined PDF from the saved combined summary CSV.
REMAKE_SUMMARY_PLOT_FROM_CSV = False

OUTPUT_ROOT = definitions.GRADIENT_OUTPUT_ROOT / "Fidelity"
SUMMARY_CSV = OUTPUT_ROOT / "local_unitary_fidelity_summary.csv"
SUMMARY_PDF = OUTPUT_ROOT / "local_unitary_fidelity_summary.pdf"
LOG_PATH = OUTPUT_ROOT / "local_unitary_fidelity.log"

FINAL_PARTICLES = ("lepton", "proton", "gamma")
TARGET_SCREEN_OBSERVABLES = {"W": "D_W", "GHZ": "dGHZ"}
INFIDELITY_PLOT_FLOOR = 1.0e-15


def _canonical_target(target_name):
    """Return one canonical target in (e_out,p_out,gamma_out) ordering."""
    target = np.zeros(8, dtype=complex)
    states = outgoing_spin_states()
    if target_name == "W":
        labels = ((+1, -1, -1), (-1, +1, -1), (-1, -1, +1))
        for state_labels in labels:
            target[states.index(state_labels)] = 1.0 / np.sqrt(3.0)
        return target
    if target_name == "GHZ":
        for state_labels in ((-1, -1, -1), (+1, +1, +1)):
            target[states.index(state_labels)] = 1.0 / np.sqrt(2.0)
        return target
    raise ValueError(f"Unknown local-unitary target {target_name!r}.")


def _local_operator(rotations):
    """Return U_e tensor U_p tensor U_gamma in project basis order."""
    return np.kron(np.kron(rotations[0], rotations[1]), rotations[2])


def _search_local_unitaries(
    state,
    target_name,
    *,
    seed_label,
    restarts,
    max_iterations,
):
    """Numerically maximize canonical-target fidelity over three SU(2)s."""
    state = np.asarray(state, dtype=complex)
    norm = float(np.vdot(state, state).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The final state must have positive finite norm.")
    state = state / np.sqrt(norm)
    target = _canonical_target(target_name)
    if target_name == "W":
        bases = _local_eigenbasis_rotations(state)
    else:
        # Maximal-GHZ one-particle density matrices are degenerate, so their
        # numerical eigenvectors do not define a stable preferred basis.
        bases = tuple(np.eye(2, dtype=complex) for _ in range(3))

    def rotations(parameters):
        return tuple(
            _su2_zyz(parameters[3 * index:3 * index + 3]) @ bases[index]
            for index in range(3)
        )

    def objective(parameters):
        overlap = np.vdot(target, _local_operator(rotations(parameters)) @ state)
        return -float(abs(overlap) ** 2)

    seed = zlib.crc32(f"{target_name}:{seed_label}".encode("utf-8"))
    random_generator = np.random.default_rng(seed)
    starts = [np.zeros(9, dtype=float)]
    starts.extend(
        random_generator.uniform(-np.pi, np.pi, size=9)
        for _ in range(restarts - 1)
    )
    bounds = ((-np.pi, np.pi),) * 9
    best = None
    total_evaluations = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1.0e-14,
                "gtol": 1.0e-10,
                "maxiter": max_iterations,
            },
        )
        total_evaluations += int(result.nfev)
        if best is None or result.fun < best.fun:
            best = result
    local_rotations = rotations(best.x)
    transformed_state = _local_operator(local_rotations) @ state
    fidelity = float(np.clip(abs(np.vdot(target, transformed_state)) ** 2, 0.0, 1.0))
    return {
        "fidelity": fidelity,
        "one_minus_fidelity": 1.0 - fidelity,
        "fixed_basis_fidelity": float(abs(np.vdot(target, state)) ** 2),
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
        "optimizer_iterations": int(best.nit),
        "optimizer_evaluations": total_evaluations,
        "rotations": local_rotations,
    }


def _rotation_columns(target_name, result):
    """Flatten optimized rotations and Euler angles into auditable CSV fields."""
    columns = {}
    prefix = f"{target_name}_LU"
    for particle, rotation in zip(FINAL_PARTICLES, result["rotations"]):
        alpha, beta, gamma = _su2_zyz_angles(rotation)
        columns[f"{prefix}_{particle}_alpha_rad"] = alpha
        columns[f"{prefix}_{particle}_beta_rad"] = beta
        columns[f"{prefix}_{particle}_gamma_rad"] = gamma
        for row_index in range(2):
            for column_index in range(2):
                value = complex(rotation[row_index, column_index])
                stem = (
                    f"{prefix}_{particle}_U{row_index}{column_index}"
                )
                columns[f"{stem}_real"] = value.real
                columns[f"{stem}_imag"] = value.imag
    return columns


def _target_result_columns(target_name, result):
    """Return scalar search diagnostics followed by local rotations."""
    prefix = f"{target_name}_LU"
    columns = {
        f"{prefix}_fidelity": result["fidelity"],
        f"{prefix}_one_minus_fidelity": result["one_minus_fidelity"],
        f"{target_name}_fixed_basis_fidelity": result["fixed_basis_fidelity"],
        f"{prefix}_optimizer_success": result["optimizer_success"],
        f"{prefix}_optimizer_status": result["optimizer_status"],
        f"{prefix}_optimizer_message": result["optimizer_message"],
        f"{prefix}_optimizer_iterations": result["optimizer_iterations"],
        f"{prefix}_optimizer_evaluations": result["optimizer_evaluations"],
    }
    columns.update(_rotation_columns(target_name, result))
    return columns


def _fidelity_task(task):
    """Reconstruct one saved minimum and run both requested LU searches."""
    (
        row,
        lepton_name,
        scan_key,
        scan_objective_name,
        selected_for,
        target_names,
        restarts,
        max_iterations,
    ) = task
    minimum_id = int(row.get("local_minimum_id", -1))
    prefix = f"lepton_{lepton_name}_alpha_e_mix_proton_alpha_p_mix"
    output = {
        "status": "ok",
        "error": "",
        "lepton": lepton_name,
        "scan_key": scan_key,
        "scan_objective_name": scan_objective_name,
        "scan_target_observable": source_observable_name(scan_objective_name),
        "scan_objective_value": row.get(
            f"{prefix}_{scan_objective_name}", ""
        ),
        "local_minimum_id": minimum_id,
        "selected_for_W_screen": "W" in selected_for,
        "selected_for_GHZ_screen": "GHZ" in selected_for,
        "final_state_order": "e_out,p_out,gamma_out",
        "single_particle_basis_order": "-,+",
        "local_rotation_action": (
            "(U_e tensor U_p tensor U_gamma)|psi> -> canonical target"
        ),
        "sqrt_s": row.get("sqrt_s", ""),
        "theta_p_out": row.get("theta_p_out", ""),
        "theta_gamma_out": row.get("theta_gamma_out", ""),
        "qOut": row.get("qOut", ""),
        "phi_p_out": row.get("phi_p_out", ""),
        "phi_gamma_out": row.get("phi_gamma_out", ""),
        "alpha_e": row.get("alpha_e", ""),
        "alpha_p": row.get("alpha_p", ""),
    }
    try:
        source = dict(row)
        source.setdefault(
            "detail_id", f"{lepton_name}_{scan_key}_{minimum_id}"
        )
        state = _normalized_mixed_final_state(source)
        measures = entanglement_measures_from_state(state)
        for name, value in measures.items():
            output[f"reconstructed_{name}"] = value
        seed_label = f"{lepton_name}:{scan_key}:{minimum_id}"
        for target_name in target_names:
            result = _search_local_unitaries(
                state,
                target_name,
                seed_label=seed_label,
                restarts=restarts,
                max_iterations=max_iterations,
            )
            output.update(_target_result_columns(target_name, result))
    except Exception as exc:  # Preserve failures in the audit CSV.
        output["status"] = "error"
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def _read_csv(path):
    """Read one nonempty dictionary CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV contains no rows: {path}")
    return rows


def _write_csv(path, rows):
    """Write the union of dictionary fields in deterministic encounter order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                headers.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        if headers:
            writer.writeheader()
            writer.writerows(rows)
    return path


def _scan_input_path(definition, lepton_name):
    """Return one definition/species stage-1 minima file."""
    return (
        Path(definition.output_root)
        / lepton_name
        / "Data"
        / definition.key
        / "scan"
        / "local_minima.csv"
    )


def _scan_output_paths(definition, lepton_name):
    """Return per-minimum, summary, and distribution-plot output paths."""
    directory = (
        Path(definition.output_root)
        / lepton_name
        / "Data"
        / definition.key
        / "fidelity"
    )
    plot_directory = (
        Path(definition.output_root)
        / lepton_name
        / "Plots"
        / definition.key
        / "fidelity"
    )
    return {
        "details": directory / "local_unitary_fidelity_by_minimum.csv",
        "summary": directory / "local_unitary_fidelity_summary.csv",
        "plot": plot_directory / "local_unitary_fidelity_distribution.pdf",
        "cut_plot": (
            plot_directory
            / "local_unitary_fidelity_distribution_after_objective_cut.pdf"
        ),
    }


def _ranked_candidate_indices(rows, lepton_name, target_name, count):
    """Select minima with the smallest matching LU-invariant distance."""
    prefix = f"lepton_{lepton_name}_alpha_e_mix_proton_alpha_p_mix"
    observable = TARGET_SCREEN_OBSERVABLES[target_name]
    key = f"{prefix}_{observable}"
    ranked = []
    for index, row in enumerate(rows):
        value = _finite_float(row, key)
        if np.isfinite(value):
            ranked.append((value, index))
    if not ranked:
        raise ValueError(
            f"No finite {key} values are available for {target_name} screening."
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    if count is not None:
        ranked = ranked[:count]
    return tuple(index for _value, index in ranked)


def _selected_candidates(rows, definition, lepton_name):
    """Return the union of W- and GHZ-screened minima with selection labels."""
    selected_for = {}
    for target_name in TARGETS_TO_CHECK:
        for index in _ranked_candidate_indices(
            rows,
            lepton_name,
            target_name,
            CANDIDATES_PER_TARGET_PER_SCAN,
        ):
            selected_for.setdefault(index, set()).add(target_name)

    # Always include the best row for the scan's own minimization objective.
    prefix = f"lepton_{lepton_name}_alpha_e_mix_proton_alpha_p_mix"
    objective_key = f"{prefix}_{definition.objective_name}"
    finite_objective = []
    for index, row in enumerate(rows):
        value = _finite_float(row, objective_key)
        if np.isfinite(value):
            finite_objective.append((value, index))
    if finite_objective:
        _value, best_index = min(finite_objective)
        selected_for.setdefault(best_index, set()).add("scan_objective")
    return tuple(
        (index, tuple(sorted(labels)))
        for index, labels in sorted(selected_for.items())
    )


def _run_tasks(tasks):
    """Run independent minima with a Windows-safe sequential fallback."""
    workers = min(FIDELITY_WORKERS, len(tasks))
    if workers <= 1:
        return [_fidelity_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_fidelity_task, tasks, chunksize=1))
    except (OSError, PermissionError, BrokenProcessPool):
        return [_fidelity_task(task) for task in tasks]


def _finite_float(row, key):
    """Parse one finite result value or return NaN."""
    try:
        value = float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _scan_summary(
    definition,
    lepton_name,
    input_path,
    details_path,
    total_minima,
    detail_rows,
):
    """Summarize maximum W and GHZ LU fidelities for one saved scan."""
    successful = [row for row in detail_rows if row["status"] == "ok"]
    summary = {
        "status": "ok" if successful else "error",
        "lepton": lepton_name,
        "scan_key": definition.key,
        "scan_objective_name": definition.objective_name,
        "scan_target_observable": source_observable_name(
            definition.objective_name
        ),
        "input_path": str(input_path),
        "details_path": str(details_path),
        "total_local_minima": total_minima,
        "checked_candidate_minima": len(detail_rows),
        "successful_candidate_minima": len(successful),
        "failed_candidate_minima": len(detail_rows) - len(successful),
        "candidate_limit_per_target": (
            "all"
            if CANDIDATES_PER_TARGET_PER_SCAN is None
            else CANDIDATES_PER_TARGET_PER_SCAN
        ),
        "exhaustive": CANDIDATES_PER_TARGET_PER_SCAN is None,
        "local_unitary_restarts": LOCAL_UNITARY_RESTARTS,
        "final_state_order": "e_out,p_out,gamma_out",
        "single_particle_basis_order": "-,+",
    }
    for target_name in TARGETS_TO_CHECK:
        fidelity_key = f"{target_name}_LU_fidelity"
        candidates = [
            (_finite_float(row, fidelity_key), row)
            for row in successful
        ]
        candidates = [item for item in candidates if np.isfinite(item[0])]
        if not candidates:
            summary[f"max_{fidelity_key}"] = ""
            summary[f"best_{target_name}_local_minimum_id"] = ""
            continue
        values = np.asarray([item[0] for item in candidates], dtype=float)
        summary[f"min_{fidelity_key}"] = float(np.min(values))
        summary[f"mean_{fidelity_key}"] = float(np.mean(values))
        summary[f"median_{fidelity_key}"] = float(np.median(values))
        summary[f"std_{fidelity_key}"] = float(np.std(values))
        for percentile in (10, 25, 75, 90, 99):
            summary[f"p{percentile}_{fidelity_key}"] = float(
                np.percentile(values, percentile)
            )
        for threshold, label in (
            (0.9, "0p9"),
            (0.99, "0p99"),
            (0.999, "0p999"),
        ):
            passing = int(np.count_nonzero(values >= threshold))
            summary[f"count_{fidelity_key}_ge_{label}"] = passing
            summary[f"fraction_{fidelity_key}_ge_{label}"] = (
                passing / len(values)
            )
        fidelity, best = max(candidates, key=lambda item: item[0])
        summary[f"max_{fidelity_key}"] = fidelity
        summary[f"min_{target_name}_LU_one_minus_fidelity"] = 1.0 - fidelity
        summary[f"best_{target_name}_local_minimum_id"] = best[
            "local_minimum_id"
        ]
        summary[f"best_{target_name}_fixed_basis_fidelity"] = best[
            f"{target_name}_fixed_basis_fidelity"
        ]
        summary[f"best_{target_name}_scan_objective_value"] = best[
            "scan_objective_value"
        ]
    return summary


def _missing_summary(definition, lepton_name, input_path):
    """Return an explicit summary row for one unavailable scan input."""
    return {
        "status": "missing_input",
        "lepton": lepton_name,
        "scan_key": definition.key,
        "scan_objective_name": definition.objective_name,
        "scan_target_observable": source_observable_name(
            definition.objective_name
        ),
        "input_path": str(input_path),
        "details_path": "",
        "distribution_plot_path": "",
        "cut_distribution_plot_path": "",
        "objective_cut_above_minimum": POLARIZATION_CLUSTER_CUT,
        "cut_retained_minima": 0,
        "total_local_minima": 0,
        "checked_candidate_minima": 0,
        "successful_candidate_minima": 0,
        "failed_candidate_minima": 0,
        "candidate_limit_per_target": (
            "all"
            if CANDIDATES_PER_TARGET_PER_SCAN is None
            else CANDIDATES_PER_TARGET_PER_SCAN
        ),
        "exhaustive": CANDIDATES_PER_TARGET_PER_SCAN is None,
        "local_unitary_restarts": LOCAL_UNITARY_RESTARTS,
        "final_state_order": "e_out,p_out,gamma_out",
        "single_particle_basis_order": "-,+",
    }


def _write_scan_distribution_plot(
    definition,
    lepton_name,
    detail_rows,
    path,
    selection_note="all saved local minima",
):
    """Plot full LU-infidelity histograms and CDFs on logarithmic axes."""
    successful = [row for row in detail_rows if row.get("status") == "ok"]
    if not successful:
        raise ValueError(
            f"No successful fidelity rows for {lepton_name}/{definition.key}."
        )
    plt, PdfPages = require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matching_target = (
        definition.key if definition.key in TARGETS_TO_CHECK else None
    )
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(13.5, 9.5),
            constrained_layout=True,
            sharex="col",
        )
        for column, target_name in enumerate(TARGETS_TO_CHECK):
            key = f"{target_name}_LU_fidelity"
            fidelities = np.asarray(
                [_finite_float(row, key) for row in successful], dtype=float
            )
            fidelities = fidelities[np.isfinite(fidelities)]
            if fidelities.size == 0:
                raise ValueError(
                    f"No finite {key} values for {lepton_name}/{definition.key}."
                )
            raw_infidelities = 1.0 - fidelities
            infidelities = np.maximum(
                raw_infidelities, INFIDELITY_PLOT_FLOOR
            )
            median = float(np.median(infidelities))
            p10, p90 = np.percentile(infidelities, (10.0, 90.0))
            minimum = float(np.min(infidelities))
            count_099 = int(np.count_nonzero(fidelities >= 0.99))
            floored = int(
                np.count_nonzero(raw_infidelities < INFIDELITY_PLOT_FLOOR)
            )

            histogram_ax = axes[0, column]
            histogram_ax.hist(
                infidelities,
                bins=np.logspace(
                    np.log10(INFIDELITY_PLOT_FLOOR), 0.0, 61
                ),
                color="tab:blue",
                alpha=0.78,
                edgecolor="white",
                linewidth=0.35,
            )
            histogram_ax.axvspan(
                p10, p90, color="tab:orange", alpha=0.16,
                label="10th-90th percentile",
            )
            histogram_ax.axvline(
                median, color="tab:orange", linewidth=1.8,
                label=f"median $1-F$={median:.3e}",
            )
            histogram_ax.axvline(
                minimum, color="tab:red", linestyle="--", linewidth=1.5,
                label=f"minimum $1-F$={minimum:.3e}",
            )
            histogram_ax.set_xscale("log")
            histogram_ax.set_xlim(INFIDELITY_PLOT_FLOOR, 1.0)
            histogram_ax.set_ylabel("local minima")
            histogram_ax.set_title(
                f"Canonical {target_name} LU infidelity"
                + (" (matching target)" if target_name == matching_target else "")
            )
            histogram_ax.legend(fontsize=8)
            histogram_ax.text(
                0.02,
                0.97,
                (
                    f"N={len(fidelities)}\n"
                    f"F>=0.99: {count_099}/{len(fidelities)}\n"
                    f"at 1e-15 floor: {floored}"
                ),
                transform=histogram_ax.transAxes,
                va="top",
                fontsize=9,
                family="monospace",
            )

            cdf_ax = axes[1, column]
            sorted_values = np.sort(infidelities)
            cumulative = (
                np.arange(1, len(infidelities) + 1, dtype=float)
                / len(infidelities)
            )
            cdf_ax.step(
                sorted_values,
                cumulative,
                where="post",
                color="tab:blue",
                linewidth=1.5,
            )
            cdf_ax.axvline(
                0.25, color="gray", linestyle=":", linewidth=1.0,
                label="exact W-GHZ cross-orbit $1-F=0.25$",
            )
            cdf_ax.axvline(
                0.01, color="black", linestyle="--", linewidth=1.0,
                label="$F=0.99$ threshold",
            )
            cdf_ax.set_xscale("log")
            cdf_ax.set_xlim(INFIDELITY_PLOT_FLOOR, 1.0)
            cdf_ax.set_ylim(0.0, 1.02)
            cdf_ax.set_xlabel(rf"$1-F_{{\mathrm{{LU}}}}({target_name})$")
            cdf_ax.set_ylabel("fraction of minima")
            cdf_ax.grid(alpha=0.25)
            cdf_ax.legend(fontsize=8, loc="upper left")

        fig.suptitle(
            f"{lepton_name} / {definition.key} gradient minima: "
            "local-unitary fidelity distributions\n"
            r"$(U_e\otimes U_p\otimes U_\gamma)|\psi\rangle$; "
            r"final order $(e_{out},p_{out},\gamma_{out})$"
            f"\nselection: {selection_note}"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _rows_within_objective_cut(detail_rows, objective_cut):
    """Apply the stage-3 objective-above-global-minimum selection."""
    successful = [row for row in detail_rows if row.get("status") == "ok"]
    values = np.asarray(
        [_finite_float(row, "scan_objective_value") for row in successful],
        dtype=float,
    )
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("No finite scan objective values are available.")
    optimum = float(np.min(values[finite]))
    selected = [
        row
        for row, value in zip(successful, values)
        if np.isfinite(value)
        and value - optimum <= objective_cut + 1.0e-12
    ]
    if not selected:
        raise RuntimeError("The objective cut retained no fidelity rows.")
    return selected, optimum


def _write_summary_plot(summary_rows, path):
    """Plot scan-level LU-infidelity summaries on logarithmic axes."""
    plt, PdfPages = require_matplotlib()
    complete = [row for row in summary_rows if row.get("status") == "ok"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5), constrained_layout=True)
        labels = [f"{row['lepton']} / {row['scan_key']}" for row in complete]
        positions = np.arange(len(complete))
        for ax, target_name in zip(axes, TARGETS_TO_CHECK):
            stem = f"{target_name}_LU_fidelity"
            fidelity_p10 = np.asarray(
                [_finite_float(row, f"p10_{stem}") for row in complete]
            )
            fidelity_p90 = np.asarray(
                [_finite_float(row, f"p90_{stem}") for row in complete]
            )
            fidelity_medians = np.asarray(
                [_finite_float(row, f"median_{stem}") for row in complete]
            )
            fidelity_maxima = np.asarray(
                [_finite_float(row, f"max_{stem}") for row in complete]
            )
            infidelity_p10 = np.maximum(
                1.0 - fidelity_p90, INFIDELITY_PLOT_FLOOR
            )
            infidelity_p90 = np.maximum(
                1.0 - fidelity_p10, INFIDELITY_PLOT_FLOOR
            )
            median_infidelity = np.maximum(
                1.0 - fidelity_medians, INFIDELITY_PLOT_FLOOR
            )
            best_infidelity = np.maximum(
                1.0 - fidelity_maxima, INFIDELITY_PLOT_FLOOR
            )
            finite_band = np.isfinite(infidelity_p10) & np.isfinite(
                infidelity_p90
            )
            ax.hlines(
                positions[finite_band],
                infidelity_p10[finite_band],
                infidelity_p90[finite_band],
                color="tab:blue",
                linewidth=3.0,
                alpha=0.45,
                label="10th-90th percentile",
            )
            finite_median = np.isfinite(median_infidelity)
            ax.scatter(
                median_infidelity[finite_median],
                positions[finite_median],
                s=52,
                color="tab:blue",
                label="median",
                zorder=3,
            )
            finite_maximum = np.isfinite(best_infidelity)
            ax.scatter(
                best_infidelity[finite_maximum],
                positions[finite_maximum],
                s=75,
                marker="*",
                color="tab:red",
                label="minimum $1-F$",
                zorder=4,
            )
            ax.axvline(0.25, color="gray", linestyle=":", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_xlim(INFIDELITY_PLOT_FLOOR, 1.0)
            ax.set_yticks(positions, labels=labels)
            ax.invert_yaxis()
            ax.set_xlabel(rf"$1-F_{{\mathrm{{LU}}}}({target_name})$")
            ax.grid(axis="x", which="both", alpha=0.3)
            ax.set_title(
                f"Canonical {target_name} infidelity after local rotations"
            )
            ax.legend(loc="lower left", fontsize=8)
        if not complete:
            for ax in axes:
                ax.text(0.5, 0.5, "No completed scan summaries", ha="center")
        saved_limits = sorted(
            {
                str(row.get("candidate_limit_per_target", "unknown"))
                for row in complete
            }
        )
        saved_restarts = sorted(
            {
                str(row.get("local_unitary_restarts", "unknown"))
                for row in complete
            }
        )
        mode = (
            "all minima"
            if saved_limits == ["all"]
            else "candidate limits/target=" + ",".join(saved_limits)
        )
        fig.suptitle(
            "Gradient-scan local-unitary fidelity summary\n"
            f"candidate mode: {mode}; starts/target="
            + ",".join(saved_restarts)
        )
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _validate_controls():
    """Validate explicit controls before reading or writing scan artifacts."""
    definitions.selected_definitions(SCANS_TO_CHECK)
    definitions.validated_leptons(LEPTONS_TO_CHECK)
    definitions.validated_workers(FIDELITY_WORKERS)
    if tuple(TARGETS_TO_CHECK) != ("W", "GHZ"):
        raise ValueError("TARGETS_TO_CHECK must be exactly ('W', 'GHZ').")
    if not isinstance(LOCAL_UNITARY_RESTARTS, int) or isinstance(
        LOCAL_UNITARY_RESTARTS, bool
    ) or LOCAL_UNITARY_RESTARTS < 1:
        raise ValueError("LOCAL_UNITARY_RESTARTS must be a positive integer.")
    if not isinstance(LOCAL_UNITARY_MAX_ITERATIONS, int) or isinstance(
        LOCAL_UNITARY_MAX_ITERATIONS, bool
    ) or LOCAL_UNITARY_MAX_ITERATIONS < 1:
        raise ValueError(
            "LOCAL_UNITARY_MAX_ITERATIONS must be a positive integer."
        )
    if CANDIDATES_PER_TARGET_PER_SCAN is not None and (
        not isinstance(CANDIDATES_PER_TARGET_PER_SCAN, int)
        or isinstance(CANDIDATES_PER_TARGET_PER_SCAN, bool)
        or CANDIDATES_PER_TARGET_PER_SCAN < 1
    ):
        raise ValueError(
            "CANDIDATES_PER_TARGET_PER_SCAN must be None or a positive integer."
        )
    for name, value in (
        ("SKIP_MISSING_SCAN_INPUTS", SKIP_MISSING_SCAN_INPUTS),
        ("REMAKE_SUMMARY_PLOT_FROM_CSV", REMAKE_SUMMARY_PLOT_FROM_CSV),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool.")


def run_fidelity_checks():
    """Run or redraw the configured local-unitary fidelity summaries."""
    _validate_controls()
    if REMAKE_SUMMARY_PLOT_FROM_CSV:
        summary_rows = _read_csv(SUMMARY_CSV)
        plot_path = _write_summary_plot(summary_rows, SUMMARY_PDF)
        report = (
            "Remade gradient local-unitary fidelity summary plot\n"
            f"  source summary: {SUMMARY_CSV}\n"
            f"  summary plot: {plot_path}\n"
        )
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(report, encoding="utf-8")
        print_console_text(report)
        return summary_rows

    scan_definitions = definitions.selected_definitions(SCANS_TO_CHECK)
    leptons = definitions.validated_leptons(LEPTONS_TO_CHECK)
    summary_rows = []
    report_lines = ["Gradient local-unitary W/GHZ fidelity check"]
    total_jobs = len(scan_definitions) * len(leptons)
    job_index = 0
    for lepton_name in leptons:
        for definition in scan_definitions:
            job_index += 1
            input_path = _scan_input_path(definition, lepton_name)
            print_console_text(
                f"[{job_index}/{total_jobs}] {lepton_name} / "
                f"{definition.key}\n"
            )
            if not input_path.exists():
                if not SKIP_MISSING_SCAN_INPUTS:
                    raise FileNotFoundError(
                        f"Missing gradient minima input: {input_path}"
                    )
                summary_rows.append(
                    _missing_summary(definition, lepton_name, input_path)
                )
                report_lines.append(f"  missing input: {input_path}")
                continue

            rows = _read_csv(input_path)
            selected = _selected_candidates(rows, definition, lepton_name)
            tasks = [
                (
                    rows[index],
                    lepton_name,
                    definition.key,
                    definition.objective_name,
                    labels,
                    TARGETS_TO_CHECK,
                    LOCAL_UNITARY_RESTARTS,
                    LOCAL_UNITARY_MAX_ITERATIONS,
                )
                for index, labels in selected
            ]
            detail_rows = _run_tasks(tasks)
            output_paths = _scan_output_paths(definition, lepton_name)
            details_path = _write_csv(output_paths["details"], detail_rows)
            distribution_plot_path = _write_scan_distribution_plot(
                definition,
                lepton_name,
                detail_rows,
                output_paths["plot"],
            )
            cut_rows, cut_optimum = _rows_within_objective_cut(
                detail_rows,
                POLARIZATION_CLUSTER_CUT,
            )
            cut_selection_note = (
                f"{definition.objective_name} - minimum <= "
                f"{POLARIZATION_CLUSTER_CUT:g}; retained "
                f"{len(cut_rows)}/{len(detail_rows)}; minimum="
                f"{cut_optimum:.8g}"
            )
            cut_distribution_plot_path = _write_scan_distribution_plot(
                definition,
                lepton_name,
                cut_rows,
                output_paths["cut_plot"],
                selection_note=cut_selection_note,
            )
            summary = _scan_summary(
                definition,
                lepton_name,
                input_path,
                details_path,
                len(rows),
                detail_rows,
            )
            summary["distribution_plot_path"] = str(
                distribution_plot_path
            )
            summary["cut_distribution_plot_path"] = str(
                cut_distribution_plot_path
            )
            summary["objective_cut_above_minimum"] = (
                POLARIZATION_CLUSTER_CUT
            )
            summary["cut_global_minimum"] = cut_optimum
            summary["cut_retained_minima"] = len(cut_rows)
            _write_csv(output_paths["summary"], [summary])
            summary_rows.append(summary)
            report_lines.extend(
                (
                    (
                        f"  checked {len(detail_rows)}/{len(rows)} minima; "
                        f"failures={summary['failed_candidate_minima']}"
                    ),
                    (
                        "  W LU fidelity mean/median/max: "
                        f"{summary.get('mean_W_LU_fidelity', '')} / "
                        f"{summary.get('median_W_LU_fidelity', '')} / "
                        f"{summary.get('max_W_LU_fidelity', '')}"
                    ),
                    (
                        "  GHZ LU fidelity mean/median/max: "
                        f"{summary.get('mean_GHZ_LU_fidelity', '')} / "
                        f"{summary.get('median_GHZ_LU_fidelity', '')} / "
                        f"{summary.get('max_GHZ_LU_fidelity', '')}"
                    ),
                    f"  details: {details_path}",
                    f"  scan summary: {output_paths['summary']}",
                    f"  distribution plot: {distribution_plot_path}",
                    (
                        f"  post-cut distribution plot: "
                        f"{cut_distribution_plot_path}"
                    ),
                )
            )

    summary_path = _write_csv(SUMMARY_CSV, summary_rows)
    plot_path = _write_summary_plot(summary_rows, SUMMARY_PDF)
    report_lines.extend(
        (
            f"Combined summary: {summary_path}",
            f"Combined summary plot: {plot_path}",
        )
    )
    report = "\n".join(report_lines) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)
    return summary_rows


def main():
    """Run using only the explicit script controls."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "GradientLocalUnitaryFidelity.py accepts no command-line "
            "arguments; edit its explicit controls instead."
        )
    run_fidelity_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
