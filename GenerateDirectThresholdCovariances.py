"""Generate direct fixed-width full covariances for threshold-selected minima.

For every retained electron minimum satisfying ``D < 0.01``, directly trace
the ``D_i + 0.02`` contour, following the earlier CEGAMMA
``contour_width_comparison`` benchmark.  Failed rays are replaced with new
deterministic directions until 1,536 successful boundary points are retained,
then both the contour and its periodic-aware full covariance are saved.  The
output is separate from the historical fixed-delta contour files and is
resumable at individual-minimum granularity.

The default production command is

    python GenerateDirectThresholdCovariances.py --threshold 0.01
"""

import argparse
import csv
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool
from GaussianSmearWorker import full_covariance_from_contour
from GradientContourWorker import configuration_complete_contour_task


STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTON_NAME = "electron"
DEFAULT_THRESHOLD = 0.01
DIRECT_DELTA = 0.02
TARGET_SUCCESS_COUNT = 1536
MAXIMUM_ATTEMPT_COUNT = 20 * TARGET_SUCCESS_COUNT
REPLACEMENT_BATCH_SIZE = 256
SCAN_DIMENSION = 8
DIRECT_BISECTION_ITERATIONS = (
    gradient_tool.CONFIG_CONTOUR_BISECTION_ITERATIONS
)
SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"
OUTPUT_ROOT = (
    SCAN_ROOT / LEPTON_NAME / "Data"
    / "contour_width_comparison_delta0.02_all_threshold0.01"
)
SUMMARY_PATH = OUTPUT_ROOT / "generation_summary.csv"
OUTPUT_VARIANT = "delta0.02_all_threshold0.01"
PREVIOUS_OUTPUT_VARIANT = None
PREVIOUS_SUCCESS_COUNT = 0
INCREMENTAL_MODE = False

OBJECTIVE_COLUMNS = {
    "W": "final_D_W",
    "GHZ": "lepton_electron_alpha_e_mix_proton_alpha_p_mix_dGHZ",
    "CEP": "final_one_minus_C_e_p",
    "CEGAMMA": "final_one_minus_C_e_gamma",
    "CPGAMMA": "final_one_minus_C_p_gamma",
}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _covariance_path(state_key):
    return _state_output_dir(state_key) / "covariances.csv"


def _state_output_dir(state_key):
    return (
        SCAN_ROOT / LEPTON_NAME / "Data" / state_key
        / "contour_width_comparison" / OUTPUT_VARIANT
    )


def _contour_path(state_key, minimum_id):
    return (
        _state_output_dir(state_key) / "local_minima"
        / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
    )


def _contour_path_for_variant(state_key, minimum_id, variant):
    return (
        SCAN_ROOT / LEPTON_NAME / "Data" / state_key
        / "contour_width_comparison" / variant / "local_minima"
        / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
    )


def _load_previous_contour(descriptor):
    """Load and strictly validate the preceding incremental contour."""
    if PREVIOUS_SUCCESS_COUNT == 0:
        return np.empty((0, SCAN_DIMENSION), dtype=float), 0
    if not PREVIOUS_OUTPUT_VARIANT:
        raise RuntimeError("An incremental predecessor variant is required.")
    path = _contour_path_for_variant(
        descriptor["state"],
        descriptor["minimum_id"],
        PREVIOUS_OUTPUT_VARIANT,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing incremental predecessor contour: {path}"
        )
    metadata = None
    points = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] == "minimum":
                metadata = row
            elif row["record_type"] == "sample":
                points.append(
                    [float(row[f"u{index}"]) for index in range(SCAN_DIMENSION)]
                )
    if metadata is None:
        raise ValueError(f"Predecessor contour has no metadata row: {path}")
    points = np.asarray(points, dtype=float).reshape((-1, SCAN_DIMENSION))
    valid = (
        int(metadata["local_minimum_id"]) == int(descriptor["minimum_id"])
        and metadata["objective_name"] == descriptor["objective_name"]
        and np.isclose(
            float(metadata["contour_delta"]), DIRECT_DELTA,
            rtol=0.0, atol=1.0e-15,
        )
        and int(metadata["target_success_count"]) == PREVIOUS_SUCCESS_COUNT
        and int(metadata["contour_point_count"]) == PREVIOUS_SUCCESS_COUNT
        and len(points) == PREVIOUS_SUCCESS_COUNT
        and all(
            np.isclose(
                float(metadata[f"center_u{index}"]),
                descriptor["center"][index],
                rtol=0.0,
                atol=1.0e-12,
            )
            for index in range(SCAN_DIMENSION)
        )
    )
    if not valid:
        raise ValueError(f"Incompatible incremental predecessor: {path}")
    boundary_count = int(metadata.get("domain_boundary_point_count", 0) or 0)
    return points, boundary_count


def audit_incremental_predecessors(descriptors):
    """Validate every required predecessor before starting a top-up stage."""
    if PREVIOUS_SUCCESS_COUNT == 0:
        return 0
    for descriptor in descriptors:
        _load_previous_contour(descriptor)
    return len(descriptors)


def _fieldnames():
    return (
        "state",
        "local_minimum_id",
        "status",
        "objective_name",
        "objective_file_tag",
        "absolute_objective",
        "absolute_threshold",
        "contour_delta",
        "target_success_count",
        "initial_direction_count",
        "initial_success_count",
        "attempted_direction_count",
        "replacement_direction_count",
        "contour_point_count",
        "initial_success_fraction",
        "overall_success_fraction",
        "physics_evaluation_count",
        "direct_contour_seconds",
        "covariance_rank",
        "covariance_min_eigenvalue",
        "covariance_max_eigenvalue",
        *(f"center_u{index}" for index in range(SCAN_DIMENSION)),
        *(
            f"covariance_u{row}_u{column}"
            for row in range(SCAN_DIMENSION)
            for column in range(SCAN_DIMENSION)
        ),
    )


FIELDNAMES = _fieldnames()


def _append_csv_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _write_contour(descriptor, points, diagnostics):
    """Atomically write one direct contour in the established CSV schema."""
    path = _contour_path(descriptor["state"], descriptor["minimum_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    base = {
        "objective_name": descriptor["objective_name"],
        "objective_file_tag": descriptor["objective_file_tag"],
        "local_minimum_index": descriptor["minimum_id"],
        "local_minimum_id": descriptor["minimum_id"],
        "selection_absolute_threshold": DEFAULT_THRESHOLD,
        "contour_delta": DIRECT_DELTA,
        "configured_direction_count": TARGET_SUCCESS_COUNT,
        "target_success_count": TARGET_SUCCESS_COUNT,
        "previous_success_count": PREVIOUS_SUCCESS_COUNT,
        "added_success_count": diagnostics["added_success_count"],
        "previous_output_variant": PREVIOUS_OUTPUT_VARIANT or "",
        "initial_success_count": diagnostics["initial_success_count"],
        "attempted_direction_count": diagnostics["attempted_direction_count"],
        "replacement_direction_count": diagnostics["replacement_direction_count"],
        "contour_point_count": len(points),
        "domain_boundary_point_count": diagnostics.get(
            "domain_boundary_point_count", 0
        ),
        **{
            f"center_u{coordinate}": float(descriptor["center"][coordinate])
            for coordinate in range(SCAN_DIMENSION)
        },
    }
    empty_point = {f"u{coordinate}": "" for coordinate in range(SCAN_DIMENSION)}
    physical_names = tuple(gradient_tool._plot_coordinate_values(
        descriptor["center"]
    ))
    fieldnames = (
        *base,
        "record_type",
        "contour_sample_id",
        *(f"u{coordinate}" for coordinate in range(SCAN_DIMENSION)),
        *physical_names,
    )
    with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            **base,
            "record_type": "minimum",
            "contour_sample_id": "",
            **empty_point,
            **{name: "" for name in physical_names},
        })
        for sample_index, point in enumerate(points):
            writer.writerow({
                **base,
                "record_type": "sample",
                "contour_sample_id": sample_index,
                **{
                    f"u{coordinate}": float(point[coordinate])
                    for coordinate in range(SCAN_DIMENSION)
                },
                **gradient_tool._plot_coordinate_values(point),
            })
    os.replace(temporary_path, path)


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer fields for empty CSV {path}.")
        fieldnames = tuple(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_descriptors(states, threshold, workers):
    selected_definitions = {
        definition.key: definition
        for definition in definitions.selected_definitions(tuple(states))
    }
    descriptors = []
    for state_key in states:
        definition = selected_definitions[state_key]
        gradient_tool.configure_scan(
            definition,
            leptons_to_process=(LEPTON_NAME,),
            gradient_workers=workers,
        )
        path = (
            SCAN_ROOT / LEPTON_NAME / "Data" / state_key
            / "cluster" / "clustered_minima.csv"
        )
        objective_column = OBJECTIVE_COLUMNS[state_key]
        rows = [
            row for row in _read_csv(path)
            if row["within_polarization_cluster_cut"] == "True"
            and float(row[objective_column]) < threshold
        ]
        rows.sort(key=lambda row: int(row["local_minimum_id"]))
        for row in rows:
            objective = float(row[objective_column])
            descriptors.append({
                "state": state_key,
                "definition": definition,
                "row": row,
                "minimum_id": row["local_minimum_id"],
                "objective": objective,
                "objective_name": definition.objective_name,
                "objective_file_tag": definition.file_tag,
                "coordinate_system": definition.coordinate_system,
                "contour_delta": DIRECT_DELTA,
                "center": gradient_tool._unit_point_from_minimum_row(row),
            })
        print(
            f"{state_key}: selected {len(rows)} minima at absolute D < "
            f"{threshold:g}",
            flush=True,
        )
    return descriptors


def _completed_rows(descriptors, threshold):
    expected = {
        (item["state"], item["minimum_id"]): item
        for item in descriptors
    }
    completed = {}
    for state_key in STATES:
        path = _covariance_path(state_key)
        if not path.exists():
            continue
        for row in _read_csv(path):
            key = (row["state"], row["local_minimum_id"])
            descriptor = expected.get(key)
            if descriptor is None or row["status"] != "ok":
                continue
            valid = (
                np.isclose(
                    float(row["absolute_threshold"]), threshold,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(row["contour_delta"]),
                    DIRECT_DELTA,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and int(row["target_success_count"]) == TARGET_SUCCESS_COUNT
                and int(row["contour_point_count"]) == TARGET_SUCCESS_COUNT
                and row["objective_name"] == descriptor["objective_name"]
                and (
                    PREVIOUS_SUCCESS_COUNT == 0
                    or (
                        int(row["previous_success_count"])
                        == PREVIOUS_SUCCESS_COUNT
                        and row["previous_output_variant"]
                        == PREVIOUS_OUTPUT_VARIANT
                    )
                )
                and _contour_path(
                    descriptor["state"], descriptor["minimum_id"]
                ).exists()
                and all(
                    np.isclose(
                        float(row[f"center_u{index}"]),
                        descriptor["center"][index],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    for index in range(SCAN_DIMENSION)
                )
            )
            if not valid:
                raise ValueError(
                    "Existing covariance checkpoint is incompatible with the "
                    f"current run: {path}, minimum {row['local_minimum_id']}"
                )
            completed[key] = row
    return completed


def _direct_contour_directions(seed, start_count=0):
    """Return a nested deterministic slice ending at the configured count."""
    if TARGET_SUCCESS_COUNT < 2 * SCAN_DIMENSION:
        raise ValueError(
            "TARGET_SUCCESS_COUNT must include the 16 signed coordinate axes."
        )
    axes = np.vstack((np.eye(SCAN_DIMENSION), -np.eye(SCAN_DIMENSION)))
    random_count = TARGET_SUCCESS_COUNT - len(axes)
    rng = np.random.default_rng(seed)
    random_directions = rng.normal(size=(random_count, SCAN_DIMENSION))
    if random_count:
        random_directions /= np.linalg.norm(
            random_directions, axis=1, keepdims=True,
        )
    directions = np.vstack((axes, random_directions))
    return directions[int(start_count):TARGET_SUCCESS_COUNT]


def _worker_task(descriptor, task_index):
    directions = _direct_contour_directions(
        gradient_tool.ENTANGLEMENT_GRADIENT_RANDOM_SEED
        + int(descriptor["minimum_id"]),
        start_count=PREVIOUS_SUCCESS_COUNT,
    )
    added_success_count = TARGET_SUCCESS_COUNT - PREVIOUS_SUCCESS_COUNT
    return (
        task_index,
        descriptor["center"],
        descriptor["objective"],
        LEPTON_NAME,
        gradient_tool.GRADIENT_LEPTON_SPECS[LEPTON_NAME]["mass"],
        descriptor["objective_name"],
        directions,
        DIRECT_DELTA,
        gradient_tool.CONFIG_CONTOUR_INITIAL_RADIUS,
        DIRECT_BISECTION_ITERATIONS,
        added_success_count,
        20 * added_success_count,
        (
            gradient_tool.ENTANGLEMENT_GRADIENT_RANDOM_SEED
            + 10_000_000
            + 100_000 * STATES.index(descriptor["state"])
            + int(descriptor["minimum_id"])
            + 1_000 * PREVIOUS_SUCCESS_COUNT
        ),
        REPLACEMENT_BATCH_SIZE,
        descriptor["coordinate_system"],
    )


def _result_row(descriptor, points, diagnostics, elapsed_seconds):
    points = np.asarray(points, dtype=float).reshape((-1, SCAN_DIMENSION))
    covariance = np.full((SCAN_DIMENSION, SCAN_DIMENSION), np.nan)
    rank = 0
    min_eigenvalue = np.nan
    max_eigenvalue = np.nan
    status = "insufficient_points"
    if len(points) >= 2:
        covariance = full_covariance_from_contour(
            descriptor["center"],
            points,
            periodic_coordinates=(
                (6, 7)
                if descriptor["coordinate_system"] == "cartesian"
                else (4, 5, 6, 7)
            ),
        )
        if np.all(np.isfinite(covariance)):
            eigenvalues = np.linalg.eigvalsh(covariance)
            rank = int(np.linalg.matrix_rank(covariance))
            min_eigenvalue = float(eigenvalues[0])
            max_eigenvalue = float(eigenvalues[-1])
            status = (
                "ok"
                if rank == SCAN_DIMENSION and len(points) == TARGET_SUCCESS_COUNT
                else "degenerate" if rank < SCAN_DIMENSION
                else "incomplete"
            )
        else:
            status = "nonfinite_covariance"
    row = {
        "state": descriptor["state"],
        "local_minimum_id": descriptor["minimum_id"],
        "status": status,
        "objective_name": descriptor["objective_name"],
        "objective_file_tag": descriptor["objective_file_tag"],
        "absolute_objective": descriptor["objective"],
        "absolute_threshold": DEFAULT_THRESHOLD,
        "contour_delta": DIRECT_DELTA,
        "target_success_count": TARGET_SUCCESS_COUNT,
        "initial_direction_count": diagnostics["added_success_count"],
        "initial_success_count": diagnostics["initial_success_count"],
        "attempted_direction_count": diagnostics["attempted_direction_count"],
        "replacement_direction_count": diagnostics["replacement_direction_count"],
        "contour_point_count": len(points),
        "initial_success_fraction": (
            diagnostics["initial_success_count"]
            / diagnostics["added_success_count"]
        ),
        "overall_success_fraction": (
            diagnostics["total_success_count"]
            / diagnostics["attempted_direction_count"]
        ),
        "physics_evaluation_count": diagnostics["physics_evaluation_count"],
        "direct_contour_seconds": elapsed_seconds,
        "covariance_rank": rank,
        "covariance_min_eigenvalue": min_eigenvalue,
        "covariance_max_eigenvalue": max_eigenvalue,
        **{
            f"center_u{index}": float(descriptor["center"][index])
            for index in range(SCAN_DIMENSION)
        },
        **{
            f"covariance_u{row}_u{column}": float(covariance[row, column])
            for row in range(SCAN_DIMENSION)
            for column in range(SCAN_DIMENSION)
        },
    }
    if INCREMENTAL_MODE:
        row.update({
            "previous_success_count": PREVIOUS_SUCCESS_COUNT,
            "added_success_count": diagnostics["added_success_count"],
            "previous_output_variant": PREVIOUS_OUTPUT_VARIANT or "",
        })
    if descriptor["coordinate_system"] == "cartesian":
        row["domain_boundary_point_count"] = diagnostics[
            "domain_boundary_point_count"
        ]
        row["domain_boundary_point_fraction"] = (
            diagnostics["domain_boundary_point_count"] / len(points)
            if len(points) else np.nan
        )
    return row


def _write_summary(descriptors, rows_by_key, started, workers):
    output = []
    for state_key in STATES:
        selected = [item for item in descriptors if item["state"] == state_key]
        rows = [
            rows_by_key[(item["state"], item["minimum_id"])]
            for item in selected
            if (item["state"], item["minimum_id"]) in rows_by_key
        ]
        ok_rows = [row for row in rows if row["status"] == "ok"]
        success_fractions = np.asarray(
            [float(row["initial_success_fraction"]) for row in ok_rows],
            dtype=float,
        )
        attempts = np.asarray(
            [int(row["attempted_direction_count"]) for row in ok_rows],
            dtype=float,
        )
        output.append({
            "state": state_key,
            "absolute_threshold": DEFAULT_THRESHOLD,
            "direct_contour_delta": DIRECT_DELTA,
            "target_success_count": TARGET_SUCCESS_COUNT,
            "maximum_attempt_count": MAXIMUM_ATTEMPT_COUNT,
            "configured_workers": workers,
            "selected_minimum_count": len(selected),
            "recorded_minimum_count": len(rows),
            "usable_covariance_count": len(ok_rows),
            "nonusable_covariance_count": len(rows) - len(ok_rows),
            "mean_initial_success_fraction": (
                float(np.mean(success_fractions))
                if len(success_fractions) else np.nan
            ),
            "minimum_initial_success_fraction": (
                float(np.min(success_fractions))
                if len(success_fractions) else np.nan
            ),
            "mean_attempted_direction_count": (
                float(np.mean(attempts)) if len(attempts) else np.nan
            ),
            "maximum_attempted_direction_count": (
                int(np.max(attempts)) if len(attempts) else 0
            ),
            "elapsed_hours": (perf_counter() - started) / 3600.0,
        })
    _write_csv(SUMMARY_PATH, output)


def _stratified_descriptor_subset(descriptors, per_state, round_index=1):
    """Select deterministic interior quantiles in photon-energy fraction."""
    if int(round_index) < 1:
        raise ValueError("round_index must be positive.")
    selected = []
    for state_key in STATES:
        state_descriptors = [
            item for item in descriptors if item["state"] == state_key
        ]
        if not state_descriptors:
            continue
        state_descriptors.sort(
            key=lambda item: (
                float(item["row"]["qOut"])
                / gradient_tool.phase_scan._qout_max(
                    float(item["row"]["s"])
                ),
                int(item["minimum_id"]),
            )
        )
        requested_count = (
            per_state.get(state_key, 0)
            if isinstance(per_state, dict)
            else per_state
        )
        count = min(int(requested_count), len(state_descriptors))
        if count == 0:
            continue
        first_round_indices = np.rint(
            np.linspace(0, len(state_descriptors) - 1, count + 2)[1:-1]
        ).astype(int)
        if int(round_index) == 1:
            indices = first_round_indices
        else:
            # Later rounds use an interleaved quantile grid and explicitly
            # avoid the first-round rows.  Choose the closest still-unused
            # row if rounding lands on a previous selection.
            target_fractions = (
                np.arange(count, dtype=float) + 0.5
            ) / count
            targets = target_fractions * (len(state_descriptors) - 1)
            unavailable = set(int(index) for index in first_round_indices)
            indices = []
            for target in targets:
                candidates = sorted(
                    (
                        index for index in range(len(state_descriptors))
                        if index not in unavailable
                    ),
                    key=lambda index: (abs(index - target), index),
                )
                chosen = candidates[0]
                indices.append(chosen)
                unavailable.add(chosen)
            indices = np.asarray(indices, dtype=int)
        selected.extend(state_descriptors[index] for index in indices)
    return selected


def run(
    states=STATES,
    threshold=DEFAULT_THRESHOLD,
    workers=None,
    limit=None,
    benchmark_per_state=None,
    benchmark_round=1,
):
    workers = max(1, int(workers or os.cpu_count() or 1))
    if not np.isclose(threshold, DEFAULT_THRESHOLD, rtol=0.0, atol=1.0e-15):
        raise ValueError(
            "This production output is reserved for absolute threshold 0.01."
        )
    if TARGET_SUCCESS_COUNT < 2 * SCAN_DIMENSION:
        raise RuntimeError("At least 16 direct-contour directions are required.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    descriptors = _selected_descriptors(states, threshold, workers)
    completed = _completed_rows(descriptors, threshold)
    requested_descriptors = descriptors
    if benchmark_per_state is not None:
        if isinstance(benchmark_per_state, dict):
            if any(int(value) < 0 for value in benchmark_per_state.values()):
                raise ValueError("benchmark state counts must be nonnegative.")
        elif benchmark_per_state < 1:
            raise ValueError("benchmark_per_state must be positive.")
        requested_descriptors = _stratified_descriptor_subset(
            descriptors, benchmark_per_state, round_index=benchmark_round,
        )
    pending = [
        item for item in requested_descriptors
        if (item["state"], item["minimum_id"]) not in completed
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        pending = pending[:limit]
    predecessor_count = audit_incremental_predecessors(pending)
    print(
        f"resume audit: {len(completed)}/{len(descriptors)} compatible "
        f"checkpoints; requested {len(requested_descriptors)}; "
        f"processing {len(pending)} with {workers} workers",
        flush=True,
    )
    if predecessor_count:
        print(
            f"incremental predecessor audit: {predecessor_count} contours at "
            f"{PREVIOUS_SUCCESS_COUNT} points from {PREVIOUS_OUTPUT_VARIANT}",
            flush=True,
        )
    if not pending:
        _write_summary(descriptors, completed, started, workers)
        return completed

    task_index = {
        (item["state"], item["minimum_id"]): index
        for index, item in enumerate(descriptors)
    }
    pending_iterator = iter(pending)
    active = {}
    processed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        def submit_next():
            try:
                descriptor = next(pending_iterator)
            except StopIteration:
                return False
            key = (descriptor["state"], descriptor["minimum_id"])
            future = executor.submit(
                configuration_complete_contour_task,
                _worker_task(descriptor, task_index[key]),
            )
            active[future] = (descriptor, perf_counter())
            return True

        for _index in range(min(workers, len(pending))):
            submit_next()
        while active:
            done, _not_done = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                descriptor, task_started = active.pop(future)
                (
                    _row_index,
                    added_points,
                    attempted_count,
                    initial_success_count,
                    total_success_count,
                    evaluation_count,
                    domain_boundary_point_count,
                ) = future.result()
                diagnostics = {
                    "attempted_direction_count": attempted_count,
                    "initial_success_count": initial_success_count,
                    "total_success_count": total_success_count,
                    "replacement_direction_count": (
                        attempted_count
                        - (TARGET_SUCCESS_COUNT - PREVIOUS_SUCCESS_COUNT)
                    ),
                    "physics_evaluation_count": evaluation_count,
                    "domain_boundary_point_count": domain_boundary_point_count,
                    "added_success_count": (
                        TARGET_SUCCESS_COUNT - PREVIOUS_SUCCESS_COUNT
                    ),
                }
                previous_points, previous_boundary_count = (
                    _load_previous_contour(descriptor)
                )
                points = np.vstack((previous_points, added_points))
                diagnostics["domain_boundary_point_count"] += (
                    previous_boundary_count
                )
                row = _result_row(
                    descriptor,
                    points,
                    diagnostics,
                    perf_counter() - task_started,
                )
                _write_contour(descriptor, points, diagnostics)
                _append_csv_row(_covariance_path(descriptor["state"]), row)
                key = (descriptor["state"], descriptor["minimum_id"])
                completed[key] = row
                processed += 1
                elapsed = perf_counter() - started
                remaining = (
                    elapsed / processed * (len(pending) - processed)
                    if processed else np.nan
                )
                print(
                    f"[{processed}/{len(pending)}] {descriptor['state']} "
                    f"minimum {descriptor['minimum_id']}: status={row['status']}, "
                    f"initial={float(row['initial_success_fraction']):.1%}, "
                    f"attempts={int(row['attempted_direction_count'])}, "
                    f"task={float(row['direct_contour_seconds']) / 60.0:.1f} min, "
                    f"ETA={remaining / 3600.0:.2f} h",
                    flush=True,
                )
                if processed % 10 == 0:
                    _write_summary(descriptors, completed, started, workers)
                submit_next()
    _write_summary(descriptors, completed, started, workers)
    nonusable = [row for row in completed.values() if row["status"] != "ok"]
    print(
        f"complete: {len(completed)}/{len(descriptors)} recorded; "
        f"nonusable={len(nonusable)}; summary={SUMMARY_PATH}",
        flush=True,
    )
    return completed


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", choices=STATES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--limit", type=int,
        help="Process only the first N not-yet-completed minima.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(
        states=tuple(args.state or STATES),
        threshold=args.threshold,
        workers=args.workers,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
