"""Generic gradient phase-space search, configuration, and contour tool.

Each optimization uses the same eight continuous coordinates and coherent
incoming-spin preparation as :mod:`PhaseSpaceScan`. The coordinates are
normalized to a unit box before SciPy's L-BFGS-B minimizer estimates numerical
gradients. A periodic-aware multiscale direct search then follows unresolved
descent directions down to the requested scan precision. Distinct verified
minima are written as phase-space rows and converted into ConfigGen-style
momentum and amplitude configurations. Objective definitions, physics anchors,
and output roots are supplied explicitly by the calling interface.
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import qmc

import ConfigGen as config_gen
from GradientContourWorker import configuration_contour_task
import PhaseSpaceConfigScan as config_scan
import PhaseSpaceScan as phase_scan
from AlignmentScan import LEPTON_SPECS
from config import (
    ENTANGLEMENT_GRADIENT_MAX_ITERATIONS,
    ENTANGLEMENT_GRADIENT_MINIMUM_SEPARATION,
    ENTANGLEMENT_GRADIENT_RANDOM_SEED,
    ENTANGLEMENT_GRADIENT_RANDOM_STARTS,
    ENTANGLEMENT_GRADIENT_SCAN_PRECISION,
    ENTANGLEMENT_GRADIENT_SCREENED_STARTS,
    ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES,
    ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION,
    ENTANGLEMENT_GRADIENT_TOLERANCE,
    ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP,
    ENTANGLEMENT_LOCAL_SEARCH_MAX_POLLS,
    ENTANGLEMENT_LOCAL_SEARCH_OBJECTIVE_TOLERANCE,
    ENTANGLEMENT_LOCAL_SEARCH_RANDOM_DIRECTIONS,
    ENTANGLEMENT_LOCAL_SEARCH_STEP_REDUCTION,
    PHASE_SPACE_CONFIG_CONTOUR_DELTA,
    PHASE_SPACE_CONFIG_CONTOUR_SAMPLES,
    SCAN_INITIAL_MIXING_ANGLES,
    SCAN_WORKERS,
)
from PlotUtils import configure_named_angle_axes, print_console_text


# Runtime state is populated only through :func:`configure_scan`.
LEPTONS_TO_PROCESS = ()
GRADIENT_WORKERS = SCAN_WORKERS
OUTPUT_ROOT = None
OBJECTIVE_NAME = ""
OBJECTIVE_FILE_TAG = ""
OBJECTIVE_LATEX = ""
OBJECTIVE_STATE_FILE_LABEL = ""
SCAN_KEY = ""
PHYSICS_ANCHOR_STARTS = {}

INVALID_OBJECTIVE = 1.0e3
SCAN_DIMENSION = 8
PERIODIC_UNIT_COORDINATES = (4, 5, 6, 7)
CONFIG_CONTOUR_BISECTION_ITERATIONS = 8
CONFIG_CONTOUR_INITIAL_RADIUS = 0.01
PLOT_PANELS = (
    ("theta_p_out", "theta_gamma_out", r"$\theta_{p'}$", r"$\theta_\gamma$"),
    ("sqrt_s", "qOut", r"$\sqrt{s}$ [GeV]", r"$E_\gamma$ [GeV]"),
    ("phi_p_out", "phi_gamma_out", r"$\phi_{p'}$", r"$\phi_\gamma$"),
    ("alpha_e", "alpha_p", r"$\alpha_e$", r"$\alpha_p$"),
    ("theta_p_out", "qOut", r"$\theta_{p'}$", r"$E_\gamma$ [GeV]"),
    ("theta_gamma_out", "qOut", r"$\theta_\gamma$", r"$E_\gamma$ [GeV]"),
    ("sqrt_s", "alpha_e", r"$\sqrt{s}$ [GeV]", r"$\alpha_e$"),
    ("sqrt_s", "alpha_p", r"$\sqrt{s}$ [GeV]", r"$\alpha_p$"),
)
PLOT_PERIODS = {
    "phi_p_out": 2.0 * np.pi,
    "phi_gamma_out": 2.0 * np.pi,
    "alpha_e": np.pi,
    "alpha_p": np.pi,
}
POLARIZATION_CLUSTER_STYLES = (
    ("#0057B8", "o"),
    ("#FF8C00", "s"),
    ("#008A45", "^"),
    ("#E60026", "D"),
    ("#7A1FA2", "v"),
    ("#00A6A6", "X"),
)


@dataclass(frozen=True)
class GradientScanDefinition:
    """Describe one objective and its independent output tree."""

    key: str
    objective_name: str
    file_tag: str
    latex: str
    state_file_label: str
    output_root: Path
    physics_anchor_starts: dict


def configure_scan(
    definition,
    *,
    leptons_to_process,
    gradient_workers,
):
    """Configure the tool from one explicit scan definition."""
    if not isinstance(definition, GradientScanDefinition):
        raise TypeError("definition must be a GradientScanDefinition.")

    global OBJECTIVE_NAME, OBJECTIVE_FILE_TAG, OBJECTIVE_LATEX
    global OBJECTIVE_STATE_FILE_LABEL, SCAN_KEY
    global OUTPUT_ROOT
    global PHYSICS_ANCHOR_STARTS
    global LEPTONS_TO_PROCESS, GRADIENT_WORKERS

    OBJECTIVE_NAME = str(definition.objective_name)
    OBJECTIVE_FILE_TAG = str(definition.file_tag)
    OBJECTIVE_LATEX = str(definition.latex)
    OBJECTIVE_STATE_FILE_LABEL = str(definition.state_file_label)
    SCAN_KEY = str(definition.key)
    OUTPUT_ROOT = Path(definition.output_root)
    PHYSICS_ANCHOR_STARTS = {
        str(lepton_name): tuple(dict(anchor) for anchor in anchors)
        for lepton_name, anchors in definition.physics_anchor_starts.items()
    }
    LEPTONS_TO_PROCESS = tuple(leptons_to_process)
    GRADIENT_WORKERS = int(gradient_workers)


def stage_log_path(stage):
    """Return the active state's log path for one independent stage."""
    return (
        OUTPUT_ROOT
        / "Logs"
        / f"{SCAN_KEY}_gradient_phase_space_{stage}.log"
    )


def _objective_key(lepton_name, objective_name=None):
    """Return the coherent-angle CSV key for the selected objective."""
    name = OBJECTIVE_NAME if objective_name is None else objective_name
    return f"{config_scan.mixing_prefix(lepton_name)}_{name}"


def _configuration_plot_path(lepton_name, polarization_cluster_id):
    """Return one parent-polarization configuration PDF path."""
    species_label = LEPTON_SPECS[lepton_name]["label"].title().replace(" ", "_")
    parent_label = f"Polarization_Cluster_{polarization_cluster_id + 1:02d}"
    filename = (
        f"{OBJECTIVE_STATE_FILE_LABEL}_State_Search_and_Config_"
        f"{species_label}_{parent_label}.pdf"
    )
    return (
        species_output_dirs(lepton_name)["plots"]
        / f"polarization_cluster_{polarization_cluster_id + 1:02d}"
        / filename
    )


def species_output_dirs(lepton_name):
    """Return the organized state/lepton data and plot directories."""
    if lepton_name not in LEPTON_SPECS:
        raise ValueError(
            f"Unknown lepton {lepton_name!r}; choose from "
            f"{tuple(LEPTON_SPECS)}."
        )
    root = OUTPUT_ROOT / lepton_name
    return {
        "root": root,
        "data": root / "Data" / SCAN_KEY,
        "scan_data": root / "Data" / SCAN_KEY / "scan",
        "cluster_data": root / "Data" / SCAN_KEY / "cluster",
        "plots": root / "Plots" / SCAN_KEY,
    }


def configuration_data_paths(lepton_name, polarization_cluster_id):
    """Return one parent-polarization configuration data package."""
    parent_tag = f"polarization_cluster_{polarization_cluster_id + 1:02d}"
    prefix = f"min_{OBJECTIVE_FILE_TAG}_{parent_tag}"
    combined_dir = (
        species_output_dirs(lepton_name)["data"]
        / OBJECTIVE_FILE_TAG
        / parent_tag
        / "combined"
    )
    return {
        "selected": combined_dir / f"{prefix}_selected_minima.csv",
        "examples": (
            combined_dir / f"{prefix}_configuration_examples.csv"
        ),
        "clusters": combined_dir / f"{prefix}_cluster_summary.csv",
        "momenta": (
            combined_dir / f"{prefix}_momentum_configurations.csv"
        ),
        "amplitudes": (
            combined_dir
            / f"{prefix}_final_state_amplitude_decomposition.csv"
        ),
        "contours": combined_dir / f"{prefix}_contour_samples.csv",
    }


def _run_objective_key(stage, objective_name=None):
    """Return one objective-specific optimization-run column name."""
    name = OBJECTIVE_NAME if objective_name is None else objective_name
    return f"{stage}_{name}"


def _normalized_to_point(unit_point):
    """Map the optimizer's unit box to the eight physical scan coordinates."""
    unit_point = np.asarray(unit_point, dtype=float)
    sqrt_s = (
        phase_scan.SQRT_S_RANGE[0]
        + unit_point[0]
        * (phase_scan.SQRT_S_RANGE[1] - phase_scan.SQRT_S_RANGE[0])
    )
    s = sqrt_s**2
    qout_fraction = (
        phase_scan.QOUT_FRACTION_RANGE[0]
        + unit_point[3]
        * (
            phase_scan.QOUT_FRACTION_RANGE[1]
            - phase_scan.QOUT_FRACTION_RANGE[0]
        )
    )
    return np.asarray(
        (
            s,
            phase_scan.THETA_P_OUT_RANGE[0]
            + unit_point[1]
            * (
                phase_scan.THETA_P_OUT_RANGE[1]
                - phase_scan.THETA_P_OUT_RANGE[0]
            ),
            phase_scan.THETA_GAMMA_OUT_RANGE[0]
            + unit_point[2]
            * (
                phase_scan.THETA_GAMMA_OUT_RANGE[1]
                - phase_scan.THETA_GAMMA_OUT_RANGE[0]
            ),
            qout_fraction * phase_scan._qout_max(s),
            unit_point[4] * 2.0 * np.pi,
            unit_point[5] * 2.0 * np.pi,
            unit_point[6] * np.pi,
            unit_point[7] * np.pi,
        ),
        dtype=float,
    )


def _objective_evaluation(
    unit_point,
    lepton_name,
    evaluation_id,
    objective_name=None,
):
    """Evaluate the selected objective and return its coherent-angle row."""
    result = phase_scan._evaluate_sample(
        _normalized_to_point(unit_point),
        sample_id=evaluation_id,
        stage="gradient",
        lepton_name=lepton_name,
        lepton_mass=LEPTON_SPECS[lepton_name]["mass"],
    )
    if result is None or result[1] is None:
        return INVALID_OBJECTIVE, None
    row = result[1]
    key = _objective_key(lepton_name, objective_name)
    value = float(row.get(key, np.nan))
    if not np.isfinite(value):
        return INVALID_OBJECTIVE, None
    return value, row


def _optimize_start(task):
    """Optimize one random initial point in a worker-safe species context."""
    (
        lepton_name,
        run_index,
        start,
        start_source,
        screening_value,
        objective_name,
    ) = task
    phase_scan._configure_lepton(lepton_name)
    cache = {}
    evaluation_count = 0

    def evaluate(unit_point):
        nonlocal evaluation_count
        clipped = np.clip(np.asarray(unit_point, dtype=float), 0.0, 1.0)
        key = clipped.tobytes()
        if key not in cache:
            cache[key] = _objective_evaluation(
                clipped,
                lepton_name,
                evaluation_id=run_index * 1_000_000 + evaluation_count,
                objective_name=objective_name,
            )
            evaluation_count += 1
        return cache[key]

    start_value, _start_row = evaluate(start)
    result = minimize(
        lambda point: evaluate(point)[0],
        np.asarray(start, dtype=float),
        method="L-BFGS-B",
        bounds=((0.0, 1.0),) * SCAN_DIMENSION,
        options={
            "maxiter": ENTANGLEMENT_GRADIENT_MAX_ITERATIONS,
            "ftol": ENTANGLEMENT_GRADIENT_TOLERANCE,
            "gtol": ENTANGLEMENT_GRADIENT_TOLERANCE,
            "eps": ENTANGLEMENT_GRADIENT_SCAN_PRECISION,
        },
    )
    lbfgs_point = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    lbfgs_value, _lbfgs_row = evaluate(lbfgs_point)
    (
        final_point,
        final_value,
        final_row,
        local_search,
    ) = _multiscale_local_search(
        evaluate,
        lbfgs_point,
        direction_seed=ENTANGLEMENT_GRADIENT_RANDOM_SEED + run_index,
        objective_name=objective_name,
    )
    gradient_norm = (
        float(np.linalg.norm(np.asarray(result.jac, dtype=float)))
        if result.jac is not None
        else np.nan
    )
    run = {
        "optimization_run": run_index,
        "start_source": start_source,
        _run_objective_key("screening", objective_name): screening_value,
        "success": local_search["local_minimum_verified"],
        "lbfgs_success": bool(result.success),
        "lbfgs_status": int(result.status),
        "lbfgs_message": str(result.message),
        "lbfgs_iterations": int(result.nit),
        "function_evaluations": evaluation_count,
        "lbfgs_function_evaluations": int(result.nfev),
        "lbfgs_gradient_norm": gradient_norm,
        _run_objective_key("initial", objective_name): start_value,
        _run_objective_key("lbfgs", objective_name): lbfgs_value,
        _run_objective_key("final", objective_name): final_value,
        **local_search,
    }
    for index, value in enumerate(start):
        run[f"initial_u{index}"] = float(value)
    for index, value in enumerate(final_point):
        run[f"final_u{index}"] = float(value)
    if final_row is not None:
        final_row = dict(final_row)
        final_row.update(run)
    return run, final_point, final_row


def _move_unit_point(point, displacement):
    """Apply a displacement with bounds and periodic-axis wrapping."""
    neighbor = np.asarray(point, dtype=float).copy()
    neighbor += np.asarray(displacement, dtype=float)
    neighbor[:4] = np.clip(neighbor[:4], 0.0, 1.0)
    neighbor[4:] %= 1.0
    return neighbor


def _poll_neighbors(point, step, extra_directions=()):
    """Return coordinate and exploratory-direction unit-box neighbors."""
    point = np.asarray(point, dtype=float)
    neighbors = []
    for coordinate in range(point.size):
        for direction in (-1.0, 1.0):
            displacement = np.zeros(point.size)
            displacement[coordinate] = direction * step
            neighbor = _move_unit_point(point, displacement)
            if not np.array_equal(neighbor, point):
                neighbors.append(neighbor)
    for direction in extra_directions:
        for sign in (-1.0, 1.0):
            neighbor = _move_unit_point(point, sign * step * direction)
            if not np.array_equal(neighbor, point):
                neighbors.append(neighbor)
    return neighbors


def _multiscale_local_search(
    evaluate,
    start,
    direction_seed=0,
    objective_name=None,
):
    """Polish a result until no poll direction improves the objective.

    Coordinate directions form a positive-spanning set. Repeating the poll
    while shrinking its mesh makes this robust to branch-sensitive or
    nonsmooth regions where L-BFGS-B can stop on relative function reduction.
    Periodic azimuth and mixing coordinates wrap across the unit-box boundary.
    """
    point = np.asarray(start, dtype=float).copy()
    value, row = evaluate(point)
    direction_rng = np.random.default_rng(direction_seed)
    extra_directions = direction_rng.normal(
        size=(ENTANGLEMENT_LOCAL_SEARCH_RANDOM_DIRECTIONS, point.size)
    )
    if len(extra_directions):
        extra_directions /= np.linalg.norm(
            extra_directions, axis=1, keepdims=True
        )
    step = ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP
    polls = 0
    accepted_moves = 0
    smallest_tested_step = step
    while polls < ENTANGLEMENT_LOCAL_SEARCH_MAX_POLLS:
        neighbors = _poll_neighbors(point, step, extra_directions)
        evaluated = [
            (evaluate(neighbor)[0], neighbor)
            for neighbor in neighbors
        ]
        polls += 1
        smallest_tested_step = step
        best_value, best_point = min(evaluated, key=lambda item: item[0])
        if (
            best_value
            < value - ENTANGLEMENT_LOCAL_SEARCH_OBJECTIVE_TOLERANCE
        ):
            point = best_point
            value, row = evaluate(point)
            accepted_moves += 1
            continue
        if (
            step
            <= ENTANGLEMENT_GRADIENT_SCAN_PRECISION * (1.0 + 1.0e-12)
        ):
            break
        step = max(
            ENTANGLEMENT_GRADIENT_SCAN_PRECISION,
            step * ENTANGLEMENT_LOCAL_SEARCH_STEP_REDUCTION,
        )

    verification_neighbors = _poll_neighbors(
        point,
        ENTANGLEMENT_GRADIENT_SCAN_PRECISION,
        extra_directions,
    )
    neighbor_values = [
        evaluate(neighbor)[0] for neighbor in verification_neighbors
    ]
    best_neighbor = min(neighbor_values, default=value)
    verified = (
        best_neighbor
        >= value - ENTANGLEMENT_LOCAL_SEARCH_OBJECTIVE_TOLERANCE
    )
    return point, value, row, {
        "local_search_polls": polls,
        "local_search_moves": accepted_moves,
        "local_search_poll_limit_reached": (
            polls >= ENTANGLEMENT_LOCAL_SEARCH_MAX_POLLS
            and step >= ENTANGLEMENT_GRADIENT_SCAN_PRECISION
        ),
        "smallest_tested_step": smallest_tested_step,
        _run_objective_key("best_neighbor", objective_name): best_neighbor,
        "local_minimum_verified": verified,
    }


def _run_tasks(tasks):
    """Run one species' independent starts in a process pool."""
    workers = min(max(1, int(GRADIENT_WORKERS)), len(tasks))
    if workers <= 1:
        return [_optimize_start(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_optimize_start, tasks, chunksize=1))
    except (OSError, PermissionError, BrokenProcessPool):
        # Sequential fallback preserves the worker-safe species configuration.
        return [_optimize_start(task) for task in tasks]


def _unit_distance(first, second):
    """Return normalized distance with periodic azimuth and mixing axes."""
    delta = np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))
    for index in PERIODIC_UNIT_COORDINATES:
        delta[index] = min(delta[index], 1.0 - delta[index])
    return float(np.linalg.norm(delta))


def _deduplicate_minima(results):
    """Keep the lowest-objective representative of each converged basin."""
    final_key = _run_objective_key("final")
    finite = [
        result for result in results
        if (
            result[0]["local_minimum_verified"]
            and result[2] is not None
            and np.isfinite(result[0][final_key])
        )
    ]
    finite.sort(key=lambda result: result[0][final_key])
    selected = []
    for result in finite:
        if any(
            _unit_distance(result[1], prior[1])
            <= ENTANGLEMENT_GRADIENT_MINIMUM_SEPARATION
            for prior in selected
        ):
            continue
        selected.append(result)
    return selected


def _write_csv(path, rows):
    """Write dictionaries while preserving the first row's column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        if headers:
            writer.writeheader()
            writer.writerows(rows)
    return path


def _read_csv(path):
    """Load a saved dictionary CSV and require at least one data row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required gradient-stage data is missing: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(
            f"Required gradient-stage data contains no rows: {path}"
        )
    return rows


def _as_bool(value):
    """Parse one bool or CSV bool field without truthy-string ambiguity."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean value.")


def _objective_values(rows, lepton_name):
    """Return the local-minimum objective values in row order."""
    key = _objective_key(lepton_name)
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _unit_point_from_minimum_row(row):
    """Recover one local minimum in the optimizer's normalized coordinates."""
    return np.asarray([
        float(row[f"final_u{index}"]) for index in range(SCAN_DIMENSION)
    ])


def _plot_coordinate_values(unit_point):
    """Return the named physical coordinates used by the projection panels."""
    point = _normalized_to_point(unit_point)
    return {
        "sqrt_s": float(np.sqrt(point[0])),
        "theta_p_out": float(point[1]),
        "theta_gamma_out": float(point[2]),
        "qOut": float(point[3]),
        "phi_p_out": float(point[4]),
        "phi_gamma_out": float(point[5]),
        "alpha_e": float(point[6]),
        "alpha_p": float(point[7]),
    }


def _contour_directions(seed):
    """Return deterministic directions spanning the eight-dimensional sphere."""
    axes = np.vstack((np.eye(SCAN_DIMENSION), -np.eye(SCAN_DIMENSION)))
    random_count = PHASE_SPACE_CONFIG_CONTOUR_SAMPLES - len(axes)
    rng = np.random.default_rng(seed)
    random_directions = rng.normal(size=(random_count, SCAN_DIMENSION))
    if random_count:
        random_directions /= np.linalg.norm(
            random_directions,
            axis=1,
            keepdims=True,
        )
    return np.vstack((axes, random_directions))


def _configuration_contours(rows, lepton_name):
    """Evaluate one high-dimensional contour for each selected minimum."""
    worker_count = max(1, int(GRADIENT_WORKERS))
    row_count = max(1, len(rows))
    chunks_per_minimum = min(
        max(1, (worker_count + row_count - 1) // row_count),
        PHASE_SPACE_CONFIG_CONTOUR_SAMPLES,
    )
    tasks = []
    for row_index, row in enumerate(rows):
        direction_chunks = np.array_split(
            _contour_directions(
                ENTANGLEMENT_GRADIENT_RANDOM_SEED
                + int(row.get("local_minimum_id", row_index))
            ),
            chunks_per_minimum,
        )
        tasks.extend(
            (
                row_index,
                chunk_index,
                row,
                lepton_name,
                LEPTON_SPECS[lepton_name]["mass"],
                OBJECTIVE_NAME,
                directions,
                PHASE_SPACE_CONFIG_CONTOUR_DELTA,
                CONFIG_CONTOUR_INITIAL_RADIUS,
                CONFIG_CONTOUR_BISECTION_ITERATIONS,
            )
            for chunk_index, directions in enumerate(direction_chunks)
            if len(directions)
        )
    workers = min(max(1, int(GRADIENT_WORKERS)), len(tasks))
    if workers <= 1:
        results = [configuration_contour_task(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(
                    executor.map(
                        configuration_contour_task,
                        tasks,
                        chunksize=1,
                    )
                )
        except (OSError, PermissionError, BrokenProcessPool):
            results = [configuration_contour_task(task) for task in tasks]
    grouped = {row_index: [] for row_index in range(len(rows))}
    for row_index, chunk_index, boundary_points in sorted(
        results,
        key=lambda result: (result[0], result[1]),
    ):
        grouped[row_index].append(boundary_points)
    return {
        row_index: (
            np.vstack(chunks)
            if chunks
            else np.empty((0, SCAN_DIMENSION))
        )
        for row_index, chunks in grouped.items()
    }


def _contour_csv_rows(selected_rows, contours):
    """Serialize contour centers and samples with compatibility metadata."""
    csv_rows = []
    for row_index, row in enumerate(selected_rows):
        center = _unit_point_from_minimum_row(row)
        minimum_id = row.get("local_minimum_id", row_index)
        base = {
            "objective_name": OBJECTIVE_NAME,
            "objective_file_tag": OBJECTIVE_FILE_TAG,
            "local_minimum_index": row_index,
            "local_minimum_id": minimum_id,
            "contour_delta": PHASE_SPACE_CONFIG_CONTOUR_DELTA,
            "configured_direction_count": PHASE_SPACE_CONFIG_CONTOUR_SAMPLES,
            "contour_point_count": len(contours[row_index]),
            **{
                f"center_u{coordinate}": float(center[coordinate])
                for coordinate in range(SCAN_DIMENSION)
            },
        }
        csv_rows.append(
            {
                **base,
                "record_type": "minimum",
                "contour_sample_id": "",
                **{
                    f"u{coordinate}": ""
                    for coordinate in range(SCAN_DIMENSION)
                },
                **{name: "" for name in _plot_coordinate_values(center)},
            }
        )
        for sample_index, point in enumerate(contours[row_index]):
            physical = _plot_coordinate_values(point)
            csv_rows.append(
                {
                    **base,
                    "record_type": "sample",
                    "contour_sample_id": sample_index,
                    **{
                        f"u{coordinate}": float(point[coordinate])
                        for coordinate in range(SCAN_DIMENSION)
                    },
                    **physical,
                }
            )
    return csv_rows


def _write_contour_data(path, selected_rows, contours):
    """Save reusable contour samples before generating their PDF."""
    return _write_csv(path, _contour_csv_rows(selected_rows, contours))


def _load_contour_data(path, selected_rows):
    """Load selected minima from saved contours, allowing unused extra rows."""
    saved_rows = _read_csv(path)
    metadata = {}
    samples = {}
    sample_ids = {}
    for saved in saved_rows:
        if saved["objective_name"] != OBJECTIVE_NAME:
            raise ValueError(
                f"Saved contour objective {saved['objective_name']!r} does "
                f"not match active objective {OBJECTIVE_NAME!r}: {path}"
            )
        if saved["objective_file_tag"] != OBJECTIVE_FILE_TAG:
            raise ValueError(
                f"Saved contour tag {saved['objective_file_tag']!r} does "
                f"not match active tag {OBJECTIVE_FILE_TAG!r}: {path}"
            )
        if not np.isclose(
            float(saved["contour_delta"]),
            PHASE_SPACE_CONFIG_CONTOUR_DELTA,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                "Saved contour delta does not match "
                f"PHASE_SPACE_CONFIG_CONTOUR_DELTA: {path}"
            )
        if (
            int(saved["configured_direction_count"])
            != PHASE_SPACE_CONFIG_CONTOUR_SAMPLES
        ):
            raise ValueError(
                "Saved contour direction count does not match "
                f"PHASE_SPACE_CONFIG_CONTOUR_SAMPLES: {path}"
            )
        row_index = int(saved["local_minimum_index"])
        samples.setdefault(row_index, [])
        sample_ids.setdefault(row_index, [])
        if saved["record_type"] == "minimum":
            if row_index in metadata:
                raise ValueError(
                    f"Saved contour repeats metadata for minimum "
                    f"{row_index}: {path}"
                )
            metadata[row_index] = saved
        elif saved["record_type"] == "sample":
            sample_ids[row_index].append(int(saved["contour_sample_id"]))
            samples[row_index].append(
                np.asarray(
                    [
                        float(saved[f"u{coordinate}"])
                        for coordinate in range(SCAN_DIMENSION)
                    ],
                    dtype=float,
                )
            )
        else:
            raise ValueError(
                f"Unknown saved contour record type "
                f"{saved['record_type']!r}: {path}"
            )

    saved_index_by_id = {}
    for saved_index, saved in metadata.items():
        saved_id = saved["local_minimum_id"]
        if saved_id in saved_index_by_id:
            raise ValueError(
                f"Saved contour repeats local-minimum ID {saved_id!r}: {path}"
            )
        saved_index_by_id[saved_id] = saved_index

    loaded = {}
    for output_index, row in enumerate(selected_rows):
        expected_id = str(row.get("local_minimum_id", output_index))
        if expected_id not in saved_index_by_id:
            raise ValueError(
                f"Saved contour data is missing local-minimum ID "
                f"{expected_id!r}: {path}"
            )
        saved_index = saved_index_by_id[expected_id]
        saved = metadata[saved_index]
        saved_center = np.asarray(
            [
                float(saved[f"center_u{coordinate}"])
                for coordinate in range(SCAN_DIMENSION)
            ],
            dtype=float,
        )
        if not np.allclose(
            saved_center,
            _unit_point_from_minimum_row(row),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Saved contour center does not match local minimum "
                f"{expected_id}: {path}"
            )
        expected_count = int(saved["contour_point_count"])
        if sample_ids[saved_index] != list(range(expected_count)):
            raise ValueError(
                f"Saved contour samples for minimum {expected_id} are "
                f"incomplete or out of order: {path}"
            )
        loaded[output_index] = np.asarray(
            samples[saved_index],
            dtype=float,
        ).reshape((-1, SCAN_DIMENSION))
    return loaded


def _unwrap_about_center(values, center, period):
    """Represent periodic values on the branch nearest the selected minimum."""
    values = np.asarray(values, dtype=float)
    if period is None:
        return values
    return center + (values - center + 0.5 * period) % period - 0.5 * period


def _project_high_dimensional_contour(
    boundary_points,
    center,
    x_name,
    y_name,
):
    """Project a sampled eight-dimensional contour into one plot panel."""
    center_coordinates = _plot_coordinate_values(center)
    projected = [
        _plot_coordinate_values(point)
        for point in boundary_points
    ]
    x_values = np.asarray(
        [coordinates[x_name] for coordinates in projected],
        dtype=float,
    )
    y_values = np.asarray(
        [coordinates[y_name] for coordinates in projected],
        dtype=float,
    )
    x_values = _unwrap_about_center(
        x_values,
        center_coordinates[x_name],
        PLOT_PERIODS.get(x_name),
    )
    y_values = _unwrap_about_center(
        y_values,
        center_coordinates[y_name],
        PLOT_PERIODS.get(y_name),
    )
    return (
        x_values,
        y_values,
        center_coordinates[x_name],
        center_coordinates[y_name],
    )


def _projected_contour_envelope(x_values, y_values):
    """Return the convex outer envelope of one contour projection."""
    points = np.column_stack((x_values, y_values))
    points = np.unique(points[np.all(np.isfinite(points), axis=1)], axis=0)
    if len(points) < 3:
        return np.empty(0), np.empty(0)
    try:
        hull = ConvexHull(points)
    except QhullError:
        return np.empty(0), np.empty(0)
    vertices = np.concatenate((hull.vertices, hull.vertices[:1]))
    return points[vertices, 0], points[vertices, 1]


def _wrap_projected_values(values, coordinate_name):
    """Map locally unwrapped contour coordinates to their physical branch."""
    period = PLOT_PERIODS.get(coordinate_name)
    values = np.asarray(values, dtype=float)
    return values % period if period is not None else values


def _split_wrapped_projection(
    x_values,
    y_values,
    x_name,
    y_name,
):
    """Break a projected envelope where it crosses a periodic boundary."""
    x_values = _wrap_projected_values(x_values, x_name)
    y_values = _wrap_projected_values(y_values, y_name)
    x_period = PLOT_PERIODS.get(x_name)
    y_period = PLOT_PERIODS.get(y_name)
    split_x = []
    split_y = []
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        if index:
            wrapped = (
                x_period is not None
                and abs(x_value - x_values[index - 1]) > 0.5 * x_period
            ) or (
                y_period is not None
                and abs(y_value - y_values[index - 1]) > 0.5 * y_period
            )
            if wrapped:
                split_x.append(np.nan)
                split_y.append(np.nan)
        split_x.append(x_value)
        split_y.append(y_value)
    return np.asarray(split_x), np.asarray(split_y)


def _full_phase_space_plot_limits(lepton_name):
    """Return the complete configured physical range for every plot axis."""
    phase_scan._configure_lepton(lepton_name)
    qout_upper = max(
        phase_scan._qout_max(sqrt_s**2)
        for sqrt_s in phase_scan.SQRT_S_RANGE
    ) * phase_scan.QOUT_FRACTION_RANGE[1]
    return {
        "sqrt_s": tuple(phase_scan.SQRT_S_RANGE),
        "theta_p_out": tuple(phase_scan.THETA_P_OUT_RANGE),
        "theta_gamma_out": tuple(phase_scan.THETA_GAMMA_OUT_RANGE),
        "qOut": (0.0, float(qout_upper)),
        "phi_p_out": tuple(phase_scan.AZIMUTH_RANGE),
        "phi_gamma_out": tuple(phase_scan.AZIMUTH_RANGE),
        "alpha_e": tuple(phase_scan.ALPHA_E_RANGE),
        "alpha_p": tuple(phase_scan.ALPHA_P_RANGE),
    }


def _plot_all_local_minima(rows, lepton_name, path, reference_rows=()):
    """Plot every minimum and overlay the exact unoptimized references."""
    plt, PdfPages = config_gen._require_matplotlib()
    values = _objective_values(rows, lepton_name)
    cmap, vmin, vmax = config_gen.observable_plot_style(OBJECTIVE_NAME)
    best_index = int(np.argmin(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(
            3, 3, figsize=(14.0, 11.5), constrained_layout=True
        )
        image = None
        for ax, (x_name, y_name, x_label, y_label) in zip(
            axes.ravel()[:8], PLOT_PANELS
        ):
            x = np.asarray([float(row[x_name]) for row in rows])
            y = np.asarray([float(row[y_name]) for row in rows])
            image = ax.scatter(
                x, y, c=values, s=42, cmap=cmap, vmin=vmin, vmax=vmax,
                edgecolors="black", linewidths=0.35,
            )
            ax.scatter(
                x[best_index], y[best_index], marker="*", s=180,
                c="red", edgecolors="black", label="global minimum",
            )
            for reference_index, reference in enumerate(reference_rows):
                ax.scatter(
                    [float(reference[x_name])],
                    [float(reference[y_name])],
                    marker="D",
                    s=72,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.2,
                    label=(
                        reference["reference_name"]
                        if ax is axes[0, 0]
                        else None
                    ),
                    zorder=5,
                )
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            configure_named_angle_axes(ax, x_name, y_name)
            ax.tick_params(labelsize=10)
        axes[0, 0].legend()
        axes[2, 2].hist(values, bins=min(60, max(5, len(values))))
        axes[2, 2].axvline(
            values[best_index], color="red", linestyle="--",
            label=(
                rf"$({OBJECTIVE_LATEX})_{{\min}}="
                rf"{values[best_index]:.5g}$"
            ),
        )
        axes[2, 2].set_xlabel(rf"local-minimum ${OBJECTIVE_LATEX}$")
        axes[2, 2].set_ylabel("distinct minima")
        axes[2, 2].legend()
        fig.suptitle(
            f"{lepton_name}: all {len(rows)} distinct gradient-search "
            rf"local minima of ${OBJECTIVE_LATEX}$"
        )
        if image is not None:
            fig.colorbar(
                image,
                ax=axes.ravel()[:8].tolist(),
                label=rf"${OBJECTIVE_LATEX}$",
            )
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _pi_quarter_label(angle):
    """Return the nearest named quarter-pi direction."""
    labels = ("0", "pi/4", "pi/2", "3pi/4", "pi")
    index = int(np.clip(np.rint(4.0 * float(angle) / np.pi), 0, 4))
    return labels[index]


def _pi_quarter_math_label(angle):
    """Return the nearest quarter-pi direction as matplotlib math text."""
    labels = ("0", r"\pi/4", r"\pi/2", r"3\pi/4", r"\pi")
    index = int(np.clip(np.rint(4.0 * float(angle) / np.pi), 0, 4))
    return labels[index]


def _circular_mixing_center(angles):
    """Return the pi-periodic circular center of alpha_e and alpha_p."""
    return np.mod(
        0.5
        * np.arctan2(
            np.mean(np.sin(2.0 * angles), axis=0),
            np.mean(np.cos(2.0 * angles), axis=0),
        ),
        np.pi,
    )


def _mixing_distance(angle, center):
    """Return normalized distance on the alpha_e/alpha_p pi-periodic torus."""
    difference = np.abs(np.asarray(angle) - np.asarray(center))
    wrapped = np.minimum(difference, np.pi - difference) / np.pi
    return float(np.linalg.norm(wrapped))


def _cluster_polarization_minima(
    rows,
    lepton_name,
    *,
    objective_cut,
    cluster_count,
    random_seed,
):
    """Classify low-objective minima in periodic mixing-angle space."""
    values = _objective_values(rows, lepton_name)
    optimum = float(np.min(values))
    eligible_indices = np.flatnonzero(
        values - optimum <= objective_cut + 1.0e-12
    )
    if len(eligible_indices) < cluster_count:
        raise RuntimeError(
            f"The {OBJECTIVE_NAME} polarization cut retained "
            f"{len(eligible_indices)} minima, fewer than the requested "
            f"{cluster_count} clusters."
        )

    angles = np.asarray(
        [
            (float(rows[index]["alpha_e"]), float(rows[index]["alpha_p"]))
            for index in eligible_indices
        ],
        dtype=float,
    )
    embedded = np.column_stack(
        (
            np.cos(2.0 * angles[:, 0]),
            np.sin(2.0 * angles[:, 0]),
            np.cos(2.0 * angles[:, 1]),
            np.sin(2.0 * angles[:, 1]),
        )
    )
    _embedded_centers, raw_labels = kmeans2(
        embedded,
        cluster_count,
        iter=100,
        minit="++",
        seed=random_seed,
    )
    raw_cluster_ids = sorted(set(int(label) for label in raw_labels))
    if len(raw_cluster_ids) != cluster_count:
        raise RuntimeError(
            f"Periodic mixing-angle clustering produced "
            f"{len(raw_cluster_ids)} nonempty groups instead of "
            f"{cluster_count}; change the seed or cut."
        )

    raw_centers = {
        cluster_id: _circular_mixing_center(
            angles[raw_labels == cluster_id]
        )
        for cluster_id in raw_cluster_ids
    }
    ordered_raw_ids = sorted(
        raw_cluster_ids,
        key=lambda cluster_id: (
            int(np.rint(4.0 * raw_centers[cluster_id][0] / np.pi)),
            int(np.rint(4.0 * raw_centers[cluster_id][1] / np.pi)),
            raw_centers[cluster_id][0],
            raw_centers[cluster_id][1],
        ),
    )
    stable_id = {
        raw_id: output_id
        for output_id, raw_id in enumerate(ordered_raw_ids)
    }
    assignments = {
        int(row_index): stable_id[int(raw_label)]
        for row_index, raw_label in zip(eligible_indices, raw_labels)
    }

    members_by_cluster = {
        cluster_id: [
            row_index
            for row_index, assigned_id in assignments.items()
            if assigned_id == cluster_id
        ]
        for cluster_id in range(cluster_count)
    }
    centers = {
        stable_id[raw_id]: raw_centers[raw_id]
        for raw_id in ordered_raw_ids
    }
    representatives = {
        cluster_id: min(
            members,
            key=lambda row_index: values[row_index],
        )
        for cluster_id, members in members_by_cluster.items()
    }

    annotated = []
    objective_delta_key = f"{OBJECTIVE_NAME}_above_global_minimum"
    for row_index, source in enumerate(rows):
        item = dict(source)
        cluster_id = assignments.get(row_index)
        item[objective_delta_key] = float(values[row_index] - optimum)
        item["within_polarization_cluster_cut"] = cluster_id is not None
        if cluster_id is None:
            item.update(
                {
                    "polarization_cluster_id": "",
                    "polarization_cluster_size": "",
                    "polarization_cluster_distance": "",
                    "polarization_cluster_representative": False,
                    "polarization_configuration": "",
                }
            )
        else:
            center = centers[cluster_id]
            configuration = (
                f"alpha_e~{_pi_quarter_label(center[0])},"
                f"alpha_p~{_pi_quarter_label(center[1])}"
            )
            item.update(
                {
                    "polarization_cluster_id": cluster_id,
                    "polarization_cluster_size": len(
                        members_by_cluster[cluster_id]
                    ),
                    "polarization_cluster_distance": _mixing_distance(
                        (float(source["alpha_e"]), float(source["alpha_p"])),
                        center,
                    ),
                    "polarization_cluster_representative": (
                        row_index == representatives[cluster_id]
                    ),
                    "polarization_configuration": configuration,
                }
            )
        annotated.append(item)

    summary = []
    objective_key = _objective_key(lepton_name)
    for cluster_id in range(cluster_count):
        members = members_by_cluster[cluster_id]
        center = centers[cluster_id]
        representative = rows[representatives[cluster_id]]
        color, marker = POLARIZATION_CLUSTER_STYLES[cluster_id]
        summary.append(
            {
                "polarization_cluster_id": cluster_id,
                "polarization_configuration": (
                    f"alpha_e~{_pi_quarter_label(center[0])},"
                    f"alpha_p~{_pi_quarter_label(center[1])}"
                ),
                "member_count": len(members),
                "representative_local_minimum_id": representative.get(
                    "local_minimum_id",
                    representatives[cluster_id],
                ),
                "objective_name": OBJECTIVE_NAME,
                "objective_cut_above_global_minimum": objective_cut,
                "global_minimum": optimum,
                "best_objective": float(values[members].min()),
                "mean_objective": float(values[members].mean()),
                "alpha_e_center": float(center[0]),
                "alpha_p_center": float(center[1]),
                "alpha_e_center_over_pi": float(center[0] / np.pi),
                "alpha_p_center_over_pi": float(center[1] / np.pi),
                "color": color,
                "marker": marker,
                "random_seed": random_seed,
                "distance_coordinates": "alpha_e,alpha_p (pi-periodic)",
                objective_key: float(representative[objective_key]),
            }
        )
    return annotated, summary, optimum


def _polarization_cluster_plot(
    rows,
    polarization_clusters,
    lepton_name,
    objective_cut,
    optimum,
    path,
):
    """Plot low-objective phase space using six polarization styles."""
    plt, PdfPages = config_gen._require_matplotlib()
    selected_rows = [
        row for row in rows
        if _as_bool(row["within_polarization_cluster_cut"])
    ]
    if not selected_rows:
        raise RuntimeError("No rows passed the polarization-cluster cut.")
    objective_key = _objective_key(lepton_name)
    best_row = min(selected_rows, key=lambda row: float(row[objective_key]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(
            3, 3, figsize=(14.0, 11.5), constrained_layout=True
        )
        legend_handles = {}
        legend_labels = {}
        plot_order = sorted(
            polarization_clusters,
            key=lambda cluster: int(cluster["member_count"]),
            reverse=True,
        )
        for ax, (x_name, y_name, x_label, y_label) in zip(
            axes.ravel()[:8],
            PLOT_PANELS,
        ):
            for draw_index, cluster in enumerate(plot_order):
                cluster_id = int(cluster["polarization_cluster_id"])
                cluster_rows = [
                    row for row in selected_rows
                    if int(row["polarization_cluster_id"]) == cluster_id
                ]
                color, marker = POLARIZATION_CLUSTER_STYLES[cluster_id]
                artist = ax.scatter(
                    [float(row[x_name]) for row in cluster_rows],
                    [float(row[y_name]) for row in cluster_rows],
                    s=42,
                    marker=marker,
                    color=color,
                    edgecolors="black",
                    linewidths=0.45,
                    alpha=0.72 if len(cluster_rows) > 100 else 0.95,
                    rasterized=True,
                    zorder=2 + draw_index,
                )
                if ax is axes.ravel()[0]:
                    legend_handles[cluster_id] = artist
                    legend_labels[cluster_id] = (
                        rf"$P_{{{cluster_id + 1}}}: "
                        rf"(\alpha_e,\alpha_p)\approx"
                        rf"({_pi_quarter_math_label(float(cluster['alpha_e_center']))},"
                        rf"{_pi_quarter_math_label(float(cluster['alpha_p_center']))})$ "
                        f"(n={cluster['member_count']})"
                    )
            ax.scatter(
                [float(best_row[x_name])],
                [float(best_row[y_name])],
                marker="*",
                s=190,
                color="gold",
                edgecolors="black",
                linewidths=0.8,
                zorder=20,
            )
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            configure_named_angle_axes(ax, x_name, y_name)
            ax.tick_params(labelsize=10)

        summary_ax = axes[2, 2]
        summary_ax.axis("off")
        legend_order = sorted(legend_handles)
        summary_ax.legend(
            [legend_handles[cluster_id] for cluster_id in legend_order],
            [legend_labels[cluster_id] for cluster_id in legend_order],
            loc="upper left",
            frameon=False,
            fontsize=9,
            handletextpad=0.8,
        )
        summary_ax.text(
            0.0,
            0.03,
            (
                rf"$D_W-(D_W)_{{\min}}\leq {objective_cut:g}$"
                "\n"
                rf"$(D_W)_{{\min}}={optimum:.7g}$"
                "\n"
                f"{len(selected_rows)}/{len(rows)} minima retained"
                "\n"
                r"$\alpha_e,\alpha_p$ clustered with period $\pi$"
            ),
            transform=summary_ax.transAxes,
            va="bottom",
            fontsize=10,
        )
        fig.suptitle(
            f"{lepton_name}: low-$D_W$ phase space colored by six "
            "major polarization configurations"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _configuration_rows(minimum_rows, lepton_name):
    """Annotate every polarization-cluster member for ConfigGen."""
    prefix = config_scan.mixing_prefix(lepton_name)
    key = _objective_key(lepton_name)
    details = []
    for index, source in enumerate(minimum_rows):
        row = dict(source)
        value = float(row[key])
        parent_id = int(row["polarization_cluster_id"])
        minimum_id = int(row.get("local_minimum_id", index))
        row.update(
            {
                "selected_observable": OBJECTIVE_NAME,
                "selected_observable_label": config_gen.observable_label(
                    OBJECTIVE_NAME
                ),
                "selected_spin_case": "mixing_angles",
                "selected_spin_label": (
                    f"alpha_e={float(row['alpha_e']):.8g}, "
                    f"alpha_p={float(row['alpha_p']):.8g}"
                ),
                "selected_concurrence_key": key,
                "selected_concurrence": value,
                "selected_purity": float(row[f"{prefix}_purity"]),
                "pair_delta_xy": np.nan,
                "scan_phi_p_out": float(row["phi_p_out"]),
                "scan_phi_gamma_out": float(row["phi_gamma_out"]),
                "cluster_id": index,
                "energy_band_cluster_id": index,
                "selected_region": (
                    f"polarization_cluster_{parent_id + 1:02d}_"
                    f"local_minimum_{minimum_id}"
                ),
                "detail_id": (
                    f"{OBJECTIVE_FILE_TAG}_mixing_angles_"
                    f"polarization_{parent_id + 1:02d}_"
                    f"local_minimum_{minimum_id}"
                ),
                "detail_source": "polarization_cluster_member",
                "qOut_regime": "gradient_local_minimum",
            }
        )
        details.append(row)
    return details


def _write_configuration_plot(
    selected_rows,
    detail_rows,
    lepton_name,
    optimum,
    path,
    contours,
):
    """Write configuration pages from supplied 8D contour samples."""
    plt, PdfPages = config_gen._require_matplotlib()
    selected_rows = list(selected_rows)
    if len(selected_rows) != len(detail_rows):
        raise ValueError(
            "Selected local-minimum rows and configuration details disagree."
        )
    full_limits = _full_phase_space_plot_limits(lepton_name)
    objective_key = _objective_key(lepton_name)
    parent_id = int(selected_rows[0]["polarization_cluster_id"])
    parent_configuration = selected_rows[0]["polarization_configuration"]
    if any(
        int(row["polarization_cluster_id"]) != parent_id
        for row in selected_rows
    ):
        raise ValueError(
            "One configuration PDF cannot mix parent polarization clusters."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        overview_fig, overview_axes = plt.subplots(
            3, 3, figsize=(14.0, 11.5), constrained_layout=True
        )
        overview_colors = plt.get_cmap("tab20")(
            np.linspace(0.0, 1.0, max(1, len(selected_rows)))
        )
        for ax, (x_name, y_name, x_label, y_label) in zip(
            overview_axes.ravel()[:8],
            PLOT_PANELS,
        ):
            for selected_index, row in enumerate(selected_rows):
                center = _unit_point_from_minimum_row(row)
                (
                    contour_x,
                    contour_y,
                    center_x,
                    center_y,
                ) = _project_high_dimensional_contour(
                    contours[selected_index],
                    center,
                    x_name,
                    y_name,
                )
                envelope_x, envelope_y = _projected_contour_envelope(
                    contour_x,
                    contour_y,
                )
                envelope_x, envelope_y = _split_wrapped_projection(
                    envelope_x,
                    envelope_y,
                    x_name,
                    y_name,
                )
                color = overview_colors[selected_index]
                if len(envelope_x):
                    ax.plot(
                        envelope_x,
                        envelope_y,
                        color=color,
                        linewidth=1.0,
                        alpha=0.9,
                        zorder=1,
                    )
                ax.scatter(
                    [center_x],
                    [center_y],
                    marker="*",
                    s=80,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=2,
                )
            ax.set_xlim(*full_limits[x_name])
            ax.set_ylim(*full_limits[y_name])
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            configure_named_angle_axes(ax, x_name, y_name)
            ax.tick_params(labelsize=10)

        overview_summary = overview_axes[2, 2]
        overview_summary.axis("off")
        overview_lines = [
            (
                f"polarization cluster P{parent_id + 1}: "
                f"{parent_configuration}"
            ),
            f"{len(selected_rows)} local minima in this polarization cluster",
            (
                f"contour delta={PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}"
            ),
            (
                f"configured 8D samples/minimum="
                f"{PHASE_SPACE_CONFIG_CONTOUR_SAMPLES}"
            ),
            (
                f"{OBJECTIVE_NAME} range="
                f"{min(float(row[objective_key]) for row in selected_rows):.7g}"
                " to "
                f"{max(float(row[objective_key]) for row in selected_rows):.7g}"
            ),
            (
                "global-minimum offsets="
                f"{min(float(row[objective_key]) - optimum for row in selected_rows):.7g}"
                " to "
                f"{max(float(row[objective_key]) - optimum for row in selected_rows):.7g}"
            ),
        ]
        overview_summary.text(
            0.01,
            0.99,
            "\n".join(overview_lines),
            transform=overview_summary.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            family="monospace",
        )
        overview_fig.suptitle(
            f"{lepton_name}: polarization cluster P{parent_id + 1}; "
            "all local-minimum configurations and pairwise projections "
            "of their 8D contours"
        )
        pdf.savefig(overview_fig)
        plt.close(overview_fig)

        for selected_index, (row, detail_row) in enumerate(
            zip(selected_rows, detail_rows)
        ):
            # For each minimum, present the reconstructed configuration before
            # its high-dimensional contour projections.
            display_detail = dict(detail_row)
            display_detail["kinematic_point"] = "gradient_local_minimum"
            config_scan._save_mixing_detail_pages(pdf, plt, [display_detail])
            center = _unit_point_from_minimum_row(row)
            boundary_points = contours[selected_index]
            local_value = float(row[objective_key])
            fig, axes = plt.subplots(
                3, 3, figsize=(14.0, 11.5), constrained_layout=True
            )
            for panel_index, (
                ax,
                (x_name, y_name, x_label, y_label),
            ) in enumerate(zip(axes.ravel()[:8], PLOT_PANELS)):
                (
                    contour_x,
                    contour_y,
                    center_x,
                    center_y,
                ) = _project_high_dimensional_contour(
                    boundary_points,
                    center,
                    x_name,
                    y_name,
                )
                envelope_x, envelope_y = _projected_contour_envelope(
                    contour_x,
                    contour_y,
                )
                display_x = _wrap_projected_values(contour_x, x_name)
                display_y = _wrap_projected_values(contour_y, y_name)
                envelope_x, envelope_y = _split_wrapped_projection(
                    envelope_x,
                    envelope_y,
                    x_name,
                    y_name,
                )
                ax.scatter(
                    display_x,
                    display_y,
                    s=7,
                    color="tab:red",
                    alpha=0.18,
                    linewidths=0.0,
                    label=(
                        "projected 8D contour samples"
                        if panel_index == 0 else None
                    ),
                    zorder=1,
                )
                if len(envelope_x):
                    ax.plot(
                        envelope_x,
                        envelope_y,
                        color="tab:red",
                        linewidth=1.5,
                        label=(
                            rf"${OBJECTIVE_LATEX}="
                            rf"({OBJECTIVE_LATEX})_{{\mathrm{{local}}}}"
                            rf"+{PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}$ projection"
                            if panel_index == 0 else None
                        ),
                        zorder=2,
                    )
                ax.scatter(
                    [center_x],
                    [center_y],
                    marker="*",
                    s=180,
                    c="gold",
                    edgecolors="black",
                    label=(
                        "selected minimum"
                        if panel_index == 0 else None
                    ),
                    zorder=3,
                )
                ax.set_xlim(*full_limits[x_name])
                ax.set_ylim(*full_limits[y_name])
                ax.set_xlabel(x_label, fontsize=11)
                ax.set_ylabel(y_label, fontsize=11)
                configure_named_angle_axes(ax, x_name, y_name)
                ax.tick_params(labelsize=10)

            axes[0, 0].legend(fontsize=8)
            summary_ax = axes[2, 2]
            summary_ax.axis("off")
            coordinates = _plot_coordinate_values(center)
            summary_lines = [
                "selected local minimum",
                "",
                f"{OBJECTIVE_NAME}(local) = {local_value:.8g}",
                (
                    f"{OBJECTIVE_NAME} contour = "
                    f"{local_value + PHASE_SPACE_CONFIG_CONTOUR_DELTA:.8g}"
                ),
                (
                    f"above global minimum = "
                    f"{local_value - optimum:.8g}"
                ),
                f"8D contour samples = {len(boundary_points)}",
                "",
                "selected phase-space point:",
            ]
            summary_lines.extend(
                f"{name:>10s} = {coordinates[name]:.7g}"
                for name in (
                    "sqrt_s",
                    "theta_p_out",
                    "theta_gamma_out",
                    "qOut",
                    "phi_p_out",
                    "phi_gamma_out",
                    "alpha_e",
                    "alpha_p",
                )
            )
            summary_ax.text(
                0.03,
                0.97,
                "\n".join(summary_lines),
                transform=summary_ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig.suptitle(
                f"{lepton_name}: polarization cluster P{parent_id + 1}, "
                f"local minimum {row.get('local_minimum_id', selected_index)}; "
                "pairwise projections of the 8D "
                rf"${OBJECTIVE_LATEX}="
                rf"({OBJECTIVE_LATEX})_{{\mathrm{{local}}}}"
                rf"+{PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}$ contour"
            )
            pdf.savefig(fig)
            plt.close(fig)
    return path


def _write_configurations(
    lepton_name,
    polarization_cluster_id,
    selected_minimum_rows,
    optimum,
    *,
    save_contour_data,
    use_saved_contour_data,
):
    """Generate one parent polarization's data and PDF package."""
    paths = configuration_data_paths(
        lepton_name,
        polarization_cluster_id,
    )
    selected_path = _write_csv(paths["selected"], selected_minimum_rows)
    config_gen.configure_lepton(
        lepton_name,
        input_path=selected_path,
    )
    details = _configuration_rows(selected_minimum_rows, lepton_name)
    config_scan._plain_write_csv(paths["examples"], details)
    config_scan._plain_write_csv(
        paths["clusters"], config_scan._mixing_cluster_rows(details)
    )
    config_scan._plain_write_csv(
        paths["momenta"], config_scan._mixing_momentum_rows(details)
    )
    config_scan._plain_write_csv(
        paths["amplitudes"], config_scan._mixing_amplitude_rows(details)
    )
    contour_started = perf_counter()
    if use_saved_contour_data:
        contours = _load_contour_data(
            paths["contours"],
            selected_minimum_rows,
        )
        contour_source = f"loaded saved contour samples: {paths['contours']}"
        contour_timing_label = "contour data load time"
    else:
        contours = _configuration_contours(
            selected_minimum_rows,
            lepton_name,
        )
        if save_contour_data:
            _write_contour_data(
                paths["contours"],
                selected_minimum_rows,
                contours,
            )
            contour_source = f"saved contour samples: {paths['contours']}"
        else:
            contour_source = "saved contour samples: disabled"
        contour_timing_label = "contour generation time"
    contour_seconds = perf_counter() - contour_started
    plot_path = _configuration_plot_path(
        lepton_name,
        polarization_cluster_id,
    )
    _write_configuration_plot(
        selected_minimum_rows,
        details,
        lepton_name,
        optimum,
        plot_path,
        contours,
    )
    return (
        paths,
        plot_path,
        contour_seconds,
        contour_timing_label,
        contour_source,
    )


def _physical_start_to_unit_point(start):
    """Map a readable physical start dictionary into the optimizer unit box."""
    sqrt_s = float(start["sqrt_s"])
    s = sqrt_s**2
    qout_fraction = float(start["qOut"]) / phase_scan._qout_max(s)
    return np.asarray(
        (
            (
                sqrt_s - phase_scan.SQRT_S_RANGE[0]
            ) / (
                phase_scan.SQRT_S_RANGE[1]
                - phase_scan.SQRT_S_RANGE[0]
            ),
            (
                float(start["theta_p_out"])
                - phase_scan.THETA_P_OUT_RANGE[0]
            ) / (
                phase_scan.THETA_P_OUT_RANGE[1]
                - phase_scan.THETA_P_OUT_RANGE[0]
            ),
            (
                float(start["theta_gamma_out"])
                - phase_scan.THETA_GAMMA_OUT_RANGE[0]
            ) / (
                phase_scan.THETA_GAMMA_OUT_RANGE[1]
                - phase_scan.THETA_GAMMA_OUT_RANGE[0]
            ),
            (
                qout_fraction - phase_scan.QOUT_FRACTION_RANGE[0]
            ) / (
                phase_scan.QOUT_FRACTION_RANGE[1]
                - phase_scan.QOUT_FRACTION_RANGE[0]
            ),
            float(start["phi_p_out"]) / (2.0 * np.pi),
            float(start["phi_gamma_out"]) / (2.0 * np.pi),
            float(start["alpha_e"]) / np.pi,
            float(start["alpha_p"]) / np.pi,
        ),
        dtype=float,
    )


def _screen_start_task(task):
    """Evaluate one low-discrepancy candidate without optimizing it."""
    lepton_name, candidate_index, point, objective_name = task
    phase_scan._configure_lepton(lepton_name)
    value, _row = _objective_evaluation(
        point,
        lepton_name,
        evaluation_id=2_000_000_000 + candidate_index,
        objective_name=objective_name,
    )
    return candidate_index, value, point


def _run_screening_tasks(tasks):
    """Evaluate global screening candidates with the shared worker policy."""
    workers = min(max(1, int(GRADIENT_WORKERS)), len(tasks))
    if workers <= 1:
        return [_screen_start_task(task) for task in tasks]
    chunksize = max(1, len(tasks) // (8 * workers))
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    _screen_start_task,
                    tasks,
                    chunksize=chunksize,
                )
            )
    except (OSError, PermissionError, BrokenProcessPool):
        return [_screen_start_task(task) for task in tasks]


def _screened_sobol_starts(lepton_name, species_seed):
    """Select low-objective, spatially separated Sobol starts."""
    exponent = int(np.log2(ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES))
    candidates = qmc.Sobol(
        d=SCAN_DIMENSION,
        scramble=True,
        seed=species_seed + 10_000,
    ).random_base2(exponent)
    tasks = [
        (lepton_name, index, point, OBJECTIVE_NAME)
        for index, point in enumerate(candidates)
    ]
    evaluated = sorted(
        _run_screening_tasks(tasks),
        key=lambda result: result[1],
    )
    selected = []
    deferred = []
    for _candidate_index, value, point in evaluated:
        if not np.isfinite(value) or value >= INVALID_OBJECTIVE:
            continue
        if all(
            _unit_distance(point, prior[1])
            >= ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION
            for prior in selected
        ):
            selected.append((value, point))
            if len(selected) == ENTANGLEMENT_GRADIENT_SCREENED_STARTS:
                break
        else:
            deferred.append((value, point))
    if len(selected) < ENTANGLEMENT_GRADIENT_SCREENED_STARTS:
        for value, point in deferred:
            if any(np.array_equal(point, prior[1]) for prior in selected):
                continue
            selected.append((value, point))
            if len(selected) == ENTANGLEMENT_GRADIENT_SCREENED_STARTS:
                break
    return selected


def _species_tasks(lepton_name):
    """Return hybrid global, screened, and physics-anchored start tasks."""
    phase_scan._configure_lepton(lepton_name)
    species_seed = (
        ENTANGLEMENT_GRADIENT_RANDOM_SEED
        + tuple(LEPTON_SPECS).index(lepton_name)
    )
    latin_starts = qmc.LatinHypercube(
        d=SCAN_DIMENSION,
        seed=species_seed,
    ).random(
        ENTANGLEMENT_GRADIENT_RANDOM_STARTS
    )
    starts = [
        (point, "latin_hypercube", np.nan)
        for point in latin_starts
    ]
    starts.extend(
        (point, "sobol_screened", value)
        for value, point in _screened_sobol_starts(
            lepton_name,
            species_seed,
        )
    )
    for anchor in PHYSICS_ANCHOR_STARTS.get(lepton_name, ()):
        point = _physical_start_to_unit_point(anchor)
        if np.all(np.isfinite(point)) and np.all(
            (0.0 <= point) & (point <= 1.0)
        ):
            starts.append(
                (
                    point,
                    f"physics_anchor:{anchor['name']}",
                    np.nan,
                )
            )
    return [
        (
            lepton_name,
            run_index,
            point,
            source,
            screening_value,
            OBJECTIVE_NAME,
        )
        for run_index, (point, source, screening_value) in enumerate(starts)
    ]


def _physics_reference_rows(lepton_name):
    """Evaluate and retain every exact physics anchor before optimization."""
    rows = []
    for reference_index, anchor in enumerate(
        PHYSICS_ANCHOR_STARTS.get(lepton_name, ())
    ):
        unit_point = _physical_start_to_unit_point(anchor)
        value, row = _objective_evaluation(
            unit_point,
            lepton_name,
            evaluation_id=3_000_000_000 + reference_index,
        )
        if row is None:
            raise RuntimeError(
                f"Physics reference {anchor['name']!r} is not a valid "
                "eight-dimensional phase-space point."
            )
        item = dict(row)
        item.update({
            "record_type": "exact_physics_reference",
            "reference_name": anchor["name"],
            "reference_index": reference_index,
            "reference_objective": value,
            "optimization_start_source": (
                f"physics_anchor:{anchor['name']}"
            ),
        })
        for coordinate, coordinate_value in enumerate(unit_point):
            item[f"reference_u{coordinate}"] = float(coordinate_value)
        rows.append(item)
    return rows


def scan_species_minima(lepton_name, results=None):
    """Find and save every distinct verified minimum for one species."""
    phase_scan._configure_lepton(lepton_name)
    minimum_scan_started = perf_counter()
    tasks = _species_tasks(lepton_name)
    if results is None:
        results = _run_tasks(tasks)
    minima = _deduplicate_minima(results)
    minimum_scan_seconds = perf_counter() - minimum_scan_started
    if not minima:
        raise RuntimeError(
            f"No converged, locally verified {OBJECTIVE_NAME} minimum was "
            f"found for "
            f"{lepton_name}; inspect the optimizer settings or increase "
            "ENTANGLEMENT_GRADIENT_MAX_ITERATIONS."
        )

    output_dirs = species_output_dirs(lepton_name)
    scan_data_dir = output_dirs["scan_data"]
    plot_dir = output_dirs["plots"]
    run_path = _write_csv(
        scan_data_dir / "optimization_runs.csv",
        [result[0] for result in results],
    )
    reference_rows = _physics_reference_rows(lepton_name)
    references_path = _write_csv(
        scan_data_dir / "physics_reference_points.csv",
        reference_rows,
    )
    minimum_rows = []
    for minimum_index, (_run, _unit_point, row) in enumerate(minima):
        item = dict(row)
        item["local_minimum_id"] = minimum_index
        item["kinematic_point"] = f"gradient_local_minimum_{minimum_index:04d}"
        minimum_rows.append(item)
    optimum = float(np.min(_objective_values(minimum_rows, lepton_name)))
    minima_path = _write_csv(
        scan_data_dir / "local_minima.csv",
        minimum_rows,
    )
    minima_plot = _plot_all_local_minima(
        minimum_rows,
        lepton_name,
        plot_dir / "all_local_minima.pdf",
        reference_rows=reference_rows,
    )
    lbfgs_converged = sum(
        bool(result[0]["lbfgs_success"]) for result in results
    )
    verified = sum(
        bool(result[0]["local_minimum_verified"])
        for result in results
    )
    latin_count = sum(task[3] == "latin_hypercube" for task in tasks)
    screened_count = sum(task[3] == "sobol_screened" for task in tasks)
    anchor_count = sum(
        str(task[3]).startswith("physics_anchor:")
        for task in tasks
    )
    screened_values = [
        float(task[4])
        for task in tasks
        if task[3] == "sobol_screened" and np.isfinite(task[4])
    ]
    report_lines = [
        f"Hybrid global gradient {OBJECTIVE_NAME} search ({lepton_name})",
        f"  optimization starts: {len(tasks)}",
        f"  Latin-hypercube starts: {latin_count}",
        (
            f"  Sobol screening: "
            f"{ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES} "
            f"candidates -> {screened_count} optimized starts"
        ),
        (
            f"  best screened {OBJECTIVE_NAME}: "
            f"{min(screened_values):.10g}"
            if screened_values
            else f"  best screened {OBJECTIVE_NAME}: unavailable"
        ),
        f"  deterministic physics anchors: {anchor_count}",
        f"  shared optimization workers: {GRADIENT_WORKERS}",
        f"  L-BFGS-B-converged runs: {lbfgs_converged}/{len(tasks)}",
        f"  multiscale-verified runs: {verified}/{len(tasks)}",
        f"  distinct finite minima: {len(minima)}",
        f"  minimum scan time: {minimum_scan_seconds:.3f} s",
        f"  best {OBJECTIVE_NAME}: {optimum:.10g}",
        f"  optimization runs: {run_path}",
        f"  exact physics reference points: {references_path}",
        f"  local minima: {minima_path}",
        f"  all-local-minima plot: {minima_plot}",
    ]
    return "\n".join(report_lines)


def remake_species_minima_plot(lepton_name):
    """Rebuild the minima plot from stage-1 CSV data without searching."""
    output_dirs = species_output_dirs(lepton_name)
    minima_path = output_dirs["scan_data"] / "local_minima.csv"
    minimum_rows = _read_csv(minima_path)
    references_path = (
        output_dirs["scan_data"] / "physics_reference_points.csv"
    )
    reference_rows = (
        _read_csv(references_path) if references_path.exists() else []
    )
    plot_started = perf_counter()
    minima_plot = _plot_all_local_minima(
        minimum_rows,
        lepton_name,
        output_dirs["plots"] / "all_local_minima.pdf",
        reference_rows=reference_rows,
    )
    plot_seconds = perf_counter() - plot_started
    return "\n".join(
        (
            f"Remade gradient {OBJECTIVE_NAME} minima plot ({lepton_name})",
            f"  source minima: {minima_path}",
            f"  local minima loaded: {len(minimum_rows)}",
            f"  plot generation time: {plot_seconds:.3f} s",
            f"  all-local-minima plot: {minima_plot}",
            "  screening and gradient optimization were not rerun",
        )
    )


def cluster_species_minima(
    lepton_name,
    *,
    polarization_objective_cut,
    polarization_cluster_count,
    polarization_cluster_seed,
):
    """Cluster low-objective minima only in polarization space."""
    output_dirs = species_output_dirs(lepton_name)
    minima_path = output_dirs["scan_data"] / "local_minima.csv"
    cluster_started = perf_counter()
    minimum_rows = _read_csv(minima_path)
    clustered_rows, polarization_clusters, optimum = _cluster_polarization_minima(
        minimum_rows,
        lepton_name,
        objective_cut=polarization_objective_cut,
        cluster_count=polarization_cluster_count,
        random_seed=polarization_cluster_seed,
    )
    polarization_path = _write_csv(
        output_dirs["cluster_data"] / "polarization_clusters.csv",
        polarization_clusters,
    )
    polarization_plot = _polarization_cluster_plot(
        clustered_rows,
        polarization_clusters,
        lepton_name,
        polarization_objective_cut,
        optimum,
        output_dirs["plots"] / "polarization_cluster_phase_space.pdf",
    )
    retained = sum(
        _as_bool(row["within_polarization_cluster_cut"])
        for row in clustered_rows
    )
    clustered_path = _write_csv(
        output_dirs["cluster_data"] / "clustered_minima.csv",
        clustered_rows,
    )
    cluster_seconds = perf_counter() - cluster_started
    return "\n".join(
        (
            (
                f"Gradient {OBJECTIVE_NAME} polarization clustering "
                f"({lepton_name})"
            ),
            f"  source minima: {minima_path}",
            f"  local minima loaded: {len(minimum_rows)}",
            (
                f"  polarization cut: {OBJECTIVE_NAME} - "
                f"{OBJECTIVE_NAME}_min <= {polarization_objective_cut:g}"
            ),
            f"  minima passing polarization cut: {retained}",
            (
                f"  pi-periodic polarization clusters: "
                f"{len(polarization_clusters)}"
            ),
            (
                "  one best-objective representative selected per "
                "polarization cluster"
            ),
            f"  clustering time: {cluster_seconds:.3f} s",
            f"  clustered minima: {clustered_path}",
            f"  polarization cluster summary: {polarization_path}",
            f"  polarization phase-space plot: {polarization_plot}",
        )
    )


def configure_species_clusters(
    lepton_name,
    *,
    polarization_clusters_to_configure,
    save_contour_data,
    use_saved_contour_data,
):
    """Configure every member of every polarization cluster."""
    output_dirs = species_output_dirs(lepton_name)
    clustered_path = output_dirs["cluster_data"] / "clustered_minima.csv"
    clustered_rows = _read_csv(clustered_path)
    required = {
        "polarization_cluster_id",
        "polarization_configuration",
        "polarization_cluster_representative",
    }
    missing = required - set(clustered_rows[0])
    if missing:
        raise ValueError(
            f"Clustered minima are missing required columns "
            f"{sorted(missing)}; rerun GradientPhaseSpaceCluster.py."
        )
    objective_values = _objective_values(clustered_rows, lepton_name)
    optimum = float(np.min(objective_values))
    parent_ids = sorted(
        {
            int(row["polarization_cluster_id"])
            for row in clustered_rows
            if row["polarization_cluster_id"] not in ("", None)
        }
    )
    if not parent_ids:
        raise RuntimeError(
            f"No parent polarization clusters are available for "
            f"{lepton_name}; rerun GradientPhaseSpaceCluster.py."
        )
    if polarization_clusters_to_configure is not None:
        requested_ids = {
            int(cluster_number) - 1
            for cluster_number in polarization_clusters_to_configure
        }
        invalid_ids = requested_ids - set(parent_ids)
        if invalid_ids:
            invalid_numbers = sorted(cluster_id + 1 for cluster_id in invalid_ids)
            raise ValueError(
                f"Requested polarization clusters {invalid_numbers} are "
                f"not available for {lepton_name}."
            )
        parent_ids = [
            parent_id for parent_id in parent_ids
            if parent_id in requested_ids
        ]

    package_reports = []
    total_selected = 0
    total_contour_seconds = 0.0
    for parent_id in parent_ids:
        parent_rows = [
            row for row in clustered_rows
            if (
                row["polarization_cluster_id"] not in ("", None)
                and int(row["polarization_cluster_id"]) == parent_id
            )
        ]
        selected_rows = list(parent_rows)
        if not selected_rows:
            raise RuntimeError(
                f"Polarization cluster P{parent_id + 1} has no minima "
                f"for {lepton_name}."
            )
        (
            paths,
            config_plot,
            contour_seconds,
            contour_timing_label,
            contour_source,
        ) = _write_configurations(
            lepton_name,
            parent_id,
            selected_rows,
            optimum,
            save_contour_data=save_contour_data,
            use_saved_contour_data=use_saved_contour_data,
        )
        total_selected += len(selected_rows)
        total_contour_seconds += contour_seconds
        package_reports.extend(
            (
                (
                    f"  P{parent_id + 1}: "
                    f"{selected_rows[0]['polarization_configuration']}"
                ),
                f"    retained minima: {len(parent_rows)}",
                (
                    "    configured local minima: "
                    f"{len(selected_rows)}"
                ),
                f"    selected minima: {paths['selected']}",
                f"    {contour_source}",
                (
                    f"    {contour_timing_label}: "
                    f"{contour_seconds:.3f} s"
                ),
                f"    configuration examples: {paths['examples']}",
                f"    cluster summary: {paths['clusters']}",
                f"    momentum configurations: {paths['momenta']}",
                f"    amplitude decomposition: {paths['amplitudes']}",
                f"    configuration PDF: {config_plot}",
            )
        )
    return "\n".join(
        (
            f"Gradient {OBJECTIVE_NAME} cluster ConfigGen ({lepton_name})",
            f"  source clustered minima: {clustered_path}",
            f"  local minima loaded: {len(clustered_rows)}",
            f"  parent polarization PDFs: {len(parent_ids)}",
            f"  configured polarization-cluster minima: {total_selected}",
            (
                f"  local-minimum contour delta: "
                f"{PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}"
            ),
            (
                f"  total contour processing time: "
                f"{total_contour_seconds:.3f} s"
            ),
            *package_reports,
        )
    )


def _validate_scan_settings():
    """Validate controls used by the local-minimum search stage."""
    if not all(
        (
            OBJECTIVE_NAME,
            OBJECTIVE_FILE_TAG,
            OBJECTIVE_LATEX,
            OBJECTIVE_STATE_FILE_LABEL,
            SCAN_KEY,
        )
    ):
        raise ValueError(
            "The gradient tool must be configured with a complete scan "
            "definition before validation."
        )
    if OUTPUT_ROOT is None:
        raise ValueError("The scan definition must provide an output root.")
    if not SCAN_INITIAL_MIXING_ANGLES:
        raise ValueError(
            "The gradient phase-space tool requires "
            "SCAN_INITIAL_MIXING_ANGLES=True."
        )
    unknown = set(LEPTONS_TO_PROCESS) - set(LEPTON_SPECS)
    if unknown:
        raise ValueError(f"Unknown lepton species: {sorted(unknown)}")
    if not LEPTONS_TO_PROCESS:
        raise ValueError("The active lepton selection must not be empty.")
    if GRADIENT_WORKERS < 1:
        raise ValueError("GRADIENT_WORKERS must be positive.")
    if ENTANGLEMENT_GRADIENT_RANDOM_STARTS < 1:
        raise ValueError(
            "ENTANGLEMENT_GRADIENT_RANDOM_STARTS must be positive."
        )
    if (
        not isinstance(
            ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES,
            (int, np.integer),
        )
        or isinstance(
            ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES,
            (bool, np.bool_),
        )
        or ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES < 2
        or (
            ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES
            & (ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES - 1)
        )
    ):
        raise ValueError(
            "ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES must be a "
            "power-of-two integer."
        )
    if (
        not isinstance(
            ENTANGLEMENT_GRADIENT_SCREENED_STARTS,
            (int, np.integer),
        )
        or isinstance(
            ENTANGLEMENT_GRADIENT_SCREENED_STARTS,
            (bool, np.bool_),
        )
        or not 1
        <= ENTANGLEMENT_GRADIENT_SCREENED_STARTS
        <= ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES
    ):
        raise ValueError(
            "ENTANGLEMENT_GRADIENT_SCREENED_STARTS must be an integer "
            "between 1 and ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES."
        )
    if (
        not np.isfinite(ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION)
        or not 0.0
        < ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION
        <= 1.0
    ):
        raise ValueError(
            "ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION must lie "
            "in (0, 1]."
        )
    if ENTANGLEMENT_GRADIENT_MAX_ITERATIONS < 1:
        raise ValueError(
            "ENTANGLEMENT_GRADIENT_MAX_ITERATIONS must be positive."
        )
    for name, value in (
        (
            "ENTANGLEMENT_GRADIENT_TOLERANCE",
            ENTANGLEMENT_GRADIENT_TOLERANCE,
        ),
        (
            "ENTANGLEMENT_GRADIENT_SCAN_PRECISION",
            ENTANGLEMENT_GRADIENT_SCAN_PRECISION,
        ),
        (
            "ENTANGLEMENT_GRADIENT_MINIMUM_SEPARATION",
            ENTANGLEMENT_GRADIENT_MINIMUM_SEPARATION,
        ),
        (
            "ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP",
            ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP,
        ),
        (
            "ENTANGLEMENT_LOCAL_SEARCH_OBJECTIVE_TOLERANCE",
            ENTANGLEMENT_LOCAL_SEARCH_OBJECTIVE_TOLERANCE,
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if ENTANGLEMENT_GRADIENT_SCAN_PRECISION > 1.0:
        raise ValueError(
            "ENTANGLEMENT_GRADIENT_SCAN_PRECISION must not exceed "
            "the normalized scan width of 1."
        )
    if not 0.0 < ENTANGLEMENT_LOCAL_SEARCH_STEP_REDUCTION < 1.0:
        raise ValueError(
            "ENTANGLEMENT_LOCAL_SEARCH_STEP_REDUCTION must lie "
            "strictly between 0 and 1."
        )
    if (
        ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP
        < ENTANGLEMENT_GRADIENT_SCAN_PRECISION
    ):
        raise ValueError(
            "ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP must be at least "
            "ENTANGLEMENT_GRADIENT_SCAN_PRECISION."
        )
    if ENTANGLEMENT_LOCAL_SEARCH_MAX_POLLS < 1:
        raise ValueError(
            "ENTANGLEMENT_LOCAL_SEARCH_MAX_POLLS must be positive."
        )
    if ENTANGLEMENT_LOCAL_SEARCH_RANDOM_DIRECTIONS < 0:
        raise ValueError(
            "ENTANGLEMENT_LOCAL_SEARCH_RANDOM_DIRECTIONS must be "
            "non-negative."
        )
    for lepton_name, anchors in PHYSICS_ANCHOR_STARTS.items():
        if lepton_name not in LEPTON_SPECS:
            raise ValueError(
                f"Unknown physics-anchor lepton species: {lepton_name!r}"
            )
        phase_scan._configure_lepton(lepton_name)
        for anchor in anchors:
            point = _physical_start_to_unit_point(anchor)
            if (
                not np.all(np.isfinite(point))
                or np.any(point < 0.0)
                or np.any(point > 1.0)
            ):
                raise ValueError(
                    f"Physics anchor {anchor.get('name')!r} for "
                    f"{lepton_name} lies outside the configured scan box."
                )


def _validate_stage_runtime():
    """Validate definition, lepton selection, and worker state."""
    if not all(
        (
            OBJECTIVE_NAME,
            OBJECTIVE_FILE_TAG,
            OBJECTIVE_LATEX,
            OBJECTIVE_STATE_FILE_LABEL,
            SCAN_KEY,
        )
    ):
        raise ValueError(
            "The gradient tool requires a complete scan definition."
        )
    if OUTPUT_ROOT is None:
        raise ValueError("The scan definition must provide an output root.")
    if not SCAN_INITIAL_MIXING_ANGLES:
        raise ValueError(
            "The gradient workflow requires "
            "SCAN_INITIAL_MIXING_ANGLES=True."
        )
    unknown = set(LEPTONS_TO_PROCESS) - set(LEPTON_SPECS)
    if unknown:
        raise ValueError(f"Unknown lepton species: {sorted(unknown)}")
    if not LEPTONS_TO_PROCESS:
        raise ValueError("The active lepton selection must not be empty.")
    if GRADIENT_WORKERS < 1:
        raise ValueError("The worker count must be positive.")


def _validate_cluster_settings():
    """Validate only settings consumed by the clustering stage."""
    _validate_stage_runtime()


def _validate_config_settings():
    """Validate only settings consumed by ConfigGen and contour plotting."""
    _validate_stage_runtime()
    if (
        not np.isfinite(PHASE_SPACE_CONFIG_CONTOUR_DELTA)
        or PHASE_SPACE_CONFIG_CONTOUR_DELTA <= 0.0
    ):
        raise ValueError(
            "PHASE_SPACE_CONFIG_CONTOUR_DELTA must be finite and positive."
        )
    if (
        not isinstance(PHASE_SPACE_CONFIG_CONTOUR_SAMPLES, (int, np.integer))
        or isinstance(PHASE_SPACE_CONFIG_CONTOUR_SAMPLES, (bool, np.bool_))
        or PHASE_SPACE_CONFIG_CONTOUR_SAMPLES < 2 * SCAN_DIMENSION
    ):
        raise ValueError(
            "PHASE_SPACE_CONFIG_CONTOUR_SAMPLES must be an integer "
            f"of at least {2 * SCAN_DIMENSION}."
        )
    if CONFIG_CONTOUR_BISECTION_ITERATIONS < 1:
        raise ValueError(
            "CONFIG_CONTOUR_BISECTION_ITERATIONS must be positive."
        )
    if (
        not np.isfinite(CONFIG_CONTOUR_INITIAL_RADIUS)
        or CONFIG_CONTOUR_INITIAL_RADIUS <= 0.0
    ):
        raise ValueError(
            "CONFIG_CONTOUR_INITIAL_RADIUS must be finite and positive."
        )


def _write_stage_report(stage, reports):
    """Write and print one independent stage report."""
    report = "\n\n".join(reports) + "\n"
    log_path = stage_log_path(stage)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report, encoding="utf-8")
    print_console_text(report)
    return report


def run_minima_scan(
    definition,
    *,
    leptons_to_scan,
    gradient_workers,
    remake_plot_from_csv=False,
):
    """Search for minima, or rebuild their plot from the saved stage-1 CSV."""
    if not isinstance(remake_plot_from_csv, bool):
        raise TypeError("remake_plot_from_csv must be a bool.")
    configure_scan(
        definition,
        leptons_to_process=leptons_to_scan,
        gradient_workers=gradient_workers,
    )
    if remake_plot_from_csv:
        _validate_stage_runtime()
        reports = [
            remake_species_minima_plot(lepton_name)
            for lepton_name in LEPTONS_TO_PROCESS
        ]
    else:
        _validate_scan_settings()
        reports = [
            scan_species_minima(lepton_name)
            for lepton_name in LEPTONS_TO_PROCESS
        ]
    return _write_stage_report("scan", reports)


def run_phase_space_clustering(
    definition,
    *,
    leptons_to_cluster,
    polarization_objective_cut,
    polarization_cluster_count,
    polarization_cluster_seed,
):
    """Cluster phase space and optionally classify low-objective polarization."""
    if not isinstance(polarization_cluster_count, int):
        raise TypeError("polarization_cluster_count must be an int.")
    if polarization_cluster_count < 1:
        raise ValueError("polarization_cluster_count must be positive.")
    if polarization_cluster_count > len(POLARIZATION_CLUSTER_STYLES):
        raise ValueError(
            f"At most {len(POLARIZATION_CLUSTER_STYLES)} distinct "
            "polarization marker/color styles are available."
        )
    if (
        polarization_objective_cut is None
        or not np.isfinite(polarization_objective_cut)
        or polarization_objective_cut < 0.0
    ):
        raise ValueError(
            "polarization_objective_cut must be finite and non-negative."
        )
    if not isinstance(polarization_cluster_seed, int):
        raise TypeError("polarization_cluster_seed must be an int.")
    configure_scan(
        definition,
        leptons_to_process=leptons_to_cluster,
        gradient_workers=1,
    )
    _validate_cluster_settings()
    reports = [
        cluster_species_minima(
            lepton_name,
            polarization_objective_cut=polarization_objective_cut,
            polarization_cluster_count=polarization_cluster_count,
            polarization_cluster_seed=polarization_cluster_seed,
        )
        for lepton_name in LEPTONS_TO_PROCESS
    ]
    return _write_stage_report("cluster", reports)


def run_cluster_configgen(
    definition,
    *,
    leptons_to_configure,
    config_workers,
    polarization_clusters_to_configure,
    save_contour_data,
    use_saved_contour_data,
):
    """Run only ConfigGen and contour plotting from clustered minima."""
    if not isinstance(save_contour_data, bool):
        raise TypeError("save_contour_data must be a bool.")
    if not isinstance(use_saved_contour_data, bool):
        raise TypeError("use_saved_contour_data must be a bool.")
    configure_scan(
        definition,
        leptons_to_process=leptons_to_configure,
        gradient_workers=config_workers,
    )
    _validate_config_settings()
    reports = [
        configure_species_clusters(
            lepton_name,
            polarization_clusters_to_configure=(
                polarization_clusters_to_configure
            ),
            save_contour_data=save_contour_data,
            use_saved_contour_data=use_saved_contour_data,
        )
        for lepton_name in LEPTONS_TO_PROCESS
    ]
    return _write_stage_report("config", reports)
