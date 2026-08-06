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
import shutil
import subprocess
import tempfile
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
from PlotUtils import configure_phase_space_axes, print_console_text
from GradientObjective import (
    PAIRWISE_CONCURRENCE_NAMES,
    objective_value,
    source_observable_name,
)


# Runtime state is populated only through :func:`configure_scan`.
GRADIENT_LEPTON_NAMES = ("electron", "muon")
GRADIENT_LEPTON_SPECS = {
    name: LEPTON_SPECS[name] for name in GRADIENT_LEPTON_NAMES
}
LEPTONS_TO_PROCESS = ()
GRADIENT_WORKERS = SCAN_WORKERS
OUTPUT_ROOT = None
OBJECTIVE_NAME = ""
OBJECTIVE_FILE_TAG = ""
OBJECTIVE_LATEX = ""
OBJECTIVE_STATE_FILE_LABEL = ""
TARGET_OBSERVABLE_NAME = ""
SCAN_KEY = ""
PHYSICS_ANCHOR_STARTS = {}

INVALID_OBJECTIVE = 1.0e3
SCAN_DIMENSION = 8
PERIODIC_UNIT_COORDINATES = (4, 5, 6, 7)
CONFIG_CONTOUR_BISECTION_ITERATIONS = 8
CONFIG_CONTOUR_INITIAL_RADIUS = 0.01
PHASE_SPACE_PLOT_PADDING_FRACTION = 0.025
CONTOUR_AXIS_LABEL_FONTSIZE = 13
CONTOUR_TICK_FONTSIZE = 12
CONTOUR_SUMMARY_FONTSIZE = 12
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
# High-contrast, color-blind-safe styles used by the standalone correlation
# PDFs.  These are intentionally independent of the compact summary-page
# styles, where smaller markers work better in the 3x3 layout.
POLARIZATION_CORRELATION_STYLES = (
    ("#0072B2", "o"),
    ("#D55E00", "s"),
    ("#009E73", "^"),
    ("#ED63FF", "D"),
    ("#E69F00", "P"),
    ("#000000", "X"),
)
POLARIZATION_CLUSTER_RESTARTS = 16
W_POLARIZATION_ALPHA_E_LINE_CENTERS = (
    np.pi / 4.0,
    3.0 * np.pi / 4.0,
)
W_POLARIZATION_ALPHA_E_STRATUM_CLUSTERS = (1, 2, 1, 2)
GHZ_ALPHA_P_CLUSTERS_PER_ALPHA_E_REGION = 2
FIXED_AXIS_POLARIZATION_CENTERS = {
    "CPGAMMA": ("alpha_e", (0.0, np.pi / 2.0, np.pi)),
    "CEGAMMA": (
        "alpha_p",
        (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi),
    ),
}
EXAMPLE_POLARIZATION_CLUSTER_IDS = {
    "GHZ": 1,      # P2
    "CEP": 1,      # P2, matching the GHZ-style partition
    "CPGAMMA": 1,  # P2, alpha_e around pi/2
    "CEGAMMA": 2,  # P3, alpha_p around pi/2
    "W": 3,        # P4
}


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
    global OBJECTIVE_STATE_FILE_LABEL, TARGET_OBSERVABLE_NAME, SCAN_KEY
    global OUTPUT_ROOT
    global PHYSICS_ANCHOR_STARTS
    global LEPTONS_TO_PROCESS, GRADIENT_WORKERS

    OBJECTIVE_NAME = str(definition.objective_name)
    OBJECTIVE_FILE_TAG = str(definition.file_tag)
    OBJECTIVE_LATEX = str(definition.latex)
    OBJECTIVE_STATE_FILE_LABEL = str(definition.state_file_label)
    TARGET_OBSERVABLE_NAME = source_observable_name(OBJECTIVE_NAME)
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
    species_label = (
        GRADIENT_LEPTON_SPECS[lepton_name]["label"].title().replace(" ", "_")
    )
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
    if lepton_name not in GRADIENT_LEPTON_SPECS:
        raise ValueError(
            f"Unknown lepton {lepton_name!r}; choose from "
            f"{GRADIENT_LEPTON_NAMES}."
        )
    root = OUTPUT_ROOT / lepton_name
    return {
        "root": root,
        "data": root / "Data" / SCAN_KEY,
        "scan_data": root / "Data" / SCAN_KEY / "scan",
        "contour_data": root / "Data" / SCAN_KEY / "contour",
        "cluster_data": root / "Data" / SCAN_KEY / "cluster",
        "plots": root / "Plots" / SCAN_KEY,
    }


def minimum_contour_data_paths(lepton_name):
    """Return the pre-cluster contour package keyed by raw minimum ID."""
    contour_dir = species_output_dirs(lepton_name)["contour_data"]
    return {
        "minima": contour_dir / "local_minima",
        "index": contour_dir / "local_minimum_contour_index.csv",
    }


def minimum_contour_data_path(lepton_name, minimum_id):
    """Return one raw minimum's only authoritative contour file."""
    return (
        minimum_contour_data_paths(lepton_name)["minima"]
        / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
    )


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
        lepton_mass=GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
    )
    if result is None or result[1] is None:
        return INVALID_OBJECTIVE, None
    row = result[1]
    value = objective_value(
        row,
        config_scan.mixing_prefix(lepton_name),
        objective_name or OBJECTIVE_NAME,
        store=True,
    )
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
    target_name = source_observable_name(objective_name)
    if target_name != objective_name:
        target_key = _objective_key(lepton_name, target_name)
        for stage, stage_row in (
            ("initial", _start_row),
            ("lbfgs", _lbfgs_row),
            ("final", final_row),
        ):
            run[_run_objective_key(stage, target_name)] = (
                np.nan
                if stage_row is None
                else float(stage_row.get(target_key, np.nan))
            )
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
    """Map one current-schema physical minimum back to the optimizer box."""
    sqrt_s = float(row["sqrt_s"])
    s = sqrt_s**2
    qout_fraction = float(row["qOut"]) / phase_scan._qout_max(s)
    unit_point = np.asarray(
        (
            (
                sqrt_s - phase_scan.SQRT_S_RANGE[0]
            )
            / (
                phase_scan.SQRT_S_RANGE[1]
                - phase_scan.SQRT_S_RANGE[0]
            ),
            (
                float(row["theta_p_out"])
                - phase_scan.THETA_P_OUT_RANGE[0]
            )
            / (
                phase_scan.THETA_P_OUT_RANGE[1]
                - phase_scan.THETA_P_OUT_RANGE[0]
            ),
            (
                float(row["theta_gamma_out"])
                - phase_scan.THETA_GAMMA_OUT_RANGE[0]
            )
            / (
                phase_scan.THETA_GAMMA_OUT_RANGE[1]
                - phase_scan.THETA_GAMMA_OUT_RANGE[0]
            ),
            (
                qout_fraction - phase_scan.QOUT_FRACTION_RANGE[0]
            )
            / (
                phase_scan.QOUT_FRACTION_RANGE[1]
                - phase_scan.QOUT_FRACTION_RANGE[0]
            ),
            float(row["phi_p_out"]) / (2.0 * np.pi),
            float(row["phi_gamma_out"]) / (2.0 * np.pi),
            float(row["alpha_e"]) / np.pi,
            float(row["alpha_p"]) / np.pi,
        ),
        dtype=float,
    )
    unit_point[:4] = np.clip(unit_point[:4], 0.0, 1.0)
    unit_point[4:] %= 1.0
    return unit_point


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
        center = _unit_point_from_minimum_row(row)
        base_value = float(row[_objective_key(lepton_name)])
        direction_chunks = np.array_split(
            _contour_directions(
                ENTANGLEMENT_GRADIENT_RANDOM_SEED
                + int(row["local_minimum_id"])
            ),
            chunks_per_minimum,
        )
        tasks.extend(
            (
                row_index,
                chunk_index,
                center,
                base_value,
                lepton_name,
                GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
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
    """Serialize contour centers, samples, and validation metadata."""
    csv_rows = []
    for row_index, row in enumerate(selected_rows):
        center = _unit_point_from_minimum_row(row)
        minimum_id = row["local_minimum_id"]
        base = {
            "objective_name": OBJECTIVE_NAME,
            "objective_file_tag": OBJECTIVE_FILE_TAG,
            # This remains stable when independently generated per-minimum
            # files are combined or later reordered by clustering.
            "local_minimum_index": minimum_id,
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


def _load_contour_data(
    path,
    selected_rows,
    *,
    return_settings=False,
):
    """Load contours by minimum ID and validate their saved metadata."""
    path = Path(path)
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
        saved_delta = float(saved["contour_delta"])
        if not np.isfinite(saved_delta) or saved_delta <= 0.0:
            raise ValueError(
                f"Saved contour delta must be finite and positive: {path}"
            )
        saved_direction_count = int(saved["configured_direction_count"])
        if saved_direction_count < 2 * SCAN_DIMENSION:
            raise ValueError(
                f"Saved contour direction count must be at least "
                f"{2 * SCAN_DIMENSION}: {path}"
            )
        if not np.isclose(
            saved_delta,
            PHASE_SPACE_CONFIG_CONTOUR_DELTA,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                "Saved contour delta does not match the current contour "
                f"generation setting: {path}"
            )
        if (
            saved_direction_count != PHASE_SPACE_CONFIG_CONTOUR_SAMPLES
        ):
            raise ValueError(
                "Saved contour direction count does not match the current "
                f"contour generation setting: {path}"
            )
        minimum_id = str(saved["local_minimum_id"])
        samples.setdefault(minimum_id, [])
        sample_ids.setdefault(minimum_id, [])
        if saved["record_type"] == "minimum":
            if minimum_id in metadata:
                raise ValueError(
                    f"Saved contour repeats metadata for local-minimum ID "
                    f"{minimum_id!r}: {path}"
                )
            metadata[minimum_id] = saved
        elif saved["record_type"] == "sample":
            sample_ids[minimum_id].append(int(saved["contour_sample_id"]))
            samples[minimum_id].append(
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

    loaded = {}
    settings = {}
    for output_index, row in enumerate(selected_rows):
        expected_id = str(row["local_minimum_id"])
        if expected_id not in metadata:
            raise ValueError(
                f"Saved contour data is missing local-minimum ID "
                f"{expected_id!r}: {path}"
            )
        saved = metadata[expected_id]
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
        if sample_ids[expected_id] != list(range(expected_count)):
            raise ValueError(
                f"Saved contour samples for minimum {expected_id} are "
                f"incomplete or out of order: {path}"
            )
        loaded[output_index] = np.asarray(
            samples[expected_id],
            dtype=float,
        ).reshape((-1, SCAN_DIMENSION))
        settings[output_index] = {
            "contour_delta": float(saved["contour_delta"]),
            "configured_direction_count": int(
                saved["configured_direction_count"]
            ),
        }
    if return_settings:
        return loaded, settings
    return loaded


def _load_minimum_contour_data(lepton_name, selected_rows):
    """Load each contour from its raw-minimum-owned CSV."""
    loaded = {}
    settings = {}
    for output_index, row in enumerate(selected_rows):
        path = minimum_contour_data_path(
            lepton_name,
            row["local_minimum_id"],
        )
        one_loaded, one_settings = _load_contour_data(
            path,
            [row],
            return_settings=True,
        )
        loaded[output_index] = one_loaded[0]
        settings[output_index] = one_settings[0]
    return loaded, settings


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


def _padded_plot_limits(limits):
    """Return finite ordered limits with marker-safe padding on both sides."""
    lower, upper = (float(value) for value in limits)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Plot limits must be finite and strictly increasing.")
    padding = PHASE_SPACE_PLOT_PADDING_FRACTION * (upper - lower)
    return lower - padding, upper + padding


def _full_phase_space_plot_limits(lepton_name):
    """Return every physical plot range with marker-safe boundary padding."""
    phase_scan._configure_lepton(lepton_name)
    qout_upper = max(
        phase_scan._qout_max(sqrt_s**2)
        for sqrt_s in phase_scan.SQRT_S_RANGE
    ) * phase_scan.QOUT_FRACTION_RANGE[1]
    physical_limits = {
        "sqrt_s": tuple(phase_scan.SQRT_S_RANGE),
        "theta_p_out": tuple(phase_scan.THETA_P_OUT_RANGE),
        "theta_gamma_out": tuple(phase_scan.THETA_GAMMA_OUT_RANGE),
        "qOut": (0.0, float(qout_upper)),
        "phi_p_out": tuple(phase_scan.AZIMUTH_RANGE),
        "phi_gamma_out": tuple(phase_scan.AZIMUTH_RANGE),
        "alpha_e": tuple(phase_scan.ALPHA_E_RANGE),
        "alpha_p": tuple(phase_scan.ALPHA_P_RANGE),
    }
    return {
        coordinate: _padded_plot_limits(limits)
        for coordinate, limits in physical_limits.items()
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
            configure_phase_space_axes(
                ax,
                x_name,
                y_name,
                lepton_mass=GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
            )
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
    """Return the pi-periodic circular center of each supplied angle."""
    return np.mod(
        0.5
        * np.arctan2(
            np.mean(np.sin(2.0 * angles), axis=0),
            np.mean(np.cos(2.0 * angles), axis=0),
        ),
        np.pi,
    )


def _mixing_distance(angle, center):
    """Return the normalized distance used by the active cluster scheme."""
    difference = np.abs(np.asarray(angle) - np.asarray(center))
    fixed_axis_spec = FIXED_AXIS_POLARIZATION_CENTERS.get(SCAN_KEY)
    if fixed_axis_spec is not None:
        axis_name, _fixed_centers = fixed_axis_spec
        axis_index = 0 if axis_name == "alpha_e" else 1
        return float(difference[axis_index] / np.pi)
    if SCAN_KEY in ("GHZ", "CEP"):
        if np.isclose(np.mod(center[0], np.pi), 0.0):
            difference[0] = min(difference[0], np.pi - difference[0])
        difference[1] = min(difference[1], np.pi - difference[1])
        return float(np.linalg.norm(difference / np.pi))
    if TARGET_OBSERVABLE_NAME in PAIRWISE_CONCURRENCE_NAMES:
        difference = np.minimum(difference, np.pi - difference)
        return float(np.linalg.norm(difference / np.pi))
    if np.isclose(np.mod(center[0], np.pi), 0.0):
        difference[0] = min(difference[0], np.pi - difference[0])
    difference[1] = min(difference[1], np.pi - difference[1])
    return float(np.linalg.norm(difference / np.pi))


def _best_periodic_kmeans_labels(
    embedded,
    cluster_count,
    random_seed,
):
    """Return the lowest-distortion result from deterministic k-means restarts."""
    random_generator = np.random.default_rng(random_seed)
    best_labels = None
    best_distortion = np.inf
    for _ in range(POLARIZATION_CLUSTER_RESTARTS):
        embedded_centers, labels = kmeans2(
            embedded,
            cluster_count,
            iter=100,
            minit="++",
            seed=random_generator,
        )
        if len(set(int(label) for label in labels)) != cluster_count:
            continue
        distortion = float(
            np.sum((embedded - embedded_centers[labels]) ** 2)
        )
        if distortion < best_distortion:
            best_distortion = distortion
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError(
            "Periodic mixing-angle clustering did not produce a complete "
            f"{cluster_count}-cluster result in "
            f"{POLARIZATION_CLUSTER_RESTARTS} deterministic restarts."
        )
    return best_labels


def _polarization_alpha_e_strata(
    alpha_e,
    *,
    cluster_count,
    alpha_e_line_half_width,
    alpha_e_boundaries,
):
    """Return state-specific alpha_e strata and clustering metadata."""
    if SCAN_KEY == "W":
        expected_cluster_count = sum(
            W_POLARIZATION_ALPHA_E_STRATUM_CLUSTERS
        )
        if cluster_count != expected_cluster_count:
            raise ValueError(
                f"W polarization cluster count must be "
                f"{expected_cluster_count}."
            )
        first_line, second_line = W_POLARIZATION_ALPHA_E_LINE_CENTERS
        near_first_line = (
            np.abs(alpha_e - first_line) <= alpha_e_line_half_width
        )
        near_second_line = (
            np.abs(alpha_e - second_line) <= alpha_e_line_half_width
        )
        midpoint = (
            ~(near_first_line | near_second_line)
            & (alpha_e >= first_line)
            & (alpha_e < second_line)
        )
        endpoint = ~(near_first_line | midpoint | near_second_line)
        strata = (
            ("0/pi", 0.0, endpoint, 1),
            (
                "pi/4 line",
                first_line,
                near_first_line,
                W_POLARIZATION_ALPHA_E_STRATUM_CLUSTERS[1],
            ),
            ("pi/2", np.pi / 2.0, midpoint, 1),
            (
                "3pi/4 line",
                second_line,
                near_second_line,
                W_POLARIZATION_ALPHA_E_STRATUM_CLUSTERS[3],
            ),
        )
        return strata, "W narrow alpha_e capture bands"

    if SCAN_KEY in ("GHZ", "CEP"):
        boundaries = np.asarray(alpha_e_boundaries, dtype=float)
        expected_cluster_count = (
            (len(boundaries) - 1)
            * GHZ_ALPHA_P_CLUSTERS_PER_ALPHA_E_REGION
        )
        if cluster_count != expected_cluster_count:
            raise ValueError(
                f"{SCAN_KEY} polarization cluster count must be "
                f"{expected_cluster_count} for alpha_e boundaries "
                f"{tuple(boundaries)}."
            )
        strata = []
        for region_index, (lower, upper) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            if region_index + 1 == len(boundaries) - 1:
                mask = (alpha_e >= lower) & (alpha_e <= upper)
                closing = "]"
            else:
                mask = (alpha_e >= lower) & (alpha_e < upper)
                closing = ")"
            region_name = (
                f"[{_pi_quarter_label(lower)},"
                f"{_pi_quarter_label(upper)}{closing}"
            )
            strata.append(
                (
                    region_name,
                    0.5 * (lower + upper),
                    mask,
                    GHZ_ALPHA_P_CLUSTERS_PER_ALPHA_E_REGION,
                )
            )
        return tuple(strata), f"{SCAN_KEY} alpha_e boundary regions"

    if TARGET_OBSERVABLE_NAME in PAIRWISE_CONCURRENCE_NAMES:
        return (
            ("full periodic plane", np.nan, np.ones_like(alpha_e, dtype=bool),
             cluster_count),
        ), "pairwise-concurrence periodic (alpha_e, alpha_p) k-means"

    raise ValueError(f"No polarization clustering algorithm for {SCAN_KEY!r}.")


def _polarization_configuration_label(alpha_e_region, center):
    """Return one state-specific polarization configuration label."""
    if SCAN_KEY in ("GHZ", "CEP"):
        return (
            f"alpha_e in {alpha_e_region},"
            f"alpha_p~{_pi_quarter_label(center[1])}"
        )
    if SCAN_KEY == "CPGAMMA":
        return f"alpha_e~{_pi_quarter_label(center[0])}"
    if SCAN_KEY == "CEGAMMA":
        return f"alpha_p~{_pi_quarter_label(center[1])}"
    return (
        f"alpha_e~{_pi_quarter_label(center[0])},"
        f"alpha_p~{_pi_quarter_label(center[1])}"
    )


def _cluster_polarization_minima(
    rows,
    lepton_name,
    *,
    objective_cut,
    cluster_count,
    random_seed,
    alpha_e_line_half_width,
    alpha_e_boundaries,
):
    """Classify minima using the active state's alpha_e partition."""
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
    alpha_e = angles[:, 0]
    assignments = {}
    members_by_cluster = {}
    centers = {}
    alpha_e_regions = {}
    fixed_axis_spec = FIXED_AXIS_POLARIZATION_CENTERS.get(SCAN_KEY)
    if fixed_axis_spec is not None:
        axis_name, fixed_centers = fixed_axis_spec
        axis_index = 0 if axis_name == "alpha_e" else 1
        fixed_centers = np.asarray(fixed_centers, dtype=float)
        if cluster_count != len(fixed_centers):
            raise ValueError(
                f"{SCAN_KEY} polarization cluster count must be "
                f"{len(fixed_centers)}."
            )
        fixed_labels = np.argmin(
            np.abs(angles[:, axis_index, None] - fixed_centers[None, :]),
            axis=1,
        )
        clustering_scheme = (
            f"{SCAN_KEY} nearest fixed {axis_name} quarter-pi center"
        )
        for cluster_id, fixed_center in enumerate(fixed_centers):
            member_positions = np.flatnonzero(fixed_labels == cluster_id)
            if not len(member_positions):
                raise RuntimeError(
                    f"The {SCAN_KEY} {axis_name} cluster around "
                    f"{_pi_quarter_label(fixed_center)} retained no minima."
                )
            members = [
                int(eligible_indices[position])
                for position in member_positions
            ]
            members_by_cluster[cluster_id] = members
            center = np.asarray(
                _circular_mixing_center(angles[member_positions]),
                dtype=float,
            )
            center[axis_index] = fixed_center
            centers[cluster_id] = center
            alpha_e_regions[cluster_id] = (
                f"around {_pi_quarter_label(fixed_center)}"
                if axis_name == "alpha_e"
                else "all alpha_e"
            )
            assignments.update(
                {row_index: cluster_id for row_index in members}
            )
    else:
        strata, clustering_scheme = _polarization_alpha_e_strata(
            alpha_e,
            cluster_count=cluster_count,
            alpha_e_line_half_width=alpha_e_line_half_width,
            alpha_e_boundaries=alpha_e_boundaries,
        )
        next_cluster_id = 0
        for stratum_id, (
            stratum_name,
            alpha_e_center,
            stratum_mask,
            stratum_cluster_count,
        ) in enumerate(strata):
            stratum_positions = np.flatnonzero(stratum_mask)
            if len(stratum_positions) < stratum_cluster_count:
                raise RuntimeError(
                    f"The alpha_e={stratum_name} stratum retained "
                    f"{len(stratum_positions)} minima, fewer than its requested "
                    f"{stratum_cluster_count} clusters. Increase the objective "
                    "cut or revise the state-specific alpha_e partition."
                )

            stratum_angles = angles[stratum_positions]
            alpha_p = stratum_angles[:, 1]
            if (
                TARGET_OBSERVABLE_NAME in PAIRWISE_CONCURRENCE_NAMES
                and SCAN_KEY != "CEP"
            ):
                embedded_angles = np.column_stack(
                    (
                        np.cos(2.0 * stratum_angles[:, 0]),
                        np.sin(2.0 * stratum_angles[:, 0]),
                        np.cos(2.0 * alpha_p),
                        np.sin(2.0 * alpha_p),
                    )
                )
            else:
                embedded_angles = np.column_stack(
                    (np.cos(2.0 * alpha_p), np.sin(2.0 * alpha_p))
                )
            stratum_labels = _best_periodic_kmeans_labels(
                embedded_angles,
                stratum_cluster_count,
                random_seed + stratum_id,
            )
            raw_ids = sorted(set(int(label) for label in stratum_labels))
            circular_centers = {
                raw_id: _circular_mixing_center(
                    angles[stratum_positions[stratum_labels == raw_id]]
                )
                for raw_id in raw_ids
            }
            ordered_raw_ids = sorted(
                raw_ids,
                key=lambda raw_id: (
                    int(np.rint(4.0 * circular_centers[raw_id][0] / np.pi)),
                    int(np.rint(4.0 * circular_centers[raw_id][1] / np.pi)),
                    *circular_centers[raw_id],
                ),
            )
            for raw_id in ordered_raw_ids:
                cluster_id = next_cluster_id
                next_cluster_id += 1
                member_positions = stratum_positions[
                    stratum_labels == raw_id
                ]
                members = [
                    int(eligible_indices[position])
                    for position in member_positions
                ]
                members_by_cluster[cluster_id] = members
                if (
                    TARGET_OBSERVABLE_NAME in PAIRWISE_CONCURRENCE_NAMES
                    and SCAN_KEY != "CEP"
                ):
                    centers[cluster_id] = circular_centers[raw_id]
                else:
                    centers[cluster_id] = np.asarray(
                        (alpha_e_center, circular_centers[raw_id][1])
                    )
                alpha_e_regions[cluster_id] = stratum_name
                assignments.update(
                    {row_index: cluster_id for row_index in members}
                )
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
                    "polarization_alpha_e_region": "",
                    "polarization_clustering_scheme": clustering_scheme,
                }
            )
        else:
            center = centers[cluster_id]
            configuration = _polarization_configuration_label(
                alpha_e_regions[cluster_id],
                center,
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
                    "polarization_alpha_e_region": (
                        alpha_e_regions[cluster_id]
                    ),
                    "polarization_clustering_scheme": clustering_scheme,
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
                    _polarization_configuration_label(
                        alpha_e_regions[cluster_id],
                        center,
                    )
                ),
                "polarization_alpha_e_region": alpha_e_regions[cluster_id],
                "polarization_clustering_scheme": clustering_scheme,
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
                "alpha_e_line_half_width": (
                    ""
                    if alpha_e_line_half_width is None
                    else alpha_e_line_half_width
                ),
                "alpha_e_line_half_width_over_pi": (
                    ""
                    if alpha_e_line_half_width is None
                    else alpha_e_line_half_width / np.pi
                ),
                "alpha_e_boundaries": (
                    ""
                    if alpha_e_boundaries is None
                    else ",".join(
                        f"{float(boundary):.17g}"
                        for boundary in alpha_e_boundaries
                    )
                ),
                "distance_coordinates": (
                    (
                        "alpha_e (narrow pi/4 and 3pi/4 capture bands), "
                        "alpha_p (pi-periodic)"
                    )
                    if SCAN_KEY == "W"
                    else (
                        "alpha_e (regions bounded by 0, pi/2, pi), "
                        "alpha_p (pi-periodic within each region)"
                    )
                    if SCAN_KEY in ("GHZ", "CEP")
                    else (
                        "alpha_e (nearest fixed 0, pi/2, pi center)"
                        if SCAN_KEY == "CPGAMMA"
                        else (
                            "alpha_p (nearest fixed quarter-pi center)"
                            if SCAN_KEY == "CEGAMMA"
                            else "alpha_e and alpha_p (both pi-periodic)"
                        )
                    )
                ),
                objective_key: float(representative[objective_key]),
            }
        )
    return annotated, summary, optimum


def _draw_alpha_e_cluster_guides(
    ax,
    x_name,
    y_name,
    alpha_e_line_half_width,
    alpha_e_boundaries,
):
    """Draw the active state's alpha_e partition on one projection."""
    if SCAN_KEY == "W":
        if x_name == "alpha_e":
            for alpha_e_line in W_POLARIZATION_ALPHA_E_LINE_CENTERS:
                ax.axvspan(
                    alpha_e_line - alpha_e_line_half_width,
                    alpha_e_line + alpha_e_line_half_width,
                    color="black",
                    alpha=0.06,
                    zorder=0,
                )
                ax.axvline(
                    alpha_e_line,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=1,
                )
        if y_name == "alpha_e":
            for alpha_e_line in W_POLARIZATION_ALPHA_E_LINE_CENTERS:
                ax.axhspan(
                    alpha_e_line - alpha_e_line_half_width,
                    alpha_e_line + alpha_e_line_half_width,
                    color="black",
                    alpha=0.06,
                    zorder=0,
                )
                ax.axhline(
                    alpha_e_line,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=1,
                )
        return

    fixed_axis_spec = FIXED_AXIS_POLARIZATION_CENTERS.get(SCAN_KEY)
    if fixed_axis_spec is not None:
        axis_name, fixed_centers = fixed_axis_spec
        if x_name == axis_name:
            for center in fixed_centers:
                ax.axvline(
                    center,
                    color="black",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.55,
                    zorder=1,
                )
        if y_name == axis_name:
            for center in fixed_centers:
                ax.axhline(
                    center,
                    color="black",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.55,
                    zorder=1,
                )
        return

    if SCAN_KEY not in ("GHZ", "CEP"):
        return

    interior_boundaries = tuple(alpha_e_boundaries[1:-1])
    if x_name == "alpha_e":
        for boundary in interior_boundaries:
            ax.axvline(
                boundary,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )
    if y_name == "alpha_e":
        for boundary in interior_boundaries:
            ax.axhline(
                boundary,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )


def _project_polarization_contours(rows, lepton_name):
    """Load, validate, and cache the eight envelopes for each minimum."""
    # Contour metadata stores normalized coordinates whose physical map is
    # species dependent.  Plot-only callers do not otherwise enter the scan
    # evaluator, so select the species explicitly before validating centers.
    phase_scan._configure_lepton(lepton_name)
    print(
        f"Projecting {len(rows)} validated {OBJECTIVE_NAME} minimum "
        f"contours for {lepton_name}...",
        flush=True,
    )
    projected_contours = {}
    for row_number, row in enumerate(rows, start=1):
        minimum_id = int(row["local_minimum_id"])
        contours, _settings = _load_minimum_contour_data(lepton_name, [row])
        center = _unit_point_from_minimum_row(row)
        panel_projections = {}
        projection_panels = (*PLOT_PANELS, *POLARIZATION_CORRELATION_PANELS)
        for x_name, y_name, _x_label, _y_label in projection_panels:
            panel_key = (x_name, y_name)
            if panel_key in panel_projections:
                continue
            contour_x, contour_y, center_x, center_y = (
                _project_high_dimensional_contour(
                    contours[0], center, x_name, y_name
                )
            )
            envelope_x, envelope_y = _projected_contour_envelope(
                contour_x, contour_y
            )
            envelope_x, envelope_y = _split_wrapped_projection(
                envelope_x, envelope_y, x_name, y_name
            )
            panel_projections[panel_key] = (
                envelope_x,
                envelope_y,
                center_x,
                center_y,
            )
        projected_contours[minimum_id] = panel_projections
        if row_number % 100 == 0 or row_number == len(rows):
            print(
                f"  projected contours: {row_number}/{len(rows)}",
                flush=True,
            )
    return projected_contours


def _polarization_cluster_plot(
    rows,
    polarization_clusters,
    lepton_name,
    objective_cut,
    optimum,
    alpha_e_line_half_width,
    alpha_e_boundaries,
    path,
    *,
    include_contours=False,
):
    """Plot saved contour projections for all minima and each cluster.

    The existing ``polarization_cluster_phase_space.pdf`` path is retained for
    the full unclustered view.  Every cluster is exported alongside it as a
    separate one-page ``..._PXX.pdf`` document.  Contours are loaded and
    validated one minimum at a time, then reduced to their eight projected
    convex envelopes so the full contour collection does not reside in memory.
    """
    plt, _PdfPages = config_gen._require_matplotlib()
    selected_rows = [
        row for row in rows
        if _as_bool(row["within_polarization_cluster_cut"])
    ]
    if not selected_rows:
        raise RuntimeError("No rows passed the polarization-cluster cut.")
    objective_key = _objective_key(lepton_name)
    best_row = min(selected_rows, key=lambda row: float(row[objective_key]))
    path.parent.mkdir(parents=True, exist_ok=True)
    full_limits = _full_phase_space_plot_limits(lepton_name)

    projected_contours = (
        _project_polarization_contours(selected_rows, lepton_name)
        if include_contours else {}
    )

    def draw_phase_space_page(
        page_rows,
        page_path,
        *,
        title,
        summary_lines,
        marker,
        point_colors,
        contour_colors,
        representative,
        point_cmap=None,
        point_vmin=None,
        point_vmax=None,
        show_objective_colorbar=False,
    ):
        fig, axes = plt.subplots(
            3, 3, figsize=(14.0, 11.5), constrained_layout=True
        )
        contour_alpha = 0.13 if len(page_rows) > 100 else 0.32
        for panel_index, (
            ax,
            (x_name, y_name, x_label, y_label),
        ) in enumerate(zip(axes.ravel()[:8], PLOT_PANELS)):
            for row, contour_color in zip(page_rows, contour_colors):
                minimum_id = int(row["local_minimum_id"])
                projection = projected_contours.get(minimum_id, {}).get(
                    (x_name, y_name)
                )
                if projection is None:
                    continue
                envelope_x, envelope_y, _center_x, _center_y = projection
                if len(envelope_x):
                    ax.plot(
                        envelope_x,
                        envelope_y,
                        color=contour_color,
                        linewidth=0.55,
                        alpha=contour_alpha,
                        rasterized=True,
                        zorder=1,
                    )
            image = ax.scatter(
                [float(row[x_name]) for row in page_rows],
                [float(row[y_name]) for row in page_rows],
                c=point_colors,
                cmap=point_cmap,
                vmin=point_vmin,
                vmax=point_vmax,
                s=35,
                marker=marker,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.82,
                rasterized=True,
                zorder=2,
            )
            ax.scatter(
                [float(representative[x_name])],
                [float(representative[y_name])],
                marker="*",
                s=190,
                color="gold",
                edgecolors="black",
                linewidths=0.8,
                label="Example" if panel_index == 0 else None,
                zorder=4,
            )
            ax.set_xlim(*full_limits[x_name])
            ax.set_ylim(*full_limits[y_name])
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            _draw_alpha_e_cluster_guides(
                ax,
                x_name,
                y_name,
                alpha_e_line_half_width,
                alpha_e_boundaries,
            )
            configure_phase_space_axes(
                ax,
                x_name,
                y_name,
                lepton_mass=GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
            )
            ax.tick_params(labelsize=10)
            if panel_index == 0:
                ax.legend(fontsize=8, frameon=False)

        summary_ax = axes[2, 2]
        summary_ax.axis("off")
        summary_ax.text(
            0.01,
            0.99,
            "\n".join(summary_lines),
            transform=summary_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
        )
        if show_objective_colorbar:
            fig.colorbar(
                image,
                ax=axes.ravel()[:8].tolist(),
                label=rf"${OBJECTIVE_LATEX}$",
            )
        fig.suptitle(title)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_path)
        plt.close(fig)
        return page_path

    unclustered_style_index = EXAMPLE_POLARIZATION_CLUSTER_IDS[SCAN_KEY]
    unclustered_color, unclustered_marker = (
        POLARIZATION_CORRELATION_STYLES[unclustered_style_index]
    )
    draw_phase_space_page(
        selected_rows,
        path,
        title=(
            f"{lepton_name}: full low-${OBJECTIVE_LATEX}$ phase space "
            + (
                "with projected 8D contours"
                if include_contours else "(contours omitted)"
            )
        ),
        summary_lines=[
            "unclustered full retained-minimum set",
            f"minima = {len(selected_rows)}",
            (
                f"validated contours = {len(selected_rows)}"
                if include_contours else "contours omitted"
            ),
            f"total raw minima = {len(rows)}",
            f"{OBJECTIVE_NAME} minimum = {optimum:.8g}",
            f"objective cut above minimum = {objective_cut:.8g}",
            f"contour delta = {PHASE_SPACE_CONFIG_CONTOUR_DELTA:.8g}",
            f"samples per 8D contour = {PHASE_SPACE_CONFIG_CONTOUR_SAMPLES}",
            "star = best retained minimum (Example)",
        ],
        marker=unclustered_marker,
        point_colors=[unclustered_color] * len(selected_rows),
        contour_colors=[unclustered_color] * len(selected_rows),
        representative=best_row,
    )

    cmap_name, _style_vmin, _style_vmax = (
        config_gen.observable_plot_style(OBJECTIVE_NAME)
    )
    cmap = plt.get_cmap(cmap_name)
    color_upper = optimum + objective_cut
    if color_upper <= optimum:
        color_upper = optimum + max(1.0e-12, abs(optimum) * 1.0e-12)
    color_span = color_upper - optimum
    for cluster in sorted(
        polarization_clusters,
        key=lambda item: int(item["polarization_cluster_id"]),
    ):
        cluster_id = int(cluster["polarization_cluster_id"])
        cluster_rows = [
            row for row in selected_rows
            if int(row["polarization_cluster_id"]) == cluster_id
        ]
        if not cluster_rows:
            raise RuntimeError(
                f"Polarization cluster {cluster_id + 1} has no members."
            )
        cluster_values = np.asarray(
            [float(row[objective_key]) for row in cluster_rows]
        )
        normalized_values = np.clip(
            (cluster_values - optimum) / color_span, 0.0, 1.0
        )
        row_colors = [cmap(value) for value in normalized_values]
        representative = min(
            cluster_rows, key=lambda row: float(row[objective_key])
        )
        _cluster_color, cluster_marker = (
            POLARIZATION_CLUSTER_STYLES[cluster_id]
        )
        distances = np.asarray([
            float(row["polarization_cluster_distance"])
            for row in cluster_rows
        ])
        cluster_path = path.with_name(
            f"{path.stem}_P{cluster_id + 1:02d}{path.suffix}"
        )
        draw_phase_space_page(
            cluster_rows,
            cluster_path,
            title=(
                f"{lepton_name}: polarization cluster P{cluster_id + 1} "
                + (
                    "with projected 8D contours"
                    if include_contours else "(contours omitted)"
                )
            ),
            summary_lines=[
                (
                    f"polarization cluster P{cluster_id + 1}: "
                    f"{cluster['polarization_configuration']}"
                ),
                f"members = {len(cluster_rows)}",
                (
                    f"validated contours = {len(cluster_rows)}"
                    if include_contours else "contours omitted"
                ),
                (
                    "representative local minimum = "
                    f"{cluster['representative_local_minimum_id']}"
                ),
                "",
                f"{OBJECTIVE_NAME} best = {cluster_values.min():.8g}",
                f"{OBJECTIVE_NAME} mean = {cluster_values.mean():.8g}",
                f"{OBJECTIVE_NAME} max = {cluster_values.max():.8g}",
                f"center alpha_e/pi = {float(cluster['alpha_e_center_over_pi']):.8g}",
                f"center alpha_p/pi = {float(cluster['alpha_p_center_over_pi']):.8g}",
                f"mean normalized distance = {distances.mean():.8g}",
                f"max normalized distance = {distances.max():.8g}",
                f"contour delta = {PHASE_SPACE_CONFIG_CONTOUR_DELTA:.8g}",
                f"samples per 8D contour = {PHASE_SPACE_CONFIG_CONTOUR_SAMPLES}",
            ],
            marker=cluster_marker,
            point_colors=cluster_values,
            contour_colors=row_colors,
            representative=representative,
            point_cmap=cmap,
            point_vmin=optimum,
            point_vmax=color_upper,
            show_objective_colorbar=True,
        )
    return path


def _write_polarization_correlation_pdfs(
    rows,
    polarization_clusters,
    lepton_name,
    objective_cut,
    optimum,
    alpha_e_line_half_width,
    alpha_e_boundaries,
    output_dir,
    *,
    include_contours=False,
):
    """Export matching summary and individual clustered/unclustered PDFs."""
    plt, _PdfPages = config_gen._require_matplotlib()
    selected_rows = [
        row for row in rows
        if _as_bool(row["within_polarization_cluster_cut"])
    ]
    if not selected_rows:
        raise RuntimeError("No rows passed the polarization-cluster cut.")
    representative_rows = [
        row for row in selected_rows
        if _as_bool(row["polarization_cluster_representative"])
    ]
    representative_ids = {
        int(row["polarization_cluster_id"]) for row in representative_rows
    }
    expected_ids = {
        int(cluster["polarization_cluster_id"])
        for cluster in polarization_clusters
    }
    if representative_ids != expected_ids or (
        len(representative_rows) != len(expected_ids)
    ):
        raise RuntimeError(
            "Expected exactly one representative minimum per polarization "
            "cluster."
        )
    example_cluster_id = EXAMPLE_POLARIZATION_CLUSTER_IDS[SCAN_KEY]
    example_row = next(
        row for row in representative_rows
        if int(row["polarization_cluster_id"]) == example_cluster_id
    )
    output_dir = Path(output_dir)
    index_rows = []
    projected_contours = (
        _project_polarization_contours(selected_rows, lepton_name)
        if include_contours else {}
    )
    plot_order = sorted(
        polarization_clusters,
        key=lambda cluster: int(cluster["member_count"]),
        reverse=True,
    )

    def draw_panel(
        ax,
        mode,
        x_name,
        y_name,
        x_label,
        y_label,
        *,
        summary_panel=False,
    ):
        """Draw one correlation panel with mode-independent axes styling."""
        contour_alpha = 0.10 if len(selected_rows) > 100 else 0.28
        contour_linewidth = 0.45 if summary_panel else 0.60
        for row in selected_rows:
            projection = projected_contours.get(
                int(row["local_minimum_id"]), {}
            ).get((x_name, y_name))
            if projection is None:
                continue
            if mode == "clustered":
                contour_color = POLARIZATION_CORRELATION_STYLES[
                    int(row["polarization_cluster_id"])
                ][0]
            else:
                contour_color = POLARIZATION_CORRELATION_STYLES[
                    EXAMPLE_POLARIZATION_CLUSTER_IDS[SCAN_KEY]
                ][0]
            envelope_x, envelope_y, _center_x, _center_y = projection
            if len(envelope_x):
                ax.plot(
                    envelope_x,
                    envelope_y,
                    color=contour_color,
                    linewidth=contour_linewidth,
                    alpha=contour_alpha,
                    rasterized=True,
                    zorder=1,
                )
        if mode == "clustered":
            for draw_index, cluster in enumerate(plot_order):
                cluster_id = int(cluster["polarization_cluster_id"])
                cluster_rows = [
                    row for row in selected_rows
                    if int(row["polarization_cluster_id"]) == cluster_id
                ]
                color, marker = POLARIZATION_CORRELATION_STYLES[cluster_id]
                ax.scatter(
                    [float(row[x_name]) for row in cluster_rows],
                    [float(row[y_name]) for row in cluster_rows],
                    s=42 if summary_panel else 58,
                    marker=marker,
                    color=color,
                    edgecolors="black",
                    linewidths=0.55 if summary_panel else 0.7,
                    alpha=0.84 if len(cluster_rows) > 100 else 0.98,
                    rasterized=True,
                    label=(
                        f"P{cluster_id + 1} (n={len(cluster_rows)})"
                        if summary_panel else None
                    ),
                    zorder=2 + draw_index,
                )
        else:
            unclustered_style_index = EXAMPLE_POLARIZATION_CLUSTER_IDS[
                SCAN_KEY
            ]
            unclustered_color, unclustered_marker = (
                POLARIZATION_CORRELATION_STYLES[
                    unclustered_style_index
                ]
            )
            ax.scatter(
                [float(row[x_name]) for row in selected_rows],
                [float(row[y_name]) for row in selected_rows],
                s=40 if summary_panel else 52,
                marker=unclustered_marker,
                color=unclustered_color,
                edgecolors="black",
                linewidths=0.5 if summary_panel else 0.6,
                alpha=0.78 if len(selected_rows) > 100 else 0.95,
                rasterized=True,
                label=None,
                zorder=2,
            )

        displayed_representatives = (
            representative_rows if mode == "clustered" else [example_row]
        )
        for representative_index, representative in enumerate(
            displayed_representatives
        ):
            representative_color = (
                "#00A6D6"
                if SCAN_KEY == "W" and mode == "unclustered"
                else "gold"
            )
            ax.scatter(
                [float(representative[x_name])],
                [float(representative[y_name])],
                marker="*",
                s=190 if summary_panel else 230,
                color=representative_color,
                edgecolors="black",
                linewidths=0.9,
                label=(
                    "cluster representatives"
                    if summary_panel
                    and mode == "clustered"
                    and representative_index == 0
                    else "Example"
                    if mode == "unclustered"
                    and representative_index == 0
                    else None
                ),
                zorder=20,
            )
        _draw_alpha_e_cluster_guides(
            ax,
            x_name,
            y_name,
            alpha_e_line_half_width,
            alpha_e_boundaries,
        )
        label_size = 11 if summary_panel else 13
        ax.set_xlabel(x_label, fontsize=label_size, labelpad=4.0)
        ax.set_ylabel(y_label, fontsize=label_size, labelpad=4.0)
        ax.margins(0.06)
        configure_phase_space_axes(
            ax,
            x_name,
            y_name,
            lepton_mass=GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
        )
        ax.tick_params(
            labelsize=10 if summary_panel else 11,
            pad=2.0,
        )
        return displayed_representatives

    for mode in ("clustered", "unclustered"):
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        expected_panel_filenames = {
            f"{panel_index:02d}_{y_name}_vs_{x_name}_{mode}.pdf"
            for panel_index, (x_name, y_name, _x_label, _y_label)
            in enumerate(POLARIZATION_CORRELATION_PANELS, start=1)
        }
        for stale_path in mode_dir.glob(f"[0-9][0-9]_*_{mode}.pdf"):
            if stale_path.name in expected_panel_filenames:
                continue
            if stale_path.is_symlink() or not stale_path.is_file():
                raise RuntimeError(
                    f"Refusing to remove unexpected plot path: {stale_path}"
                )
            stale_path.unlink()
        summary_fig, summary_axes = plt.subplots(
            3,
            3,
            figsize=(13.3, 11.9),
        )
        summary_fig.subplots_adjust(
            left=0.079,
            right=0.984,
            bottom=0.073,
            top=0.985,
            wspace=0.18,
            hspace=0.18,
        )
        for ax, (x_name, y_name, x_label, y_label) in zip(
            summary_axes.ravel(),
            POLARIZATION_CORRELATION_PANELS,
        ):
            draw_panel(
                ax,
                mode,
                x_name,
                y_name,
                x_label,
                y_label,
                summary_panel=True,
            )
        legend_handles, legend_labels = (
            summary_axes[0, 0].get_legend_handles_labels()
        )
        legend_items = list(zip(legend_handles, legend_labels))
        if mode == "clustered":
            legend_items = [
                item for item in legend_items
                if not item[1].startswith(r"$E_\gamma^{\max}")
            ]

        def legend_order(item):
            label = item[1]
            cluster_label = label.split(" ", 1)[0]
            if cluster_label.startswith("P") and cluster_label[1:].isdigit():
                return (0, int(cluster_label[1:]))
            if label in {"cluster representatives", "Example"}:
                return (1, 0)
            return (2, 0)

        legend_items.sort(key=legend_order)
        legend_handles = [item[0] for item in legend_items]
        legend_labels = [item[1] for item in legend_items]
        if mode == "unclustered":
            axis_legend = summary_axes[0, 0].get_legend()
            if axis_legend is not None:
                axis_legend.remove()
        legend_ax = (
            summary_axes[0, 0]
            if mode == "unclustered"
            else summary_axes[1, 0]
        )
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left" if mode == "unclustered" else "upper center",
            ncol=1 if mode == "unclustered" else 2,
            frameon=True,
            framealpha=0.88,
            edgecolor="0.65",
            fontsize=11,
            borderaxespad=0.3,
            borderpad=0.35,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=0.75,
            labelspacing=0.25,
            markerscale=1.0,
        )
        summary_path = mode_dir / "00_summary.pdf"
        summary_fig.savefig(summary_path)
        plt.close(summary_fig)
        displayed_representatives = (
            representative_rows if mode == "clustered" else [example_row]
        )
        index_rows.append(
            {
                "panel_index": 0,
                "mode": mode,
                "x_name": "",
                "y_name": "",
                "x_label": "",
                "y_label": "",
                "retained_minima": len(selected_rows),
                "polarization_clusters": len(polarization_clusters),
                "representative_minima": len(displayed_representatives),
                "example_polarization_cluster": (
                    f"P{example_cluster_id + 1}"
                    if mode == "unclustered" else ""
                ),
                "objective_name": OBJECTIVE_NAME,
                "objective_cut_above_global_minimum": objective_cut,
                "global_minimum": optimum,
                "plot_path": str(summary_path),
            }
        )
        for panel_index, (x_name, y_name, x_label, y_label) in enumerate(
            POLARIZATION_CORRELATION_PANELS,
            start=1,
        ):
            fig, ax = plt.subplots(
                figsize=(5.0, 4.5),
            )
            fig.subplots_adjust(
                left=0.160,
                right=0.970,
                bottom=0.150,
                top=0.962,
            )
            displayed_representatives = draw_panel(
                ax,
                mode,
                x_name,
                y_name,
                x_label,
                y_label,
            )
            if mode == "unclustered":
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                legend_position = (
                    {
                        "loc": "center",
                        "bbox_to_anchor": (0.5, 0.63),
                    }
                    if SCAN_KEY == "W"
                    and x_name == "alpha_e"
                    and y_name == "alpha_p"
                    else {"loc": "best"}
                )
                ax.legend(
                    legend_handles,
                    legend_labels,
                    borderaxespad=0.35,
                    ncol=1,
                    fontsize=10,
                    frameon=True,
                    framealpha=0.88,
                    edgecolor="0.65",
                    borderpad=0.25,
                    handlelength=1.6,
                    handletextpad=0.45,
                    columnspacing=0.9,
                    labelspacing=0.25,
                    **legend_position,
                )
            '''
            ax.set_title(
                f"{lepton_name} / {SCAN_KEY}: {y_name} versus {x_name}\n"
                + (
                    "polarization clusters shown by color and marker"
                    if mode == "clustered"
                    else "polarization cluster labels hidden"
                ),
                fontsize=13,
            )
            '''
            filename = (
                f"{panel_index:02d}_{y_name}_vs_{x_name}_{mode}.pdf"
            )
            path = mode_dir / filename
            fig.savefig(path)
            plt.close(fig)
            index_rows.append(
                {
                    "panel_index": panel_index,
                    "mode": mode,
                    "x_name": x_name,
                    "y_name": y_name,
                    "x_label": x_label,
                    "y_label": y_label,
                    "retained_minima": len(selected_rows),
                    "polarization_clusters": len(polarization_clusters),
                    "representative_minima": len(displayed_representatives),
                    "example_polarization_cluster": (
                        f"P{example_cluster_id + 1}"
                        if mode == "unclustered" else ""
                    ),
                    "objective_name": OBJECTIVE_NAME,
                    "objective_cut_above_global_minimum": objective_cut,
                    "global_minimum": optimum,
                    "plot_path": str(path),
                }
            )
    return index_rows


def _configuration_rows(minimum_rows, lepton_name):
    """Annotate every polarization-cluster member for ConfigGen."""
    prefix = config_scan.mixing_prefix(lepton_name)
    key = f"{prefix}_{TARGET_OBSERVABLE_NAME}"
    details = []
    for index, source in enumerate(minimum_rows):
        row = dict(source)
        value = float(row[key])
        parent_id = int(row["polarization_cluster_id"])
        minimum_id = int(row["local_minimum_id"])
        row.update(
            {
                "selected_observable": TARGET_OBSERVABLE_NAME,
                "selected_observable_label": config_gen.observable_label(
                    TARGET_OBSERVABLE_NAME
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
    contour_settings,
):
    """Write configuration pages from supplied 8D contour samples."""
    plt, PdfPages = config_gen._require_matplotlib()
    selected_rows = list(selected_rows)
    if len(selected_rows) != len(detail_rows):
        raise ValueError(
            "Selected local-minimum rows and configuration details disagree."
        )
    if set(contour_settings) != set(range(len(selected_rows))):
        raise ValueError(
            "Every selected minimum must have saved contour settings."
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
            3, 3, figsize=(14.5, 11.5), constrained_layout=True
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
            ax.set_xlabel(x_label, fontsize=CONTOUR_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(y_label, fontsize=CONTOUR_AXIS_LABEL_FONTSIZE)
            configure_phase_space_axes(
                ax,
                x_name,
                y_name,
                lepton_mass=GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
            )
            ax.tick_params(labelsize=CONTOUR_TICK_FONTSIZE)

        overview_summary = overview_axes[2, 2]
        overview_summary.axis("off")
        overview_deltas = sorted(
            {
                settings["contour_delta"]
                for settings in contour_settings.values()
            }
        )
        overview_direction_counts = sorted(
            {
                settings["configured_direction_count"]
                for settings in contour_settings.values()
            }
        )
        overview_lines = [
            (
                f"polarization cluster P{parent_id + 1}: "
                f"{parent_configuration}"
            ),
            f"{len(selected_rows)} local minima in this polarization cluster",
            (
                "contour delta/minimum="
                + ",".join(f"{value:g}" for value in overview_deltas)
            ),
            (
                "configured 8D directions/minimum="
                + ",".join(str(value) for value in overview_direction_counts)
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
            fontsize=CONTOUR_SUMMARY_FONTSIZE,
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
            contour_delta = contour_settings[selected_index]["contour_delta"]
            configured_direction_count = contour_settings[selected_index][
                "configured_direction_count"
            ]
            local_value = float(row[objective_key])
            fig, axes = plt.subplots(
                3, 3, figsize=(14.5, 11.5), constrained_layout=True
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
                            rf"+{contour_delta:g}$ projection"
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
                ax.set_xlabel(
                    x_label,
                    fontsize=CONTOUR_AXIS_LABEL_FONTSIZE,
                )
                ax.set_ylabel(
                    y_label,
                    fontsize=CONTOUR_AXIS_LABEL_FONTSIZE,
                )
                configure_phase_space_axes(
                    ax,
                    x_name,
                    y_name,
                    lepton_mass=GRADIENT_LEPTON_SPECS[lepton_name]["mass"],
                )
                ax.tick_params(labelsize=CONTOUR_TICK_FONTSIZE)

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
                    f"{local_value + contour_delta:.8g}"
                ),
                (
                    f"above global minimum = "
                    f"{local_value - optimum:.8g}"
                ),
                f"8D contour samples = {len(boundary_points)}",
                f"configured 8D directions = {configured_direction_count}",
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
                fontsize=CONTOUR_SUMMARY_FONTSIZE,
                family="monospace",
            )
            fig.suptitle(
                f"{lepton_name}: polarization cluster P{parent_id + 1}, "
                f"local minimum {row['local_minimum_id']}; "
                "pairwise projections of the 8D "
                rf"${OBJECTIVE_LATEX}="
                rf"({OBJECTIVE_LATEX})_{{\mathrm{{local}}}}"
                rf"+{contour_delta:g}$ contour"
            )
            pdf.savefig(fig)
            plt.close(fig)
    return path


def _write_representative_configuration_pdfs(
    rows,
    lepton_name,
    optimum,
    output_dir,
):
    """Extract two configuration pages for the starred unclustered example."""
    # Contour centers are normalized with species-dependent sqrt(s) bounds.
    # Reassert the requested species because plotting another species can
    # legitimately change PhaseSpaceScan's active global range.
    phase_scan._configure_lepton(lepton_name)
    pdfseparate = shutil.which("pdfseparate")
    pdfunite = shutil.which("pdfunite")
    if pdfseparate is None or pdfunite is None:
        raise RuntimeError(
            "Representative configuration extraction requires pdfseparate "
            "and pdfunite on PATH."
        )

    example_cluster_id = EXAMPLE_POLARIZATION_CLUSTER_IDS[SCAN_KEY]
    representatives = [
        row for row in rows
        if _as_bool(row.get("within_polarization_cluster_cut"))
        and _as_bool(row.get("polarization_cluster_representative"))
        and int(row["polarization_cluster_id"]) == example_cluster_id
    ]
    if len(representatives) != 1:
        raise RuntimeError(
            f"Expected one P{example_cluster_id + 1} representative example."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "P??_local_minimum_*_representative_configuration.pdf",
        "Example_P??_local_minimum_*_configuration.pdf",
    ):
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()
    index_rows = []
    for representative in representatives:
        cluster_id = int(representative["polarization_cluster_id"])
        minimum_id = int(representative["local_minimum_id"])
        details = _configuration_rows([representative], lepton_name)
        try:
            contours, contour_settings = _load_minimum_contour_data(
                lepton_name,
                [representative],
            )
            contour_source = "validated saved local-minimum contour"
        except (FileNotFoundError, ValueError):
            # A scan rerun can make an older normalized contour center stale.
            # Recompute only this single displayed example; do not silently
            # reuse a contour belonging to a different physical point.
            phase_scan._configure_lepton(lepton_name)
            global GRADIENT_WORKERS
            clustering_worker_count = GRADIENT_WORKERS
            try:
                GRADIENT_WORKERS = SCAN_WORKERS
                contours = _configuration_contours(
                    [representative],
                    lepton_name,
                )
            finally:
                GRADIENT_WORKERS = clustering_worker_count
            contour_settings = {
                0: {
                    "contour_delta": PHASE_SPACE_CONFIG_CONTOUR_DELTA,
                    "configured_direction_count": (
                        PHASE_SPACE_CONFIG_CONTOUR_SAMPLES
                    ),
                }
            }
            contour_source = "fresh contour for displayed example"
        filename = (
            f"Example_P{cluster_id + 1:02d}_local_minimum_{minimum_id:04d}_"
            "configuration.pdf"
        )
        output_path = output_dir / filename

        with tempfile.TemporaryDirectory(
            prefix="gradient_representative_config_"
        ) as temporary_directory:
            temporary_directory = Path(temporary_directory)
            temporary_package = temporary_directory / "package.pdf"
            _write_configuration_plot(
                [representative],
                details,
                lepton_name,
                optimum,
                temporary_package,
                contours,
                contour_settings,
            )
            page_pattern = temporary_directory / "page-%d.pdf"
            subprocess.run(
                [
                    pdfseparate,
                    "-f",
                    "2",
                    "-l",
                    "3",
                    str(temporary_package),
                    str(page_pattern),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_path.unlink(missing_ok=True)
            subprocess.run(
                [
                    pdfunite,
                    str(temporary_directory / "page-2.pdf"),
                    str(temporary_directory / "page-3.pdf"),
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        index_rows.append(
            {
                "polarization_cluster_id": cluster_id,
                "polarization_cluster_label": f"P{cluster_id + 1}",
                "local_minimum_id": minimum_id,
                "page_count": 2,
                "page_1": "reconstructed configuration and amplitudes",
                "page_2": "pairwise projections of the 8D contour",
                "contour_source": contour_source,
                "configuration_path": str(output_path),
            }
        )

    _write_csv(output_dir / "representative_configuration_index.csv", index_rows)
    return index_rows


def _write_configurations(
    lepton_name,
    polarization_cluster_id,
    selected_minimum_rows,
    optimum,
):
    """Generate one cluster package from pre-cluster minimum contours."""
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
    # Normalized contour centers use species-dependent sqrt(s) and qOut
    # bounds.  Configuration reconstruction can change module-global species
    # state, so reassert it immediately before validating saved centers.
    phase_scan._configure_lepton(lepton_name)
    contour_started = perf_counter()
    contours, contour_settings = _load_minimum_contour_data(
        lepton_name,
        selected_minimum_rows,
    )
    contour_source = (
        "loaded pre-cluster per-minimum contour samples by "
        "local_minimum_id"
    )
    contour_timing_label = "pre-cluster contour data load time"
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
        contour_settings,
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
        + GRADIENT_LEPTON_NAMES.index(lepton_name)
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
    # Validate deterministic references before starting the expensive Sobol
    # screening and optimization.  A stale frame-dependent anchor must fail
    # immediately rather than after every optimization run has completed.
    reference_rows = _physics_reference_rows(lepton_name)
    minimum_scan_started = perf_counter()
    if results is None:
        tasks = _species_tasks(lepton_name)
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
    references_path = scan_data_dir / "physics_reference_points.csv"
    if reference_rows:
        _write_csv(references_path, reference_rows)
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
    latin_count = sum(
        result[0]["start_source"] == "latin_hypercube"
        for result in results
    )
    screened_count = sum(
        result[0]["start_source"] == "sobol_screened"
        for result in results
    )
    anchor_count = sum(
        str(result[0]["start_source"]).startswith("physics_anchor:")
        for result in results
    )
    screening_key = _run_objective_key("screening")
    screened_values = [
        float(result[0][screening_key])
        for result in results
        if (
            result[0]["start_source"] == "sobol_screened"
            and np.isfinite(float(result[0][screening_key]))
        )
    ]
    report_lines = [
        f"Hybrid global gradient {OBJECTIVE_NAME} search ({lepton_name})",
        f"  optimization starts: {len(results)}",
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
        f"  L-BFGS-B-converged runs: {lbfgs_converged}/{len(results)}",
        f"  multiscale-verified runs: {verified}/{len(results)}",
        f"  distinct finite minima: {len(minima)}",
        f"  minimum scan time: {minimum_scan_seconds:.3f} s",
        f"  best {OBJECTIVE_NAME}: {optimum:.10g}",
        *(
            (f"  maximum {TARGET_OBSERVABLE_NAME}: {1.0 - optimum:.10g}",)
            if TARGET_OBSERVABLE_NAME != OBJECTIVE_NAME
            else ()
        ),
        f"  optimization runs: {run_path}",
        (
            f"  exact physics reference points: {references_path}"
            if reference_rows
            else "  exact physics reference points: none configured"
        ),
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
        _read_csv(references_path)
        if PHYSICS_ANCHOR_STARTS.get(lepton_name) and references_path.exists()
        else []
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


def contour_species_minima(lepton_name, *, reuse_saved_minima):
    """Generate resumable contours for raw minima before any clustering."""
    # Both the physical-to-unit center and the worker interpretation use
    # species-dependent sqrt(s) and qOut bounds.  Reassert the species before
    # validating or generating any saved contour package.
    phase_scan._configure_lepton(lepton_name)
    output_dirs = species_output_dirs(lepton_name)
    minima_path = output_dirs["scan_data"] / "local_minima.csv"
    minimum_rows = _read_csv(minima_path)
    if "local_minimum_id" not in minimum_rows[0]:
        raise ValueError(
            f"Raw minima are missing local_minimum_id: {minima_path}"
        )
    minimum_ids = [str(row["local_minimum_id"]) for row in minimum_rows]
    if len(set(minimum_ids)) != len(minimum_ids):
        raise ValueError(
            f"Raw minima repeat local_minimum_id values: {minima_path}"
        )

    paths = minimum_contour_data_paths(lepton_name)
    index_rows = []
    generated_count = 0
    regenerated_stale_count = 0
    reused_count = 0
    contour_started = perf_counter()
    for minimum_number, row in enumerate(minimum_rows, start=1):
        minimum_id = str(row["local_minimum_id"])
        minimum_path = minimum_contour_data_path(lepton_name, minimum_id)
        minimum_started = perf_counter()
        if reuse_saved_minima and minimum_path.exists():
            try:
                _load_contour_data(minimum_path, [row])
            except ValueError:
                contours = _configuration_contours([row], lepton_name)
                minimum_csv_rows = _contour_csv_rows([row], contours)
                _write_csv(minimum_path, minimum_csv_rows)
                generated_count += 1
                regenerated_stale_count += 1
                action = "regenerated stale"
            else:
                minimum_csv_rows = _read_csv(minimum_path)
                reused_count += 1
                action = "reused"
        else:
            contours = _configuration_contours([row], lepton_name)
            minimum_csv_rows = _contour_csv_rows([row], contours)
            _write_csv(minimum_path, minimum_csv_rows)
            generated_count += 1
            action = "generated"
        metadata = next(
            item
            for item in minimum_csv_rows
            if item["record_type"] == "minimum"
        )
        index_rows.append(
            {
                "local_minimum_id": minimum_id,
                "contour_status": "complete",
                "contour_file": str(minimum_path),
                "contour_point_count": metadata["contour_point_count"],
                "configured_direction_count": (
                    metadata["configured_direction_count"]
                ),
                "contour_delta": metadata["contour_delta"],
            }
        )
        # Refresh the lightweight index after each minimum. The authoritative
        # numerical samples remain in their per-minimum files.
        _write_csv(paths["index"], index_rows)
        elapsed = perf_counter() - contour_started
        average = elapsed / minimum_number
        remaining = average * (len(minimum_rows) - minimum_number)
        print(
            f"[{minimum_number}/{len(minimum_rows)}] {action} contour for "
            f"local_minimum_id={minimum_id} in "
            f"{perf_counter() - minimum_started:.3f} s; "
            f"ETA {remaining / 60.0:.1f} min",
            flush=True,
        )

    contour_seconds = perf_counter() - contour_started
    return "\n".join(
        (
            f"Gradient {OBJECTIVE_NAME} pre-cluster contours ({lepton_name})",
            f"  source raw minima: {minima_path}",
            f"  local minima loaded: {len(minimum_rows)}",
            f"  contours generated: {generated_count}",
            f"  stale contours regenerated: {regenerated_stale_count}",
            f"  contours reused after validation: {reused_count}",
            (
                f"  directions per minimum: "
                f"{PHASE_SPACE_CONFIG_CONTOUR_SAMPLES}"
            ),
            (
                "  local-minimum contour settings: read from each "
                "minimum-owned contour file"
            ),
            f"  contour generation time: {contour_seconds:.3f} s",
            f"  per-minimum contour data: {paths['minima']}",
            f"  contour index: {paths['index']}",
            (
                "  ownership key: local_minimum_id from the raw "
                "local_minima.csv"
            ),
        )
    )


def cluster_species_minima(
    lepton_name,
    *,
    polarization_objective_cut,
    polarization_cluster_count,
    polarization_cluster_seed,
    polarization_alpha_e_line_half_width,
    polarization_alpha_e_boundaries,
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
        alpha_e_line_half_width=polarization_alpha_e_line_half_width,
        alpha_e_boundaries=polarization_alpha_e_boundaries,
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
        polarization_alpha_e_line_half_width,
        polarization_alpha_e_boundaries,
        output_dirs["plots"] / "polarization_cluster_phase_space.pdf",
    )
    correlation_rows = _write_polarization_correlation_pdfs(
        clustered_rows,
        polarization_clusters,
        lepton_name,
        polarization_objective_cut,
        optimum,
        polarization_alpha_e_line_half_width,
        polarization_alpha_e_boundaries,
        output_dirs["plots"] / "polarization_correlations",
    )
    representative_configuration_dir = (
        output_dirs["plots"]
        / "polarization_correlations"
        / "unclustered"
    )
    representative_configuration_rows = (
        _write_representative_configuration_pdfs(
            clustered_rows,
            lepton_name,
            optimum,
            representative_configuration_dir,
        )
    )
    example_configuration_path = representative_configuration_rows[0][
        "configuration_path"
    ]
    for correlation_row in correlation_rows:
        correlation_row["example_configuration_path"] = (
            example_configuration_path
            if correlation_row["mode"] == "unclustered"
            else ""
        )
    correlation_index_path = _write_csv(
        output_dirs["cluster_data"]
        / "polarization_correlation_plot_index.csv",
        correlation_rows,
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
                f"  {SCAN_KEY} polarization clusters: "
                f"{len(polarization_clusters)}"
            ),
            (
                (
                    "  alpha_e line half-width: "
                    f"{polarization_alpha_e_line_half_width:.8g} rad "
                    f"({polarization_alpha_e_line_half_width / np.pi:.8g}pi)"
                )
                if SCAN_KEY == "W"
                else (
                    "  alpha_e boundaries: "
                    + ", ".join(
                        f"{float(boundary):.8g}"
                        for boundary in polarization_alpha_e_boundaries
                    )
                    + " rad"
                )
                if SCAN_KEY in ("GHZ", "CEP")
                else (
                    "  fixed alpha_e centers: 0, pi/2, pi"
                    if SCAN_KEY == "CPGAMMA"
                    else (
                        "  fixed alpha_p centers: "
                        "0, pi/4, pi/2, 3pi/4, pi"
                        if SCAN_KEY == "CEGAMMA"
                        else (
                            "  polarization clustering: periodic k-means in "
                            "(alpha_e, alpha_p)"
                        )
                    )
                )
            ),
            (
                "  one best-objective representative selected per "
                "polarization cluster"
            ),
            f"  clustering time: {cluster_seconds:.3f} s",
            f"  clustered minima: {clustered_path}",
            f"  polarization cluster summary: {polarization_path}",
            f"  polarization phase-space plot: {polarization_plot}",
            (
                f"  separate correlation PDFs: {len(correlation_rows)} "
                f"({output_dirs['plots'] / 'polarization_correlations'})"
            ),
            f"  correlation plot index: {correlation_index_path}",
            (
                "  starred representative configuration PDFs: "
                f"{len(representative_configuration_rows)} "
                f"({representative_configuration_dir})"
            ),
        )
    )


def configure_species_clusters(
    lepton_name,
    *,
    polarization_clusters_to_configure,
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
    unknown = set(LEPTONS_TO_PROCESS) - set(GRADIENT_LEPTON_SPECS)
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
        if lepton_name not in GRADIENT_LEPTON_SPECS:
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
    unknown = set(LEPTONS_TO_PROCESS) - set(GRADIENT_LEPTON_SPECS)
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
    polarization_alpha_e_line_half_width,
    polarization_alpha_e_boundaries,
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
    if definition.key == "W":
        expected_cluster_count = sum(
            W_POLARIZATION_ALPHA_E_STRATUM_CLUSTERS
        )
        if polarization_cluster_count != expected_cluster_count:
            raise ValueError(
                f"W polarization_cluster_count must be "
                f"{expected_cluster_count}."
            )
        if (
            not np.isfinite(polarization_alpha_e_line_half_width)
            or polarization_alpha_e_line_half_width <= 0.0
            or polarization_alpha_e_line_half_width >= np.pi / 4.0
        ):
            raise ValueError(
                "W polarization_alpha_e_line_half_width must be finite and "
                "strictly between 0 and pi/4."
            )
        if polarization_alpha_e_boundaries is not None:
            raise ValueError("W polarization_alpha_e_boundaries must be None.")
    elif definition.key in ("GHZ", "CEP"):
        boundaries = np.asarray(
            polarization_alpha_e_boundaries,
            dtype=float,
        )
        required_boundaries = np.asarray((0.0, np.pi / 2.0, np.pi))
        if (
            boundaries.shape != required_boundaries.shape
            or not np.allclose(
                boundaries,
                required_boundaries,
                rtol=0.0,
                atol=1.0e-15,
            )
        ):
            raise ValueError(
                f"{definition.key} polarization_alpha_e_boundaries must be "
                "(0, pi/2, pi)."
            )
        expected_cluster_count = (
            (len(boundaries) - 1)
            * GHZ_ALPHA_P_CLUSTERS_PER_ALPHA_E_REGION
        )
        if polarization_cluster_count != expected_cluster_count:
            raise ValueError(
                f"{definition.key} polarization_cluster_count must be "
                f"{expected_cluster_count}."
            )
        if polarization_alpha_e_line_half_width is not None:
            raise ValueError(
                f"{definition.key} polarization_alpha_e_line_half_width "
                "must be None."
            )
    elif definition.key in FIXED_AXIS_POLARIZATION_CENTERS:
        axis_name, fixed_centers = FIXED_AXIS_POLARIZATION_CENTERS[
            definition.key
        ]
        if polarization_cluster_count != len(fixed_centers):
            raise ValueError(
                f"{definition.key} polarization_cluster_count must be "
                f"{len(fixed_centers)} for its fixed {axis_name} centers."
            )
        if polarization_alpha_e_line_half_width is not None:
            raise ValueError(
                f"{definition.key} polarization_alpha_e_line_half_width "
                "must be None."
            )
        if polarization_alpha_e_boundaries is not None:
            raise ValueError(
                f"{definition.key} polarization_alpha_e_boundaries must "
                "be None."
            )
    elif (
        source_observable_name(definition.objective_name)
        in PAIRWISE_CONCURRENCE_NAMES
    ):
        if polarization_alpha_e_line_half_width is not None:
            raise ValueError(
                "Pairwise-concurrence polarization_alpha_e_line_half_width "
                "must be None."
            )
        if polarization_alpha_e_boundaries is not None:
            raise ValueError(
                "Pairwise-concurrence polarization_alpha_e_boundaries must "
                "be None."
            )
    else:
        raise ValueError(
            f"No polarization clustering setup for {definition.key!r}."
        )
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
            polarization_alpha_e_line_half_width=(
                polarization_alpha_e_line_half_width
            ),
            polarization_alpha_e_boundaries=(
                polarization_alpha_e_boundaries
            ),
        )
        for lepton_name in LEPTONS_TO_PROCESS
    ]
    return _write_stage_report("cluster", reports)


def run_minimum_contours(
    definition,
    *,
    leptons_to_contour,
    contour_workers,
    reuse_saved_minima,
):
    """Generate contours for raw local minima before clustering."""
    if not isinstance(reuse_saved_minima, bool):
        raise TypeError("reuse_saved_minima must be a bool.")
    configure_scan(
        definition,
        leptons_to_process=leptons_to_contour,
        gradient_workers=contour_workers,
    )
    _validate_config_settings()
    reports = [
        contour_species_minima(
            lepton_name,
            reuse_saved_minima=reuse_saved_minima,
        )
        for lepton_name in LEPTONS_TO_PROCESS
    ]
    return _write_stage_report("contour", reports)


def run_cluster_configgen(
    definition,
    *,
    leptons_to_configure,
    config_workers,
    polarization_clusters_to_configure,
):
    """Run ConfigGen from clusters and pre-cluster contour samples."""
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
        )
        for lepton_name in LEPTONS_TO_PROCESS
    ]
    return _write_stage_report("config", reports)
