"""Find local D_W minima with hybrid starts and bounded gradient searches.

Each optimization uses the same seven continuous coordinates and coherent
incoming-spin preparation as :mod:`PhaseSpaceScan`. The coordinates are
normalized to a unit box before SciPy's L-BFGS-B minimizer estimates numerical
gradients. A periodic-aware multiscale direct search then follows unresolved
descent directions down to the requested scan precision. Distinct verified
minima are written as scan-compatible rows and converted into ConfigGen-style
momentum and amplitude configurations.
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import qmc

import ConfigGen as config_gen
import PhaseSpaceConfigScan as config_scan
import PhaseSpaceScan as phase_scan
from AlignmentScan import LEPTON_SPECS
from config import (
    DW_GRADIENT_MAX_ITERATIONS,
    DW_GRADIENT_MINIMUM_SEPARATION,
    DW_GRADIENT_RANDOM_SEED,
    DW_GRADIENT_RANDOM_STARTS,
    DW_GRADIENT_SCAN_PRECISION,
    DW_GRADIENT_SCREENED_STARTS,
    DW_GRADIENT_SCREENING_SAMPLES,
    DW_GRADIENT_SCREENING_SEPARATION,
    DW_GRADIENT_TOLERANCE,
    DW_LOCAL_SEARCH_INITIAL_STEP,
    DW_LOCAL_SEARCH_MAX_POLLS,
    DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE,
    DW_LOCAL_SEARCH_RANDOM_DIRECTIONS,
    DW_LOCAL_SEARCH_STEP_REDUCTION,
    PHASE_SPACE_CONFIG_CONTOUR_DELTA,
    PHASE_SPACE_CONFIG_CONTOUR_SAMPLES,
    PHASE_SPACE_CONFIG_THRESHOLD,
    SCAN_INITIAL_MIXING_ANGLES,
    SCAN_WORKERS,
)
from PlotUtils import print_console_text


# Script controls.
LEPTONS_TO_OPTIMIZE = ("electron", "muon", "heavy", "massless")
GRADIENT_WORKERS = SCAN_WORKERS
# Set True to rebuild the phase-space and configuration PDFs from each
# species' existing local_minima.csv without rerunning the gradient search.
REGENERATE_PLOTS_FROM_CSV = False
OUTPUT_ROOT = Path("Output") / "GradientPhaseSpaceScan"
CONFIG_OUTPUT_ROOT = Path("Output") / "GradientPhaseSpaceConfig"
LOG_PATH = OUTPUT_ROOT / "GradientPhaseSpaceScan.log"
INVALID_OBJECTIVE = 1.0e3
PERIODIC_UNIT_COORDINATES = (3, 4, 5, 6)
CONFIG_CONTOUR_BISECTION_ITERATIONS = 8
CONFIG_CONTOUR_INITIAL_RADIUS = 0.01
PLOT_PANELS = (
    ("theta_in", "qOut", r"$\theta_{in}$", r"$E_\gamma$ [GeV]"),
    ("sqrt_s", "qOut", r"$\sqrt{s}$ [GeV]", r"$E_\gamma$ [GeV]"),
    ("phi_in", "phiOut", r"$\phi_{P,in}$", r"$\phi_\gamma$"),
    ("theta_e", "theta_p", r"$\theta_e$", r"$\theta_p$"),
    ("sqrt_s", "theta_e", r"$\sqrt{s}$ [GeV]", r"$\theta_e$"),
    ("sqrt_s", "theta_p", r"$\sqrt{s}$ [GeV]", r"$\theta_p$"),
    ("qOut", "theta_e", r"$E_\gamma$ [GeV]", r"$\theta_e$"),
    ("qOut", "theta_p", r"$E_\gamma$ [GeV]", r"$\theta_p$"),
)
PLOT_PERIODS = {
    "phi_in": 2.0 * np.pi,
    "phiOut": 2.0 * np.pi,
    "theta_e": np.pi,
    "theta_p": np.pi,
}
# Deterministic physical seeds supplement the generic global designs. Values
# are user-frame coordinates, not optimizer coordinates, so they remain
# readable and are remapped if species scan ranges change.
PHYSICS_ANCHOR_STARTS = {
    "electron": (
        {
            "name": "epcm_standard_W",
            "sqrt_s": 1.1518524360498226,
            "theta_in": 0.5 * np.pi,
            "qOut": 0.1771320126293574,
            "phi_in_lepton": 1.5 * np.pi,
            "phiOut": (0.5 * np.pi - 3.032) % (2.0 * np.pi),
            "theta_e": 0.834,
            "theta_p": (-0.036) % np.pi,
        },
    ),
}


def _normalized_to_point(unit_point):
    """Map the optimizer's unit box to PhaseSpaceScan's seven coordinates."""
    unit_point = np.asarray(unit_point, dtype=float)
    sqrt_s = (
        phase_scan.SQRT_S_RANGE[0]
        + unit_point[0]
        * (phase_scan.SQRT_S_RANGE[1] - phase_scan.SQRT_S_RANGE[0])
    )
    s = sqrt_s**2
    qout_fraction = (
        phase_scan.QOUT_FRACTION_RANGE[0]
        + unit_point[2]
        * (
            phase_scan.QOUT_FRACTION_RANGE[1]
            - phase_scan.QOUT_FRACTION_RANGE[0]
        )
    )
    return np.asarray(
        (
            s,
            phase_scan.THETA_IN_RANGE[0]
            + unit_point[1]
            * (
                phase_scan.THETA_IN_RANGE[1]
                - phase_scan.THETA_IN_RANGE[0]
            ),
            qout_fraction * phase_scan._qout_max(s),
            unit_point[3] * 2.0 * np.pi,
            unit_point[4] * 2.0 * np.pi,
            unit_point[5] * np.pi,
            unit_point[6] * np.pi,
        ),
        dtype=float,
    )


def _d_w_evaluation(unit_point, lepton_name, evaluation_id):
    """Evaluate D_W and return its complete coherent-angle result row."""
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
    key = f"{config_scan.mixing_prefix(lepton_name)}_D_W"
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
        screening_d_w,
    ) = task
    phase_scan._configure_lepton(lepton_name)
    cache = {}
    evaluation_count = 0

    def evaluate(unit_point):
        nonlocal evaluation_count
        clipped = np.clip(np.asarray(unit_point, dtype=float), 0.0, 1.0)
        key = clipped.tobytes()
        if key not in cache:
            cache[key] = _d_w_evaluation(
                clipped,
                lepton_name,
                evaluation_id=run_index * 1_000_000 + evaluation_count,
            )
            evaluation_count += 1
        return cache[key]

    start_value, _start_row = evaluate(start)
    result = minimize(
        lambda point: evaluate(point)[0],
        np.asarray(start, dtype=float),
        method="L-BFGS-B",
        bounds=((0.0, 1.0),) * 7,
        options={
            "maxiter": DW_GRADIENT_MAX_ITERATIONS,
            "ftol": DW_GRADIENT_TOLERANCE,
            "gtol": DW_GRADIENT_TOLERANCE,
            "eps": DW_GRADIENT_SCAN_PRECISION,
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
        direction_seed=DW_GRADIENT_RANDOM_SEED + run_index,
    )
    gradient_norm = (
        float(np.linalg.norm(np.asarray(result.jac, dtype=float)))
        if result.jac is not None
        else np.nan
    )
    run = {
        "optimization_run": run_index,
        "start_source": start_source,
        "screening_D_W": screening_d_w,
        "success": local_search["local_minimum_verified"],
        "lbfgs_success": bool(result.success),
        "lbfgs_status": int(result.status),
        "lbfgs_message": str(result.message),
        "lbfgs_iterations": int(result.nit),
        "function_evaluations": evaluation_count,
        "lbfgs_function_evaluations": int(result.nfev),
        "lbfgs_gradient_norm": gradient_norm,
        "initial_D_W": start_value,
        "lbfgs_D_W": lbfgs_value,
        "final_D_W": final_value,
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
    neighbor[:3] = np.clip(neighbor[:3], 0.0, 1.0)
    neighbor[3:] %= 1.0
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


def _multiscale_local_search(evaluate, start, direction_seed=0):
    """Polish a gradient result until no poll direction improves D_W.

    Coordinate directions form a positive-spanning set. Repeating the poll
    while shrinking its mesh makes this robust to branch-sensitive or
    nonsmooth regions where L-BFGS-B can stop on relative function reduction.
    Periodic azimuth and mixing coordinates wrap across the unit-box boundary.
    """
    point = np.asarray(start, dtype=float).copy()
    value, row = evaluate(point)
    direction_rng = np.random.default_rng(direction_seed)
    extra_directions = direction_rng.normal(
        size=(DW_LOCAL_SEARCH_RANDOM_DIRECTIONS, point.size)
    )
    if len(extra_directions):
        extra_directions /= np.linalg.norm(
            extra_directions, axis=1, keepdims=True
        )
    step = DW_LOCAL_SEARCH_INITIAL_STEP
    polls = 0
    accepted_moves = 0
    smallest_tested_step = step
    while polls < DW_LOCAL_SEARCH_MAX_POLLS:
        neighbors = _poll_neighbors(point, step, extra_directions)
        evaluated = [
            (evaluate(neighbor)[0], neighbor)
            for neighbor in neighbors
        ]
        polls += 1
        smallest_tested_step = step
        best_value, best_point = min(evaluated, key=lambda item: item[0])
        if best_value < value - DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE:
            point = best_point
            value, row = evaluate(point)
            accepted_moves += 1
            continue
        if step <= DW_GRADIENT_SCAN_PRECISION * (1.0 + 1.0e-12):
            break
        step = max(
            DW_GRADIENT_SCAN_PRECISION,
            step * DW_LOCAL_SEARCH_STEP_REDUCTION,
        )

    verification_neighbors = _poll_neighbors(
        point,
        DW_GRADIENT_SCAN_PRECISION,
        extra_directions,
    )
    neighbor_values = [
        evaluate(neighbor)[0] for neighbor in verification_neighbors
    ]
    best_neighbor = min(neighbor_values, default=value)
    verified = (
        best_neighbor >= value - DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE
    )
    return point, value, row, {
        "local_search_polls": polls,
        "local_search_moves": accepted_moves,
        "local_search_poll_limit_reached": (
            polls >= DW_LOCAL_SEARCH_MAX_POLLS
            and step >= DW_GRADIENT_SCAN_PRECISION
        ),
        "smallest_tested_step": smallest_tested_step,
        "best_neighbor_D_W": best_neighbor,
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
    """Keep the lowest-D_W representative of each converged basin."""
    finite = [
        result for result in results
        if (
            result[0]["local_minimum_verified"]
            and result[2] is not None
            and np.isfinite(result[0]["final_D_W"])
        )
    ]
    finite.sort(key=lambda result: result[0]["final_D_W"])
    selected = []
    for result in finite:
        if any(
            _unit_distance(result[1], prior[1])
            <= DW_GRADIENT_MINIMUM_SEPARATION
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
            f"Cannot regenerate gradient plots; missing saved data: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(
            f"Cannot regenerate gradient plots; {path} contains no data rows."
        )
    return rows


def _d_w_values(rows, lepton_name):
    """Return the local-minimum D_W values in row order."""
    key = f"{config_scan.mixing_prefix(lepton_name)}_D_W"
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _unit_point_from_minimum_row(row):
    """Recover one local minimum in the optimizer's normalized coordinates."""
    return np.asarray([float(row[f"final_u{index}"]) for index in range(7)])


def _plot_coordinate_values(unit_point):
    """Return the named physical coordinates used by the projection panels."""
    point = _normalized_to_point(unit_point)
    return {
        "sqrt_s": float(np.sqrt(point[0])),
        "theta_in": float(point[1]),
        "qOut": float(point[2]),
        "phi_in": float(point[3]),
        "phiOut": float(point[4]),
        "theta_e": float(point[5]),
        "theta_p": float(point[6]),
    }


def _maximum_contour_radius(center, direction):
    """Return the non-repeating radial extent along one unit-box direction."""
    limits = []
    for index, component in enumerate(direction):
        if abs(component) <= 1.0e-15:
            continue
        if index in PERIODIC_UNIT_COORDINATES:
            limits.append(0.5 / abs(component))
        elif component > 0.0:
            limits.append((1.0 - center[index]) / component)
        else:
            limits.append(center[index] / -component)
    return max(0.0, float(min(limits, default=0.0)))


def _contour_directions(seed):
    """Return deterministic directions spanning the seven-dimensional sphere."""
    axes = np.vstack((np.eye(7), -np.eye(7)))
    random_count = PHASE_SPACE_CONFIG_CONTOUR_SAMPLES - len(axes)
    rng = np.random.default_rng(seed)
    random_directions = rng.normal(size=(random_count, 7))
    if random_count:
        random_directions /= np.linalg.norm(
            random_directions,
            axis=1,
            keepdims=True,
        )
    return np.vstack((axes, random_directions))


def _trace_high_dimensional_contour(
    evaluate,
    center,
    base_value,
    directions,
):
    """Sample the local D_W isosurface along seven-dimensional rays."""
    target = float(base_value) + PHASE_SPACE_CONFIG_CONTOUR_DELTA
    boundary_points = []
    for direction in directions:
        maximum_radius = _maximum_contour_radius(center, direction)
        if maximum_radius <= 1.0e-14:
            continue

        low_radius = 0.0
        high_radius = min(CONFIG_CONTOUR_INITIAL_RADIUS, maximum_radius)
        high_point = _move_unit_point(center, high_radius * direction)
        high_value = float(evaluate(high_point))
        while high_value < target and high_radius < maximum_radius:
            low_radius = high_radius
            high_radius = min(2.0 * high_radius, maximum_radius)
            high_point = _move_unit_point(center, high_radius * direction)
            high_value = float(evaluate(high_point))

        if high_value < target:
            continue

        for _iteration in range(CONFIG_CONTOUR_BISECTION_ITERATIONS):
            middle_radius = 0.5 * (low_radius + high_radius)
            middle_point = _move_unit_point(
                center,
                middle_radius * direction,
            )
            if float(evaluate(middle_point)) < target:
                low_radius = middle_radius
            else:
                high_radius = middle_radius

        boundary_points.append(
            _move_unit_point(center, high_radius * direction)
        )
    return np.asarray(boundary_points, dtype=float).reshape(
        (-1, center.size)
    )


def _configuration_contour_task(task):
    """Compute one direction chunk of a local contour in a worker."""
    row_index, chunk_index, row, lepton_name, directions = task
    phase_scan._configure_lepton(lepton_name)
    center = _unit_point_from_minimum_row(row)
    base_value = float(
        row[f"{config_scan.mixing_prefix(lepton_name)}_D_W"]
    )
    evaluation_id = row_index * 10_000_000 + chunk_index * 100_000
    evaluation_count = 0

    def evaluate(unit_point):
        nonlocal evaluation_count
        value, _row = _d_w_evaluation(
            unit_point,
            lepton_name,
            evaluation_id + evaluation_count,
        )
        evaluation_count += 1
        return value

    boundary_points = _trace_high_dimensional_contour(
        evaluate,
        center,
        base_value,
        directions,
    )
    return row_index, chunk_index, boundary_points


def _configuration_contours(rows, lepton_name):
    """Evaluate one high-dimensional contour for each selected minimum."""
    chunks_per_minimum = min(
        8,
        max(1, PHASE_SPACE_CONFIG_CONTOUR_SAMPLES // 64),
    )
    tasks = []
    for row_index, row in enumerate(rows):
        direction_chunks = np.array_split(
            _contour_directions(DW_GRADIENT_RANDOM_SEED + row_index),
            chunks_per_minimum,
        )
        tasks.extend(
            (
                row_index,
                chunk_index,
                row,
                lepton_name,
                directions,
            )
            for chunk_index, directions in enumerate(direction_chunks)
            if len(directions)
        )
    workers = min(max(1, int(GRADIENT_WORKERS)), len(tasks))
    if workers <= 1:
        results = [_configuration_contour_task(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(
                    executor.map(
                        _configuration_contour_task,
                        tasks,
                        chunksize=1,
                    )
                )
        except (OSError, PermissionError, BrokenProcessPool):
            results = [_configuration_contour_task(task) for task in tasks]
    grouped = {row_index: [] for row_index in range(len(rows))}
    for row_index, chunk_index, boundary_points in sorted(
        results,
        key=lambda result: (result[0], result[1]),
    ):
        grouped[row_index].append(boundary_points)
    return {
        row_index: np.vstack(chunks) if chunks else np.empty((0, 7))
        for row_index, chunks in grouped.items()
    }


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
    """Project a sampled seven-dimensional contour into one plot panel."""
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
        "theta_in": tuple(phase_scan.THETA_IN_RANGE),
        "qOut": (0.0, float(qout_upper)),
        "phi_in": tuple(phase_scan.AZIMUTH_RANGE),
        "phiOut": tuple(phase_scan.AZIMUTH_RANGE),
        "theta_e": tuple(phase_scan.THETA_E_MIX_RANGE),
        "theta_p": tuple(phase_scan.THETA_P_MIX_RANGE),
    }


def _annotate_minimum_ids(ax, x, y, rows):
    """Label phase-space points with their persistent local-minimum IDs."""
    for x_value, y_value, row in zip(x, y, rows):
        minimum_id = row.get("local_minimum_id")
        if minimum_id is None:
            continue
        ax.annotate(
            f"ID {minimum_id}",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )


def _plot_all_local_minima(rows, lepton_name, path):
    """Plot every distinct local minimum before configuration selection."""
    plt, PdfPages = config_gen._require_matplotlib()
    values = _d_w_values(rows, lepton_name)
    cmap, vmin, vmax = config_gen.observable_plot_style("D_W")
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
            _annotate_minimum_ids(ax, x, y, rows)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.tick_params(labelsize=10)
        axes[0, 0].legend()
        axes[2, 2].hist(values, bins=min(60, max(5, len(values))))
        axes[2, 2].axvline(
            values[best_index], color="red", linestyle="--",
            label=rf"$D_{{W,\min}}={values[best_index]:.5g}$",
        )
        axes[2, 2].set_xlabel(r"local-minimum $D_W$")
        axes[2, 2].set_ylabel("distinct minima")
        axes[2, 2].legend()
        fig.suptitle(
            f"{lepton_name}: all {len(rows)} distinct gradient-search "
            r"local minima of $D_W$"
        )
        if image is not None:
            fig.colorbar(
                image,
                ax=axes.ravel()[:8].tolist(),
                label=r"$D_W$",
            )
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _mark_configuration_minima(rows, lepton_name):
    """Mark minima no farther than the threshold above the global minimum."""
    values = _d_w_values(rows, lepton_name)
    optimum = float(np.min(values))
    marked = []
    selected = []
    for row, value in zip(rows, values):
        item = dict(row)
        delta = float(value - optimum)
        eligible = delta <= PHASE_SPACE_CONFIG_THRESHOLD + 1.0e-12
        item["D_W_above_global_minimum"] = delta
        item["within_config_threshold"] = eligible
        marked.append(item)
        if eligible:
            selected.append(item)
    return marked, selected, optimum


def _configuration_rows(minimum_rows, lepton_name):
    """Annotate every distinct minimum for the coherent ConfigGen helpers."""
    prefix = config_scan.mixing_prefix(lepton_name)
    key = f"{prefix}_D_W"
    details = []
    for index, source in enumerate(minimum_rows):
        row = dict(source)
        value = float(row[key])
        row.update(
            {
                "selected_observable": "D_W",
                "selected_observable_label": config_gen.observable_label("D_W"),
                "selected_spin_case": "mixing_angles",
                "selected_spin_label": (
                    f"theta_e={float(row['theta_e']):.8g}, "
                    f"theta_p={float(row['theta_p']):.8g}"
                ),
                "selected_concurrence_key": key,
                "selected_concurrence": value,
                "selected_purity": float(row[f"{prefix}_purity"]),
                "pair_delta_xy": np.nan,
                "scan_phi_lepton_in": float(row["phi_in_lepton"]),
                "scan_phi_p_in": float(row["phi_in"]),
                "scan_phi_gamma": float(row["phiOut"]),
                "cluster_id": index,
                "energy_band_cluster_id": index,
                "selected_region": f"local_minimum_{index}",
                "detail_id": f"dw_mixing_angles_local_minimum_{index}",
                "detail_source": "random_start_gradient_search",
                "qOut_regime": "gradient_local_minimum",
            }
        )
        details.append(row)
    return details


def _write_configuration_plot(
    all_rows,
    detail_rows,
    lepton_name,
    optimum,
    path,
):
    """Write one projected 7D-contour page per selected local minimum."""
    plt, PdfPages = config_gen._require_matplotlib()
    selected_rows = [
        row for row in all_rows
        if bool(row["within_config_threshold"])
    ]
    if len(selected_rows) != len(detail_rows):
        raise ValueError(
            "Selected local-minimum rows and configuration details disagree."
        )
    full_limits = _full_phase_space_plot_limits(lepton_name)
    contours = _configuration_contours(selected_rows, lepton_name)
    d_w_key = f"{config_scan.mixing_prefix(lepton_name)}_D_W"
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
                minimum_id = row.get(
                    "local_minimum_id",
                    selected_index,
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
                ax.annotate(
                    f"ID {minimum_id}",
                    (center_x, center_y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
            ax.set_xlim(*full_limits[x_name])
            ax.set_ylim(*full_limits[y_name])
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.tick_params(labelsize=10)

        overview_summary = overview_axes[2, 2]
        overview_summary.axis("off")
        overview_lines = [
            f"{len(selected_rows)} selected local minima",
            (
                f"Threshold={PHASE_SPACE_CONFIG_THRESHOLD:g}, "
                f"contour delta={PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}"
            ),
            (
                f"configured 7D samples/minimum="
                f"{PHASE_SPACE_CONFIG_CONTOUR_SAMPLES}"
            ),
            "",
            " ID       D_W(local)    above global   contour pts",
        ]
        for selected_index, row in enumerate(selected_rows):
            minimum_id = row.get("local_minimum_id", selected_index)
            local_value = float(row[d_w_key])
            overview_lines.append(
                f"{str(minimum_id):>3s}  {local_value:14.7g}  "
                f"{local_value - optimum:12.6g}  "
                f"{len(contours[selected_index]):11d}"
            )
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
            f"{lepton_name}: summary of all selected local minima and "
            "pairwise projections of their 7D contours"
        )
        pdf.savefig(overview_fig)
        plt.close(overview_fig)

        for selected_index, (row, detail_row) in enumerate(
            zip(selected_rows, detail_rows)
        ):
            # For each minimum, present the reconstructed configuration before
            # its high-dimensional contour projections.
            config_scan._save_mixing_detail_pages(pdf, plt, [detail_row])
            center = _unit_point_from_minimum_row(row)
            boundary_points = contours[selected_index]
            minimum_id = row.get("local_minimum_id", selected_index)
            local_value = float(row[d_w_key])
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
                    rasterized=True,
                    label=(
                        "projected 7D contour samples"
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
                            rf"$D_W=D_{{W,\mathrm{{local}}}}"
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
                        f"selected minimum ID {minimum_id}"
                        if panel_index == 0 else None
                    ),
                    zorder=3,
                )
                ax.set_xlim(*full_limits[x_name])
                ax.set_ylim(*full_limits[y_name])
                ax.set_xlabel(x_label, fontsize=11)
                ax.set_ylabel(y_label, fontsize=11)
                ax.tick_params(labelsize=10)

            axes[0, 0].legend(fontsize=8)
            summary_ax = axes[2, 2]
            summary_ax.axis("off")
            coordinates = _plot_coordinate_values(center)
            summary_lines = [
                f"selected local minimum ID {minimum_id}",
                "",
                f"D_W(local) = {local_value:.8g}",
                (
                    f"D_W contour = "
                    f"{local_value + PHASE_SPACE_CONFIG_CONTOUR_DELTA:.8g}"
                ),
                (
                    f"above global minimum = "
                    f"{local_value - optimum:.8g}"
                ),
                (
                    f"selection Threshold = "
                    f"{PHASE_SPACE_CONFIG_THRESHOLD:g}"
                ),
                f"7D contour samples = {len(boundary_points)}",
                "",
                "selected phase-space point:",
            ]
            summary_lines.extend(
                f"{name:>10s} = {coordinates[name]:.7g}"
                for name in (
                    "sqrt_s",
                    "theta_in",
                    "qOut",
                    "phi_in",
                    "phiOut",
                    "theta_e",
                    "theta_p",
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
                f"{lepton_name}: selected local minimum ID {minimum_id}; "
                "pairwise projections of the 7D "
                rf"$D_W=D_{{W,\mathrm{{local}}}}"
                rf"+{PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}$ contour"
            )
            pdf.savefig(fig)
            plt.close(fig)
    return path


def _write_configurations(
    lepton_name,
    selected_path,
    all_minimum_rows,
    selected_minimum_rows,
    optimum,
):
    """Generate configuration, momentum, amplitude, and PDF outputs."""
    config_gen.configure_lepton(
        lepton_name,
        input_path=selected_path,
        output_root=CONFIG_OUTPUT_ROOT,
    )
    config_gen.clean_egamma_config_outputs()
    config_gen.clean_data_outputs()
    details = _configuration_rows(selected_minimum_rows, lepton_name)
    paths = config_gen.target_paths("dw")
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
    plot_path = (
        config_gen.OUTPUT_DIR
        / config_scan.mixing_prefix(lepton_name)
        / "dw_gradient_local_minima.pdf"
    )
    _write_configuration_plot(
        all_minimum_rows,
        details,
        lepton_name,
        optimum,
        plot_path,
    )
    return paths, plot_path


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
                float(start["theta_in"]) - phase_scan.THETA_IN_RANGE[0]
            ) / (
                phase_scan.THETA_IN_RANGE[1]
                - phase_scan.THETA_IN_RANGE[0]
            ),
            (
                qout_fraction - phase_scan.QOUT_FRACTION_RANGE[0]
            ) / (
                phase_scan.QOUT_FRACTION_RANGE[1]
                - phase_scan.QOUT_FRACTION_RANGE[0]
            ),
            float(start["phi_in_lepton"]) / (2.0 * np.pi),
            float(start["phiOut"]) / (2.0 * np.pi),
            float(start["theta_e"]) / np.pi,
            float(start["theta_p"]) / np.pi,
        ),
        dtype=float,
    )


def _screen_start_task(task):
    """Evaluate one low-discrepancy candidate without optimizing it."""
    lepton_name, candidate_index, point = task
    phase_scan._configure_lepton(lepton_name)
    value, _row = _d_w_evaluation(
        point,
        lepton_name,
        evaluation_id=2_000_000_000 + candidate_index,
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
    """Select low-D_W, spatially separated starts from a Sobol design."""
    exponent = int(np.log2(DW_GRADIENT_SCREENING_SAMPLES))
    candidates = qmc.Sobol(
        d=7,
        scramble=True,
        seed=species_seed + 10_000,
    ).random_base2(exponent)
    tasks = [
        (lepton_name, index, point)
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
            >= DW_GRADIENT_SCREENING_SEPARATION
            for prior in selected
        ):
            selected.append((value, point))
            if len(selected) == DW_GRADIENT_SCREENED_STARTS:
                break
        else:
            deferred.append((value, point))
    if len(selected) < DW_GRADIENT_SCREENED_STARTS:
        for value, point in deferred:
            if any(np.array_equal(point, prior[1]) for prior in selected):
                continue
            selected.append((value, point))
            if len(selected) == DW_GRADIENT_SCREENED_STARTS:
                break
    return selected


def _species_tasks(lepton_name):
    """Return hybrid global, screened, and physics-anchored start tasks."""
    phase_scan._configure_lepton(lepton_name)
    species_seed = (
        DW_GRADIENT_RANDOM_SEED + tuple(LEPTON_SPECS).index(lepton_name)
    )
    latin_starts = qmc.LatinHypercube(d=7, seed=species_seed).random(
        DW_GRADIENT_RANDOM_STARTS
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
            screening_d_w,
        )
        for run_index, (point, source, screening_d_w) in enumerate(starts)
    ]


def run_species(lepton_name, results=None):
    """Find or consume minima, then write one species' outputs safely."""
    phase_scan._configure_lepton(lepton_name)
    tasks = _species_tasks(lepton_name)
    if results is None:
        results = _run_tasks(tasks)
    minima = _deduplicate_minima(results)
    if not minima:
        raise RuntimeError(
            f"No converged, locally verified D_W minimum was found for "
            f"{lepton_name}; inspect the optimizer settings or increase "
            "DW_GRADIENT_MAX_ITERATIONS."
        )

    species_dir = OUTPUT_ROOT / lepton_name
    run_path = _write_csv(
        species_dir / "optimization_runs.csv",
        [result[0] for result in results],
    )
    minimum_rows = []
    for minimum_index, (_run, _unit_point, row) in enumerate(minima):
        item = dict(row)
        item["local_minimum_id"] = minimum_index
        item["kinematic_point"] = f"gradient_local_minimum_{minimum_index:04d}"
        minimum_rows.append(item)
    minimum_rows, selected_rows, optimum = _mark_configuration_minima(
        minimum_rows, lepton_name
    )
    minima_path = _write_csv(species_dir / "local_minima.csv", minimum_rows)
    minima_plot = _plot_all_local_minima(
        minimum_rows,
        lepton_name,
        species_dir / "all_local_minima.pdf",
    )
    selected_path = _write_csv(
        species_dir / "config_selected_minima.csv",
        selected_rows,
    )
    paths, plot_path = _write_configurations(
        lepton_name,
        selected_path,
        minimum_rows,
        selected_rows,
        optimum,
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
    d_w_key = f"{config_scan.mixing_prefix(lepton_name)}_D_W"
    return "\n".join(
        (
            f"Hybrid global gradient D_W search ({lepton_name})",
            f"  optimization starts: {len(tasks)}",
            f"  Latin-hypercube starts: {latin_count}",
            (
                f"  Sobol screening: {DW_GRADIENT_SCREENING_SAMPLES} "
                f"candidates -> {screened_count} optimized starts"
            ),
            (
                f"  best screened D_W: "
                f"{min(screened_values):.10g}"
                if screened_values
                else "  best screened D_W: unavailable"
            ),
            f"  deterministic physics anchors: {anchor_count}",
            f"  shared optimization workers: {GRADIENT_WORKERS}",
            f"  L-BFGS-B-converged runs: {lbfgs_converged}/{len(tasks)}",
            f"  multiscale-verified runs: {verified}/{len(tasks)}",
            f"  distinct finite minima: {len(minima)}",
            (
                f"  minima within Threshold={PHASE_SPACE_CONFIG_THRESHOLD:g}: "
                f"{len(selected_rows)}/{len(minimum_rows)}"
            ),
            f"  best D_W: {minimum_rows[0][d_w_key]:.10g}",
            f"  optimization runs: {run_path}",
            f"  local minima: {minima_path}",
            f"  all-local-minima plot: {minima_plot}",
            (
                f"  local-minimum contour delta: "
                f"{PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}"
            ),
            f"  configuration-selected minima: {selected_path}",
            f"  configuration examples: {paths['examples']}",
            f"  momentum configurations: {paths['momenta']}",
            f"  amplitude decomposition: {paths['amplitudes']}",
            f"  configuration PDF: {plot_path}",
        )
    )


def regenerate_species_plots(lepton_name):
    """Rebuild one species' PDFs from its saved local-minimum CSV."""
    species_dir = OUTPUT_ROOT / lepton_name
    minima_path = species_dir / "local_minima.csv"
    minimum_rows = _read_csv(minima_path)
    minimum_rows, selected_rows, optimum = _mark_configuration_minima(
        minimum_rows, lepton_name
    )
    minima_plot = _plot_all_local_minima(
        minimum_rows,
        lepton_name,
        species_dir / "all_local_minima.pdf",
    )
    config_gen.configure_lepton(
        lepton_name,
        input_path=minima_path,
        output_root=CONFIG_OUTPUT_ROOT,
    )
    detail_rows = _configuration_rows(selected_rows, lepton_name)
    config_plot = (
        config_gen.OUTPUT_DIR
        / config_scan.mixing_prefix(lepton_name)
        / "dw_gradient_local_minima.pdf"
    )
    _write_configuration_plot(
        minimum_rows,
        detail_rows,
        lepton_name,
        optimum,
        config_plot,
    )
    return "\n".join(
        (
            f"Regenerated gradient D_W plots ({lepton_name})",
            f"  source data: {minima_path}",
            f"  local minima loaded: {len(minimum_rows)}",
            (
                f"  minima within Threshold={PHASE_SPACE_CONFIG_THRESHOLD:g}: "
                f"{len(selected_rows)}/{len(minimum_rows)}"
            ),
            f"  all-local-minima plot: {minima_plot}",
            (
                f"  local-minimum contour delta: "
                f"{PHASE_SPACE_CONFIG_CONTOUR_DELTA:g}"
            ),
            f"  configuration PDF: {config_plot}",
            "  gradient optimization was not rerun",
        )
    )


def validate_settings():
    """Validate controls before starting expensive optimization work."""
    if not SCAN_INITIAL_MIXING_ANGLES:
        raise ValueError(
            "GradientPhaseSpaceScan requires SCAN_INITIAL_MIXING_ANGLES=True."
        )
    unknown = set(LEPTONS_TO_OPTIMIZE) - set(LEPTON_SPECS)
    if unknown:
        raise ValueError(f"Unknown lepton species: {sorted(unknown)}")
    if not LEPTONS_TO_OPTIMIZE:
        raise ValueError("LEPTONS_TO_OPTIMIZE must not be empty.")
    if GRADIENT_WORKERS < 1:
        raise ValueError("GRADIENT_WORKERS must be positive.")
    if DW_GRADIENT_RANDOM_STARTS < 1:
        raise ValueError("DW_GRADIENT_RANDOM_STARTS must be positive.")
    if (
        not isinstance(DW_GRADIENT_SCREENING_SAMPLES, (int, np.integer))
        or isinstance(DW_GRADIENT_SCREENING_SAMPLES, (bool, np.bool_))
        or DW_GRADIENT_SCREENING_SAMPLES < 2
        or (
            DW_GRADIENT_SCREENING_SAMPLES
            & (DW_GRADIENT_SCREENING_SAMPLES - 1)
        )
    ):
        raise ValueError(
            "DW_GRADIENT_SCREENING_SAMPLES must be a power-of-two integer."
        )
    if (
        not isinstance(DW_GRADIENT_SCREENED_STARTS, (int, np.integer))
        or isinstance(DW_GRADIENT_SCREENED_STARTS, (bool, np.bool_))
        or not 1
        <= DW_GRADIENT_SCREENED_STARTS
        <= DW_GRADIENT_SCREENING_SAMPLES
    ):
        raise ValueError(
            "DW_GRADIENT_SCREENED_STARTS must be an integer between 1 "
            "and DW_GRADIENT_SCREENING_SAMPLES."
        )
    if (
        not np.isfinite(DW_GRADIENT_SCREENING_SEPARATION)
        or not 0.0 < DW_GRADIENT_SCREENING_SEPARATION <= 1.0
    ):
        raise ValueError(
            "DW_GRADIENT_SCREENING_SEPARATION must lie in (0, 1]."
        )
    if DW_GRADIENT_MAX_ITERATIONS < 1:
        raise ValueError("DW_GRADIENT_MAX_ITERATIONS must be positive.")
    for name, value in (
        ("DW_GRADIENT_TOLERANCE", DW_GRADIENT_TOLERANCE),
        (
            "DW_GRADIENT_SCAN_PRECISION",
            DW_GRADIENT_SCAN_PRECISION,
        ),
        ("DW_GRADIENT_MINIMUM_SEPARATION", DW_GRADIENT_MINIMUM_SEPARATION),
        ("DW_LOCAL_SEARCH_INITIAL_STEP", DW_LOCAL_SEARCH_INITIAL_STEP),
        (
            "DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE",
            DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE,
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if DW_GRADIENT_SCAN_PRECISION > 1.0:
        raise ValueError(
            "DW_GRADIENT_SCAN_PRECISION must not exceed the normalized "
            "scan width of 1."
        )
    if not 0.0 < DW_LOCAL_SEARCH_STEP_REDUCTION < 1.0:
        raise ValueError(
            "DW_LOCAL_SEARCH_STEP_REDUCTION must lie strictly between 0 and 1."
        )
    if DW_LOCAL_SEARCH_INITIAL_STEP < DW_GRADIENT_SCAN_PRECISION:
        raise ValueError(
            "DW_LOCAL_SEARCH_INITIAL_STEP must be at least "
            "DW_GRADIENT_SCAN_PRECISION."
        )
    if DW_LOCAL_SEARCH_MAX_POLLS < 1:
        raise ValueError("DW_LOCAL_SEARCH_MAX_POLLS must be positive.")
    if DW_LOCAL_SEARCH_RANDOM_DIRECTIONS < 0:
        raise ValueError(
            "DW_LOCAL_SEARCH_RANDOM_DIRECTIONS must be non-negative."
        )
    if (
        not np.isfinite(PHASE_SPACE_CONFIG_THRESHOLD)
        or PHASE_SPACE_CONFIG_THRESHOLD < 0.0
    ):
        raise ValueError(
            "PHASE_SPACE_CONFIG_THRESHOLD must be finite and non-negative."
        )
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
        or PHASE_SPACE_CONFIG_CONTOUR_SAMPLES < 14
    ):
        raise ValueError(
            "PHASE_SPACE_CONFIG_CONTOUR_SAMPLES must be an integer "
            "of at least 14."
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


def main():
    """Optimize D_W and generate local-minimum configurations."""
    if REGENERATE_PLOTS_FROM_CSV:
        reports = [
            regenerate_species_plots(lepton_name)
            for lepton_name in LEPTONS_TO_OPTIMIZE
        ]
    else:
        validate_settings()
        reports = [
            run_species(lepton_name)
            for lepton_name in LEPTONS_TO_OPTIMIZE
        ]
    report = "\n\n".join(reports) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
