"""Find local D_W minima with random starts and bounded gradient searches.

Each optimization uses the same seven continuous coordinates and coherent
incoming-spin preparation as :mod:`PhaseSpaceScan`. The coordinates are
normalized to a unit box before SciPy's L-BFGS-B minimizer estimates numerical
gradients. A periodic-aware multiscale direct search then follows unresolved
descent directions down to the requested scan precision. Distinct verified
minima are written as scan-compatible rows and converted into ConfigGen-style
momentum and amplitude configurations.
"""

from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
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
    DW_GRADIENT_TOLERANCE,
    DW_LOCAL_SEARCH_INITIAL_STEP,
    DW_LOCAL_SEARCH_MAX_POLLS,
    DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE,
    DW_LOCAL_SEARCH_RANDOM_DIRECTIONS,
    DW_LOCAL_SEARCH_STEP_REDUCTION,
    PHASE_SPACE_CONFIG_STEP,
    SCAN_INITIAL_MIXING_ANGLES,
    SCAN_WORKERS,
)
from PlotUtils import print_console_text


# Script controls.
LEPTONS_TO_OPTIMIZE = ("electron", "muon", "heavy", "massless")
GRADIENT_WORKERS = SCAN_WORKERS
OUTPUT_ROOT = Path("Output") / "GradientPhaseSpaceScan"
CONFIG_OUTPUT_ROOT = Path("Output") / "GradientPhaseSpaceConfig"
LOG_PATH = OUTPUT_ROOT / "GradientPhaseSpaceScan.log"
INVALID_OBJECTIVE = 1.0e3
PERIODIC_UNIT_COORDINATES = (3, 4, 5, 6)
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
    lepton_name, run_index, start = task
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
    except (OSError, PermissionError):
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


def _d_w_values(rows, lepton_name):
    """Return the local-minimum D_W values in row order."""
    key = f"{config_scan.mixing_prefix(lepton_name)}_D_W"
    return np.asarray([float(row[key]) for row in rows], dtype=float)


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
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
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
    """Mark minima no farther than STEP above the global D_W minimum."""
    values = _d_w_values(rows, lepton_name)
    optimum = float(np.min(values))
    marked = []
    selected = []
    for row, value in zip(rows, values):
        item = dict(row)
        delta = float(value - optimum)
        eligible = delta <= PHASE_SPACE_CONFIG_STEP + 1.0e-12
        item["D_W_above_global_minimum"] = delta
        item["within_config_STEP"] = eligible
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
    """Identify STEP-selected minima and append only their config pages."""
    plt, PdfPages = config_gen._require_matplotlib()
    values = _d_w_values(all_rows, lepton_name)
    eligible = np.asarray(
        [bool(row["within_config_STEP"]) for row in all_rows],
        dtype=bool,
    )
    threshold = optimum + PHASE_SPACE_CONFIG_STEP
    cmap, vmin, vmax = config_gen.observable_plot_style("D_W")
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(
            3, 3, figsize=(14.0, 11.5), constrained_layout=True
        )
        image = None
        for ax, (x_name, y_name, x_label, y_label) in zip(
            axes.ravel()[:8], PLOT_PANELS
        ):
            x = np.asarray([float(row[x_name]) for row in all_rows])
            y = np.asarray([float(row[y_name]) for row in all_rows])
            ax.scatter(
                x[~eligible], y[~eligible], s=32, c="lightgray",
                edgecolors="gray", linewidths=0.3, label="outside STEP",
            )
            image = ax.scatter(
                x[eligible], y[eligible], c=values[eligible], marker="*",
                s=150, cmap=cmap, vmin=vmin, vmax=vmax,
                edgecolors="black", label="configuration selected",
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        axes[0, 0].legend()
        axes[2, 2].hist(
            values[~eligible],
            bins=min(60, max(5, len(values))),
            color="lightgray",
            label="outside STEP",
        )
        axes[2, 2].hist(
            values[eligible],
            bins=min(60, max(5, len(values))),
            color="tab:blue",
            alpha=0.8,
            label="configuration selected",
        )
        axes[2, 2].axvline(
            threshold, color="red", linestyle="--",
            label=rf"$D_{{W,\min}}+\mathrm{{STEP}}={threshold:.5g}$",
        )
        axes[2, 2].set_xlabel(r"local-minimum $D_W$")
        axes[2, 2].set_ylabel("distinct minima")
        axes[2, 2].legend()
        fig.suptitle(
            f"{lepton_name}: {len(detail_rows)}/{len(all_rows)} local minima "
            f"within STEP={PHASE_SPACE_CONFIG_STEP:g} of the global minimum"
        )
        if image is not None:
            fig.colorbar(
                image,
                ax=axes.ravel()[:8].tolist(),
                label=r"$D_W$",
            )
        pdf.savefig(fig)
        plt.close(fig)
        config_scan._save_mixing_detail_pages(pdf, plt, detail_rows)
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


def _species_tasks(lepton_name):
    """Return randomized stratified optimization tasks for one species."""
    species_seed = (
        DW_GRADIENT_RANDOM_SEED + tuple(LEPTON_SPECS).index(lepton_name)
    )
    starts = qmc.LatinHypercube(d=7, seed=species_seed).random(
        DW_GRADIENT_RANDOM_STARTS
    )
    return [
        (lepton_name, index, start)
        for index, start in enumerate(starts)
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
    d_w_key = f"{config_scan.mixing_prefix(lepton_name)}_D_W"
    return "\n".join(
        (
            f"Random-start gradient D_W search ({lepton_name})",
            f"  random starts: {len(tasks)}",
            f"  shared optimization workers: {GRADIENT_WORKERS}",
            f"  L-BFGS-B-converged runs: {lbfgs_converged}/{len(tasks)}",
            f"  multiscale-verified runs: {verified}/{len(tasks)}",
            f"  distinct finite minima: {len(minima)}",
            (
                f"  minima within STEP={PHASE_SPACE_CONFIG_STEP:g}: "
                f"{len(selected_rows)}/{len(minimum_rows)}"
            ),
            f"  best D_W: {minimum_rows[0][d_w_key]:.10g}",
            f"  optimization runs: {run_path}",
            f"  local minima: {minima_path}",
            f"  all-local-minima plot: {minima_plot}",
            f"  configuration-selected minima: {selected_path}",
            f"  configuration examples: {paths['examples']}",
            f"  momentum configurations: {paths['momenta']}",
            f"  amplitude decomposition: {paths['amplitudes']}",
            f"  configuration PDF: {plot_path}",
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
    if not np.isfinite(PHASE_SPACE_CONFIG_STEP) or PHASE_SPACE_CONFIG_STEP < 0.0:
        raise ValueError(
            "PHASE_SPACE_CONFIG_STEP must be finite and non-negative."
        )


def main():
    """Optimize D_W and generate local-minimum configurations."""
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
