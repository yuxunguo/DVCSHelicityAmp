"""Adaptive phase-space scan for all AlignmentScan entanglement observables.

The scan covers ``sqrt(s)``, ``theta_in``, ``E_gamma``, the incoming-lepton
azimuth, and the outgoing-photon azimuth.  It evaluates the same polarization
cases and entanglement observables as :mod:`AlignmentScan`, then refines around
the best point for every observable/polarization pair.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from pathlib import Path
import shutil

import numpy as np

from AlignmentScan import (
    ALIGNMENT_ANGLE_MAX_RAD,
    ALIGNMENT_SPIN_CASES,
    SCAN_OBSERVABLE_NAMES,
    SIGNED_CONCURRENCE_OBSERVABLES,
    _concurrence_csv_headers,
    _concurrence_csv_row,
    _evaluate_kinematic_sample,
    explicit_polarization_name,
    observable_latex_label,
    observable_is_minimized,
    observable_optimum_label,
    observable_rank_value,
    observable_text_label,
    save_concurrence_top_csv,
    species_spin_label,
)
from config import (
    HEAVY_LEPTON_MASS_GEV,
    ELECTRON_MASS_GEV,
    MASSLESS_LEPTON_MASS_GEV,
    MUON_MASS_GEV,
    NORMALIZE_TRACE,
    PROTON_MASS_GEV,
    SCAN_WORKERS,
)
from PlotUtils import print_console_text, require_matplotlib


# Explicit script settings. Edit these values before running PhaseSpaceScan.py.
LEPTONS_TO_SCAN = ("electron", "muon", "heavy", "massless")
PHASE_SPACE_SCAN_WORKERS = SCAN_WORKERS
PHASE_SPACE_PLOT_WORKERS = max(1, min(SCAN_WORKERS, 24))
RANDOM_SEED = 271828
PHASE_SPACE_SAMPLES = 8192
REFINEMENT_SAMPLES = 4096
REFINEMENT_CENTERS = len(ALIGNMENT_SPIN_CASES) * len(SCAN_OBSERVABLE_NAMES)
ALIGNMENT_SEED_CENTERS = REFINEMENT_CENTERS
TOP_POINTS_PER_POLARIZATION = 100
THETA_IN_RANGE = (0.35, 2.80)
QOUT_FRACTION_RANGE = (0.05, 0.95)
AZIMUTH_RANGE = (0.0, 2.0 * np.pi)
OUTPUT_ROOT = Path("Output") / "PhaseSpaceScan"
LOG_PATH = OUTPUT_ROOT / "PhaseSpaceScan.log"

LEPTON_SETTINGS = {
    "electron": {
        "mass": ELECTRON_MASS_GEV,
        "sqrt_s_range": (1.05 * (PROTON_MASS_GEV + ELECTRON_MASS_GEV), 5.00),
        "file_stem": "electron_photon",
    },
    "muon": {
        "mass": MUON_MASS_GEV,
        "sqrt_s_range": (1.05 * (PROTON_MASS_GEV + MUON_MASS_GEV), 5.00),
        "file_stem": "muon_photon",
    },
    "heavy": {
        "mass": HEAVY_LEPTON_MASS_GEV,
        "sqrt_s_range": (1.001 * (PROTON_MASS_GEV + HEAVY_LEPTON_MASS_GEV), 100.0),
        "file_stem": "heavy_photon",
    },
    "massless": {
        "mass": MASSLESS_LEPTON_MASS_GEV,
        "sqrt_s_range": (1.05 * PROTON_MASS_GEV, 5.00),
        "file_stem": "massless_photon",
    },
}

LEPTON_NAME = "electron"
LEPTON_MASS_GEV = ELECTRON_MASS_GEV
COM_THRESHOLD = PROTON_MASS_GEV + LEPTON_MASS_GEV
SQRT_S_RANGE = LEPTON_SETTINGS[LEPTON_NAME]["sqrt_s_range"]
S_RANGE = tuple(value**2 for value in SQRT_S_RANGE)
# Sample E_gamma relative to its s-dependent kinematic ceiling.  This keeps
# near-threshold points physical as the available photon energy approaches zero.
OUTPUT_DIR = OUTPUT_ROOT / "electron"
FULL_CSV = OUTPUT_DIR / "electron_photon_entanglement_phase_space.csv"
ALIGNED_CSV = OUTPUT_DIR / "electron_photon_entanglement_aligned.csv"
TOP_CSV = OUTPUT_DIR / "electron_photon_entanglement_top.csv"
PLOT_DIR = OUTPUT_DIR


def _qout_max(s):
    """Return the photon-energy ceiling for the user-frame parametrization."""
    sqrt_s = np.sqrt(s)
    available_energy = sqrt_s - PROTON_MASS_GEV
    return (
        available_energy**2 - LEPTON_MASS_GEV**2
    ) / (2.0 * available_energy)


QOUT_RANGE = (0.0, float(_qout_max(S_RANGE[1])))


def _configure_lepton(name):
    """Configure masses, threshold, and independent output paths."""
    global LEPTON_NAME, LEPTON_MASS_GEV, COM_THRESHOLD
    global SQRT_S_RANGE, S_RANGE, QOUT_RANGE
    global OUTPUT_DIR, FULL_CSV, ALIGNED_CSV, TOP_CSV, PLOT_DIR

    if name not in LEPTON_SETTINGS:
        raise ValueError(
            f"Unknown lepton {name!r}; choose from {tuple(LEPTON_SETTINGS)}."
        )
    settings = LEPTON_SETTINGS[name]
    LEPTON_NAME = name
    LEPTON_MASS_GEV = settings["mass"]
    COM_THRESHOLD = PROTON_MASS_GEV + LEPTON_MASS_GEV
    SQRT_S_RANGE = settings["sqrt_s_range"]
    S_RANGE = tuple(value**2 for value in SQRT_S_RANGE)
    QOUT_RANGE = (0.0, float(_qout_max(S_RANGE[1])))

    stem = settings["file_stem"]
    species_dir = OUTPUT_ROOT / name
    OUTPUT_DIR = species_dir
    FULL_CSV = OUTPUT_DIR / f"{stem}_entanglement_phase_space.csv"
    ALIGNED_CSV = OUTPUT_DIR / f"{stem}_entanglement_aligned.csv"
    TOP_CSV = OUTPUT_DIR / f"{stem}_entanglement_top.csv"
    PLOT_DIR = OUTPUT_DIR


def _uniform_samples(rng, count):
    """Return a randomized stratified design covering every scan coordinate."""
    unit_samples = np.column_stack([
        (rng.permutation(count) + rng.random(count)) / count
        for _dimension in range(5)
    ])
    sqrt_s_values = (
        SQRT_S_RANGE[0]
        + unit_samples[:, 0] * (SQRT_S_RANGE[1] - SQRT_S_RANGE[0])
    )
    s_values = sqrt_s_values**2
    qout_max = _qout_max(s_values)
    qout_fractions = (
        QOUT_FRACTION_RANGE[0]
        + unit_samples[:, 2]
        * (QOUT_FRACTION_RANGE[1] - QOUT_FRACTION_RANGE[0])
    )
    qout_values = qout_fractions * qout_max
    return np.column_stack((
        s_values,
        THETA_IN_RANGE[0]
        + unit_samples[:, 1] * (THETA_IN_RANGE[1] - THETA_IN_RANGE[0]),
        qout_values,
        AZIMUTH_RANGE[0]
        + unit_samples[:, 3] * (AZIMUTH_RANGE[1] - AZIMUTH_RANGE[0]),
        AZIMUTH_RANGE[0]
        + unit_samples[:, 4] * (AZIMUTH_RANGE[1] - AZIMUTH_RANGE[0]),
    ))


def _circular_delta(first, second):
    return (first - second + np.pi) % (2.0 * np.pi) - np.pi


def _normalized_distance(first, second):
    scales = np.array((SQRT_S_RANGE[1] - SQRT_S_RANGE[0],
                       THETA_IN_RANGE[1] - THETA_IN_RANGE[0],
                       QOUT_RANGE[1] - QOUT_RANGE[0], np.pi, np.pi))
    delta = first - second
    delta[0] = np.sqrt(first[0]) - np.sqrt(second[0])
    delta[3:] = [_circular_delta(first[i], second[i]) for i in (3, 4)]
    return float(np.linalg.norm(delta / scales))


def _point_from_row(row):
    """Return the five independent scan coordinates stored in a result row."""
    return np.asarray([
        float(row["s"]), float(row["theta_in"]), float(row["qOut"]),
        float(row["phi_in_lepton"]), float(row["phiOut"]),
    ])


def _point_in_scan_range(point):
    """Return whether a seed lies inside the configured physical search box."""
    return (
        S_RANGE[0] <= point[0] <= S_RANGE[1]
        and THETA_IN_RANGE[0] <= point[1] <= THETA_IN_RANGE[1]
        and QOUT_RANGE[0] <= point[2] <= min(QOUT_RANGE[1], _qout_max(point[0]))
    )


def _append_separated_center(centers, point, separation):
    if not _point_in_scan_range(point):
        return False
    if all(
        _normalized_distance(point.copy(), other.copy()) > separation
        for other in centers
    ):
        centers.append(point)
        return True
    return False


def _select_refinement_centers(rows, count=REFINEMENT_CENTERS):
    """Select fairly across every observable and initial polarization.

    One best candidate per observable/polarization pair is considered first.
    Additional ranked candidates fill holes caused by nearby optima, so no
    single observable controls the refinement stage.
    """
    centers = []
    rankings = []
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        for observable in SCAN_OBSERVABLE_NAMES:
            key = f"{prefix}_{observable}"
            finite = [row for row in rows if np.isfinite(float(row.get(key, np.nan)))]
            rankings.append(sorted(
                finite,
                key=lambda row: observable_rank_value(row[key], observable),
                reverse=not observable_is_minimized(observable),
            ))

    # Round-robin ranks prevent the first polarization/observable from
    # consuming all centers when several optima coincide.
    max_rank = max((len(ranking) for ranking in rankings), default=0)
    for rank in range(max_rank):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            _append_separated_center(centers, _point_from_row(ranking[rank]), 0.035)
            if len(centers) >= count:
                return centers
    return centers


def _alignment_seed_path():
    return (
        Path("Output") / "AlignmentScan" / LEPTON_NAME
        / f"{LEPTON_SETTINGS[LEPTON_NAME]['file_stem']}_concurrence_top.csv"
    )


def _alignment_seed_points():
    """Return separated top points from the matching AlignmentScan species."""
    path = _alignment_seed_path()
    if not path.exists():
        return np.empty((0, 5))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row.get("rank", 0) or 0))
    centers = []
    for row in rows:
        _append_separated_center(centers, _point_from_row(row), 0.035)
        if len(centers) == ALIGNMENT_SEED_CENTERS:
            break
    return np.asarray(centers)


def _refinement_samples(rng, centers, count):
    if not len(centers):
        return np.empty((0, 5))
    scales = np.array((0.16, 0.08 * QOUT_RANGE[1], 0.24, 0.24))
    sqrt_s_scale = 0.06 * (SQRT_S_RANGE[1] - SQRT_S_RANGE[0])
    samples = []
    for index in range(count):
        point = centers[index % len(centers)].copy()
        point[0] = np.clip(
            np.sqrt(point[0]) + rng.normal() * sqrt_s_scale, *SQRT_S_RANGE
        ) ** 2
        point[1:] += rng.normal(size=4) * scales
        point[1] = np.clip(point[1], *THETA_IN_RANGE)
        qout_max = _qout_max(point[0])
        point[2] = np.clip(point[2], 0.01 * qout_max, 0.99 * qout_max)
        point[3:] %= 2.0 * np.pi
        samples.append(point)
    return np.asarray(samples)


def _evaluate_sample(
    point,
    sample_id,
    stage,
    lepton_name,
    lepton_mass,
):
    """Evaluate one sample using explicit worker-safe species settings."""
    s, theta_in, qout, phi_e, phi_gamma = map(float, point)
    anchor = {
        "kinematic_point": f"{stage}_{sample_id:05d}",
        "s_regime": stage,
        "theta_in_regime": stage,
        "qOut_regime": stage,
        "s": s,
        "theta_in": theta_in,
        "qOut": qout,
    }
    settings = {
        "m": PROTON_MASS_GEV,
        "lepton_mass": lepton_mass,
        "lepton_name": lepton_name,
        "normalize_trace": NORMALIZE_TRACE,
        "angle_max_rad": ALIGNMENT_ANGLE_MAX_RAD,
    }
    try:
        result = _evaluate_kinematic_sample((anchor, phi_e, phi_gamma, settings))
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return None
    if not result["ok"]:
        return None
    row = result["row"]
    row["search_stage"] = stage
    row["sample_id"] = sample_id
    return row


def _evaluate_sample_task(task):
    """Picklable adapter for process and thread executors."""
    point, sample_id, stage, lepton_name, lepton_mass = task
    return _evaluate_sample(point, sample_id, stage, lepton_name, lepton_mass)


def _parallel_chunksize(task_count, worker_count):
    """Aim for four work batches per process to balance IPC and tail latency."""
    target_chunks = max(1, worker_count * 4)
    return max(1, (task_count + target_chunks - 1) // target_chunks)


def _build_kinematic_sample_tasks(samples, stage, start_id):
    """Build one task per kinematic sample for the configured lepton type."""
    return [
        (point, start_id + offset, stage, LEPTON_NAME, LEPTON_MASS_GEV)
        for offset, point in enumerate(samples)
    ]


def _run_kinematic_sample_tasks(tasks, max_workers):
    """Distribute kinematics across workers while preserving sample order."""
    if not tasks:
        return []
    if max_workers and max_workers > 1 and len(tasks) > 1:
        worker_count = min(int(max_workers), len(tasks))
        chunksize = _parallel_chunksize(len(tasks), worker_count)
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(_evaluate_sample_task, tasks, chunksize=chunksize)
                )
        except (OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(_evaluate_sample_task, tasks))
    else:
        results = [_evaluate_sample_task(task) for task in tasks]
    return results


def _evaluate_samples(samples, stage, start_id, max_workers):
    """Evaluate all polarizations for each parallelized kinematic sample."""
    tasks = _build_kinematic_sample_tasks(samples, stage, start_id)
    results = _run_kinematic_sample_tasks(tasks, max_workers)
    return [row for row in results if row is not None]


def run_phase_space_scan():
    rng = np.random.default_rng(RANDOM_SEED)
    phase_space_rows = _evaluate_samples(
        _uniform_samples(rng, PHASE_SPACE_SAMPLES),
        "phase_space",
        start_id=0,
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    seed_points = _alignment_seed_points()
    seed_rows = _evaluate_samples(
        seed_points,
        "alignment_seed",
        start_id=PHASE_SPACE_SAMPLES,
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    centers = _select_refinement_centers(phase_space_rows + seed_rows)
    refined = _refinement_samples(rng, centers, REFINEMENT_SAMPLES)
    refinement_rows = _evaluate_samples(
        refined,
        "refined",
        start_id=PHASE_SPACE_SAMPLES + len(seed_points),
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    return (
        phase_space_rows + seed_rows + refinement_rows,
        len(phase_space_rows),
        len(seed_rows),
        len(refinement_rows),
    )


def _write_alignment_style_csv(path, rows):
    """Write the same concurrence columns and ordering as AlignmentScan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_concurrence_csv_headers(LEPTON_NAME))
        for row in rows:
            writer.writerow(_concurrence_csv_row(row))


def write_outputs(rows):
    """Write full, aligned, and per-observable ranked AlignmentScan-style CSVs."""
    _write_alignment_style_csv(FULL_CSV, rows)
    _write_alignment_style_csv(ALIGNED_CSV, [row for row in rows if row["aligned"]])
    save_concurrence_top_csv(
        rows,
        TOP_CSV,
        top_n=TOP_POINTS_PER_POLARIZATION,
        observables=SCAN_OBSERVABLE_NAMES,
        lepton_name=LEPTON_NAME,
    )
    return {
        "all_csv": FULL_CSV,
        "aligned_csv": ALIGNED_CSV,
        "top_csv": TOP_CSV,
    }


def _write_polarization_plot(rows, prefix, spin_label, lepton_name, plot_dir):
    """Write one rasterized multi-page PDF for a polarization case."""
    plt, PdfPages = require_matplotlib()
    plot_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        ("theta_in", "qOut", r"$\theta_{in}$", r"$E_\gamma$ [GeV]"),
        ("sqrt_s", "qOut", r"$\sqrt{s}$ [GeV]", r"$E_\gamma$ [GeV]"),
        ("phi_in", "phiOut", r"$\phi_{P,in}$", r"$\phi_\gamma$"),
    )
    coordinates = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in {item for panel in panels for item in panel[:2]}
    }
    output_prefix = explicit_polarization_name(prefix, lepton_name)
    path = plot_dir / f"phase_space_scan_{output_prefix}.pdf"
    with PdfPages(path) as pdf:
        for observable in SCAN_OBSERVABLE_NAMES:
            key = f"{prefix}_{observable}"
            values = np.asarray(
                [float(row.get(key, np.nan)) for row in rows],
                dtype=float,
            )
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            finite_values = values[finite]
            signed = observable in SIGNED_CONCURRENCE_OBSERVABLES
            if observable == "D_W":
                vmin, vmax, cmap = 0.0, 2.0 / np.sqrt(3.0), "viridis_r"
            elif observable == "M2_magic":
                vmin, vmax, cmap = (
                    -3.0 * np.log(2.0),
                    3.0 * np.log(2.0),
                    "coolwarm",
                )
            else:
                vmin, vmax = (-1.0, 1.0) if signed else (0.0, 1.0)
                cmap = "coolwarm" if signed else "viridis"
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            image = None
            for ax, (x_name, y_name, x_label, y_label) in zip(
                axes.ravel()[:3], panels
            ):
                image = ax.scatter(
                    coordinates[x_name][finite],
                    coordinates[y_name][finite],
                    c=finite_values,
                    s=6,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            observable_label = observable_latex_label(observable, lepton_name)
            axes[1, 1].hist(finite_values, bins=60, color="tab:blue", alpha=0.8)
            axes[1, 1].set_xlabel(observable_label)
            axes[1, 1].set_ylabel("samples")
            fig.colorbar(
                image, ax=axes.ravel()[:3].tolist(), label=observable_label
            )
            finite_indices = np.flatnonzero(finite)
            objective_values = np.asarray([
                observable_rank_value(value, observable)
                for value in finite_values
            ])
            local_best = (
                np.argmin(objective_values)
                if observable_is_minimized(observable)
                else np.argmax(objective_values)
            )
            best_index = int(finite_indices[local_best])
            best = rows[int(best_index)]
            direction = observable_optimum_label(observable)
            fig.suptitle(
                f"Phase-space scan: {species_spin_label(spin_label, lepton_name)} "
                f"[{output_prefix}], {direction} {observable_label}={float(best[key]):.5g}"
            )
            pdf.savefig(fig)
            plt.close(fig)
    return prefix, path


_PLOT_WORKER_ROWS = None
_PLOT_WORKER_LEPTON = None
_PLOT_WORKER_DIR = None


def _initialize_plot_worker(rows, lepton_name, plot_dir):
    """Load one species scan into a plotting worker once."""
    global _PLOT_WORKER_ROWS, _PLOT_WORKER_LEPTON, _PLOT_WORKER_DIR
    _PLOT_WORKER_ROWS = rows
    _PLOT_WORKER_LEPTON = lepton_name
    _PLOT_WORKER_DIR = plot_dir


def _write_polarization_plot_task(case):
    """Process-pool adapter for one independent polarization PDF."""
    prefix, spin_label = case
    return _write_polarization_plot(
        _PLOT_WORKER_ROWS,
        prefix,
        spin_label,
        _PLOT_WORKER_LEPTON,
        _PLOT_WORKER_DIR,
    )


def write_plot(rows, max_workers=PHASE_SPACE_PLOT_WORKERS):
    """Write polarization PDFs concurrently in independent processes."""
    cases = [
        (prefix, spin_label)
        for prefix, spin_label, _spin_case in ALIGNMENT_SPIN_CASES
    ]
    if not max_workers or max_workers <= 1 or len(cases) == 1:
        return dict(
            _write_polarization_plot(
                rows,
                prefix,
                spin_label,
                LEPTON_NAME,
                PLOT_DIR,
            )
            for prefix, spin_label in cases
        )

    worker_count = min(int(max_workers), len(cases))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_plot_worker,
        initargs=(rows, LEPTON_NAME, PLOT_DIR),
    ) as executor:
        return dict(
            executor.map(_write_polarization_plot_task, cases, chunksize=1)
        )


def build_report(
    rows,
    phase_space_valid,
    seed_valid,
    refinement_valid,
):
    lines = [
        f"AlignmentScan-style entanglement phase-space scan ({LEPTON_NAME})",
        f"  random seed: {RANDOM_SEED}",
        f"  parallel kinematic workers: {PHASE_SPACE_SCAN_WORKERS}",
        f"  parallel PDF workers: {PHASE_SPACE_PLOT_WORKERS}",
        f"  {LEPTON_NAME} mass: {LEPTON_MASS_GEV:.10g} GeV",
        f"  threshold: sqrt(s)={COM_THRESHOLD:.9g} GeV",
        f"  ranges: sqrt(s)={SQRT_S_RANGE} GeV, s={S_RANGE}, "
        f"theta_in={THETA_IN_RANGE}",
        f"  qOut fraction of kinematic maximum: {QOUT_FRACTION_RANGE}",
        f"  phase-space valid samples: {phase_space_valid}/{PHASE_SPACE_SAMPLES}",
        "  observables: " + ", ".join(
            observable_text_label(name, LEPTON_NAME)
            for name in SCAN_OBSERVABLE_NAMES
        ),
        f"  polarization cases: {len(ALIGNMENT_SPIN_CASES)}",
        f"  AlignmentScan seed samples: {seed_valid}/{ALIGNMENT_SEED_CENTERS}",
        f"  refinement valid samples: {refinement_valid}/{REFINEMENT_SAMPLES}",
        f"  total valid samples: {len(rows)}",
        "  best points by observable and polarization:",
    ]
    for observable in SCAN_OBSERVABLE_NAMES:
        lines.append(f"  {observable_text_label(observable, LEPTON_NAME)}:")
        for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
            key = f"{prefix}_{observable}"
            finite_rows = [row for row in rows if np.isfinite(row.get(key, np.nan))]
            if not finite_rows:
                continue
            selector = min if observable_is_minimized(observable) else max
            best = selector(
                finite_rows,
                key=lambda row: observable_rank_value(row[key], observable),
            )
            output_prefix = explicit_polarization_name(prefix, LEPTON_NAME)
            lines.append(
                f"    {species_spin_label(label, LEPTON_NAME)} [{output_prefix}]: "
                f"{float(best[key]):.8g}, sqrt(s)={best['sqrt_s']:.7g}, "
                f"theta={best['theta_in']:.7g}, qOut={best['qOut']:.7g}, "
                f"phi_lepton={best['phi_in_lepton']:.7g}, "
                f"phi_gamma={best['phiOut']:.7g}"
            )
    lines.extend((
        f"  full csv: {FULL_CSV}",
        f"  aligned csv: {ALIGNED_CSV}",
        f"  ranked csv: {TOP_CSV}",
        f"  per-polarization plots: {PLOT_DIR}",
    ))
    return "\n".join(lines) + "\n"


def _run_species(lepton):
    _configure_lepton(lepton)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    rows, phase_space_valid, seed_valid, refinement_valid = run_phase_space_scan()
    write_outputs(rows)
    write_plot(rows, max_workers=PHASE_SPACE_PLOT_WORKERS)
    report = build_report(
        rows,
        phase_space_valid,
        seed_valid,
        refinement_valid,
    )
    return report


def main():
    """Run each requested lepton species into an independent output tree."""
    unknown = set(LEPTONS_TO_SCAN) - set(LEPTON_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown lepton species: {sorted(unknown)}")
    if not LEPTONS_TO_SCAN:
        raise ValueError("LEPTONS_TO_SCAN must contain at least one species")
    if PHASE_SPACE_SCAN_WORKERS < 1:
        raise ValueError("PHASE_SPACE_SCAN_WORKERS must be positive")
    if PHASE_SPACE_PLOT_WORKERS < 1:
        raise ValueError("PHASE_SPACE_PLOT_WORKERS must be positive")
    reports = [_run_species(lepton) for lepton in LEPTONS_TO_SCAN]
    log_text = "\n\n".join(report.rstrip() for report in reports) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print_console_text(log_text)


if __name__ == "__main__":
    main()
