"""Adaptive kinematic and coherent-polarization phase-space scan.

The scan covers ``sqrt(s)``, ``theta_in``, ``E_gamma``, the incoming-lepton
azimuth, the outgoing-photon azimuth, and the coherent preparation angles
``theta_e`` and ``theta_p``. It preserves the fixed polarization cases and
entanglement observables from :mod:`AlignmentScan`, then refines around the
best kinematic points. An exact deterministic seed covers the quoted electron
reference configuration.
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
    final_lepton_photon_spin_correlations,
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
    PHASE_SPACE_SAMPLES,
    PROTON_MASS_GEV,
    REFINEMENT_SAMPLES,
    SCAN_INITIAL_MIXING_ANGLES,
    SCAN_WORKERS,
)
from PlotUtils import print_console_text, require_matplotlib
from SpinDensityMat import (
    ghz_observables_from_density_matrix,
    mixed_angle_spin_density_observables,
    outgoing_spin_states,
    w_observables_from_density_matrix,
)


# Explicit script settings. Edit these values before running PhaseSpaceScan.py.
LEPTONS_TO_SCAN = ("electron", "muon", "heavy", "massless")
PHASE_SPACE_SCAN_WORKERS = SCAN_WORKERS
PHASE_SPACE_PLOT_WORKERS = max(1, min(SCAN_WORKERS, 24))
RANDOM_SEED = 271828
REFINEMENT_CENTERS = len(ALIGNMENT_SPIN_CASES) * len(SCAN_OBSERVABLE_NAMES)
ALIGNMENT_SEED_CENTERS = REFINEMENT_CENTERS
TOP_POINTS_PER_POLARIZATION = 100
THETA_IN_RANGE = (0.0, 2.80)
QOUT_FRACTION_RANGE = (0.05, 0.95)
AZIMUTH_RANGE = (0.0, 2.0 * np.pi)
THETA_E_MIX_RANGE = (0.0, np.pi)
THETA_P_MIX_RANGE = (0.0, np.pi)
REFERENCE_PIN_GEV = 0.130
REFERENCE_POUT_GEV = 0.028
REFERENCE_THETA_P_RAD = 3.429
REFERENCE_THETA_GAMMA_RAD = 1.298
REFERENCE_QOUT_GEV = 0.07443763686223767
REFERENCE_THETA_E_MIX_RAD = 5.503 % np.pi
REFERENCE_THETA_P_MIX_RAD = 3.056 % np.pi
# The user-frame photon azimuth preserves the reference proton-photon opening
# angle after rotating the final proton onto the frame's +y axis.
REFERENCE_PHIOUT_RAD = float(
    np.arcsin(np.cos(REFERENCE_THETA_P_RAD - REFERENCE_THETA_GAMMA_RAD))
    % (2.0 * np.pi)
)
REFERENCE_THETA_IN_RAD = 0.5 * np.pi
REFERENCE_PHI_IN_LEPTON_RAD = float(
    (
        REFERENCE_PHIOUT_RAD
        - REFERENCE_THETA_GAMMA_RAD
        + np.pi
    )
    % (2.0 * np.pi)
)
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
MIXING_CSV = OUTPUT_DIR / "electron_photon_mixing_angle_phase_space.csv"
MIXING_TOP_CSV = OUTPUT_DIR / "electron_photon_mixing_angle_top.csv"
MIXING_PLOT = (
    OUTPUT_DIR
    / "phase_space_scan_lepton_electron_theta_mix_proton_theta_p_mix.pdf"
)
PLOT_DIR = OUTPUT_DIR


def _qout_max(s):
    """Return the physical three-body photon-energy ceiling."""
    sqrt_s = np.sqrt(s)
    final_mass = PROTON_MASS_GEV + LEPTON_MASS_GEV
    return (s - final_mass**2) / (2.0 * sqrt_s)


QOUT_RANGE = (0.0, float(_qout_max(S_RANGE[1])))


def _configure_lepton(name):
    """Configure masses, threshold, and independent output paths."""
    global LEPTON_NAME, LEPTON_MASS_GEV, COM_THRESHOLD
    global SQRT_S_RANGE, S_RANGE, QOUT_RANGE
    global OUTPUT_DIR, FULL_CSV, ALIGNED_CSV, TOP_CSV, PLOT_DIR
    global MIXING_CSV, MIXING_TOP_CSV, MIXING_PLOT

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
    MIXING_CSV = OUTPUT_DIR / f"{stem}_mixing_angle_phase_space.csv"
    MIXING_TOP_CSV = OUTPUT_DIR / f"{stem}_mixing_angle_top.csv"
    MIXING_PLOT = (
        OUTPUT_DIR
        / f"phase_space_scan_lepton_{name}_theta_mix_proton_theta_p_mix.pdf"
    )
    PLOT_DIR = OUTPUT_DIR


def _uniform_samples(rng, count):
    """Return a randomized stratified design covering every scan coordinate."""
    dimensions = 7 if SCAN_INITIAL_MIXING_ANGLES else 5
    unit_samples = np.column_stack([
        (rng.permutation(count) + rng.random(count)) / count
        for _dimension in range(dimensions)
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
    columns = [
        s_values,
        THETA_IN_RANGE[0]
        + unit_samples[:, 1] * (THETA_IN_RANGE[1] - THETA_IN_RANGE[0]),
        qout_values,
        AZIMUTH_RANGE[0]
        + unit_samples[:, 3] * (AZIMUTH_RANGE[1] - AZIMUTH_RANGE[0]),
        AZIMUTH_RANGE[0]
        + unit_samples[:, 4] * (AZIMUTH_RANGE[1] - AZIMUTH_RANGE[0]),
    ]
    if SCAN_INITIAL_MIXING_ANGLES:
        columns.extend((
            THETA_E_MIX_RANGE[0]
            + unit_samples[:, 5]
            * (THETA_E_MIX_RANGE[1] - THETA_E_MIX_RANGE[0]),
            THETA_P_MIX_RANGE[0]
            + unit_samples[:, 6]
            * (THETA_P_MIX_RANGE[1] - THETA_P_MIX_RANGE[0]),
        ))
    return np.column_stack(columns)


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


def _select_mixing_refinement_centers(rows, count=REFINEMENT_CENTERS):
    """Select kinematic refinement centers from coherent-angle observables."""
    centers = []
    prefix = _mixing_prefix()
    rankings = []
    for observable in SCAN_OBSERVABLE_NAMES:
        key = f"{prefix}_{observable}"
        finite = [row for row in rows if np.isfinite(float(row.get(key, np.nan)))]
        rankings.append(sorted(
            finite,
            key=lambda row: observable_rank_value(row[key], observable),
            reverse=not observable_is_minimized(observable),
        ))
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


def _reference_seed_points():
    """Return the exact electron benchmark in the seven scan coordinates."""
    if LEPTON_NAME != "electron":
        return np.empty((0, 7))
    proton_energy = np.hypot(PROTON_MASS_GEV, REFERENCE_PIN_GEV)
    lepton_energy = np.hypot(LEPTON_MASS_GEV, REFERENCE_PIN_GEV)
    return np.asarray([[
        (proton_energy + lepton_energy) ** 2,
        REFERENCE_THETA_IN_RAD,
        REFERENCE_QOUT_GEV,
        REFERENCE_PHI_IN_LEPTON_RAD,
        REFERENCE_PHIOUT_RAD,
        REFERENCE_THETA_E_MIX_RAD,
        REFERENCE_THETA_P_MIX_RAD,
    ]])


def _refinement_samples(rng, centers, count):
    if not len(centers):
        return np.empty((0, 7 if SCAN_INITIAL_MIXING_ANGLES else 5))
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
        if SCAN_INITIAL_MIXING_ANGLES:
            point = np.concatenate((
                point,
                [
                    rng.uniform(*THETA_E_MIX_RANGE),
                    rng.uniform(*THETA_P_MIX_RANGE),
                ],
            ))
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
    point = np.asarray(point, dtype=float)
    if point.size == 5:
        s, theta_in, qout, phi_e, phi_gamma = map(float, point)
        theta_e_mix = REFERENCE_THETA_E_MIX_RAD
        theta_p_mix = REFERENCE_THETA_P_MIX_RAD
    elif point.size == 7:
        (
            s, theta_in, qout, phi_e, phi_gamma,
            theta_e_mix, theta_p_mix,
        ) = map(float, point)
    else:
        raise ValueError("Phase-space samples must contain five or seven coordinates.")
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
        "return_amplitudes": SCAN_INITIAL_MIXING_ANGLES,
        "skip_fixed_polarizations": SCAN_INITIAL_MIXING_ANGLES,
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
    if not SCAN_INITIAL_MIXING_ANGLES:
        return row, None
    amplitudes = result["amplitudes"]
    try:
        spin_data = mixed_angle_spin_density_observables(
            amplitudes,
            theta_e_mix,
            theta_p_mix,
            normalize_trace=NORMALIZE_TRACE,
        )
    except (
        ValueError,
        ZeroDivisionError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ):
        return None
    prefix = f"lepton_{lepton_name}_theta_mix_proton_theta_p_mix"
    mixing_row = {
        key: row[key]
        for key in (
            "lepton", "kinematic_point", "s_regime", "theta_in_regime",
            "qOut_regime", "lepton_mass", "s", "sqrt_s", "pIn", "pOut",
            "qOut", "theta_in", "phi_in", "phi_in_lepton", "phiOut",
            "Q2", "xB", "t", "F1", "F2", "W2", "y",
            "theta_lepton_gamma_rad", "theta_lepton_gamma_deg",
            "k_dot_qout", "kp_dot_qout", "abs_k_dot_qout",
            "abs_kp_dot_qout", "aligned",
        )
    }
    mixing_row.update({
        "search_stage": stage,
        "sample_id": sample_id,
        "theta_e": theta_e_mix,
        "theta_p": theta_p_mix,
        f"{prefix}_trace": spin_data["trace"],
        f"{prefix}_spin_signal_M2": spin_data["spin_signal"],
        f"{prefix}_cross_section_ratio": spin_data["cross_section_ratio"],
        f"{prefix}_purity": spin_data["purity"],
        f"{prefix}_M2_magic": spin_data["M2_magic"],
    })
    corr = final_lepton_photon_spin_correlations(
        spin_data["rho"],
        outgoing_spin_states(),
    )
    for name, value in corr.items():
        mixing_row[f"{prefix}_{name}"] = value
    for name, value in spin_data["entanglement"].items():
        mixing_row[f"{prefix}_{name}"] = value
    ghz = ghz_observables_from_density_matrix(spin_data["rho"])
    for name, value in ghz.items():
        mixing_row[f"{prefix}_{name}"] = value
    mixing_row[f"{prefix}_GHZ_purity"] = ghz["GHZ_plus_fidelity"]
    w_state = w_observables_from_density_matrix(spin_data["rho"])
    for name, value in w_state.items():
        mixing_row[f"{prefix}_{name}"] = value
    mixing_row[f"{prefix}_W_purity"] = w_state["W_fidelity"]
    return row, mixing_row


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
    valid = [result for result in results if result is not None]
    return (
        [result[0] for result in valid],
        [result[1] for result in valid if result[1] is not None],
    )


def run_phase_space_scan():
    rng = np.random.default_rng(RANDOM_SEED)
    phase_space_rows, phase_space_mixing_rows = _evaluate_samples(
        _uniform_samples(rng, PHASE_SPACE_SAMPLES),
        "phase_space",
        start_id=0,
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    seed_points = (
        np.empty((0, 5))
        if SCAN_INITIAL_MIXING_ANGLES
        else _alignment_seed_points()
    )
    seed_rows, seed_mixing_rows = _evaluate_samples(
        seed_points,
        "alignment_seed",
        start_id=PHASE_SPACE_SAMPLES,
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    centers = (
        _select_mixing_refinement_centers(
            phase_space_mixing_rows + seed_mixing_rows
        )
        if SCAN_INITIAL_MIXING_ANGLES
        else _select_refinement_centers(phase_space_rows + seed_rows)
    )
    refined = _refinement_samples(rng, centers, REFINEMENT_SAMPLES)
    refinement_rows, refinement_mixing_rows = _evaluate_samples(
        refined,
        "refined",
        start_id=PHASE_SPACE_SAMPLES + len(seed_points),
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    reference_points = _reference_seed_points()
    reference_rows, reference_mixing_rows = _evaluate_samples(
        reference_points,
        "reference_seed",
        start_id=PHASE_SPACE_SAMPLES + len(seed_points) + len(refined),
        max_workers=PHASE_SPACE_SCAN_WORKERS,
    )
    return (
        phase_space_rows + seed_rows + refinement_rows + reference_rows,
        (
            phase_space_mixing_rows
            + seed_mixing_rows
            + refinement_mixing_rows
            + reference_mixing_rows
        ),
        len(phase_space_rows),
        len(seed_rows),
        len(refinement_rows),
        len(reference_rows),
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


def _mixing_prefix(lepton_name=None):
    species = LEPTON_NAME if lepton_name is None else lepton_name
    return f"lepton_{species}_theta_mix_proton_theta_p_mix"


def write_mixing_outputs(rows):
    """Write full and ranked coherent-polarization phase-space rows."""
    if not rows:
        raise ValueError("The coherent mixing-angle scan produced no rows.")
    MIXING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MIXING_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    prefix = _mixing_prefix()
    ranked = []
    for observable in SCAN_OBSERVABLE_NAMES:
        key = f"{prefix}_{observable}"
        finite = [row for row in rows if np.isfinite(float(row.get(key, np.nan)))]
        finite.sort(
            key=lambda row: observable_rank_value(row[key], observable),
            reverse=not observable_is_minimized(observable),
        )
        for rank, row in enumerate(
            finite[:TOP_POINTS_PER_POLARIZATION],
            start=1,
        ):
            ranked.append({
                "observable": observable,
                "rank": rank,
                "value": row[key],
                "theta_e": row["theta_e"],
                "theta_p": row["theta_p"],
                "search_stage": row["search_stage"],
                "sample_id": row["sample_id"],
                "sqrt_s": row["sqrt_s"],
                "pIn": row["pIn"],
                "pOut": row["pOut"],
                "qOut": row["qOut"],
                "theta_in": row["theta_in"],
                "phi_in_lepton": row["phi_in_lepton"],
                "phiOut": row["phiOut"],
            })
    with MIXING_TOP_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(ranked[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(ranked)
    return {"mixing_csv": MIXING_CSV, "mixing_top_csv": MIXING_TOP_CSV}


def write_mixing_plot(rows):
    """Extend the original phase-space plot format with two angle dimensions."""
    plt, PdfPages = require_matplotlib()
    prefix = _mixing_prefix()
    panels = (
        ("theta_in", "qOut", r"$\theta_{in}$", r"$E_\gamma$ [GeV]"),
        ("sqrt_s", "qOut", r"$\sqrt{s}$ [GeV]", r"$E_\gamma$ [GeV]"),
        ("phi_in", "phiOut", r"$\phi_{P,in}$", r"$\phi_\gamma$"),
        ("theta_e", "theta_p", r"$\theta_e$", r"$\theta_p$"),
        ("sqrt_s", "theta_e", r"$\sqrt{s}$ [GeV]", r"$\theta_e$"),
        ("sqrt_s", "theta_p", r"$\sqrt{s}$ [GeV]", r"$\theta_p$"),
        ("qOut", "theta_e", r"$E_\gamma$ [GeV]", r"$\theta_e$"),
        ("qOut", "theta_p", r"$E_\gamma$ [GeV]", r"$\theta_p$"),
    )
    coordinates = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in {item for panel in panels for item in panel[:2]}
    }
    with PdfPages(MIXING_PLOT) as pdf:
        for observable in SCAN_OBSERVABLE_NAMES:
            key = f"{prefix}_{observable}"
            values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            if observable == "D_W":
                vmin, vmax, cmap = 0.0, 2.0 / np.sqrt(3.0), "viridis_r"
            elif observable == "M2_magic":
                vmin, vmax, cmap = (
                    -3.0 * np.log(2.0),
                    3.0 * np.log(2.0),
                    "coolwarm",
                )
            elif observable in SIGNED_CONCURRENCE_OBSERVABLES:
                vmin, vmax, cmap = -1.0, 1.0, "coolwarm"
            else:
                vmin, vmax, cmap = 0.0, 1.0, "viridis"
            finite_values = values[finite]
            fig, axes = plt.subplots(
                3,
                3,
                figsize=(14.0, 11.5),
                constrained_layout=True,
            )
            image = None
            for ax, (x_name, y_name, x_label, y_label) in zip(
                axes.ravel()[:8],
                panels,
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
            axes[2, 2].hist(
                finite_values,
                bins=60,
                color="tab:blue",
                alpha=0.8,
            )
            observable_label = observable_text_label(observable, LEPTON_NAME)
            axes[2, 2].set_xlabel(observable_label)
            axes[2, 2].set_ylabel("samples")
            objective_values = np.asarray([
                observable_rank_value(value, observable)
                for value in finite_values
            ])
            local_best = (
                np.argmin(objective_values)
                if observable_is_minimized(observable)
                else np.argmax(objective_values)
            )
            best_value = finite_values[int(local_best)]
            direction = observable_optimum_label(observable)
            fig.suptitle(
                f"Phase-space scan: {LEPTON_NAME} coherent "
                f"theta_e x theta_p [{prefix}], "
                f"{direction} {observable_label}={best_value:.5g}"
            )
            fig.colorbar(
                image,
                ax=axes.ravel()[:8].tolist(),
                label=observable_label,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return MIXING_PLOT


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
    mixing_rows,
    phase_space_valid,
    seed_valid,
    refinement_valid,
    reference_valid,
):
    mixing_range_text = (
        f", theta_e={THETA_E_MIX_RANGE}, theta_p={THETA_P_MIX_RANGE}"
        if SCAN_INITIAL_MIXING_ANGLES
        else ""
    )
    lines = [
        f"AlignmentScan-style entanglement phase-space scan ({LEPTON_NAME})",
        "  polarization mode: "
        + (
            "coherent theta_e x theta_p only"
            if SCAN_INITIAL_MIXING_ANGLES
            else "fixed polarization cases only"
        ),
        f"  random seed: {RANDOM_SEED}",
        f"  parallel kinematic workers: {PHASE_SPACE_SCAN_WORKERS}",
        f"  parallel PDF workers: {PHASE_SPACE_PLOT_WORKERS}",
        f"  {LEPTON_NAME} mass: {LEPTON_MASS_GEV:.10g} GeV",
        f"  threshold: sqrt(s)={COM_THRESHOLD:.9g} GeV",
        f"  ranges: sqrt(s)={SQRT_S_RANGE} GeV, s={S_RANGE}, "
        f"theta_in={THETA_IN_RANGE}{mixing_range_text}",
        f"  qOut fraction of kinematic maximum: {QOUT_FRACTION_RANGE}",
        f"  phase-space valid samples: {phase_space_valid}/{PHASE_SPACE_SAMPLES}",
        "  observables: " + ", ".join(
            observable_text_label(name, LEPTON_NAME)
            for name in SCAN_OBSERVABLE_NAMES
        ),
        f"  polarization cases: "
        f"{1 if SCAN_INITIAL_MIXING_ANGLES else len(ALIGNMENT_SPIN_CASES)}",
        f"  AlignmentScan seed samples: {seed_valid}/{ALIGNMENT_SEED_CENTERS}",
        f"  refinement valid samples: {refinement_valid}/{REFINEMENT_SAMPLES}",
        f"  reference valid samples: {reference_valid}/"
        f"{len(_reference_seed_points())}",
        f"  total valid samples: {len(rows)}",
        "  best points by observable and polarization:",
    ]
    if SCAN_INITIAL_MIXING_ANGLES:
        lines.insert(-1, f"  coherent mixing-angle samples: {len(mixing_rows)}")
    if SCAN_INITIAL_MIXING_ANGLES:
        prefix = _mixing_prefix()
        for observable in SCAN_OBSERVABLE_NAMES:
            key = f"{prefix}_{observable}"
            finite_rows = [
                row for row in mixing_rows
                if np.isfinite(row.get(key, np.nan))
            ]
            if not finite_rows:
                continue
            selector = min if observable_is_minimized(observable) else max
            best = selector(
                finite_rows,
                key=lambda row: observable_rank_value(row[key], observable),
            )
            lines.append(
                f"  {observable_text_label(observable, LEPTON_NAME)} "
                f"[{prefix}]: {float(best[key]):.8g}, "
                f"theta_e={best['theta_e']:.7g}, "
                f"theta_p={best['theta_p']:.7g}, "
                f"sqrt(s)={best['sqrt_s']:.7g}, qOut={best['qOut']:.7g}"
            )
    else:
        for observable in SCAN_OBSERVABLE_NAMES:
            lines.append(f"  {observable_text_label(observable, LEPTON_NAME)}:")
            for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
                key = f"{prefix}_{observable}"
                finite_rows = [
                    row for row in rows
                    if np.isfinite(row.get(key, np.nan))
                ]
                if not finite_rows:
                    continue
                selector = min if observable_is_minimized(observable) else max
                best = selector(
                    finite_rows,
                    key=lambda row: observable_rank_value(row[key], observable),
                )
                output_prefix = explicit_polarization_name(prefix, LEPTON_NAME)
                lines.append(
                    f"    {species_spin_label(label, LEPTON_NAME)} "
                    f"[{output_prefix}]: "
                    f"{float(best[key]):.8g}, sqrt(s)={best['sqrt_s']:.7g}, "
                    f"theta={best['theta_in']:.7g}, qOut={best['qOut']:.7g}, "
                    f"phi_lepton={best['phi_in_lepton']:.7g}, "
                    f"phi_gamma={best['phiOut']:.7g}"
                )
    if SCAN_INITIAL_MIXING_ANGLES:
        lines.extend((
            f"  mixing-angle csv: {MIXING_CSV}",
            f"  mixing-angle ranked csv: {MIXING_TOP_CSV}",
            f"  mixing-angle plot: {MIXING_PLOT}",
        ))
    else:
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
    (
        rows,
        mixing_rows,
        phase_space_valid,
        seed_valid,
        refinement_valid,
        reference_valid,
    ) = run_phase_space_scan()
    if SCAN_INITIAL_MIXING_ANGLES:
        write_mixing_outputs(mixing_rows)
        write_mixing_plot(mixing_rows)
    else:
        write_outputs(rows)
        write_plot(rows, max_workers=PHASE_SPACE_PLOT_WORKERS)
    report = build_report(
        rows,
        mixing_rows,
        phase_space_valid,
        seed_valid,
        refinement_valid,
        reference_valid,
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
    for name, bounds in (
        ("theta_e mixing", THETA_E_MIX_RANGE),
        ("theta_p mixing", THETA_P_MIX_RANGE),
    ):
        if not 0.0 <= bounds[0] < bounds[1] <= np.pi:
            raise ValueError(f"{name} range must lie inside [0, pi].")
    electron_settings = LEPTON_SETTINGS["electron"]
    reference_sqrt_s = (
        np.hypot(PROTON_MASS_GEV, REFERENCE_PIN_GEV)
        + np.hypot(ELECTRON_MASS_GEV, REFERENCE_PIN_GEV)
    )
    if not (
        electron_settings["sqrt_s_range"][0]
        <= reference_sqrt_s
        <= electron_settings["sqrt_s_range"][1]
    ):
        raise ValueError("The electron sqrt(s) range excludes the reference point.")
    if not THETA_IN_RANGE[0] <= REFERENCE_THETA_IN_RAD <= THETA_IN_RANGE[1]:
        raise ValueError("THETA_IN_RANGE excludes the reference point.")
    reference_s = reference_sqrt_s**2
    electron_final_mass = PROTON_MASS_GEV + ELECTRON_MASS_GEV
    reference_qout_max = (
        reference_s - electron_final_mass**2
    ) / (2.0 * reference_sqrt_s)
    reference_qout_fraction = REFERENCE_QOUT_GEV / reference_qout_max
    if not (
        QOUT_FRACTION_RANGE[0]
        <= reference_qout_fraction
        <= QOUT_FRACTION_RANGE[1]
    ):
        raise ValueError("QOUT_FRACTION_RANGE excludes the reference point.")
    reports = [_run_species(lepton) for lepton in LEPTONS_TO_SCAN]
    log_text = "\n\n".join(report.rstrip() for report in reports) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print_console_text(log_text)


if __name__ == "__main__":
    main()
