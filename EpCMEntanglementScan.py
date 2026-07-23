"""Focused Bethe--Heitler entanglement scan with a slow recoil proton.

The construction starts in the exact electron--proton CM frame with the proton
moving along +z and the electron along -z.  A recoil proton carrying ``(1-z)``
of the incoming proton three-momentum fixes the exchanged photon
``q = p - p'``. The default grid is centered on the quoted W-state reference
point. Its recoil direction uses an oriented planar angle, allowing values
above ``pi`` at fixed azimuth. The resulting ``q + electron`` subsystem is
scattered into an on-shell photon and electron at ``theta_cm``, then boosted
back to the ep CM frame.

All external masses and four-momentum conservation are retained exactly.
The legacy fixed-polarization scan is preserved, and a separate coherent
incoming-polarization scan varies ``theta_e`` and ``theta_p`` over one physical
period. Outputs are written below ``Output/EpCMEntanglementScan``.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import math
from pathlib import Path

import numpy as np

from Algebra import mdot
from AlignmentScan import (
    ALIGNMENT_ANGLE_MAX_RAD,
    COARSE_CONCURRENCE_NAMES,
    explicit_polarization_name,
    species_observable_name,
    spatial_opening_angle,
)
from FormFactors import yahl_dirac_pauli_from_t
from Kinematics import invariant_q2_xb_t
from PlotUtils import print_console_text, require_matplotlib
from SpinDensityMat import (
    SPIN_CASES,
    SPIN_CASE_MINUS_TX_PROTON,
    SPIN_CASE_MINUS_TY_PROTON,
    SPIN_CASE_L_MINUS_TX,
    SPIN_CASE_L_MINUS_TY,
    amplitude_table,
    ghz_observables_from_density_matrix,
    mixed_angle_final_state,
    mixed_angle_spin_density_observables,
    process_density_matrix_from_amplitudes,
    spin_case_display_label,
    spin_density_observables_from_amplitudes,
    w_observables_from_density_matrix,
)
from config import (
    ELECTRON_MASS_GEV,
    NORMALIZE_TRACE,
    PROTON_MASS_GEV,
    SCAN_INITIAL_MIXING_ANGLES,
    SCAN_WORKERS,
)


# Editable scan controls centered on the reference W-state point.
BEAM_MOMENTUM_GEV = 0.130
LEPTON_MASS_GEV = ELECTRON_MASS_GEV
LEPTON_NAME = "electron"
# Descending order keeps Z_VALUES strictly increasing.  The grid contains the
# reference |p'|=0.028 GeV exactly.
FINAL_PROTON_MOMENTUM_VALUES_GEV = np.linspace(0.036, 0.020, 9)
Z_VALUES = 1.0 - FINAL_PROTON_MOMENTUM_VALUES_GEV / BEAM_MOMENTUM_GEV
# theta_gamma=1.298 in the ep CM maps to this internal q+e CM angle.
REFERENCE_THETA_CM_RAD = 1.4276943335022996
THETA_CM_VALUES = np.unique(np.concatenate((
    np.linspace(1.30, 1.55, 11),
    [REFERENCE_THETA_CM_RAD],
)))
# Use the oriented x-z-plane angle so theta_p'=3.429, phi_p'=0 is represented
# directly even though it lies above the conventional polar range [0, pi].
REFERENCE_THETA_P_RAD = 3.429
THETA_P_VALUES = np.unique(np.concatenate((
    np.linspace(3.30, 3.56, 14),
    [REFERENCE_THETA_P_RAD],
)))
PHI_CM_RAD = 0.0
PHI_P_RAD = 0.0
# Explicit coherent incoming preparation.  In the user's |p e> ordering,
#
#   |p> = cos(theta_p) |+> + sin(theta_p) |->
#   |e> = cos(theta_e) |+> + sin(theta_e) |->.
#
# ``theta_p`` here is a polarization mixing angle and is distinct from the
# final-proton recoil polar angle ``THETA_P_VALUES``.
INITIAL_LEPTON_MIXING_ANGLE_RAD = 5.503
INITIAL_PROTON_MIXING_ANGLE_RAD = 3.056
# A polarization state is unchanged by theta -> theta + pi up to a global
# phase, so one period is sufficient.  The benchmark angles are inserted as
# exact grid anchors in addition to the uniform scan.
THETA_E_MIX_VALUES_RAD = np.unique(np.concatenate((
    np.linspace(0.0, np.pi, 5, endpoint=False),
    [INITIAL_LEPTON_MIXING_ANGLE_RAD % np.pi],
)))
THETA_P_MIX_VALUES_RAD = np.unique(np.concatenate((
    np.linspace(0.0, np.pi, 5, endpoint=False),
    [INITIAL_PROTON_MIXING_ANGLE_RAD % np.pi],
)))
SPIN_CASE_MIXING_ANGLES = "mixing_angles"
SCAN_WORKER_COUNT = SCAN_WORKERS
EP_CM_FIXED_SPIN_CASES = SPIN_CASES + (
    SPIN_CASE_MINUS_TX_PROTON,
    SPIN_CASE_MINUS_TY_PROTON,
    SPIN_CASE_L_MINUS_TX,
    SPIN_CASE_L_MINUS_TY,
)
EP_CM_SPIN_CASES = EP_CM_FIXED_SPIN_CASES + (
    SPIN_CASE_MIXING_ANGLES,
)
SCAN_PLOT_WORKER_COUNT = max(1, min(SCAN_WORKERS, len(EP_CM_SPIN_CASES)))
TOP_POINTS = 10

OUTPUT_DIR = Path("Output") / "EpCMEntanglementScan"
FULL_CSV = OUTPUT_DIR / "ep_cm_entanglement_scan.csv"
TOP_CSV = OUTPUT_DIR / "ep_cm_entanglement_top.csv"
MIXING_SCAN_CSV = OUTPUT_DIR / "ep_cm_mixing_angle_scan.csv"
MIXING_TOP_CSV = OUTPUT_DIR / "ep_cm_mixing_angle_top.csv"
PLOT_DIR = OUTPUT_DIR / "Plots"
LOG_PATH = OUTPUT_DIR / "EpCMEntanglementScan.log"

PLOT_OBSERVABLES = COARSE_CONCURRENCE_NAMES
RANKED_OBSERVABLES = COARSE_CONCURRENCE_NAMES
MIXING_SCAN_KINEMATIC_FIELDS = (
    "lepton", "kinematic_point", "lepton_mass",
    "z_index", "theta_index", "theta_p_index",
    "z", "theta_cm_rad", "theta_p_rad", "theta_p_deg", "phi_p_rad",
    "mu", "p_cm_GeV", "sqrt_s_GeV", "subsystem_mass_GeV", "t_GeV2",
    "final_proton_momentum_GeV", "final_proton_energy_GeV",
    "proton_energy_loss_GeV", "proton_energy_loss_fraction",
    "F1", "F2",
    "qout_E", "qout_px", "qout_py", "qout_pz",
    "kp_E", "kp_px", "kp_py", "kp_pz",
    "pp_E", "pp_px", "pp_py", "pp_pz",
    "conservation_error", "mass_shell_error",
)


def polarization_prefix(spin_case):
    """Return the exact species-aware polarization label."""
    if spin_case == SPIN_CASE_MIXING_ANGLES:
        return f"lepton_{LEPTON_NAME}_theta_mix_proton_theta_p_mix"
    return explicit_polarization_name(spin_case, LEPTON_NAME)


def ep_cm_spin_case_display_label(spin_case):
    """Return a display label, including the explicit angle preparation."""
    if spin_case == SPIN_CASE_MIXING_ANGLES:
        return "coherent theta_e x theta_p mixing-angle scan"
    return spin_case_display_label(spin_case)


def observable_column(spin_case, observable):
    """Return an AlignmentScan-compatible observable column name."""
    return (
        f"{polarization_prefix(spin_case)}_"
        f"{species_observable_name(observable, LEPTON_NAME)}"
    )


def real_mdot(first, second):
    """Return a real Minkowski product after rejecting numerical residue."""
    value = np.real_if_close(mdot(first, second), tol=1000)
    if np.iscomplexobj(value):
        raise ValueError(f"Minkowski product is unexpectedly complex: {value}")
    return float(value)


def boost_from_rest(four_vector, beta):
    """Actively boost a four-vector from a rest frame to velocity ``beta``."""
    vector = np.asarray(four_vector, dtype=float)
    beta = np.asarray(beta, dtype=float)
    beta2 = float(np.dot(beta, beta))
    if beta2 >= 1.0:
        raise ValueError("Boost velocity must have magnitude below one.")
    if beta2 == 0.0:
        return vector.copy()
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    beta_dot_p = float(np.dot(beta, vector[1:]))
    energy = gamma * (vector[0] + beta_dot_p)
    spatial = (
        vector[1:]
        + ((gamma - 1.0) * beta_dot_p / beta2 + gamma * vector[0]) * beta
    )
    return np.concatenate(([energy], spatial))


def ep_cm_momenta(
    z,
    theta_cm,
    theta_p=0.0,
    phi_cm=PHI_CM_RAD,
    phi_p=PHI_P_RAD,
):
    """Return exact ep-CM momenta for one ``(z, theta_cm, theta_p)`` point.

    ``theta_p`` and ``phi_p`` specify the final proton in the ep-CM frame.
    ``theta_cm`` and ``phi_cm`` specify the final real photon relative to the
    incoming virtual-photon direction in the virtual-photon--lepton CM frame.
    """
    if not 0.0 < z < 1.0:
        raise ValueError("z must lie strictly between zero and one.")
    if not 0.0 <= theta_cm <= np.pi:
        raise ValueError("theta_cm must lie in [0, pi].")
    if not 0.0 <= theta_p < 2.0 * np.pi:
        raise ValueError("The oriented theta_p angle must lie in [0, 2 pi).")

    momentum = BEAM_MOMENTUM_GEV
    proton_energy = np.sqrt(momentum**2 + PROTON_MASS_GEV**2)
    lepton_energy = np.sqrt(momentum**2 + LEPTON_MASS_GEV**2)
    recoil_momentum = (1.0 - z) * momentum
    recoil_energy = np.sqrt(recoil_momentum**2 + PROTON_MASS_GEV**2)

    p = np.array((proton_energy, 0.0, 0.0, momentum))
    k = np.array((lepton_energy, 0.0, 0.0, -momentum))
    pp_direction = np.array((
        np.sin(theta_p) * np.cos(phi_p),
        np.sin(theta_p) * np.sin(phi_p),
        np.cos(theta_p),
    ))
    pp = np.concatenate(([recoil_energy], recoil_momentum * pp_direction))
    virtual_photon = p - pp
    subsystem = k + virtual_photon
    subsystem_mass2 = real_mdot(subsystem, subsystem)
    if subsystem_mass2 <= LEPTON_MASS_GEV**2:
        raise ValueError("The virtual-photon--lepton subsystem is below threshold.")
    subsystem_mass = np.sqrt(subsystem_mass2)
    cm_momentum = (
        subsystem_mass2 - LEPTON_MASS_GEV**2
    ) / (2.0 * subsystem_mass)

    beta = subsystem[1:] / subsystem[0]
    virtual_photon_cm = boost_from_rest(virtual_photon, -beta)
    photon_axis = virtual_photon_cm[1:]
    photon_axis /= np.linalg.norm(photon_axis)
    transverse_axis = np.array((1.0, 0.0, 0.0))
    transverse_axis -= np.dot(transverse_axis, photon_axis) * photon_axis
    if np.linalg.norm(transverse_axis) < 1.0e-12:
        transverse_axis = np.array((0.0, 1.0, 0.0))
        transverse_axis -= np.dot(transverse_axis, photon_axis) * photon_axis
    transverse_axis /= np.linalg.norm(transverse_axis)
    second_transverse_axis = np.cross(photon_axis, transverse_axis)
    direction = (
        np.cos(theta_cm) * photon_axis
        + np.sin(theta_cm) * (
            np.cos(phi_cm) * transverse_axis
            + np.sin(phi_cm) * second_transverse_axis
        )
    )
    qout_cm = np.concatenate(([cm_momentum], cm_momentum * direction))
    kp_cm = np.concatenate((
        [np.sqrt(cm_momentum**2 + LEPTON_MASS_GEV**2)],
        -cm_momentum * direction,
    ))
    qout = boost_from_rest(qout_cm, beta)
    kp = boost_from_rest(kp_cm, beta)
    momenta = {"k": k, "p": p, "kp": kp, "pp": pp, "qout": qout}

    residual = k + p - kp - pp - qout
    mass_shell_errors = {
        "k": abs(real_mdot(k, k) - LEPTON_MASS_GEV**2),
        "p": abs(real_mdot(p, p) - PROTON_MASS_GEV**2),
        "kp": abs(real_mdot(kp, kp) - LEPTON_MASS_GEV**2),
        "pp": abs(real_mdot(pp, pp) - PROTON_MASS_GEV**2),
        "qout": abs(real_mdot(qout, qout)),
    }
    if np.max(np.abs(residual)) > 1.0e-10 or max(mass_shell_errors.values()) > 1.0e-9:
        raise ValueError("Constructed momenta failed conservation or on-shell checks.")

    total = k + p
    t = real_mdot(virtual_photon, virtual_photon)
    return {
        "momenta": momenta,
        "virtual_photon": virtual_photon,
        "subsystem": subsystem,
        "sqrt_s": np.sqrt(real_mdot(total, total)),
        "t": t,
        "subsystem_mass": subsystem_mass,
        "cm_momentum": cm_momentum,
        "mu": cm_momentum / LEPTON_MASS_GEV,
        "recoil_momentum": recoil_momentum,
        "recoil_energy": recoil_energy,
        "proton_energy_loss": proton_energy - recoil_energy,
        "proton_energy_loss_fraction": (
            (proton_energy - recoil_energy) / proton_energy
        ),
        "theta_p": float(theta_p),
        "phi_p": float(phi_p),
        "conservation_error": float(np.max(np.abs(residual))),
        "mass_shell_error": float(max(mass_shell_errors.values())),
    }


def _evaluate_point_data(task):
    """Return common kinematics, amplitudes, and process matrix for one point."""
    z_index, theta_index, theta_p_index, z, theta_cm, theta_p = task
    kin = ep_cm_momenta(z, theta_cm, theta_p)
    if kin["t"] >= 0.0:
        raise ValueError(f"Expected spacelike proton transfer, obtained t={kin['t']}")
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    amplitudes = amplitude_table(
        kin["momenta"],
        PROTON_MASS_GEV,
        F1,
        F2,
        electron_mass=LEPTON_MASS_GEV,
    )
    process_rho = process_density_matrix_from_amplitudes(amplitudes)
    mom = kin["momenta"]
    derived = invariant_q2_xb_t(mom, PROTON_MASS_GEV)
    lepton_photon_angle = spatial_opening_angle(mom["kp"], mom["qout"])
    k_dot_qout = real_mdot(mom["k"], mom["qout"])
    kp_dot_qout = real_mdot(mom["kp"], mom["qout"])
    row = {
        "lepton": LEPTON_NAME,
        "kinematic_point": (
            f"ep_cm_z_{z:.8g}_theta_cm_{theta_cm:.8g}_theta_p_{theta_p:.8g}"
        ),
        "s_regime": "reference_low_energy_ep_cm",
        "theta_in_regime": "collinear",
        "qOut_regime": "high_energy_transfer_slow_recoil_proton",
        "lepton_mass": LEPTON_MASS_GEV,
        "z_index": z_index,
        "theta_index": theta_index,
        "theta_p_index": theta_p_index,
        "z": float(z),
        "theta_cm_rad": float(theta_cm),
        "theta_p_rad": float(theta_p),
        "theta_p_deg": float(np.degrees(theta_p)),
        "phi_p_rad": float(PHI_P_RAD),
        "mu": kin["mu"],
        "p_cm_GeV": kin["cm_momentum"],
        "s": kin["sqrt_s"] ** 2,
        "sqrt_s_GeV": kin["sqrt_s"],
        "sqrt_s": kin["sqrt_s"],
        "pIn": BEAM_MOMENTUM_GEV,
        "pOut": kin["recoil_momentum"],
        "final_proton_momentum_GeV": kin["recoil_momentum"],
        "final_proton_energy_GeV": kin["recoil_energy"],
        "proton_energy_loss_GeV": kin["proton_energy_loss"],
        "proton_energy_loss_fraction": kin["proton_energy_loss_fraction"],
        "qOut": mom["qout"][0],
        "theta_in": 0.0,
        "phi_in": 0.0,
        "phi_in_lepton": np.pi,
        "phiOut": float(np.arctan2(mom["qout"][2], mom["qout"][1]) % (2.0 * np.pi)),
        "Q2": derived["Q2"],
        "xB": derived["xB"],
        "t": kin["t"],
        "W2": derived["W2"],
        "y": derived["y"],
        "subsystem_mass_GeV": kin["subsystem_mass"],
        "t_GeV2": kin["t"],
        "F1": F1,
        "F2": F2,
        "qout_E": mom["qout"][0],
        "qout_px": mom["qout"][1],
        "qout_py": mom["qout"][2],
        "qout_pz": mom["qout"][3],
        "kp_E": mom["kp"][0],
        "kp_px": mom["kp"][1],
        "kp_py": mom["kp"][2],
        "kp_pz": mom["kp"][3],
        "pp_E": mom["pp"][0],
        "pp_px": mom["pp"][1],
        "pp_py": mom["pp"][2],
        "pp_pz": mom["pp"][3],
        "theta_lepton_gamma_rad": lepton_photon_angle,
        "theta_lepton_gamma_deg": float(np.degrees(lepton_photon_angle)),
        "k_dot_qout": k_dot_qout,
        "kp_dot_qout": kp_dot_qout,
        "abs_k_dot_qout": abs(k_dot_qout),
        "abs_kp_dot_qout": abs(kp_dot_qout),
        "aligned": lepton_photon_angle <= ALIGNMENT_ANGLE_MAX_RAD,
        "conservation_error": kin["conservation_error"],
        "mass_shell_error": kin["mass_shell_error"],
        "squared_amplitude_M2": np.nan,
        "initial_theta_rad": np.nan,
        "initial_theta_deg": np.nan,
        "initial_theta_p_rad": np.nan,
        "initial_theta_p_deg": np.nan,
    }
    return row, amplitudes, process_rho


def _store_spin_result(row, spin_case, result):
    """Store one polarization result using the shared CSV column convention."""
    prefix = polarization_prefix(spin_case)
    row[f"{prefix}_trace"] = result["trace"]
    row[f"{prefix}_spin_signal_M2"] = result["spin_signal"]
    row[f"{prefix}_cross_section_ratio"] = result["cross_section_ratio"]
    row[f"{prefix}_purity"] = result["purity"]
    row["squared_amplitude_M2"] = result["squared_amplitude"]
    for name, value in result["entanglement"].items():
        row[observable_column(spin_case, name)] = value
    ghz = ghz_observables_from_density_matrix(result["rho"])
    w_state = w_observables_from_density_matrix(result["rho"])
    row[observable_column(spin_case, "GHZ_purity")] = ghz["GHZ_plus_fidelity"]
    row[observable_column(spin_case, "W_purity")] = w_state["W_fidelity"]


def evaluate_point(task):
    """Evaluate every legacy fixed polarization at one kinematic point."""
    row, amplitudes, process_rho = _evaluate_point_data(task)
    return _evaluate_fixed_polarizations(row, amplitudes, process_rho)


def _evaluate_fixed_polarizations(row, amplitudes, process_rho):
    """Evaluate legacy cases from already-computed kinematic amplitudes."""
    row = dict(row)
    for spin_case in EP_CM_FIXED_SPIN_CASES:
        result = spin_density_observables_from_amplitudes(
            amplitudes,
            spin_case=spin_case,
            normalize_trace=NORMALIZE_TRACE,
            process_rho=process_rho,
        )
        _store_spin_result(row, spin_case, result)
    return row


def evaluate_mixing_angle_point(task):
    """Evaluate the theta_e x theta_p coherent-polarization grid at one point."""
    base_row, amplitudes, _process_rho = _evaluate_point_data(task)
    return _evaluate_mixing_polarizations(base_row, amplitudes)


def _evaluate_mixing_polarizations(base_row, amplitudes):
    """Evaluate the angle grid from already-computed kinematic amplitudes."""
    rows = []
    mixing_base = {
        field: base_row[field]
        for field in MIXING_SCAN_KINEMATIC_FIELDS
    }
    for theta_e_index, theta_e in enumerate(THETA_E_MIX_VALUES_RAD):
        for theta_p_mix_index, theta_p_mix in enumerate(THETA_P_MIX_VALUES_RAD):
            row = dict(mixing_base)
            row.update({
                "theta_e_mix_index": theta_e_index,
                "theta_p_mix_index": theta_p_mix_index,
                "initial_theta_rad": float(theta_e),
                "initial_theta_deg": float(np.degrees(theta_e)),
                "initial_theta_p_rad": float(theta_p_mix),
                "initial_theta_p_deg": float(np.degrees(theta_p_mix)),
            })
            result = mixed_angle_spin_density_observables(
                amplitudes,
                lepton_angle=theta_e,
                proton_angle=theta_p_mix,
                normalize_trace=NORMALIZE_TRACE,
            )
            _store_spin_result(row, SPIN_CASE_MIXING_ANGLES, result)
            rows.append(row)
    return rows


def evaluate_point_bundle(task):
    """Evaluate fixed and angle-scanned polarizations with one amplitude table."""
    base_row, amplitudes, process_rho = _evaluate_point_data(task)
    return (
        _evaluate_fixed_polarizations(base_row, amplitudes, process_rho),
        _evaluate_mixing_polarizations(base_row, amplitudes),
    )


def run_tasks(tasks):
    """Run tasks in processes with a thread fallback for restricted systems."""
    workers = min(max(1, int(SCAN_WORKER_COUNT)), len(tasks))
    if workers == 1:
        return [evaluate_point(task) for task in tasks]
    chunksize = max(1, math.ceil(len(tasks) / (4 * workers)))
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(evaluate_point, tasks, chunksize=chunksize))
    except (OSError, PermissionError):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(evaluate_point, tasks))


def run_mixing_angle_tasks(tasks):
    """Run the coherent angle grid and flatten its rows by kinematic point."""
    workers = min(max(1, int(SCAN_WORKER_COUNT)), len(tasks))
    if workers == 1:
        batches = [evaluate_mixing_angle_point(task) for task in tasks]
    else:
        chunksize = max(1, math.ceil(len(tasks) / (4 * workers)))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                batches = list(executor.map(
                    evaluate_mixing_angle_point,
                    tasks,
                    chunksize=chunksize,
                ))
        except (OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batches = list(executor.map(evaluate_mixing_angle_point, tasks))
    return [row for batch in batches for row in batch]


def run_scan_tasks(tasks):
    """Evaluate both scan products without recomputing helicity amplitudes."""
    workers = min(max(1, int(SCAN_WORKER_COUNT)), len(tasks))
    if workers == 1:
        bundles = [evaluate_point_bundle(task) for task in tasks]
    else:
        chunksize = max(1, math.ceil(len(tasks) / (4 * workers)))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                bundles = list(executor.map(
                    evaluate_point_bundle,
                    tasks,
                    chunksize=chunksize,
                ))
        except (OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                bundles = list(executor.map(evaluate_point_bundle, tasks))
    fixed_rows = [fixed for fixed, _mixing in bundles]
    mixing_rows = [
        row
        for _fixed, mixing_batch in bundles
        for row in mixing_batch
    ]
    return fixed_rows, mixing_rows


def write_csv(path, rows, fieldnames=None):
    """Write dictionaries with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_csv(
    rows,
    spin_cases=EP_CM_FIXED_SPIN_CASES,
    output_path=TOP_CSV,
):
    """Rank each entanglement observable for every incoming spin case."""
    ranked = []
    base_fields = (
        "z", "theta_cm_rad", "theta_p_rad", "theta_p_deg",
        "initial_theta_rad", "initial_theta_deg",
        "initial_theta_p_rad", "initial_theta_p_deg",
        "mu", "p_cm_GeV", "t_GeV2",
        "final_proton_momentum_GeV", "final_proton_energy_GeV",
        "proton_energy_loss_GeV", "proton_energy_loss_fraction",
        "qout_E", "qout_px", "qout_pz", "kp_E", "kp_px", "kp_pz",
        "pp_E", "pp_pz",
    )
    optional_fields = (
        "theta_e_mix_index", "theta_p_mix_index",
    )
    for spin_case in spin_cases:
        for observable in RANKED_OBSERVABLES:
            key = observable_column(spin_case, observable)
            reverse = observable != "D_W"
            ordered = sorted(rows, key=lambda row: row[key], reverse=reverse)
            for rank, row in enumerate(ordered[:TOP_POINTS], start=1):
                ranked.append({
                    "spin_case": spin_case,
                    "spin_label": ep_cm_spin_case_display_label(spin_case),
                    "observable": observable,
                    "rank": rank,
                    "value": row[key],
                    **{field: row[field] for field in base_fields},
                    **{
                        field: row[field]
                        for field in optional_fields
                        if field in row
                    },
                })
    write_csv(output_path, ranked)


def grid_from_rows(rows, key):
    """Return a rectangular grid for a quantity independent of ``theta_p``."""
    grid = np.full((len(Z_VALUES), len(THETA_CM_VALUES)), np.nan)
    for row in rows:
        grid[int(row["z_index"]), int(row["theta_index"])] = row[key]
    return grid


def reduced_observable_grid(rows, key, minimize=False):
    """Reduce the proton-angle axis into a ``(z, theta_cm)`` heatmap."""
    grid = np.full((len(Z_VALUES), len(THETA_CM_VALUES)), np.nan)
    for row in rows:
        index = (int(row["z_index"]), int(row["theta_index"]))
        value = abs(row[key])
        current = grid[index]
        if np.isnan(current) or (value < current if minimize else value > current):
            grid[index] = value
    return grid


def plot_output_path(spin_case, output_dir=PLOT_DIR):
    """Return the explicit per-polarization PDF path used by the plot pool."""
    return Path(output_dir) / f"ep_cm_scan_{polarization_prefix(spin_case)}.pdf"


def save_spin_plot(rows, spin_case, output_dir=PLOT_DIR):
    """Write all AlignmentScan entanglement heatmaps using absolute values."""
    plt, PdfPages = require_matplotlib()
    output_path = plot_output_path(spin_case, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recoil_values = grid_from_rows(rows, "final_proton_momentum_GeV")[:, 0]
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(4, 4, figsize=(15.5, 12.5), constrained_layout=True)
        for ax, observable in zip(axes.flat, PLOT_OBSERVABLES):
            key = observable_column(spin_case, observable)
            values = reduced_observable_grid(
                rows,
                key,
                minimize=observable == "D_W",
            )[::-1, :]
            if observable == "D_W":
                image = ax.pcolormesh(
                    THETA_CM_VALUES, recoil_values[::-1], values,
                    shading="auto", cmap="viridis_r",
                    vmin=0.0, vmax=2.0 / np.sqrt(3.0),
                )
            else:
                image = ax.pcolormesh(
                    THETA_CM_VALUES, recoil_values[::-1], values,
                    shading="auto", cmap="viridis", vmin=0.0, vmax=1.0,
                )
            ax.axhline(
                1.0,
                color="white",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
            )
            ax.set_xlabel(r"$\theta_{\gamma\ell}^{(q\ell\,\mathrm{CM})}$ [rad]")
            ax.set_ylabel(r"$|\mathbf{p}'_p|$ [GeV]")
            display_name = species_observable_name(observable, LEPTON_NAME)
            ax.set_title(f"|{display_name}|")
            fig.colorbar(image, ax=ax)
        for ax in axes.flat[len(PLOT_OBSERVABLES):]:
            ax.set_visible(False)
        fig.suptitle(
            f"ep-CM slow-recoil-proton scan: "
            f"{ep_cm_spin_case_display_label(spin_case)}\n"
            f"p={BEAM_MOMENTUM_GEV:g} GeV, m_lepton={LEPTON_MASS_GEV:g} GeV, "
            f"z={Z_VALUES[0]:.3f}--{Z_VALUES[-1]:.3f}; "
            f"optimized over recoil theta_p={THETA_P_VALUES[0]:.3f}--"
            f"{THETA_P_VALUES[-1]:.3f} rad"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return output_path


def mixing_angle_grid(rows, key, minimize=False):
    """Reduce all kinematic axes into a theta_p x theta_e polarization map."""
    grid = np.full(
        (len(THETA_P_MIX_VALUES_RAD), len(THETA_E_MIX_VALUES_RAD)),
        np.nan,
    )
    for row in rows:
        index = (
            int(row["theta_p_mix_index"]),
            int(row["theta_e_mix_index"]),
        )
        value = abs(row[key])
        current = grid[index]
        if np.isnan(current) or (value < current if minimize else value > current):
            grid[index] = value
    return grid


def save_mixing_angle_plot(rows, output_dir=PLOT_DIR):
    """Write the coherent-polarization scan reduced over all kinematic axes."""
    plt, PdfPages = require_matplotlib()
    output_path = Path(output_dir) / "ep_cm_scan_mixing_angles.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spin_case = SPIN_CASE_MIXING_ANGLES
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(4, 4, figsize=(15.5, 12.5), constrained_layout=True)
        for ax, observable in zip(axes.flat, PLOT_OBSERVABLES):
            values = mixing_angle_grid(
                rows,
                observable_column(spin_case, observable),
                minimize=observable == "D_W",
            )
            image = ax.pcolormesh(
                THETA_E_MIX_VALUES_RAD,
                THETA_P_MIX_VALUES_RAD,
                values,
                shading="auto",
                cmap="viridis_r" if observable == "D_W" else "viridis",
                vmin=0.0,
                vmax=2.0 / np.sqrt(3.0) if observable == "D_W" else 1.0,
            )
            ax.set_xlabel(r"$\theta_e$ [rad]")
            ax.set_ylabel(r"$\theta_p$ [rad]")
            ax.set_title(f"|{species_observable_name(observable, LEPTON_NAME)}|")
            fig.colorbar(image, ax=ax)
        for ax in axes.flat[len(PLOT_OBSERVABLES):]:
            ax.set_visible(False)
        fig.suptitle(
            "ep-CM coherent initial-polarization scan\n"
            r"$|e\rangle=\cos\theta_e|+\rangle+\sin\theta_e|-\rangle$, "
            r"$|p\rangle=\cos\theta_p|+\rangle+\sin\theta_p|-\rangle$; "
            "optimized over all kinematic axes"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return output_path


_PLOT_WORKER_ROWS = None
_PLOT_WORKER_OUTPUT_DIR = None


def _initialize_plot_worker(rows, output_dir):
    """Load the scan payload once in each independent plotting process."""
    global _PLOT_WORKER_ROWS, _PLOT_WORKER_OUTPUT_DIR
    _PLOT_WORKER_ROWS = rows
    _PLOT_WORKER_OUTPUT_DIR = output_dir


def _save_spin_plot_worker(spin_case):
    return spin_case, save_spin_plot(
        _PLOT_WORKER_ROWS, spin_case, _PLOT_WORKER_OUTPUT_DIR
    )


def save_plots(rows, max_workers=SCAN_PLOT_WORKER_COUNT):
    """Save independent polarization PDFs concurrently, as AlignmentScan does."""
    if not max_workers or max_workers <= 1 or len(EP_CM_FIXED_SPIN_CASES) == 1:
        return {
            spin_case: save_spin_plot(rows, spin_case, PLOT_DIR)
            for spin_case in EP_CM_FIXED_SPIN_CASES
        }
    worker_count = min(int(max_workers), len(EP_CM_FIXED_SPIN_CASES))
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_plot_worker,
            initargs=(rows, PLOT_DIR),
        ) as executor:
            return dict(
                executor.map(
                    _save_spin_plot_worker,
                    EP_CM_FIXED_SPIN_CASES,
                    chunksize=1,
                )
            )
    except (OSError, PermissionError):
        return {
            spin_case: save_spin_plot(rows, spin_case, PLOT_DIR)
            for spin_case in EP_CM_FIXED_SPIN_CASES
        }


def nearest_row(rows, z, theta_cm, theta_p=0.0):
    return min(
        rows,
        key=lambda row: (
            abs(row["z"] - z)
            + abs(row["theta_cm_rad"] - theta_cm)
            + abs(row["theta_p_rad"] - theta_p)
        ),
    )


def format_momentum(label, row, prefix):
    return (
        f"  {label} = ({row[f'{prefix}_E']:.8g}, {row[f'{prefix}_px']:.8g}, "
        f"{row[f'{prefix}_py']:.8g}, {row[f'{prefix}_pz']:.8g}) GeV"
    )


def build_report(rows, plot_paths, mixing_rows=None, mixing_plot_path=None):
    """Summarize slow-proton anchors and the strongest lepton-photon points."""
    kinematic_rows = (
        mixing_rows
        if SCAN_INITIAL_MIXING_ANGLES
        else rows
    )
    anchors = (
        nearest_row(
            kinematic_rows,
            Z_VALUES[0],
            REFERENCE_THETA_CM_RAD,
            REFERENCE_THETA_P_RAD,
        ),
        nearest_row(
            kinematic_rows,
            1.0 - 0.028 / BEAM_MOMENTUM_GEV,
            REFERENCE_THETA_CM_RAD,
            REFERENCE_THETA_P_RAD,
        ),
        nearest_row(
            kinematic_rows,
            Z_VALUES[-1],
            REFERENCE_THETA_CM_RAD,
            REFERENCE_THETA_P_RAD,
        ),
    )
    lines = [
        "Focused ep-CM slow-recoil-proton entanglement scan",
        "  polarization mode: "
        + (
            "coherent theta_e x theta_p only"
            if SCAN_INITIAL_MIXING_ANGLES
            else "fixed polarization cases only"
        ),
        f"  kinematic points: "
        f"{len(kinematic_rows) // (len(THETA_E_MIX_VALUES_RAD) * len(THETA_P_MIX_VALUES_RAD)) if SCAN_INITIAL_MIXING_ANGLES else len(kinematic_rows)} "
        f"({len(Z_VALUES)} z x {len(THETA_CM_VALUES)} theta_cm "
        f"x {len(THETA_P_VALUES)} theta_p)",
        f"  sqrt(s): {anchors[0]['sqrt_s_GeV']:.8g} GeV",
        f"  lepton mass: {LEPTON_MASS_GEV:.8g} GeV",
        f"  z range: {Z_VALUES[0]:.8g}--{Z_VALUES[-1]:.8g}",
        f"  theta_cm range: {THETA_CM_VALUES[0]:.8g}--"
        f"{THETA_CM_VALUES[-1]:.8g} rad",
        f"  theta_p range: {THETA_P_VALUES[0]:.8g}--"
        f"{THETA_P_VALUES[-1]:.8g} rad",
        f"  benchmark lepton mixing-angle anchor theta_e: "
        f"{INITIAL_LEPTON_MIXING_ANGLE_RAD:.8g} rad",
        f"  benchmark proton mixing-angle anchor theta_p: "
        f"{INITIAL_PROTON_MIXING_ANGLE_RAD:.8g} rad",
        f"  theta_e polarization scan points: {len(THETA_E_MIX_VALUES_RAD)}",
        f"  theta_p polarization scan points: {len(THETA_P_MIX_VALUES_RAD)}",
        "",
    ]
    for row in anchors:
        lines.extend([
            f"Anchor z={row['z']:.6g}, theta_cm={row['theta_cm_rad']:.6g} rad, "
            f"theta_p={row['theta_p_rad']:.6g} rad:",
            f"  p'_p={row['final_proton_momentum_GeV']:.8g} GeV, "
            f"E'_p={row['final_proton_energy_GeV']:.8g} GeV",
            f"  proton energy loss={row['proton_energy_loss_GeV']:.8g} GeV "
            f"({row['proton_energy_loss_fraction']:.6%})",
            f"  t={row['t_GeV2']:.8g} GeV^2, "
            f"p_cm={row['p_cm_GeV']:.8g} GeV, mu={row['mu']:.8g}",
            format_momentum("p'_gamma", row, "qout"),
            format_momentum("p'_lepton", row, "kp"),
            format_momentum("p'_proton", row, "pp"),
        ])
    lines.append("")
    if not SCAN_INITIAL_MIXING_ANGLES:
        for spin_case in EP_CM_FIXED_SPIN_CASES:
            key = observable_column(spin_case, "C_e_gamma")
            best = max(rows, key=lambda row: row[key])
            lines.append(
                f"  max C_lepton_gamma "
                f"({ep_cm_spin_case_display_label(spin_case)}): "
                f"{best[key]:.8g} at z={best['z']:.6g}, "
                f"theta_cm={best['theta_cm_rad']:.6g}, "
                f"theta_p={best['theta_p_rad']:.6g}, mu={best['mu']:.6g}"
            )
    if SCAN_INITIAL_MIXING_ANGLES and mixing_rows:
        key = observable_column(SPIN_CASE_MIXING_ANGLES, "C_e_gamma")
        best = max(mixing_rows, key=lambda row: row[key])
        lines.append(
            f"  max C_lepton_gamma "
            f"({ep_cm_spin_case_display_label(SPIN_CASE_MIXING_ANGLES)}): "
            f"{best[key]:.8g} at theta_e={best['initial_theta_rad']:.6g}, "
            f"theta_p_mix={best['initial_theta_p_rad']:.6g}, "
            f"z={best['z']:.6g}, theta_cm={best['theta_cm_rad']:.6g}, "
            f"theta_p_recoil={best['theta_p_rad']:.6g}"
        )
    lines.extend((
        "",
        f"  scan workers: {SCAN_WORKER_COUNT}",
        f"  plot workers: {SCAN_PLOT_WORKER_COUNT}",
        f"  plot directory: {PLOT_DIR}",
    ))
    if SCAN_INITIAL_MIXING_ANGLES:
        lines.extend((
            f"  mixing-angle CSV: {MIXING_SCAN_CSV}",
            f"  mixing-angle ranked CSV: {MIXING_TOP_CSV}",
        ))
    else:
        lines.extend((
            f"  full CSV: {FULL_CSV}",
            f"  ranked CSV: {TOP_CSV}",
        ))
    for spin_case, path in plot_paths.items():
        lines.append(f"    {spin_case}: {path}")
    if mixing_plot_path is not None:
        lines.append(f"    {SPIN_CASE_MIXING_ANGLES}: {mixing_plot_path}")
    return "\n".join(lines) + "\n"


def validate_settings():
    if BEAM_MOMENTUM_GEV <= 0.0 or LEPTON_MASS_GEV <= 0.0:
        raise ValueError("Beam momentum and lepton mass must be positive.")
    if len(Z_VALUES) < 2 or len(THETA_CM_VALUES) < 2 or len(THETA_P_VALUES) < 2:
        raise ValueError("All scan axes must contain at least two points.")
    if (
        np.any(np.diff(Z_VALUES) <= 0.0)
        or np.any(np.diff(THETA_CM_VALUES) <= 0.0)
        or np.any(np.diff(THETA_P_VALUES) <= 0.0)
    ):
        raise ValueError("Scan axes must be strictly increasing.")
    if np.any(THETA_P_VALUES < 0.0) or np.any(THETA_P_VALUES >= 2.0 * np.pi):
        raise ValueError("Oriented recoil angles must lie in [0, 2 pi).")
    if (
        np.any(FINAL_PROTON_MOMENTUM_VALUES_GEV <= 0.0)
        or np.any(FINAL_PROTON_MOMENTUM_VALUES_GEV >= BEAM_MOMENTUM_GEV)
        or np.any(np.diff(FINAL_PROTON_MOMENTUM_VALUES_GEV) >= 0.0)
    ):
        raise ValueError(
            "Final-proton momenta must be positive, below the beam momentum, "
            "and strictly decreasing."
        )
    if SCAN_WORKER_COUNT < 1 or SCAN_PLOT_WORKER_COUNT < 1:
        raise ValueError("Scan and plot worker counts must be positive.")
    if not (
        np.isfinite(INITIAL_LEPTON_MIXING_ANGLE_RAD)
        and np.isfinite(INITIAL_PROTON_MIXING_ANGLE_RAD)
    ):
        raise ValueError("Initial polarization mixing angles must be finite.")
    for name, values in (
        ("theta_e mixing", THETA_E_MIX_VALUES_RAD),
        ("theta_p mixing", THETA_P_MIX_VALUES_RAD),
    ):
        if len(values) < 2 or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"The {name} scan axis must be strictly increasing.")
        if np.any(values < 0.0) or np.any(values >= np.pi):
            raise ValueError(f"The {name} scan axis must lie in [0, pi).")


def main():
    validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            z_index,
            theta_index,
            theta_p_index,
            float(z),
            float(theta_cm),
            float(theta_p),
        )
        for z_index, z in enumerate(Z_VALUES)
        for theta_index, theta_cm in enumerate(THETA_CM_VALUES)
        for theta_p_index, theta_p in enumerate(THETA_P_VALUES)
    ]
    if SCAN_INITIAL_MIXING_ANGLES:
        rows = []
        mixing_rows = run_mixing_angle_tasks(tasks)
        write_csv(MIXING_SCAN_CSV, mixing_rows)
        write_ranked_csv(
            mixing_rows,
            spin_cases=(SPIN_CASE_MIXING_ANGLES,),
            output_path=MIXING_TOP_CSV,
        )
        plot_paths = {}
        mixing_plot_path = save_mixing_angle_plot(mixing_rows)
    else:
        rows = run_tasks(tasks)
        mixing_rows = []
        write_csv(FULL_CSV, rows)
        write_ranked_csv(rows)
        plot_paths = save_plots(rows)
        mixing_plot_path = None
    report = build_report(rows, plot_paths, mixing_rows, mixing_plot_path)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
