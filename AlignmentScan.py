"""Final electron-photon alignment phase-space scan.

This script scans characteristic user-frame kinematics with a fine
``phi_in_electron`` by ``phi_gamma`` grid and focuses the locator outputs on the
selected two-body concurrences and multipartite observables.
"""

from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from Algebra import mdot
from Kinematics import kinematics_user_from_independent
from SpinDensityMat import (
    AVERAGE_INITIAL_SPINS,
    ENTANGLEMENT_INITIAL_STATE,
    ENTANGLEMENT_NAMES,
    F1,
    F2,
    M,
    NORMALIZE_TRACE,
    OUTPUT_DIR,
    SCAN_WORKERS,
    USER_S_CENTER,
    SPIN_CASE_POLARIZED,
    SPIN_CASE_TRANSVERSE_TX,
    SPIN_CASE_TRANSVERSE_TY,
    SPIN_CASE_UNPOLARIZED,
    amplitude_table,
    density_matrix_from_amplitudes,
    entanglement_measures_from_amplitudes,
    entanglement_measures_from_state,
    initial_spin_states,
    is_transverse_spin_case,
    outgoing_spin_states,
    polarized_entanglement_difference,
    trace_value,
    transverse_electron_coefficients,
    transverse_entanglement_measures,
)


CHARACTERISTIC_S_POINTS = (
    ("low_s", 0.78 * USER_S_CENTER),
    ("mid_s", 1.00 * USER_S_CENTER),
    ("high_s", 1.18 * USER_S_CENTER),
)
CHARACTERISTIC_THETA_IN_POINTS = (
    ("low_theta_in", 3.14159/2),
    ("high_theta_in", 3.14159/2),
)
CHARACTERISTIC_QOUT_POINTS = (
    ("low_Egamma", 0.75),
    ("mid_Egamma", 1.25),
    ("high_Egamma", 1.75),
)

PHASE_SPACE_PHI_IN_VALUES = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
PHASE_SPACE_PHIOUT_VALUES = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
ALIGNMENT_ANGLE_MAX_DEG = 10.0
ALIGNMENT_ANGLE_MAX_RAD = np.deg2rad(ALIGNMENT_ANGLE_MAX_DEG)

OUTPUT_ROOT = OUTPUT_DIR.parent
LEGACY_ALIGNMENT_OUTPUT_DIR = OUTPUT_DIR / "AlignmentScan"
ALIGNMENT_OUTPUT_DIR = OUTPUT_ROOT / "AlignmentScan"
LEGACY_ALIGNMENT_LOG_PATH = ALIGNMENT_OUTPUT_DIR / "AlignmentScan.log"
ALIGNMENT_LOG_PATH = OUTPUT_ROOT / "AlignmentScan.log"
DENSITY_MATRIX_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "DensityMatScan"
CONCURRENCE_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "ConcurrenceScan"
AMPLITUDE_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "AmplitudeScan"
RUN_ALIGNMENT_DENSITY_MATRIX_SCAN = False
RUN_ALIGNMENT_AMPLITUDE_SCAN = False
SPIN_CASE_DOUBLE_TRANSVERSE = "double_transverse"
DOUBLE_TRANSVERSE_BASE_SPIN_CASE = SPIN_CASE_TRANSVERSE_TX
REDUCED_EP_BASIS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALIGNMENT_SPIN_CASES = (
    ("unpolarized", "Unpolarized", SPIN_CASE_UNPOLARIZED),
    ("longitudinal_polarized", "Longitudinal polarized", SPIN_CASE_POLARIZED),
    ("Tx", "Tx", SPIN_CASE_TRANSVERSE_TX),
    ("Ty", "Ty", SPIN_CASE_TRANSVERSE_TY),
    ("double_transverse", "Double transverse", SPIN_CASE_DOUBLE_TRANSVERSE),
)
COARSE_CONCURRENCE_NAMES = ("C12", "C13", "C23", "M1", "M2", "M3", "F3")
COARSE_CONCURRENCE_TOP_N = 60
COARSE_C13_TOP_N = COARSE_CONCURRENCE_TOP_N


def characteristic_kinematic_points():
    """Return coarse anchor kinematics for two-angle concurrence scans."""
    points = []
    for s_regime, s in CHARACTERISTIC_S_POINTS:
        for theta_regime, theta_in in CHARACTERISTIC_THETA_IN_POINTS:
            for qout_regime, qOut in CHARACTERISTIC_QOUT_POINTS:
                point_id = f"{s_regime}_{theta_regime}_{qout_regime}"
                points.append({
                    "kinematic_point": point_id,
                    "s_regime": s_regime,
                    "theta_in_regime": theta_regime,
                    "qOut_regime": qout_regime,
                    "s": float(s),
                    "theta_in": float(theta_in),
                    "qOut": float(qOut),
                })
    return points


def normalize_azimuth(angle):
    """Return an azimuth normalized to [0, 2*pi)."""
    return float(np.mod(angle, 2.0 * np.pi))


def electron_phi_from_proton(phi_in_proton):
    """Return incoming electron azimuth from the incoming proton azimuth."""
    return normalize_azimuth(phi_in_proton + np.pi)


def proton_phi_from_electron(phi_in_electron):
    """Return incoming proton azimuth from the incoming electron azimuth."""
    return normalize_azimuth(phi_in_electron - np.pi)


def _require_matplotlib():
    """Import matplotlib in headless mode with a writable cache directory."""
    cache_dir = Path(tempfile.gettempdir()) / "dvcs_helicity_amp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def clean_alignment_outputs():
    """Remove generated alignment-scan outputs before regenerating."""
    if ALIGNMENT_OUTPUT_DIR.exists():
        shutil.rmtree(ALIGNMENT_OUTPUT_DIR)
    if LEGACY_ALIGNMENT_OUTPUT_DIR.exists():
        shutil.rmtree(LEGACY_ALIGNMENT_OUTPUT_DIR)
    if LEGACY_ALIGNMENT_LOG_PATH.exists():
        LEGACY_ALIGNMENT_LOG_PATH.unlink()
    if ALIGNMENT_LOG_PATH.exists():
        ALIGNMENT_LOG_PATH.unlink()


def spatial_opening_angle(a, b):
    """Return the spatial opening angle between two four-vectors in radians."""
    a3 = np.asarray(a, dtype=float)[1:4]
    b3 = np.asarray(b, dtype=float)[1:4]
    denominator = np.linalg.norm(a3) * np.linalg.norm(b3)
    if denominator <= 1e-14:
        raise ZeroDivisionError("Cannot compute an opening angle with zero momentum.")
    cosine = np.dot(a3, b3) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def real_scalar(value, label):
    """Return a real scalar, rejecting non-negligible imaginary residue."""
    scalar = np.real_if_close(value, tol=1000)
    if np.iscomplexobj(scalar):
        real = float(np.real(scalar))
        imag = float(np.imag(scalar))
        if abs(imag) > 1.0e-10 * max(1.0, abs(real)):
            raise ValueError(f"{label} has a non-negligible imaginary part: {value}")
        return real
    return float(scalar)


def final_electron_photon_spin_correlations(rho, out_states):
    """Return final electron/photon helicity means and correlations."""
    matrix = np.asarray(rho, dtype=complex)
    populations = np.real_if_close(np.diag(matrix), tol=1000).real
    out_states = np.asarray(out_states, dtype=float)
    h_out = out_states[:, 0]
    lam = out_states[:, 2]
    h_mean = float(np.sum(populations * h_out))
    lambda_mean = float(np.sum(populations * lam))
    h_lambda = float(np.sum(populations * h_out * lam))
    return {
        "h_out_mean": h_mean,
        "lambda_mean": lambda_mean,
        "h_lambda": h_lambda,
        "h_lambda_connected": h_lambda - h_mean * lambda_mean,
    }


def electron_photon_reduced_density_matrix(rho):
    """Trace out the outgoing proton and keep the 4x4 electron-photon matrix."""
    matrix = np.asarray(rho, dtype=complex)
    tensor = matrix.reshape((2, 2, 2, 2, 2, 2))
    reduced = np.trace(tensor, axis1=1, axis2=4)
    return reduced.reshape((4, 4))


def _reduced_matrix_key(prefix, row_index, col_index, part):
    """Return a stable CSV key for one reduced electron-photon matrix entry."""
    return f"{prefix}_rho_ep_r{row_index}_c{col_index}_{part}"


def _reduced_matrix_headers(prefix):
    """Return ordered real/imag CSV headers for a 4x4 reduced matrix."""
    headers = []
    for row_index in range(4):
        for col_index in range(4):
            headers.append(_reduced_matrix_key(prefix, row_index, col_index, "real"))
            headers.append(_reduced_matrix_key(prefix, row_index, col_index, "imag"))
    return headers


def _reduced_basis_label(index):
    """Return a compact label for one electron-photon basis state."""
    h_out, lam = REDUCED_EP_BASIS[index]
    return rf"$h'={h_out:+d},\lambda={lam:+d}$"


def _electron_photon_amplitude_matrix_from_state(amplitude_row):
    """Return a 2x2 final electron-photon amplitude from one outgoing state."""
    matrix = np.zeros((2, 2), dtype=complex)
    out_states = outgoing_spin_states()
    helicity_index = {-1: 0, 1: 1}
    for out_index, (h_out, _s_out, lam) in enumerate(out_states):
        matrix[helicity_index[h_out], helicity_index[lam]] += amplitude_row[out_index]
    return matrix


def double_transverse_final_state(amplitudes):
    """Return the outgoing state for e and p both polarized along transverse x."""
    in_states = initial_spin_states()
    coefficients = transverse_electron_coefficients(DOUBLE_TRANSVERSE_BASE_SPIN_CASE)
    return sum(
        coefficients[h_in]
        * coefficients[s_in]
        * amplitudes[in_states.index((h_in, s_in))]
        for h_in in (-1, +1)
        for s_in in (-1, +1)
    )


def normalized_double_transverse_final_state(amplitudes):
    """Return the normalized double-transverse outgoing pure state."""
    state = double_transverse_final_state(amplitudes)
    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    if norm <= 1e-14:
        raise ZeroDivisionError("Cannot normalize a zero double-transverse final state.")
    return state / norm


def density_matrix_for_alignment_spin_case(
    amplitudes,
    spin_case,
    average_initial=AVERAGE_INITIAL_SPINS,
):
    """Return density-matrix data for one alignment spin case."""
    if spin_case == SPIN_CASE_DOUBLE_TRANSVERSE:
        state = double_transverse_final_state(amplitudes)
        rho = np.outer(state, state.conj())
        spin_signal = float(np.sum(np.abs(state) ** 2))
        squared_amplitude = float(np.sum(np.abs(amplitudes) ** 2))
        if average_initial:
            squared_amplitude /= amplitudes.shape[0]
        return rho, spin_signal, squared_amplitude
    return density_matrix_from_amplitudes(
        amplitudes,
        average_initial=average_initial,
        spin_case=spin_case,
    )


def electron_photon_amplitude_matrix(amplitudes, spin_case, initial_state):
    """Return a 2x2 final electron-photon amplitude for one spin case."""
    in_states = initial_spin_states()
    proton_spin = initial_state[1]
    if spin_case == SPIN_CASE_UNPOLARIZED:
        amplitude_row = np.sum(amplitudes, axis=0) / np.sqrt(len(in_states))
    elif spin_case == SPIN_CASE_POLARIZED:
        amplitude_row = (
            amplitudes[in_states.index((+1, proton_spin))]
            - amplitudes[in_states.index((-1, proton_spin))]
        ) / np.sqrt(2.0)
    elif is_transverse_spin_case(spin_case):
        coefficients = transverse_electron_coefficients(spin_case)
        amplitude_row = sum(
            coefficients[h_in] * amplitudes[in_states.index((h_in, proton_spin))]
            for h_in in (-1, +1)
        )
    elif spin_case == SPIN_CASE_DOUBLE_TRANSVERSE:
        amplitude_row = double_transverse_final_state(amplitudes)
    else:
        raise ValueError(f"Unknown alignment amplitude spin case: {spin_case}")
    return _electron_photon_amplitude_matrix_from_state(amplitude_row)


def normalized_amplitude_matrix(matrix, squared_amplitude):
    """Normalize an amplitude matrix by ``sqrt(M^2)``."""
    if squared_amplitude <= 1e-14:
        raise ZeroDivisionError("Cannot normalize amplitude with zero M^2.")
    return np.asarray(matrix, dtype=complex) / np.sqrt(squared_amplitude)


def _amplitude_matrix_headers(prefix):
    """Return ordered real/imag CSV headers for the normalized 2x2 amplitude."""
    headers = []
    for row_index in range(2):
        for col_index in range(2):
            headers.append(f"{prefix}_amp_ep_norm_r{row_index}_c{col_index}_real")
            headers.append(f"{prefix}_amp_ep_norm_r{row_index}_c{col_index}_imag")
    return headers


def _amplitude_basis_label(row_index, col_index):
    """Return a compact label for one electron-photon amplitude entry."""
    h_out = -1 if row_index == 0 else 1
    lam = -1 if col_index == 0 else 1
    return rf"$h'={h_out:+d},\lambda={lam:+d}$"


def alignment_concurrence_measures(amplitudes, spin_case, initial_state):
    """Return concurrence observables for one alignment spin case."""
    if spin_case == SPIN_CASE_UNPOLARIZED:
        return entanglement_measures_from_amplitudes(amplitudes, initial_state)
    if spin_case == SPIN_CASE_POLARIZED:
        return polarized_entanglement_difference(amplitudes, initial_state[1])
    if is_transverse_spin_case(spin_case):
        return transverse_entanglement_measures(
            amplitudes,
            initial_state[1],
            spin_case,
        )
    if spin_case == SPIN_CASE_DOUBLE_TRANSVERSE:
        return entanglement_measures_from_state(
            normalized_double_transverse_final_state(amplitudes)
        )
    raise ValueError(f"Unknown alignment spin case: {spin_case}")


def _scan_alignment_point_task(task):
    """Evaluate one final electron-photon alignment point."""
    anchor, phi_in_electron, phiOut, settings = task
    s = anchor["s"]
    theta_in = anchor["theta_in"]
    qOut = anchor["qOut"]
    phi_in_proton = proton_phi_from_electron(phi_in_electron)
    out_states = outgoing_spin_states()
    try:
        kin = kinematics_user_from_independent(
            s,
            theta_in,
            phi_in_proton,
            qOut,
            phiOut,
            settings["m"],
            label=(
                f"user alignment s={s:.6g}, theta_in={theta_in:.6g}, "
                f"qOut={qOut:.6g}"
            ),
        )
        momenta = kin["momenta"]
        user_params = kin["user_params"]
        user_independent = kin["user_independent"]
        angle_rad = spatial_opening_angle(momenta["kp"], momenta["qout"])
        k_dot_qout = real_scalar(mdot(momenta["k"], momenta["qout"]), "k dot qout")
        kp_dot_qout = real_scalar(mdot(momenta["kp"], momenta["qout"]), "kp dot qout")
        row = {
            "kinematic_point": anchor["kinematic_point"],
            "s_regime": anchor["s_regime"],
            "theta_in_regime": anchor["theta_in_regime"],
            "qOut_regime": anchor["qOut_regime"],
            "s": float(user_independent["s"]),
            "sqrt_s": float(kin["sqrt_s"]),
            "pIn": float(user_params["pIn"]),
            "pOut": float(user_params["pOut"]),
            "qOut": float(user_independent["qOut"]),
            "theta_in": float(user_independent["theta_in"]),
            "phi_in": float(user_independent["phi_in"]),
            "phi_in_electron": electron_phi_from_proton(user_independent["phi_in"]),
            "phiOut": float(user_independent["phiOut"]),
            "Q2": float(kin["Q2"]),
            "xB": float(kin["xB"]),
            "t": float(kin["t"]),
            "W2": float(kin["W2"]),
            "y": float(kin["y"]),
            "theta_e_gamma_rad": angle_rad,
            "theta_e_gamma_deg": float(np.degrees(angle_rad)),
            "k_dot_qout": k_dot_qout,
            "kp_dot_qout": kp_dot_qout,
            "abs_k_dot_qout": abs(k_dot_qout),
            "abs_kp_dot_qout": abs(kp_dot_qout),
            "aligned": angle_rad <= settings["angle_max_rad"],
            "squared_amplitude_M2": np.nan,
            "amplitude_normalization_sqrt_M2": np.nan,
        }
        for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
            row[f"{prefix}_amplitude_ep"] = np.full((2, 2), np.nan + 0.0j)
            row[f"{prefix}_trace"] = np.nan
            row[f"{prefix}_spin_signal_M2"] = np.nan
            row[f"{prefix}_h_out_mean"] = np.nan
            row[f"{prefix}_lambda_mean"] = np.nan
            row[f"{prefix}_h_lambda"] = np.nan
            row[f"{prefix}_h_lambda_connected"] = np.nan
            row[f"{prefix}_rho_ep"] = np.full((4, 4), np.nan + 0.0j)
            for name in ENTANGLEMENT_NAMES:
                row[f"{prefix}_{name}"] = np.nan

        amplitudes = amplitude_table(momenta, kin["m"], settings["F1"], settings["F2"])
        rho_unpolarized, unpolarized_signal, squared_amplitude = (
            density_matrix_from_amplitudes(
                amplitudes,
                average_initial=AVERAGE_INITIAL_SPINS,
                spin_case=SPIN_CASE_UNPOLARIZED,
            )
        )
        if settings["normalize_trace"] and squared_amplitude <= 1e-14:
            raise ZeroDivisionError("Cannot normalize a zero alignment matrix.")
        row["amplitude_normalization_sqrt_M2"] = np.sqrt(squared_amplitude)

        for prefix, _label, spin_case in ALIGNMENT_SPIN_CASES:
            amplitude_ep = electron_photon_amplitude_matrix(
                amplitudes,
                spin_case,
                settings["entanglement_initial_state"],
            )
            row[f"{prefix}_amplitude_ep"] = normalized_amplitude_matrix(
                amplitude_ep,
                squared_amplitude,
            )
            if spin_case == SPIN_CASE_UNPOLARIZED:
                rho = rho_unpolarized
                spin_signal = unpolarized_signal
            else:
                rho, spin_signal, _squared_amplitude_check = (
                    density_matrix_for_alignment_spin_case(
                        amplitudes,
                        average_initial=AVERAGE_INITIAL_SPINS,
                        spin_case=spin_case,
                    )
                )
            if settings["normalize_trace"]:
                rho = rho / squared_amplitude
            corr = final_electron_photon_spin_correlations(rho, out_states)
            concurrence = alignment_concurrence_measures(
                amplitudes,
                spin_case,
                settings["entanglement_initial_state"],
            )
            row.update({
                f"{prefix}_trace": trace_value(rho),
                f"{prefix}_spin_signal_M2": spin_signal,
                f"{prefix}_h_out_mean": corr["h_out_mean"],
                f"{prefix}_lambda_mean": corr["lambda_mean"],
                f"{prefix}_h_lambda": corr["h_lambda"],
                f"{prefix}_h_lambda_connected": corr["h_lambda_connected"],
                f"{prefix}_rho_ep": electron_photon_reduced_density_matrix(rho),
            })
            for name in ENTANGLEMENT_NAMES:
                row[f"{prefix}_{name}"] = concurrence[name]
    except Exception as exc:
        return {
            "ok": False,
            "kinematic_point": anchor["kinematic_point"],
            "s": float(s),
            "theta_in": float(theta_in),
            "phi_in": float(phi_in_proton),
            "phi_in_electron": float(phi_in_electron),
            "qOut": float(qOut),
            "phiOut": float(phiOut),
            "error": str(exc),
        }

    row["squared_amplitude_M2"] = squared_amplitude
    return {"ok": True, "row": row}


def scan_final_electron_photon_alignment(
    kinematic_points=None,
    phi_in_electron_values=PHASE_SPACE_PHI_IN_VALUES,
    phiOut_values=PHASE_SPACE_PHIOUT_VALUES,
    angle_max_rad=ALIGNMENT_ANGLE_MAX_RAD,
    m=M,
    F1=F1,
    F2=F2,
    normalize_trace=NORMALIZE_TRACE,
    entanglement_initial_state=ENTANGLEMENT_INITIAL_STATE,
    max_workers=SCAN_WORKERS,
):
    """Scan two angular variables around characteristic user-frame kinematics."""
    rows = []
    failures = []
    if kinematic_points is None:
        kinematic_points = characteristic_kinematic_points()
    settings = {
        "m": m,
        "F1": F1,
        "F2": F2,
        "normalize_trace": normalize_trace,
        "entanglement_initial_state": entanglement_initial_state,
        "angle_max_rad": angle_max_rad,
    }
    tasks = [
        (
            anchor,
            float(phi_in_electron),
            float(phiOut),
            settings,
        )
        for anchor, phi_in_electron, phiOut in product(
            kinematic_points,
            phi_in_electron_values,
            phiOut_values,
        )
    ]
    if max_workers and max_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_scan_alignment_point_task, tasks))
    else:
        results = [_scan_alignment_point_task(task) for task in tasks]

    for result in results:
        if result["ok"]:
            rows.append(result["row"])
        else:
            failures.append((
                result["kinematic_point"],
                result["s"],
                result["theta_in"],
                result["phi_in"],
                result.get("phi_in_electron", np.nan),
                result["qOut"],
                result["phiOut"],
                result["error"],
            ))

    return {
        "rows": rows,
        "failures": failures,
        "angle_max_rad": angle_max_rad,
        "angle_max_deg": float(np.degrees(angle_max_rad)),
        "kinematic_points": list(kinematic_points),
        "s_values": np.asarray([point["s"] for point in kinematic_points], dtype=float),
        "theta_in_values": np.asarray([point["theta_in"] for point in kinematic_points], dtype=float),
        "phi_in_electron_values": np.asarray(phi_in_electron_values, dtype=float),
        "qOut_values": np.asarray([point["qOut"] for point in kinematic_points], dtype=float),
        "phiOut_values": np.asarray(phiOut_values, dtype=float),
        "m": m,
        "normalized_by_squared_amplitude": normalize_trace,
        "entanglement_initial_state": entanglement_initial_state,
        "spin_cases": ALIGNMENT_SPIN_CASES,
        "scan_parameterization": "user_frame_independent",
    }


def _alignment_csv_headers():
    """Return CSV headers for the final electron-photon alignment scan."""
    headers = _kinematic_csv_headers() + [
        "aligned",
        "squared_amplitude_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend([
            f"{prefix}_trace",
            f"{prefix}_spin_signal_M2",
            f"{prefix}_h_out_mean",
            f"{prefix}_lambda_mean",
            f"{prefix}_h_lambda",
            f"{prefix}_h_lambda_connected",
        ])
    return headers


def _kinematic_csv_headers():
    """Return common user-frame and derived-invariant CSV headers."""
    return [
        "kinematic_point",
        "s_regime",
        "theta_in_regime",
        "qOut_regime",
        "s",
        "sqrt_s",
        "pIn",
        "pOut",
        "qOut",
        "theta_in",
        "phi_in",
        "phi_in_electron",
        "phiOut",
        "Q2",
        "xB",
        "t",
        "W2",
        "y",
        "theta_e_gamma_rad",
        "theta_e_gamma_deg",
        "k_dot_qout",
        "kp_dot_qout",
        "abs_k_dot_qout",
        "abs_kp_dot_qout",
    ]


def _kinematic_csv_row(row):
    """Return common formatted user-frame and invariant metadata."""
    return [
        row["kinematic_point"],
        row["s_regime"],
        row["theta_in_regime"],
        row["qOut_regime"],
        f"{row['s']:.16e}",
        f"{row['sqrt_s']:.16e}",
        f"{row['pIn']:.16e}",
        f"{row['pOut']:.16e}",
        f"{row['qOut']:.16e}",
        f"{row['theta_in']:.16e}",
        f"{row['phi_in']:.16e}",
        f"{row['phi_in_electron']:.16e}",
        f"{row['phiOut']:.16e}",
        f"{row['Q2']:.16e}",
        f"{row['xB']:.16e}",
        f"{row['t']:.16e}",
        f"{row['W2']:.16e}",
        f"{row['y']:.16e}",
        f"{row['theta_e_gamma_rad']:.16e}",
        f"{row['theta_e_gamma_deg']:.16e}",
        f"{row['k_dot_qout']:.16e}",
        f"{row['kp_dot_qout']:.16e}",
        f"{row['abs_k_dot_qout']:.16e}",
        f"{row['abs_kp_dot_qout']:.16e}",
    ]


def _alignment_csv_row(row):
    """Return one formatted CSV row for the alignment scan."""
    values = _kinematic_csv_row(row) + [
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        values.extend([
            f"{row[f'{prefix}_trace']:.16e}",
            f"{row[f'{prefix}_spin_signal_M2']:.16e}",
            f"{row[f'{prefix}_h_out_mean']:.16e}",
            f"{row[f'{prefix}_lambda_mean']:.16e}",
            f"{row[f'{prefix}_h_lambda']:.16e}",
            f"{row[f'{prefix}_h_lambda_connected']:.16e}",
        ])
    return values


def _density_matrix_csv_headers():
    """Return CSV headers for the reduced electron-photon density-matrix scan."""
    headers = _kinematic_csv_headers() + [
        "aligned",
        "squared_amplitude_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend([
            f"{prefix}_trace",
            f"{prefix}_spin_signal_M2",
            *_reduced_matrix_headers(prefix),
        ])
    return headers


def _density_matrix_csv_row(row):
    """Return one formatted CSV row for the reduced density-matrix scan."""
    values = _kinematic_csv_row(row) + [
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        values.extend([
            f"{row[f'{prefix}_trace']:.16e}",
            f"{row[f'{prefix}_spin_signal_M2']:.16e}",
        ])
        matrix = row[f"{prefix}_rho_ep"]
        for row_index in range(4):
            for col_index in range(4):
                value = matrix[row_index, col_index]
                values.extend([f"{value.real:.16e}", f"{value.imag:.16e}"])
    return values


def save_alignment_scan_csv_files(alignment_scan, output_dir=ALIGNMENT_OUTPUT_DIR):
    """Save full and aligned-only CSV files for the alignment phase-space scan."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_csv": output_dir / "electron_photon_spin_correlation_phase_space.csv",
        "aligned_csv": output_dir / "electron_photon_spin_correlation_aligned.csv",
    }
    headers = _alignment_csv_headers()
    aligned_rows = [row for row in alignment_scan["rows"] if row["aligned"]]

    for key, rows in (("all_csv", alignment_scan["rows"]), ("aligned_csv", aligned_rows)):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_alignment_csv_row(row))
    return paths


def save_density_matrix_scan_csv_files(
    alignment_scan,
    output_dir=DENSITY_MATRIX_OUTPUT_DIR,
):
    """Save full and aligned-only reduced density-matrix scan CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_csv": output_dir / "electron_photon_reduced_density_phase_space.csv",
        "aligned_csv": output_dir / "electron_photon_reduced_density_aligned.csv",
    }
    headers = _density_matrix_csv_headers()
    aligned_rows = [row for row in alignment_scan["rows"] if row["aligned"]]
    for key, rows in (("all_csv", alignment_scan["rows"]), ("aligned_csv", aligned_rows)):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_density_matrix_csv_row(row))
    return paths


def _amplitude_csv_headers():
    """Return CSV headers for the reduced electron-photon amplitude scan."""
    headers = _kinematic_csv_headers() + [
        "aligned",
        "squared_amplitude_M2",
        "amplitude_normalization_sqrt_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend(_amplitude_matrix_headers(prefix))
    return headers


def _amplitude_csv_row(row):
    """Return one formatted CSV row for the reduced amplitude scan."""
    values = _kinematic_csv_row(row) + [
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
        f"{row['amplitude_normalization_sqrt_M2']:.16e}",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        matrix = row[f"{prefix}_amplitude_ep"]
        for row_index in range(2):
            for col_index in range(2):
                value = matrix[row_index, col_index]
                values.extend([f"{value.real:.16e}", f"{value.imag:.16e}"])
    return values


def save_amplitude_scan_csv_files(
    alignment_scan,
    output_dir=AMPLITUDE_OUTPUT_DIR,
):
    """Save full and aligned-only 2x2 amplitude scan CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_csv": output_dir / "electron_photon_amplitude_phase_space.csv",
        "aligned_csv": output_dir / "electron_photon_amplitude_aligned.csv",
    }
    headers = _amplitude_csv_headers()
    aligned_rows = [row for row in alignment_scan["rows"] if row["aligned"]]
    for key, rows in (("all_csv", alignment_scan["rows"]), ("aligned_csv", aligned_rows)):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_amplitude_csv_row(row))
    return paths


def _bin_edges_from_values(values, max_bins=18):
    """Return plotting bin edges adapted to discrete or continuous values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0, 1.0])

    unique = np.unique(values)
    if unique.size == 1:
        width = max(1.0e-6, abs(unique[0]) * 1.0e-6)
        return np.asarray([unique[0] - width, unique[0] + width])
    if unique.size <= max_bins:
        midpoints = 0.5 * (unique[:-1] + unique[1:])
        first = unique[0] - 0.5 * (unique[1] - unique[0])
        last = unique[-1] + 0.5 * (unique[-1] - unique[-2])
        return np.concatenate([[first], midpoints, [last]])
    return np.linspace(values.min(), values.max(), max_bins + 1)


def user_frame_kinematic_specs(rows):
    """Return user-frame variables used as binned plot axes."""
    return (
        ("s", np.asarray([row["s"] for row in rows]), r"$s$ [GeV$^2$]"),
        ("theta_in", np.asarray([row["theta_in"] for row in rows]), r"$\theta_{\rm in}$ [rad]"),
        ("phi_in_electron", np.asarray([row["phi_in_electron"] for row in rows]), r"$\phi_{e,\rm in}$ [rad]"),
        ("qOut", np.asarray([row["qOut"] for row in rows]), r"$E_{\gamma}'$ [GeV]"),
        ("phiOut", np.asarray([row["phiOut"] for row in rows]), r"$\phi_{\gamma}'$ [rad]"),
    )


def _binned_mean_2d(x_values, y_values, z_values, x_edges, y_edges):
    """Return a masked 2D binned mean ``z`` on ``x``/``y`` bins."""
    finite = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & np.isfinite(z_values)
    )
    counts, _x_edges, _y_edges = np.histogram2d(
        x_values[finite],
        y_values[finite],
        bins=(x_edges, y_edges),
    )
    sums, _x_edges, _y_edges = np.histogram2d(
        x_values[finite],
        y_values[finite],
        bins=(x_edges, y_edges),
        weights=z_values[finite],
    )
    mean = np.full_like(sums, np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return np.ma.masked_invalid(mean.T), counts.T


def save_reduced_density_matrix_component_plot(
    alignment_scan,
    prefix,
    title_prefix,
    component,
    output_path,
    cmap,
    vmin=None,
    vmax=None,
):
    """Save 4x4 reduced electron-photon matrix heatmaps for one component."""
    plt, PdfPages = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = alignment_scan["rows"]
    if not rows:
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
            ax.text(0.5, 0.5, "No valid phase-space points", ha="center", va="center")
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close(fig)
        return output_path

    kinematic_specs = user_frame_kinematic_specs(rows)
    matrices = np.asarray([row[f"{prefix}_rho_ep"] for row in rows], dtype=complex)
    if component == "abs":
        matrix_values = np.abs(matrices)
        component_title = "magnitude"
        colorbar_label = rf"$|\rho_{{e\gamma}}|$ ({title_prefix})"
    elif component == "phase":
        matrix_values = np.angle(matrices)
        component_title = "phase"
        colorbar_label = rf"$\arg\rho_{{e\gamma}}$ [rad] ({title_prefix})"
    else:
        raise ValueError(f"Unknown reduced-matrix component: {component}")
    finite_matrix_values = matrix_values[np.isfinite(matrix_values)]
    plot_vmin = vmin
    plot_vmax = vmax
    if finite_matrix_values.size and (plot_vmin is None or plot_vmax is None):
        if plot_vmin is None:
            plot_vmin = float(np.nanmin(finite_matrix_values))
        if plot_vmax is None:
            plot_vmax = float(np.nanmax(finite_matrix_values))
    if plot_vmin is None:
        plot_vmin = 0.0
    if plot_vmax is None:
        plot_vmax = 1.0

    with PdfPages(output_path) as pdf:
        for (_x_name, x_values, x_label), (_y_name, y_values, y_label) in combinations(kinematic_specs, 2):
            base_mask = np.isfinite(x_values) & np.isfinite(y_values)
            x_edges = _bin_edges_from_values(x_values[base_mask])
            y_edges = _bin_edges_from_values(y_values[base_mask])
            fig, axes = plt.subplots(
                4,
                4,
                figsize=(13.0, 10.0),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            meshes = []
            for flat_index, ax in enumerate(axes.ravel()):
                row_index, col_index = divmod(flat_index, 4)
                values = matrix_values[:, row_index, col_index]
                finite_mask = base_mask & np.isfinite(values)
                mean_grid, _count_grid = _binned_mean_2d(
                    x_values[finite_mask],
                    y_values[finite_mask],
                    values[finite_mask],
                    x_edges,
                    y_edges,
                )
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    mean_grid,
                    shading="auto",
                    cmap=cmap,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                )
                meshes.append(mesh)
                ax.set_title(
                    f"r{row_index} c{col_index}\n"
                    f"{_reduced_basis_label(row_index)} x "
                    f"{_reduced_basis_label(col_index)}",
                    fontsize=8.0,
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            fig.suptitle(
                f"{title_prefix}: reduced electron-photon matrix {component_title}",
                fontsize=13,
            )
            fig.colorbar(
                meshes[-1],
                ax=axes,
                label=colorbar_label,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def _spin_case_plot_stem(prefix):
    """Return the filename stem for one alignment spin case."""
    return f"electron_photon_reduced_density_{prefix}_phase_space"


def save_reduced_density_matrix_plots(alignment_scan, prefix, title_prefix):
    """Save magnitude and phase PDFs for one 4x4 reduced density matrix scan."""
    stem = _spin_case_plot_stem(prefix)
    return {
        "abs": save_reduced_density_matrix_component_plot(
            alignment_scan,
            prefix,
            title_prefix,
            "abs",
            DENSITY_MATRIX_OUTPUT_DIR / f"{stem}_matrix_abs.pdf",
            "viridis",
            0.0,
            None,
        ),
        "phase": save_reduced_density_matrix_component_plot(
            alignment_scan,
            prefix,
            title_prefix,
            "phase",
            DENSITY_MATRIX_OUTPUT_DIR / f"{stem}_matrix_phase.pdf",
            "twilight",
            -np.pi,
            np.pi,
        ),
    }


def save_amplitude_component_plot(
    alignment_scan,
    prefix,
    title_prefix,
    component,
    output_path,
    cmap,
    vmin=None,
    vmax=None,
):
    """Save a 2x2 electron-photon amplitude heatmap PDF for one component."""
    plt, PdfPages = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = alignment_scan["rows"]
    if not rows:
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
            ax.text(0.5, 0.5, "No valid phase-space points", ha="center", va="center")
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close(fig)
        return output_path

    kinematic_specs = user_frame_kinematic_specs(rows)
    matrices = np.asarray([row[f"{prefix}_amplitude_ep"] for row in rows], dtype=complex)
    if component == "abs":
        matrix_values = np.abs(matrices)
        component_title = "magnitude"
        colorbar_label = r"$|M|/\sqrt{M^2_{\rm unpol}}$"
    elif component == "phase":
        matrix_values = np.angle(matrices)
        component_title = "phase"
        colorbar_label = r"$\arg(M/\sqrt{M^2_{\rm unpol}})$ [rad]"
    else:
        raise ValueError(f"Unknown amplitude component: {component}")
    finite_matrix_values = matrix_values[np.isfinite(matrix_values)]
    plot_vmin = vmin
    plot_vmax = vmax
    if finite_matrix_values.size and (plot_vmin is None or plot_vmax is None):
        if plot_vmin is None:
            plot_vmin = float(np.nanmin(finite_matrix_values))
        if plot_vmax is None:
            plot_vmax = float(np.nanmax(finite_matrix_values))
    if plot_vmin is None:
        plot_vmin = 0.0
    if plot_vmax is None:
        plot_vmax = 1.0

    with PdfPages(output_path) as pdf:
        for (_x_name, x_values, x_label), (_y_name, y_values, y_label) in combinations(kinematic_specs, 2):
            base_mask = np.isfinite(x_values) & np.isfinite(y_values)
            x_edges = _bin_edges_from_values(x_values[base_mask])
            y_edges = _bin_edges_from_values(y_values[base_mask])
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(9.0, 7.0),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            meshes = []
            for flat_index, ax in enumerate(axes.ravel()):
                row_index, col_index = divmod(flat_index, 2)
                values = matrix_values[:, row_index, col_index]
                finite_mask = base_mask & np.isfinite(values)
                mean_grid, _count_grid = _binned_mean_2d(
                    x_values[finite_mask],
                    y_values[finite_mask],
                    values[finite_mask],
                    x_edges,
                    y_edges,
                )
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    mean_grid,
                    shading="auto",
                    cmap=cmap,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                )
                meshes.append(mesh)
                ax.set_title(
                    f"r{row_index} c{col_index}\n"
                    f"{_amplitude_basis_label(row_index, col_index)}"
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            fig.suptitle(
                f"{title_prefix}: normalized electron-photon amplitude {component_title}",
                fontsize=13,
            )
            fig.colorbar(meshes[-1], ax=axes, label=colorbar_label)
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def save_amplitude_scan_plots(alignment_scan, prefix, title_prefix):
    """Save magnitude and phase PDFs for one 2x2 amplitude scan."""
    return {
        "abs": save_amplitude_component_plot(
            alignment_scan,
            prefix,
            title_prefix,
            "abs",
            AMPLITUDE_OUTPUT_DIR / f"electron_photon_amplitude_{prefix}_matrix_abs.pdf",
            "viridis",
            0.0,
            None,
        ),
        "phase": save_amplitude_component_plot(
            alignment_scan,
            prefix,
            title_prefix,
            "phase",
            AMPLITUDE_OUTPUT_DIR / f"electron_photon_amplitude_{prefix}_matrix_phase.pdf",
            "twilight",
            -np.pi,
            np.pi,
        ),
    }


def _concurrence_csv_headers():
    """Return CSV headers for the coarse concurrence locator scan."""
    headers = _kinematic_csv_headers() + [
        "aligned",
        "squared_amplitude_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend(f"{prefix}_{name}" for name in COARSE_CONCURRENCE_NAMES)
    return headers


def _concurrence_csv_row(row):
    """Return one formatted CSV row for the coarse concurrence locator scan."""
    values = _kinematic_csv_row(row) + [
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        values.extend(f"{row[f'{prefix}_{name}']:.16e}" for name in COARSE_CONCURRENCE_NAMES)
    return values


def _concurrence_top_csv_headers():
    """Return CSV headers for ranked coarse concurrence locator points."""
    return [
        "rank_group",
        "rank",
        "rank_value",
        "rank_observable",
        "rank_spin_case",
        *_kinematic_csv_headers(),
        "aligned",
        "squared_amplitude_M2",
        *(
            f"{prefix}_{name}"
            for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES
            for name in COARSE_CONCURRENCE_NAMES
        ),
    ]


def _concurrence_top_csv_row(rank_group, rank, row):
    """Return one ranked coarse concurrence CSV row."""
    prefix, observable = rank_group.rsplit("_", 1)
    return [
        rank_group,
        rank,
        f"{row[rank_group]:.16e}",
        observable,
        prefix,
        *_kinematic_csv_row(row),
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
        *(
            f"{row[f'{prefix}_{name}']:.16e}"
            for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES
            for name in COARSE_CONCURRENCE_NAMES
        ),
    ]


def save_concurrence_top_csv(
    rows,
    output_path,
    top_n=COARSE_CONCURRENCE_TOP_N,
    observables=COARSE_CONCURRENCE_NAMES,
):
    """Save top coarse concurrence rows for each observable and spin case."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_concurrence_top_csv_headers())
        for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
            for observable in observables:
                key = f"{prefix}_{observable}"
                finite_rows = [row for row in rows if np.isfinite(row.get(key, np.nan))]
                ordered = sorted(finite_rows, key=lambda row: row[key], reverse=True)
                for rank, row in enumerate(ordered[:top_n], start=1):
                    writer.writerow(_concurrence_top_csv_row(key, rank, row))
    return output_path


def save_c13_top_csv(rows, output_path, top_n=COARSE_C13_TOP_N):
    """Save top coarse C13 rows for each spin case."""
    return save_concurrence_top_csv(rows, output_path, top_n=top_n, observables=("C13",))


def save_concurrence_scan_csv_files(
    alignment_scan,
    output_dir=CONCURRENCE_OUTPUT_DIR,
):
    """Save full, aligned-only, and ranked coarse concurrence locator CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_csv": output_dir / "electron_photon_concurrence_phase_space.csv",
        "aligned_csv": output_dir / "electron_photon_concurrence_aligned.csv",
        "top_concurrence_csv": output_dir / "electron_photon_concurrence_top.csv",
        "top_c13_csv": output_dir / "electron_photon_c13_top.csv",
    }
    headers = _concurrence_csv_headers()
    aligned_rows = [row for row in alignment_scan["rows"] if row["aligned"]]
    for key, rows in (("all_csv", alignment_scan["rows"]), ("aligned_csv", aligned_rows)):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_concurrence_csv_row(row))
    save_concurrence_top_csv(alignment_scan["rows"], paths["top_concurrence_csv"])
    save_c13_top_csv(alignment_scan["rows"], paths["top_c13_csv"])
    return paths


def save_concurrence_scan_plot(
    alignment_scan,
    prefix,
    title_prefix,
    output_path=None,
):
    """Save binned concurrence heatmaps for one alignment spin case."""
    plt, PdfPages = _require_matplotlib()
    if output_path is None:
        output_path = CONCURRENCE_OUTPUT_DIR / f"concurrence_scan_{prefix}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = alignment_scan["rows"]
    if not rows:
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
            ax.text(0.5, 0.5, "No valid phase-space points", ha="center", va="center")
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close(fig)
        return output_path

    with PdfPages(output_path) as pdf:
        for name in COARSE_CONCURRENCE_NAMES:
            values = np.asarray([row[f"{prefix}_{name}"] for row in rows], dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                vmin, vmax, cmap = 0.0, 1.0, "viridis"
            elif np.nanmin(finite_values) < 0.0:
                limit = float(np.nanmax(np.abs(finite_values)))
                vmin, vmax, cmap = -limit, limit, "coolwarm"
            else:
                vmin, vmax, cmap = 0.0, float(np.nanmax(finite_values)), "viridis"
            if abs(vmax - vmin) <= 1.0e-14:
                vmax = vmin + 1.0

            anchors = alignment_scan.get("kinematic_points", [])
            if anchors:
                fig, axes = plt.subplots(
                    3,
                    6,
                    figsize=(18.0, 9.8),
                    constrained_layout=True,
                )
                axes_flat = axes.ravel()
                anchor_meshes = []
                for index, anchor in enumerate(anchors):
                    ax = axes_flat[index]
                    point_rows = [
                        row for row in rows
                        if row.get("kinematic_point") == anchor["kinematic_point"]
                        and np.isfinite(row[f"{prefix}_{name}"])
                    ]
                    if not point_rows:
                        ax.set_axis_off()
                        continue
                    x_values = np.asarray([row["phi_in_electron"] for row in point_rows], dtype=float)
                    y_values = np.asarray([row["phiOut"] for row in point_rows], dtype=float)
                    point_values = np.asarray([row[f"{prefix}_{name}"] for row in point_rows], dtype=float)
                    x_edges = _bin_edges_from_values(x_values)
                    y_edges = _bin_edges_from_values(y_values)
                    mean_grid, _count_grid = _binned_mean_2d(
                        x_values,
                        y_values,
                        point_values,
                        x_edges,
                        y_edges,
                    )
                    mesh = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        mean_grid,
                        shading="auto",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    anchor_meshes.append(mesh)
                    best = max(point_rows, key=lambda row: row[f"{prefix}_{name}"])
                    ax.set_title(
                        f"{anchor['s_regime']}, {anchor['theta_in_regime']}\n"
                        f"{anchor['qOut_regime']}, max={best[f'{prefix}_{name}']:.3f}",
                        fontsize=8,
                    )
                    if index // 6 == 2:
                        ax.set_xlabel(r"$\phi_{e,\rm in}$", fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    if index % 6 == 0:
                        ax.set_ylabel(r"$\phi_{\gamma}'$", fontsize=8)
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=7)
                for ax in axes_flat[len(anchors):]:
                    ax.set_axis_off()
                fig.suptitle(
                    f"{title_prefix}: {name} two-angle scans at characteristic kinematics",
                    fontsize=14,
                )
                if anchor_meshes:
                    fig.colorbar(anchor_meshes[-1], ax=axes, label=name, shrink=0.82)
                pdf.savefig(fig)
                plt.close(fig)
    return output_path


def save_concurrence_scan_plots(alignment_scan):
    """Save selected concurrence scan PDFs for all alignment spin cases."""
    return {
        prefix: save_concurrence_scan_plot(alignment_scan, prefix, label)
        for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES
    }


def concurrence_summary_line(row, key):
    """Return a compact coarse concurrence maximum summary line."""
    return (
        f"    {key}={row[key]:.12g}, s={row['s']:.8g}, "
        f"theta_in={row['theta_in']:.8g}, phi_e_in={row['phi_in_electron']:.8g}, "
        f"phi_p_in={row['phi_in']:.8g}, "
        f"qOut={row['qOut']:.8g}, phiOut={row['phiOut']:.8g}, "
        f"Q2={row['Q2']:.8g}, xB={row['xB']:.8g}, t={row['t']:.8g}, "
        f"theta(e',gamma)={row['theta_e_gamma_deg']:.8g} deg"
    )


def build_alignment_report(alignment_scan, alignment_paths):
    """Build a text report for the final electron-photon alignment scan."""
    rows = alignment_scan["rows"]
    aligned_rows = [row for row in rows if row["aligned"]]
    locator_label = "/".join(COARSE_CONCURRENCE_NAMES)
    lines = [
        f"{locator_label}-focused user-frame phase-space scan",
        "  anchor variables: s, theta_in, qOut",
        "  scanned variables per anchor: phi_in_electron, phi_gamma",
        "  locator observables: "
        f"{', '.join(COARSE_CONCURRENCE_NAMES)}",
        f"  angle cut: theta(e', gamma) <= {alignment_scan['angle_max_deg']:.6g} deg",
        f"  characteristic kinematic anchors: {len(alignment_scan['kinematic_points'])}",
        f"  s anchor range: {min(alignment_scan['s_values']):.6g} to {max(alignment_scan['s_values']):.6g}",
        f"  theta_in anchor range: {min(alignment_scan['theta_in_values']):.6g} to {max(alignment_scan['theta_in_values']):.6g}",
        f"  qOut/Egamma anchor range: {min(alignment_scan['qOut_values']):.6g} to {max(alignment_scan['qOut_values']):.6g}",
        f"  phi_e_in scan: {len(alignment_scan['phi_in_electron_values'])} values from "
        f"{alignment_scan['phi_in_electron_values'][0]:.6g} to {alignment_scan['phi_in_electron_values'][-1]:.6g}",
        f"  phi_gamma scan: {len(alignment_scan['phiOut_values'])} values from "
        f"{alignment_scan['phiOut_values'][0]:.6g} to {alignment_scan['phiOut_values'][-1]:.6g}",
        f"  valid points: {len(rows)}",
        f"  aligned points: {len(aligned_rows)}",
        "  amplitude normalization: M / sqrt(M^2_unpol)",
        "  density matrix scan outputs: "
        f"{'enabled' if alignment_paths.get('run_density_matrix_scan') else 'disabled'}",
        "  amplitude scan outputs: "
        f"{'enabled' if alignment_paths.get('run_amplitude_scan') else 'disabled'}",
    ]
    if rows:
        min_angle = min(row["theta_e_gamma_deg"] for row in rows)
        max_angle = max(row["theta_e_gamma_deg"] for row in rows)
        lines.append(f"  theta range: {min_angle:.6g} to {max_angle:.6g} deg")
        lines.append("")
        lines.append(f"Top {locator_label} locator points:")
        for observable in COARSE_CONCURRENCE_NAMES:
            lines.append(f"  {observable}:")
            for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
                key = f"{prefix}_{observable}"
                finite_rows = [row for row in rows if np.isfinite(row.get(key, np.nan))]
                if finite_rows:
                    best = max(finite_rows, key=lambda row: row[key])
                    lines.append(f"    {label}:")
                    lines.append(concurrence_summary_line(best, key))
    if aligned_rows:
        correlations = [row["unpolarized_h_lambda"] for row in aligned_rows]
        lines.append(
            "  aligned <hOut*lambda> range: "
            f"{min(correlations):.6g} to {max(correlations):.6g}"
        )
    lines.extend([
        f"  saved full csv: {alignment_paths['all_csv']}",
        f"  saved aligned csv: {alignment_paths['aligned_csv']}",
        "  saved concurrence full csv: "
        f"{alignment_paths['concurrence_csv']['all_csv']}",
        "  saved concurrence aligned csv: "
        f"{alignment_paths['concurrence_csv']['aligned_csv']}",
        "  saved ranked concurrence csv: "
        f"{alignment_paths['concurrence_csv']['top_concurrence_csv']}",
        "  saved legacy ranked C13 csv: "
        f"{alignment_paths['concurrence_csv']['top_c13_csv']}",
    ])
    if alignment_paths.get("run_density_matrix_scan"):
        lines.extend([
            "  saved density matrix full csv: "
            f"{alignment_paths['density_matrix_csv']['all_csv']}",
            "  saved density matrix aligned csv: "
            f"{alignment_paths['density_matrix_csv']['aligned_csv']}",
        ])
    if alignment_paths.get("run_amplitude_scan"):
        lines.extend([
            "  saved amplitude full csv: "
            f"{alignment_paths['amplitude_csv']['all_csv']}",
            "  saved amplitude aligned csv: "
            f"{alignment_paths['amplitude_csv']['aligned_csv']}",
        ])
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        if alignment_paths.get("run_amplitude_scan"):
            amplitude_paths = alignment_paths[f"{prefix}_amplitude_plots"]
            lines.append(
                f"  saved {label.lower()} amplitude magnitude plot: "
                f"{amplitude_paths['abs']}"
            )
            lines.append(
                f"  saved {label.lower()} amplitude phase plot: "
                f"{amplitude_paths['phase']}"
            )
        if alignment_paths.get("run_density_matrix_scan"):
            plot_paths = alignment_paths[f"{prefix}_reduced_density_plots"]
            lines.append(
                f"  saved {label.lower()} reduced density magnitude plot: "
                f"{plot_paths['abs']}"
            )
            lines.append(
                f"  saved {label.lower()} reduced density phase plot: "
                f"{plot_paths['phase']}"
            )
        lines.append(
            f"  saved {label.lower()} concurrence plot: "
            f"{alignment_paths['concurrence_plots'][prefix]}"
        )
    if alignment_scan["failures"]:
        lines.append(f"  invalid phase-space points: {len(alignment_scan['failures'])}")
        for point_id, s, theta_in, phi_in, phi_in_electron, qOut, phiOut, message in alignment_scan["failures"][:10]:
            lines.append(
                f"    point={point_id}, s={s:.8g}, theta_in={theta_in:.8g}, "
                f"phi_e_in={phi_in_electron:.8g}, phi_p_in={phi_in:.8g}, "
                f"qOut={qOut:.8g}, phiOut={phiOut:.8g}: {message}"
            )
    return "\n".join(lines)


def main(
    run_density_matrix_scan=RUN_ALIGNMENT_DENSITY_MATRIX_SCAN,
    run_amplitude_scan=RUN_ALIGNMENT_AMPLITUDE_SCAN,
):
    """Regenerate final electron-photon alignment scan outputs."""
    clean_alignment_outputs()
    alignment_scan = scan_final_electron_photon_alignment()
    paths = save_alignment_scan_csv_files(alignment_scan)
    paths["run_density_matrix_scan"] = run_density_matrix_scan
    paths["run_amplitude_scan"] = run_amplitude_scan
    if run_density_matrix_scan:
        paths["density_matrix_csv"] = save_density_matrix_scan_csv_files(alignment_scan)
    if run_amplitude_scan:
        paths["amplitude_csv"] = save_amplitude_scan_csv_files(alignment_scan)
    paths["concurrence_csv"] = save_concurrence_scan_csv_files(alignment_scan)
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        if run_amplitude_scan:
            paths[f"{prefix}_amplitude_plots"] = (
                save_amplitude_scan_plots(alignment_scan, prefix, label)
            )
        if run_density_matrix_scan:
            paths[f"{prefix}_reduced_density_plots"] = (
                save_reduced_density_matrix_plots(alignment_scan, prefix, label)
            )
    paths["concurrence_plots"] = save_concurrence_scan_plots(alignment_scan)

    log_text = build_alignment_report(alignment_scan, paths) + f"\n\nSaved log: {ALIGNMENT_LOG_PATH}\n"
    ALIGNMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
