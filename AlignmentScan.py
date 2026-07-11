"""Final electron-photon alignment phase-space scan.

This script scans characteristic user-frame kinematics with a fine
``phi_in_electron`` by ``phi_gamma`` grid and focuses the locator outputs on the
selected two-body concurrences and multipartite observables.
"""

from itertools import product
from concurrent.futures import ProcessPoolExecutor
import csv
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

def _progress_bar(iterable, total, desc):
    """Pass-through iterable (install tqdm for progress bars)."""
    return iterable

from Algebra import DEFAULT_TOL, mdot
from FormFactors import YAHL_MODEL_NAME, yahl_dirac_pauli_from_t
from Kinematics import kinematics_user_from_independent
from SpinDensityMat import (
    M,
    NORMALIZE_TRACE,
    OUTPUT_DIR,
    SCAN_WORKERS,
    USER_S_CENTER,
    SPIN_CASE_L_PROTON,
    SPIN_CASE_L_ELECTRON,
    SPIN_CASE_TX_PROTON,
    SPIN_CASE_TY_PROTON,
    SPIN_CASE_TX_ELECTRON,
    SPIN_CASE_TY_ELECTRON,
    SPIN_CASE_LL,
    SPIN_CASE_LTX,
    SPIN_CASE_LTY,
    SPIN_CASE_TXTX,
    SPIN_CASE_TXTY,
    SPIN_CASE_UNPOLARIZED,
    amplitude_table,
    entanglement_mode,
    outgoing_spin_states,
    initial_spin_average_divisor,
    process_density_matrix_from_amplitudes,
    spin_density_observables_from_amplitudes,
)


CHARACTERISTIC_S_POINTS = (
    #("mid_s", 1.00 * USER_S_CENTER),
    ("high_s", 1.00 * USER_S_CENTER),
)
CHARACTERISTIC_THETA_IN_POINTS = (
    ("high_theta_in", 3.14159/2),
)
CHARACTERISTIC_QOUT_POINTS = (
    ("low_Egamma", 0.5),
    ("mid_Egamma", 1.0),
    ("high_Egamma", 1.8),
)

PHASE_SPACE_PHI_IN_VALUES = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
PHASE_SPACE_PHIOUT_VALUES = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
ALIGNMENT_ANGLE_MAX_DEG = 10.0
ALIGNMENT_ANGLE_MAX_RAD = np.deg2rad(ALIGNMENT_ANGLE_MAX_DEG)

OUTPUT_ROOT = OUTPUT_DIR.parent
ALIGNMENT_OUTPUT_DIR = OUTPUT_ROOT / "AlignmentScan"
ALIGNMENT_LOG_PATH = OUTPUT_ROOT / "AlignmentScan.log"
CONCURRENCE_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "ConcurrenceScan"
CONCURRENCE_PHASE_SPACE_CSV = (
    CONCURRENCE_OUTPUT_DIR / "electron_photon_concurrence_phase_space.csv"
)
REGENERATE_PLOTS_FROM_CSV = False
REGENERATE_PLOTS_CSV_PATH = CONCURRENCE_PHASE_SPACE_CSV
ALIGNMENT_SPIN_CASES = (
    ("unpolarized", "Unpolarized", SPIN_CASE_UNPOLARIZED),
    ("L_proton", "Longitudinal proton", SPIN_CASE_L_PROTON),
    ("L_electron", "Longitudinal electron", SPIN_CASE_L_ELECTRON),
    ("Tx_proton", "Tx proton", SPIN_CASE_TX_PROTON),
    ("Ty_proton", "Ty proton", SPIN_CASE_TY_PROTON),
    ("Tx_electron", "Tx electron", SPIN_CASE_TX_ELECTRON),
    ("Ty_electron", "Ty electron", SPIN_CASE_TY_ELECTRON),
    ("LL", "LL", SPIN_CASE_LL),
    ("LTx", "LTx", SPIN_CASE_LTX),
    ("LTy", "LTy", SPIN_CASE_LTY),
    ("TxTx", "TxTx", SPIN_CASE_TXTX),
    ("TxTy", "TxTy", SPIN_CASE_TXTY),
)
SPIN_AVERAGING_VERSION = "prepared_spin_ensemble_v4"


def spin_averaging_description(spin_case):
    """Return the initial-spin ensemble convention used by AlignmentScan."""
    descriptions = {
        prefix_spin_case: (
            f"direct prepared state; {initial_spin_average_divisor(prefix_spin_case):.0f} "
            "incoherent component(s)"
        )
        for _prefix, _label, prefix_spin_case in ALIGNMENT_SPIN_CASES
    }
    if spin_case not in descriptions:
        raise ValueError(f"Unknown alignment spin case: {spin_case}")
    return descriptions[spin_case]
COARSE_CONCURRENCE_NAMES = (
    "C_e_p",
    "C_e_gamma",
    "C_p_gamma",
    "C_e_rest",
    "C_p_rest",
    "C_gamma_rest",
    "M_e",
    "M_p",
    "M_gamma",
    "F3",
)
SIGNED_CONCURRENCE_OBSERVABLES = {"M_e", "M_p", "M_gamma"}
COARSE_CONCURRENCE_TOP_N = 60
COARSE_E_GAMMA_TOP_N = COARSE_CONCURRENCE_TOP_N
OBSERVABLE_LATEX_LABELS = {
    "C_e_p": r"$C_{ep}$",
    "C_e_gamma": r"$C_{e\gamma}$",
    "C_p_gamma": r"$C_{p\gamma}$",
    "C_e_rest": r"$C_{e|p\gamma}$",
    "C_p_rest": r"$C_{p|e\gamma}$",
    "C_gamma_rest": r"$C_{\gamma|ep}$",
    "M_e": r"$M_e$",
    "M_p": r"$M_p$",
    "M_gamma": r"$M_\gamma$",
    "F3": r"$F_3$",
}
OBSERVABLE_TEXT_LABELS = {
    name: label.replace("$", "")
    for name, label in OBSERVABLE_LATEX_LABELS.items()
}
ANCHOR_TITLE_FONTSIZE = 11
PLOT_AXIS_LABEL_FONTSIZE = 12
PLOT_TICK_FONTSIZE = 10
PLOT_SUPTITLE_FONTSIZE = 17
PLOT_COLORBAR_FONTSIZE = 12
HEATMAP_MAX_BINS = 96
HEATMAP_PLOT_STYLE = "grid"  # "grid" or "contour"
HEATMAP_CONTOUR_LEVELS = 12


def observable_latex_label(name):
    """Return a LaTeX display label for an observable key."""
    return OBSERVABLE_LATEX_LABELS.get(name, name)


def observable_text_label(name):
    """Return a plain text report label using LaTeX-style subscripts."""
    return OBSERVABLE_TEXT_LABELS.get(name, name)


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


from PlotUtils import require_matplotlib as _require_matplotlib


def clean_alignment_outputs():
    """Remove generated alignment-scan outputs before regenerating."""
    if ALIGNMENT_OUTPUT_DIR.exists():
        shutil.rmtree(ALIGNMENT_OUTPUT_DIR)
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


def _scan_alignment_point_task(task):
    """Evaluate one final electron-photon alignment point."""
    anchor, phi_in_electron, phiOut, settings = task
    s = anchor["s"]
    theta_in = anchor["theta_in"]
    qOut = anchor["qOut"]
    phi_in_proton = proton_phi_from_electron(phi_in_electron)
    out_states = outgoing_spin_states()
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
    angle_rad = spatial_opening_angle(momenta["kp"], momenta["qout"])
    k_dot_qout = real_scalar(mdot(momenta["k"], momenta["qout"]), "k dot qout")
    kp_dot_qout = real_scalar(mdot(momenta["kp"], momenta["qout"]), "kp dot qout")
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], kin["m"])

    # --- explicitly guard against singular propagator / t denominators ---
    singular_reason = None
    if abs(kin["t"]) <= DEFAULT_TOL:
        singular_reason = "momentum-transfer denominator t is singular"
    elif abs(kp_dot_qout) <= DEFAULT_TOL:
        singular_reason = "final-state lepton propagator is singular (kp·qout ≈ 0)"
    elif abs(k_dot_qout) <= DEFAULT_TOL:
        singular_reason = "initial-state lepton propagator is singular (k·qout ≈ 0)"
    if singular_reason is not None:
        return {
            "ok": False,
            "kinematic_point": anchor["kinematic_point"],
            "s": float(kin["s"]),
            "theta_in": float(kin["theta_in"]),
            "phi_in": float(kin["phi_in"]),
            "phi_in_electron": electron_phi_from_proton(kin["phi_in"]),
            "qOut": float(kin["qOut"]),
            "phiOut": float(kin["phiOut"]),
            "error": singular_reason,
        }

    row = {
        "kinematic_point": anchor["kinematic_point"],
        "s_regime": anchor["s_regime"],
        "theta_in_regime": anchor["theta_in_regime"],
        "qOut_regime": anchor["qOut_regime"],
        "initial_spin_averaging_version": SPIN_AVERAGING_VERSION,
        "s": float(kin["s"]),
        "sqrt_s": float(kin["sqrt_s"]),
        "pIn": float(kin["pIn"]),
        "pOut": float(kin["pOut"]),
        "qOut": float(kin["qOut"]),
        "theta_in": float(kin["theta_in"]),
        "phi_in": float(kin["phi_in"]),
        "phi_in_electron": electron_phi_from_proton(kin["phi_in"]),
        "phiOut": float(kin["phiOut"]),
        "Q2": float(kin["Q2"]),
        "xB": float(kin["xB"]),
        "t": float(kin["t"]),
        "F1": F1,
        "F2": F2,
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
    }
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        row[f"{prefix}_trace"] = np.nan
        row[f"{prefix}_spin_signal_M2"] = np.nan
        row[f"{prefix}_cross_section_ratio"] = np.nan
        row[f"{prefix}_purity"] = np.nan
        row[f"{prefix}_h_out_mean"] = np.nan
        row[f"{prefix}_lambda_mean"] = np.nan
        row[f"{prefix}_h_lambda"] = np.nan
        row[f"{prefix}_h_lambda_connected"] = np.nan
        for name in COARSE_CONCURRENCE_NAMES:
            row[f"{prefix}_{name}"] = np.nan

    amplitudes = amplitude_table(momenta, kin["m"], F1, F2)
    process_rho = process_density_matrix_from_amplitudes(amplitudes)
    squared_amplitude = np.nan
    for prefix, _label, spin_case in ALIGNMENT_SPIN_CASES:
        spin_data = spin_density_observables_from_amplitudes(
            amplitudes,
            spin_case=spin_case,
            normalize_trace=settings["normalize_trace"],
            process_rho=process_rho,
        )
        rho = spin_data["rho"]
        squared_amplitude = spin_data["squared_amplitude"]
        corr = final_electron_photon_spin_correlations(rho, out_states)
        row.update({
            f"{prefix}_trace": spin_data["trace"],
            f"{prefix}_spin_signal_M2": spin_data["spin_signal"],
            f"{prefix}_cross_section_ratio": spin_data["cross_section_ratio"],
            f"{prefix}_purity": spin_data["purity"],
            f"{prefix}_h_out_mean": corr["h_out_mean"],
            f"{prefix}_lambda_mean": corr["lambda_mean"],
            f"{prefix}_h_lambda": corr["h_lambda"],
            f"{prefix}_h_lambda_connected": corr["h_lambda_connected"],
        })
        for name in COARSE_CONCURRENCE_NAMES:
            row[f"{prefix}_{name}"] = spin_data["entanglement"][name]

    row["squared_amplitude_M2"] = squared_amplitude
    return {"ok": True, "row": row}


def _optimal_chunksize(num_tasks, num_workers):
    """Return a chunksize that balances scheduler overhead and load imbalance.

    For ``ProcessPoolExecutor.map``, ``chunksize`` controls how many tasks are
    batched into a single serialized submission.  A value too small increases
    IPC overhead; a value too large makes the work queue lumpy and can leave
    workers idle at the tail.

    The heuristic aims for roughly ``num_workers * 4`` chunks so that every
    worker gets several batches, keeping all cores busy without excessive
    round-trips.
    """
    if num_workers < 1:
        return 1
    ideal_chunks = max(1, num_workers * 4)
    return max(1, math.ceil(num_tasks / ideal_chunks))


def scan_final_electron_photon_alignment(
    kinematic_points=None,
    phi_in_electron_values=PHASE_SPACE_PHI_IN_VALUES,
    phiOut_values=PHASE_SPACE_PHIOUT_VALUES,
    angle_max_rad=ALIGNMENT_ANGLE_MAX_RAD,
    m=M,
    normalize_trace=NORMALIZE_TRACE,
    max_workers=SCAN_WORKERS,
):
    """Scan two angular variables around characteristic user-frame kinematics."""
    rows = []
    failures = []
    if kinematic_points is None:
        kinematic_points = characteristic_kinematic_points()
    settings = {
        "m": m,
        "normalize_trace": normalize_trace,
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
        chunksize = _optimal_chunksize(len(tasks), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            _map = executor.map(
                _scan_alignment_point_task,
                tasks,
                chunksize=chunksize,
            )
            results = list(
                _progress_bar(
                    _map,
                    total=len(tasks),
                    desc="Alignment scan",
                )
            )
    else:
        results = [
            _scan_alignment_point_task(task)
            for task in _progress_bar(tasks, total=len(tasks), desc="Alignment scan")
        ]

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
        "form_factor_model": YAHL_MODEL_NAME,
        "normalized_to_unit_trace": normalize_trace,
        "spin_cases": ALIGNMENT_SPIN_CASES,
        "spin_averaging_version": SPIN_AVERAGING_VERSION,
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
            f"{prefix}_cross_section_ratio",
            f"{prefix}_purity",
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
        "initial_spin_averaging_version",
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
        "F1",
        "F2",
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
        row["initial_spin_averaging_version"],
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
        f"{row['F1']:.16e}",
        f"{row['F2']:.16e}",
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
            f"{row[f'{prefix}_cross_section_ratio']:.16e}",
            f"{row[f'{prefix}_purity']:.16e}",
            f"{row[f'{prefix}_h_out_mean']:.16e}",
            f"{row[f'{prefix}_lambda_mean']:.16e}",
            f"{row[f'{prefix}_h_lambda']:.16e}",
            f"{row[f'{prefix}_h_lambda_connected']:.16e}",
        ])
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
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_alignment_csv_row(row))
    return paths


def _bin_edges_from_values(values, max_bins=HEATMAP_MAX_BINS):
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


def _heatmap_style(style):
    """Return a normalized heatmap style name."""
    style = str(style).strip().lower()
    aliases = {
        "cell": "grid",
        "cells": "grid",
        "mesh": "grid",
        "pcolormesh": "grid",
        "filled_contour": "contour",
        "contourf": "contour",
    }
    style = aliases.get(style, style)
    if style not in {"grid", "contour"}:
        raise ValueError("Heatmap style must be 'grid' or 'contour'.")
    return style


def _draw_heatmap(ax, x_edges, y_edges, values, cmap, vmin, vmax, style):
    """Draw one binned heatmap as either cell grid or filled contours."""
    style = _heatmap_style(style)
    if style == "grid":
        return ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    if values.shape[0] < 2 or values.shape[1] < 2:
        return ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    levels = np.linspace(vmin, vmax, HEATMAP_CONTOUR_LEVELS + 1)
    return ax.contourf(x_grid, y_grid, values, levels=levels, cmap=cmap, extend="neither")


def _scan_x_phi(row):
    """Return the incoming proton azimuth used as the heatmap x coordinate."""
    value = row.get("phi_in", np.nan)
    if np.isfinite(value):
        return value
    return proton_phi_from_electron(row.get("phi_in_electron", np.nan))


def _add_pi_over_two_reference_lines(ax):
    """Draw the requested pi/2 guide lines on heatmaps."""
    ax.axvline(0.5 * np.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)
    ax.axhline(0.5 * np.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)


def _heatmap_color_scale(prefix, observable):
    """Return fixed absolute heatmap scale and colormap for an observable."""
    if observable in SIGNED_CONCURRENCE_OBSERVABLES:
        return -1.0, 1.0, "coolwarm"
    return 0.0, 1.0, "viridis"


def _concurrence_csv_headers():
    """Return CSV headers for the coarse concurrence locator scan."""
    headers = _kinematic_csv_headers() + [
        "aligned",
        "squared_amplitude_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend([
            f"{prefix}_cross_section_ratio",
            f"{prefix}_purity",
            *(f"{prefix}_{name}" for name in COARSE_CONCURRENCE_NAMES),
        ])
    return headers


def _concurrence_csv_row(row):
    """Return one formatted CSV row for the coarse concurrence locator scan."""
    values = _kinematic_csv_row(row) + [
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        values.extend([
            f"{row[f'{prefix}_cross_section_ratio']:.16e}",
            f"{row[f'{prefix}_purity']:.16e}",
            *(f"{row[f'{prefix}_{name}']:.16e}" for name in COARSE_CONCURRENCE_NAMES),
        ])
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
            key
            for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES
            for key in (
                f"{prefix}_cross_section_ratio",
                f"{prefix}_purity",
                *(f"{prefix}_{name}" for name in COARSE_CONCURRENCE_NAMES),
            )
        ),
    ]


def _concurrence_top_csv_row(rank_group, rank, row, prefix, observable):
    """Return one ranked coarse concurrence CSV row."""
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
            f"{row[key]:.16e}"
            for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES
            for key in (
                f"{prefix}_cross_section_ratio",
                f"{prefix}_purity",
                *(f"{prefix}_{name}" for name in COARSE_CONCURRENCE_NAMES),
            )
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
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_concurrence_top_csv_headers())
        for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
            for observable in observables:
                key = f"{prefix}_{observable}"
                finite_rows = [row for row in rows if np.isfinite(row.get(key, np.nan))]
                ordered = sorted(finite_rows, key=lambda row: row[key], reverse=True)
                for rank, row in enumerate(ordered[:top_n], start=1):
                    writer.writerow(_concurrence_top_csv_row(key, rank, row, prefix, observable))
    return output_path


def save_e_gamma_top_csv(rows, output_path, top_n=COARSE_E_GAMMA_TOP_N):
    """Save top coarse electron-photon concurrence rows for each spin case."""
    return save_concurrence_top_csv(rows, output_path, top_n=top_n, observables=("C_e_gamma",))


def save_concurrence_scan_csv_files(
    alignment_scan,
    output_dir=CONCURRENCE_OUTPUT_DIR,
):
    """Save full, aligned-only, and ranked coarse concurrence locator CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_csv": output_dir / CONCURRENCE_PHASE_SPACE_CSV.name,
        "aligned_csv": output_dir / "electron_photon_concurrence_aligned.csv",
        "top_concurrence_csv": output_dir / "electron_photon_concurrence_top.csv",
        "top_e_gamma_csv": output_dir / "electron_photon_e_gamma_top.csv",
    }
    headers = _concurrence_csv_headers()
    aligned_rows = [row for row in alignment_scan["rows"] if row["aligned"]]
    for key, rows in (("all_csv", alignment_scan["rows"]), ("aligned_csv", aligned_rows)):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_concurrence_csv_row(row))
    save_concurrence_top_csv(alignment_scan["rows"], paths["top_concurrence_csv"])
    save_e_gamma_top_csv(alignment_scan["rows"], paths["top_e_gamma_csv"])
    return paths


def _csv_float(row, key, default=np.nan):
    """Return a float from a CSV row, using ``default`` for absent/blank values."""
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _csv_bool(row, key):
    """Return a bool from a CSV row written by the alignment scan."""
    value = str(row.get(key, "")).strip().lower()
    return value in {"1", "true", "yes"}


def load_concurrence_scan_csv(csv_path=CONCURRENCE_PHASE_SPACE_CSV):
    """Load saved concurrence scan rows for plot regeneration."""
    csv_path = Path(csv_path)
    rows = []
    missing_observable_columns = set()
    string_columns = {
        "kinematic_point",
        "s_regime",
        "theta_in_regime",
        "qOut_regime",
        "initial_spin_averaging_version",
    }
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        for raw in reader:
            row = {}
            for key in _kinematic_csv_headers():
                if key in string_columns:
                    row[key] = raw.get(key, "")
                else:
                    row[key] = _csv_float(raw, key)
            row["aligned"] = _csv_bool(raw, "aligned")
            row["squared_amplitude_M2"] = _csv_float(raw, "squared_amplitude_M2")
            for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
                row[f"{prefix}_cross_section_ratio"] = _csv_float(
                    raw, f"{prefix}_cross_section_ratio"
                )
                row[f"{prefix}_purity"] = _csv_float(raw, f"{prefix}_purity")
                for name in COARSE_CONCURRENCE_NAMES:
                    key = f"{prefix}_{name}"
                    if key not in fieldnames:
                        missing_observable_columns.add(key)
                    row[key] = _csv_float(raw, key)
            rows.append(row)

    kinematic_points = []
    seen_points = set()
    for row in rows:
        point_id = row["kinematic_point"]
        if point_id in seen_points:
            continue
        seen_points.add(point_id)
        kinematic_points.append({
            "kinematic_point": point_id,
            "s_regime": row["s_regime"],
            "theta_in_regime": row["theta_in_regime"],
            "qOut_regime": row["qOut_regime"],
            "s": row["s"],
            "theta_in": row["theta_in"],
            "qOut": row["qOut"],
        })

    return {
        "rows": rows,
        "failures": [],
        "angle_max_rad": np.nan,
        "angle_max_deg": np.nan,
        "kinematic_points": kinematic_points,
        "s_values": np.asarray([point["s"] for point in kinematic_points], dtype=float),
        "theta_in_values": np.asarray([point["theta_in"] for point in kinematic_points], dtype=float),
        "phi_in_electron_values": np.unique(np.asarray([
            row["phi_in_electron"] for row in rows
        ], dtype=float)),
        "qOut_values": np.asarray([point["qOut"] for point in kinematic_points], dtype=float),
        "phiOut_values": np.unique(np.asarray([row["phiOut"] for row in rows], dtype=float)),
        "m": M,
        "form_factor_model": YAHL_MODEL_NAME,
        "normalized_to_unit_trace": NORMALIZE_TRACE,
        "spin_cases": ALIGNMENT_SPIN_CASES,
        "spin_averaging_version": SPIN_AVERAGING_VERSION,
        "source_csv": csv_path,
        "missing_observable_columns": sorted(missing_observable_columns),
    }


def save_concurrence_scan_plot(
    alignment_scan,
    prefix,
    title_prefix,
    output_path=None,
    plot_style=HEATMAP_PLOT_STYLE,
):
    """Save binned concurrence heatmaps for one alignment spin case."""
    plot_style = _heatmap_style(plot_style)
    plt, PdfPages = _require_matplotlib()
    if output_path is None:
        output_path = CONCURRENCE_OUTPUT_DIR / f"concurrence_scan_{prefix}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = alignment_scan["rows"]
    if not rows:
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            ax.text(
                0.5,
                0.5,
                "No valid phase-space points",
                ha="center",
                va="center",
                fontsize=PLOT_SUPTITLE_FONTSIZE,
            )
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        return output_path

    with PdfPages(output_path) as pdf:
        for name in COARSE_CONCURRENCE_NAMES:
            vmin, vmax, cmap = _heatmap_color_scale(prefix, name)

            anchors = alignment_scan.get("kinematic_points", [])
            if anchors:
                ncols = min(3, len(anchors))
                nrows = int(np.ceil(len(anchors) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(3.9 * ncols, 3.45 * nrows),
                )
                subplot_top = 0.78 if nrows == 1 else 0.86
                fig.subplots_adjust(wspace=0.01, hspace=0.0, top=subplot_top)
                axes_flat = np.atleast_1d(axes).ravel()
                anchor_meshes = []
                label = observable_latex_label(name)
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
                    x_values = np.asarray([_scan_x_phi(row) for row in point_rows], dtype=float)
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
                    mesh = _draw_heatmap(
                        ax,
                        x_edges,
                        y_edges,
                        mean_grid,
                        cmap,
                        vmin,
                        vmax,
                        plot_style,
                    )
                    ax.set_box_aspect(1)
                    _add_pi_over_two_reference_lines(ax)
                    anchor_meshes.append(mesh)
                    best = max(point_rows, key=lambda row: row[f"{prefix}_{name}"])
                    ax.set_title(
                        f"$s={anchor['s']:.3g}\\,{{\\rm GeV}}^2$, "
                        f"$\\theta_{{\\rm in}}={anchor['theta_in']:.3g}\\,{{\\rm rad}}$\n"
                        f"$E_\\gamma={anchor['qOut']:.3g}\\,{{\\rm GeV}}$, "
                        f"max {label}={best[f'{prefix}_{name}']:.3f}",
                        fontsize=ANCHOR_TITLE_FONTSIZE,
                    )
                    if index // ncols == nrows - 1:
                        ax.set_xlabel(r"$\phi_{P,\rm in}$", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    else:
                        ax.set_xticklabels([])
                    if index % ncols == 0:
                        ax.set_ylabel(r"$\phi_{\gamma}'$", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
                for ax in axes_flat[len(anchors):]:
                    ax.set_axis_off()
                fig.suptitle(
                    f"{title_prefix}: {label} two-angle scans at characteristic kinematics",
                    fontsize=PLOT_SUPTITLE_FONTSIZE,
                    y=0.98,
                )
                if anchor_meshes:
                    colorbar = fig.colorbar(
                        anchor_meshes[-1],
                        ax=axes_flat,
                        label=label,
                        shrink=0.86,
                        pad=0.01,
                    )
                    colorbar.ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
                    colorbar.set_label(label, fontsize=PLOT_COLORBAR_FONTSIZE)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return output_path


def save_concurrence_scan_plots(alignment_scan, plot_style=HEATMAP_PLOT_STYLE):
    """Save selected concurrence scan PDFs for all alignment spin cases."""
    return {
        prefix: save_concurrence_scan_plot(
            alignment_scan,
            prefix,
            label,
            plot_style=plot_style,
        )
        for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES
    }


def concurrence_summary_line(row, key, display_key=None):
    """Return a compact coarse concurrence maximum summary line."""
    if display_key is None:
        display_key = key
    return (
        f"    {display_key}={row[key]:.12g}, s={row['s']:.8g}, "
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
    locator_label = "/".join(observable_text_label(name) for name in COARSE_CONCURRENCE_NAMES)
    lines = [
        f"{locator_label}-focused user-frame phase-space scan",
        "  anchor variables: s, theta_in, qOut",
        "  scanned variables per anchor: phi_in_electron, phi_gamma",
        "  locator observables: "
        f"{', '.join(observable_text_label(name) for name in COARSE_CONCURRENCE_NAMES)}",
        f"  angle cut: theta(e', gamma) <= {alignment_scan['angle_max_deg']:.6g} deg",
        f"  characteristic kinematic anchors: {len(alignment_scan['kinematic_points'])}",
        f"  s anchor range: {min(alignment_scan['s_values']):.6g} to {max(alignment_scan['s_values']):.6g}",
        f"  theta_in anchor range: {min(alignment_scan['theta_in_values']):.6g} to {max(alignment_scan['theta_in_values']):.6g}",
        f"  qOut/Egamma anchor range: {min(alignment_scan['qOut_values']):.6g} to {max(alignment_scan['qOut_values']):.6g}",
        f"  form factor model: {alignment_scan['form_factor_model']} with F1(t), F2(t)",
        f"  initial-spin averaging version: {alignment_scan.get('spin_averaging_version', SPIN_AVERAGING_VERSION)}",
        f"  phi_e_in scan: {len(alignment_scan['phi_in_electron_values'])} values from "
        f"{alignment_scan['phi_in_electron_values'][0]:.6g} to {alignment_scan['phi_in_electron_values'][-1]:.6g}",
        f"  phi_gamma scan: {len(alignment_scan['phiOut_values'])} values from "
        f"{alignment_scan['phiOut_values'][0]:.6g} to {alignment_scan['phiOut_values'][-1]:.6g}",
        "  heatmap x coordinate: phi_p_in",
        "  heatmap guide lines: phi_p_in=pi/2 and phi_gamma=pi/2",
        "  heatmap color scales: 0..1 for nonnegative observables, -1..1 for signed observables",
        f"  valid points: {len(rows)}",
        f"  aligned points: {len(aligned_rows)}",
    ]
    lines.append("  initial-spin ensemble conventions:")
    for prefix, label, spin_case in ALIGNMENT_SPIN_CASES:
        lines.append(
            f"    {label} ({prefix}): {spin_averaging_description(spin_case)}; "
            f"density divisor={initial_spin_average_divisor(spin_case):.0f}; "
            f"entanglement mode={entanglement_mode(spin_case)}"
        )
    if rows:
        min_angle = min(row["theta_e_gamma_deg"] for row in rows)
        max_angle = max(row["theta_e_gamma_deg"] for row in rows)
        lines.append(f"  theta range: {min_angle:.6g} to {max_angle:.6g} deg")
        lines.append("")
        lines.append(f"Top {locator_label} locator points:")
        for observable in COARSE_CONCURRENCE_NAMES:
            observable_label = observable_text_label(observable)
            lines.append(f"  {observable_label}:")
            for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
                key = f"{prefix}_{observable}"
                finite_rows = [row for row in rows if np.isfinite(row.get(key, np.nan))]
                if finite_rows:
                    best = max(finite_rows, key=lambda row: row[key])
                    lines.append(f"    {label}:")
                    lines.append(concurrence_summary_line(best, key, f"{prefix}_{observable_label}"))
    if aligned_rows:
        correlations = [row["unpolarized_h_lambda"] for row in aligned_rows]
        lines.append(
            "  aligned <hOut*lambda> range: "
            f"{min(correlations):.6g} to {max(correlations):.6g}"
        )
    elif rows:
        lines.append(
            "  aligned <hOut*lambda> range: none; no valid points pass the "
            "configured angle cut"
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
        "  saved ranked electron-photon csv: "
        f"{alignment_paths['concurrence_csv']['top_e_gamma_csv']}",
    ])
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
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


def regenerate_concurrence_plots_from_csv(
    csv_path=CONCURRENCE_PHASE_SPACE_CSV,
    plot_style=HEATMAP_PLOT_STYLE,
):
    """Regenerate concurrence plot PDFs from a saved concurrence scan CSV."""
    alignment_scan = load_concurrence_scan_csv(csv_path)
    plots = save_concurrence_scan_plots(alignment_scan, plot_style=plot_style)
    return alignment_scan, plots


def main():
    """Regenerate final electron-photon alignment scan outputs."""
    if REGENERATE_PLOTS_FROM_CSV:
        alignment_scan, plots = regenerate_concurrence_plots_from_csv(REGENERATE_PLOTS_CSV_PATH)
        lines = [
            "Regenerated concurrence plots from saved CSV without recalculating amplitudes.",
            f"  source csv: {alignment_scan['source_csv']}",
            f"  heatmap plot style: {_heatmap_style(HEATMAP_PLOT_STYLE)}",
            "  heatmap x coordinate: phi_p_in",
            "  heatmap guide lines: phi_p_in=pi/2 and phi_gamma=pi/2",
            "  heatmap color scales: 0..1 for nonnegative observables, -1..1 for signed observables",
            f"  rows loaded: {len(alignment_scan['rows'])}",
            f"  characteristic kinematic anchors: {len(alignment_scan['kinematic_points'])}",
        ]
        if alignment_scan["missing_observable_columns"]:
            lines.extend([
                "  missing observable columns in source csv; affected plots use NaN values:",
                "    " + ", ".join(alignment_scan["missing_observable_columns"]),
            ])
        for prefix, plot_path in plots.items():
            lines.append(f"  saved {prefix} plot: {plot_path}")
        print("\n".join(lines))
        return

    clean_alignment_outputs()
    alignment_scan = scan_final_electron_photon_alignment()
    paths = save_alignment_scan_csv_files(alignment_scan)
    paths["concurrence_csv"] = save_concurrence_scan_csv_files(alignment_scan)
    paths["concurrence_plots"] = save_concurrence_scan_plots(alignment_scan)

    log_text = build_alignment_report(alignment_scan, paths) + f"\n\nSaved log: {ALIGNMENT_LOG_PATH}\n"
    ALIGNMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
