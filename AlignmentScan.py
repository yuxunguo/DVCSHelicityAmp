"""Final electron-photon alignment phase-space scan.

This script scans phase space for small opening angle between the outgoing
electron and final real photon, then evaluates electron-photon spin
correlation observables and reduced 4x4 electron-photon matrices for all valid
points. It writes CSV files plus amplitude, density-matrix, and concurrence
heatmap PDFs under ``Output/AlignmentScan``.
"""

from itertools import product
import csv
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from Kinematics import kinematics_user_from_scalar_inputs
from SpinDensityMat import (
    AVERAGE_INITIAL_SPINS,
    AZIMUTH_INPUT,
    EB,
    ENTANGLEMENT_INITIAL_STATE,
    ENTANGLEMENT_NAMES,
    F1,
    F2,
    M,
    NORMALIZE_TRACE,
    OUTPUT_DIR,
    PHI_VALUES,
    SPIN_CASE_POLARIZED,
    SPIN_CASE_TRANSVERSE,
    SPIN_CASE_UNPOLARIZED,
    amplitude_table,
    density_matrix_from_amplitudes,
    entanglement_measures_from_amplitudes,
    initial_spin_states,
    outgoing_spin_states,
    polarized_entanglement_difference,
    trace_value,
    transverse_entanglement_measures,
)


PHASE_SPACE_Q2_VALUES = np.linspace(0.5, 6.0, 12)
PHASE_SPACE_XB_VALUES = np.linspace(0.10, 0.60, 6)
PHASE_SPACE_T_VALUES = np.linspace(-2.5, -0.1, 13)
PHASE_SPACE_PHI_VALUES = PHI_VALUES
ALIGNMENT_ANGLE_MAX_DEG = 10.0
ALIGNMENT_ANGLE_MAX_RAD = np.deg2rad(ALIGNMENT_ANGLE_MAX_DEG)

OUTPUT_ROOT = OUTPUT_DIR.parent
LEGACY_ALIGNMENT_OUTPUT_DIR = OUTPUT_DIR / "AlignmentScan"
LEGACY_ALIGNMENT_LOG_PATH = OUTPUT_ROOT / "AlignmentScan.log"
ALIGNMENT_OUTPUT_DIR = OUTPUT_ROOT / "AlignmentScan"
ALIGNMENT_LOG_PATH = ALIGNMENT_OUTPUT_DIR / "AlignmentScan.log"
DENSITY_MATRIX_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "DensityMatScan"
CONCURRENCE_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "ConcurrenceScan"
AMPLITUDE_OUTPUT_DIR = ALIGNMENT_OUTPUT_DIR / "AmplitudeScan"
REDUCED_EP_BASIS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALIGNMENT_SPIN_CASES = (
    ("unpolarized", "Unpolarized", SPIN_CASE_UNPOLARIZED),
    ("longitudinal_polarized", "Longitudinal polarized", SPIN_CASE_POLARIZED),
    ("transverse_polarized", "Transverse polarized", SPIN_CASE_TRANSVERSE),
)


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


def spatial_opening_angle(a, b):
    """Return the spatial opening angle between two four-vectors in radians."""
    a3 = np.asarray(a, dtype=float)[1:4]
    b3 = np.asarray(b, dtype=float)[1:4]
    denominator = np.linalg.norm(a3) * np.linalg.norm(b3)
    if denominator <= 1e-14:
        raise ZeroDivisionError("Cannot compute an opening angle with zero momentum.")
    cosine = np.dot(a3, b3) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


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
    elif spin_case == SPIN_CASE_TRANSVERSE:
        amplitude_row = (
            amplitudes[in_states.index((+1, proton_spin))]
            + amplitudes[in_states.index((-1, proton_spin))]
        ) / np.sqrt(2.0)
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
    if spin_case == SPIN_CASE_TRANSVERSE:
        return transverse_entanglement_measures(amplitudes, initial_state[1])
    raise ValueError(f"Unknown alignment spin case: {spin_case}")


def scan_final_electron_photon_alignment(
    Q2_values=PHASE_SPACE_Q2_VALUES,
    xB_values=PHASE_SPACE_XB_VALUES,
    t_values=PHASE_SPACE_T_VALUES,
    phi_values=PHASE_SPACE_PHI_VALUES,
    angle_max_rad=ALIGNMENT_ANGLE_MAX_RAD,
    Eb=EB,
    m=M,
    F1=F1,
    F2=F2,
    azimuth_input=AZIMUTH_INPUT,
    normalize_trace=NORMALIZE_TRACE,
    entanglement_initial_state=ENTANGLEMENT_INITIAL_STATE,
):
    """Scan phase space for final electron-photon spin data and alignment."""
    rows = []
    failures = []
    out_states = outgoing_spin_states()

    for Q2, xB, t, phi in product(Q2_values, xB_values, t_values, phi_values):
        try:
            kin = kinematics_user_from_scalar_inputs(
                Eb,
                Q2,
                xB,
                t,
                phi,
                m,
                azimuth_input=azimuth_input,
                label=f"alignment Q2={Q2:.6g}, xB={xB:.6g}, t={t:.6g}",
            )
            momenta = kin["momenta"]
            angle_rad = spatial_opening_angle(momenta["kp"], momenta["qout"])
            aligned = angle_rad <= angle_max_rad
            row = {
                "Q2": float(Q2),
                "xB": float(xB),
                "t": float(t),
                "phi": float(phi),
                "theta_e_gamma_rad": angle_rad,
                "theta_e_gamma_deg": float(np.degrees(angle_rad)),
                "aligned": aligned,
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

            amplitudes = amplitude_table(momenta, kin["m"], F1, F2)
            rho_unpolarized, unpolarized_signal, squared_amplitude = (
                density_matrix_from_amplitudes(
                    amplitudes,
                    average_initial=AVERAGE_INITIAL_SPINS,
                    spin_case=SPIN_CASE_UNPOLARIZED,
                )
            )
            if normalize_trace:
                if squared_amplitude <= 1e-14:
                    raise ZeroDivisionError("Cannot normalize a zero alignment matrix.")
            row["amplitude_normalization_sqrt_M2"] = np.sqrt(squared_amplitude)

            for prefix, _label, spin_case in ALIGNMENT_SPIN_CASES:
                amplitude_ep = electron_photon_amplitude_matrix(
                    amplitudes,
                    spin_case,
                    entanglement_initial_state,
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
                        density_matrix_from_amplitudes(
                            amplitudes,
                            average_initial=AVERAGE_INITIAL_SPINS,
                            spin_case=spin_case,
                        )
                    )
                if normalize_trace:
                    rho = rho / squared_amplitude
                corr = final_electron_photon_spin_correlations(rho, out_states)
                concurrence = alignment_concurrence_measures(
                    amplitudes,
                    spin_case,
                    entanglement_initial_state,
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
            failures.append((Q2, xB, t, phi, str(exc)))
            continue

        row["squared_amplitude_M2"] = squared_amplitude
        rows.append(row)

    return {
        "rows": rows,
        "failures": failures,
        "angle_max_rad": angle_max_rad,
        "angle_max_deg": float(np.degrees(angle_max_rad)),
        "Q2_values": np.asarray(Q2_values, dtype=float),
        "xB_values": np.asarray(xB_values, dtype=float),
        "t_values": np.asarray(t_values, dtype=float),
        "phi_values": np.asarray(phi_values, dtype=float),
        "Eb": Eb,
        "m": m,
        "normalized_by_squared_amplitude": normalize_trace,
        "entanglement_initial_state": entanglement_initial_state,
        "spin_cases": ALIGNMENT_SPIN_CASES,
    }


def _alignment_csv_headers():
    """Return CSV headers for the final electron-photon alignment scan."""
    headers = [
        "Q2",
        "xB",
        "t",
        "phi",
        "theta_e_gamma_rad",
        "theta_e_gamma_deg",
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


def _alignment_csv_row(row):
    """Return one formatted CSV row for the alignment scan."""
    values = [
        f"{row['Q2']:.16e}",
        f"{row['xB']:.16e}",
        f"{row['t']:.16e}",
        f"{row['phi']:.16e}",
        f"{row['theta_e_gamma_rad']:.16e}",
        f"{row['theta_e_gamma_deg']:.16e}",
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
    headers = [
        "Q2",
        "xB",
        "t",
        "phi",
        "theta_e_gamma_rad",
        "theta_e_gamma_deg",
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
    values = [
        f"{row['Q2']:.16e}",
        f"{row['xB']:.16e}",
        f"{row['t']:.16e}",
        f"{row['phi']:.16e}",
        f"{row['theta_e_gamma_rad']:.16e}",
        f"{row['theta_e_gamma_deg']:.16e}",
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
    headers = [
        "Q2",
        "xB",
        "t",
        "phi",
        "theta_e_gamma_rad",
        "theta_e_gamma_deg",
        "aligned",
        "squared_amplitude_M2",
        "amplitude_normalization_sqrt_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend(_amplitude_matrix_headers(prefix))
    return headers


def _amplitude_csv_row(row):
    """Return one formatted CSV row for the reduced amplitude scan."""
    values = [
        f"{row['Q2']:.16e}",
        f"{row['xB']:.16e}",
        f"{row['t']:.16e}",
        f"{row['phi']:.16e}",
        f"{row['theta_e_gamma_rad']:.16e}",
        f"{row['theta_e_gamma_deg']:.16e}",
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

    theta = np.asarray([row["theta_e_gamma_deg"] for row in rows])
    kinematic_specs = (
        (np.asarray([row["Q2"] for row in rows]), r"$Q^2$ [GeV$^2$]"),
        (np.asarray([row["xB"] for row in rows]), r"$x_B$"),
        (np.asarray([row["t"] for row in rows]), r"$t$ [GeV$^2$]"),
        (np.asarray([row["phi"] for row in rows]), r"$\phi$ [rad]"),
    )
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

    theta_edges = np.linspace(0.0, 180.0, 19)

    with PdfPages(output_path) as pdf:
        for variable, y_label in kinematic_specs:
            base_mask = np.isfinite(theta) & np.isfinite(variable)
            y_edges = _bin_edges_from_values(variable[base_mask])
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
                    theta[finite_mask],
                    variable[finite_mask],
                    values[finite_mask],
                    theta_edges,
                    y_edges,
                )
                mesh = ax.pcolormesh(
                    theta_edges,
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
                ax.set_xlabel(r"$\theta(e', \gamma)$ [deg]")
                ax.set_ylabel(y_label)
                ax.set_xlim(0.0, 180.0)
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

    theta = np.asarray([row["theta_e_gamma_deg"] for row in rows])
    kinematic_specs = (
        (np.asarray([row["Q2"] for row in rows]), r"$Q^2$ [GeV$^2$]"),
        (np.asarray([row["xB"] for row in rows]), r"$x_B$"),
        (np.asarray([row["t"] for row in rows]), r"$t$ [GeV$^2$]"),
        (np.asarray([row["phi"] for row in rows]), r"$\phi$ [rad]"),
    )
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

    theta_edges = np.linspace(0.0, 180.0, 19)

    with PdfPages(output_path) as pdf:
        for variable, y_label in kinematic_specs:
            base_mask = np.isfinite(theta) & np.isfinite(variable)
            y_edges = _bin_edges_from_values(variable[base_mask])
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
                    theta[finite_mask],
                    variable[finite_mask],
                    values[finite_mask],
                    theta_edges,
                    y_edges,
                )
                mesh = ax.pcolormesh(
                    theta_edges,
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
                ax.set_xlabel(r"$\theta(e', \gamma)$ [deg]")
                ax.set_ylabel(y_label)
                ax.set_xlim(0.0, 180.0)
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
    """Return CSV headers for the alignment concurrence scan."""
    headers = [
        "Q2",
        "xB",
        "t",
        "phi",
        "theta_e_gamma_rad",
        "theta_e_gamma_deg",
        "aligned",
        "squared_amplitude_M2",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend(f"{prefix}_{name}" for name in ENTANGLEMENT_NAMES)
    return headers


def _concurrence_csv_row(row):
    """Return one formatted CSV row for the alignment concurrence scan."""
    values = [
        f"{row['Q2']:.16e}",
        f"{row['xB']:.16e}",
        f"{row['t']:.16e}",
        f"{row['phi']:.16e}",
        f"{row['theta_e_gamma_rad']:.16e}",
        f"{row['theta_e_gamma_deg']:.16e}",
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
    ]
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        values.extend(f"{row[f'{prefix}_{name}']:.16e}" for name in ENTANGLEMENT_NAMES)
    return values


def save_concurrence_scan_csv_files(
    alignment_scan,
    output_dir=CONCURRENCE_OUTPUT_DIR,
):
    """Save full and aligned-only concurrence scan CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_csv": output_dir / "electron_photon_concurrence_phase_space.csv",
        "aligned_csv": output_dir / "electron_photon_concurrence_aligned.csv",
    }
    headers = _concurrence_csv_headers()
    aligned_rows = [row for row in alignment_scan["rows"] if row["aligned"]]
    for key, rows in (("all_csv", alignment_scan["rows"]), ("aligned_csv", aligned_rows)):
        with paths[key].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_concurrence_csv_row(row))
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

    theta = np.asarray([row["theta_e_gamma_deg"] for row in rows])
    kinematic_specs = (
        (np.asarray([row["Q2"] for row in rows]), r"$Q^2$ [GeV$^2$]"),
        (np.asarray([row["xB"] for row in rows]), r"$x_B$"),
        (np.asarray([row["t"] for row in rows]), r"$t$ [GeV$^2$]"),
        (np.asarray([row["phi"] for row in rows]), r"$\phi$ [rad]"),
    )
    theta_edges = np.linspace(0.0, 180.0, 19)

    with PdfPages(output_path) as pdf:
        for name in ENTANGLEMENT_NAMES:
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

            for variable, y_label in kinematic_specs:
                finite_mask = (
                    np.isfinite(theta)
                    & np.isfinite(variable)
                    & np.isfinite(values)
                )
                y_edges = _bin_edges_from_values(variable[finite_mask])
                mean_grid, count_grid = _binned_mean_2d(
                    theta[finite_mask],
                    variable[finite_mask],
                    values[finite_mask],
                    theta_edges,
                    y_edges,
                )
                fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
                mesh = ax.pcolormesh(
                    theta_edges,
                    y_edges,
                    mean_grid,
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                occupied_y, occupied_x = np.nonzero(count_grid > 0)
                if occupied_x.size:
                    x_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
                    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                    ax.scatter(
                        x_centers[occupied_x],
                        y_centers[occupied_y],
                        s=10 + 4 * count_grid[occupied_y, occupied_x],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.6,
                    )
                ax.set_xlabel(r"$\theta(e', \gamma)$ [deg]")
                ax.set_ylabel(y_label)
                ax.set_xlim(0.0, 180.0)
                ax.set_title(f"{title_prefix}: {name}")
                fig.colorbar(mesh, ax=ax, label=name)
                pdf.savefig(fig)
                plt.close(fig)
    return output_path


def save_concurrence_scan_plots(alignment_scan):
    """Save concurrence scan PDFs for all alignment spin cases."""
    return {
        prefix: save_concurrence_scan_plot(alignment_scan, prefix, label)
        for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES
    }


def build_alignment_report(alignment_scan, alignment_paths):
    """Build a text report for the final electron-photon alignment scan."""
    rows = alignment_scan["rows"]
    aligned_rows = [row for row in rows if row["aligned"]]
    lines = [
        "Final electron-photon alignment phase-space scan",
        f"  angle cut: theta(e', gamma) <= {alignment_scan['angle_max_deg']:.6g} deg",
        f"  Q2 grid: {alignment_scan['Q2_values'][0]:.6g} to {alignment_scan['Q2_values'][-1]:.6g}",
        f"  xB grid: {alignment_scan['xB_values'][0]:.6g} to {alignment_scan['xB_values'][-1]:.6g}",
        f"  t grid: {alignment_scan['t_values'][0]:.6g} to {alignment_scan['t_values'][-1]:.6g}",
        f"  phi grid: {alignment_scan['phi_values'][0]:.6g} to {alignment_scan['phi_values'][-1]:.6g}",
        f"  valid points: {len(rows)}",
        f"  aligned points: {len(aligned_rows)}",
        "  amplitude normalization: M / sqrt(M^2_unpol)",
    ]
    if rows:
        min_angle = min(row["theta_e_gamma_deg"] for row in rows)
        max_angle = max(row["theta_e_gamma_deg"] for row in rows)
        lines.append(f"  theta range: {min_angle:.6g} to {max_angle:.6g} deg")
    if aligned_rows:
        correlations = [row["unpolarized_h_lambda"] for row in aligned_rows]
        lines.append(
            "  aligned <hOut*lambda> range: "
            f"{min(correlations):.6g} to {max(correlations):.6g}"
        )
    lines.extend([
        f"  saved full csv: {alignment_paths['all_csv']}",
        f"  saved aligned csv: {alignment_paths['aligned_csv']}",
        "  saved density matrix full csv: "
        f"{alignment_paths['density_matrix_csv']['all_csv']}",
        "  saved density matrix aligned csv: "
        f"{alignment_paths['density_matrix_csv']['aligned_csv']}",
        "  saved amplitude full csv: "
        f"{alignment_paths['amplitude_csv']['all_csv']}",
        "  saved amplitude aligned csv: "
        f"{alignment_paths['amplitude_csv']['aligned_csv']}",
        "  saved concurrence full csv: "
        f"{alignment_paths['concurrence_csv']['all_csv']}",
        "  saved concurrence aligned csv: "
        f"{alignment_paths['concurrence_csv']['aligned_csv']}",
    ])
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        amplitude_paths = alignment_paths[f"{prefix}_amplitude_plots"]
        lines.append(
            f"  saved {label.lower()} amplitude magnitude plot: "
            f"{amplitude_paths['abs']}"
        )
        lines.append(
            f"  saved {label.lower()} amplitude phase plot: "
            f"{amplitude_paths['phase']}"
        )
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
    return "\n".join(lines)


def main():
    """Regenerate final electron-photon alignment scan outputs."""
    clean_alignment_outputs()
    alignment_scan = scan_final_electron_photon_alignment()
    paths = save_alignment_scan_csv_files(alignment_scan)
    paths["density_matrix_csv"] = save_density_matrix_scan_csv_files(alignment_scan)
    paths["amplitude_csv"] = save_amplitude_scan_csv_files(alignment_scan)
    paths["concurrence_csv"] = save_concurrence_scan_csv_files(alignment_scan)
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        paths[f"{prefix}_amplitude_plots"] = (
            save_amplitude_scan_plots(alignment_scan, prefix, label)
        )
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
