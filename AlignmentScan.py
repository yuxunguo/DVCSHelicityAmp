"""Final electron-photon alignment phase-space scan.

This script scans phase space for small opening angle between the outgoing
electron and final real photon, then evaluates electron-photon spin
correlation observables for the aligned points. It writes CSV files and
separate unpolarized/polarized heatmap PDFs under
``Output/SpinDensityMat/AlignmentScan``.
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
    F1,
    F2,
    M,
    NORMALIZE_TRACE,
    OUTPUT_DIR,
    PHI_VALUES,
    SPIN_CASE_POLARIZED,
    SPIN_CASE_UNPOLARIZED,
    amplitude_table,
    density_matrix_from_amplitudes,
    entanglement_measures_from_amplitudes,
    outgoing_spin_states,
    polarized_entanglement_difference,
    trace_value,
)


PHASE_SPACE_Q2_VALUES = np.linspace(0.5, 6.0, 12)
PHASE_SPACE_XB_VALUES = np.linspace(0.10, 0.60, 6)
PHASE_SPACE_T_VALUES = np.linspace(-2.5, -0.1, 13)
PHASE_SPACE_PHI_VALUES = PHI_VALUES
ALIGNMENT_ANGLE_MAX_DEG = 10.0
ALIGNMENT_ANGLE_MAX_RAD = np.deg2rad(ALIGNMENT_ANGLE_MAX_DEG)

ALIGNMENT_OUTPUT_DIR = OUTPUT_DIR / "AlignmentScan"
ALIGNMENT_LOG_PATH = ALIGNMENT_OUTPUT_DIR / "AlignmentScan.log"


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
    """Scan phase space for final electron-photon spin correlation near alignment."""
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
                "unpolarized_trace": np.nan,
                "unpolarized_h_out_mean": np.nan,
                "unpolarized_lambda_mean": np.nan,
                "unpolarized_h_lambda": np.nan,
                "unpolarized_h_lambda_connected": np.nan,
                "unpolarized_C13": np.nan,
                "polarized_trace": np.nan,
                "polarized_spin_signal_M2": np.nan,
                "polarized_h_lambda": np.nan,
                "polarized_delta_C13": np.nan,
            }
            if not aligned:
                rows.append(row)
                continue

            amplitudes = amplitude_table(momenta, kin["m"], F1, F2)
            rho_unpolarized, _unpolarized_signal, squared_amplitude = (
                density_matrix_from_amplitudes(
                    amplitudes,
                    average_initial=AVERAGE_INITIAL_SPINS,
                    spin_case=SPIN_CASE_UNPOLARIZED,
                )
            )
            rho_polarized, polarized_signal, _squared_amplitude_check = (
                density_matrix_from_amplitudes(
                    amplitudes,
                    average_initial=AVERAGE_INITIAL_SPINS,
                    spin_case=SPIN_CASE_POLARIZED,
                )
            )
            if normalize_trace:
                if squared_amplitude <= 1e-14:
                    raise ZeroDivisionError("Cannot normalize a zero alignment matrix.")
                rho_unpolarized = rho_unpolarized / squared_amplitude
                rho_polarized = rho_polarized / squared_amplitude

            unpolarized_corr = final_electron_photon_spin_correlations(
                rho_unpolarized,
                out_states,
            )
            polarized_corr = final_electron_photon_spin_correlations(
                rho_polarized,
                out_states,
            )
            unpolarized_entanglement = entanglement_measures_from_amplitudes(
                amplitudes,
                entanglement_initial_state,
            )
            polarized_entanglement = polarized_entanglement_difference(
                amplitudes,
                entanglement_initial_state[1],
            )
        except Exception as exc:
            failures.append((Q2, xB, t, phi, str(exc)))
            continue

        row.update({
            "squared_amplitude_M2": squared_amplitude,
            "unpolarized_trace": trace_value(rho_unpolarized),
            "unpolarized_h_out_mean": unpolarized_corr["h_out_mean"],
            "unpolarized_lambda_mean": unpolarized_corr["lambda_mean"],
            "unpolarized_h_lambda": unpolarized_corr["h_lambda"],
            "unpolarized_h_lambda_connected": unpolarized_corr["h_lambda_connected"],
            "unpolarized_C13": unpolarized_entanglement["C13"],
            "polarized_trace": trace_value(rho_polarized),
            "polarized_spin_signal_M2": polarized_signal,
            "polarized_h_lambda": polarized_corr["h_lambda"],
            "polarized_delta_C13": polarized_entanglement["C13"],
        })
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
    }


def _alignment_csv_headers():
    """Return CSV headers for the final electron-photon alignment scan."""
    return [
        "Q2",
        "xB",
        "t",
        "phi",
        "theta_e_gamma_rad",
        "theta_e_gamma_deg",
        "aligned",
        "squared_amplitude_M2",
        "unpolarized_trace",
        "unpolarized_h_out_mean",
        "unpolarized_lambda_mean",
        "unpolarized_h_lambda",
        "unpolarized_h_lambda_connected",
        "unpolarized_C13",
        "polarized_trace",
        "polarized_spin_signal_M2",
        "polarized_h_lambda",
        "polarized_delta_C13",
    ]


def _alignment_csv_row(row):
    """Return one formatted CSV row for the alignment scan."""
    return [
        f"{row['Q2']:.16e}",
        f"{row['xB']:.16e}",
        f"{row['t']:.16e}",
        f"{row['phi']:.16e}",
        f"{row['theta_e_gamma_rad']:.16e}",
        f"{row['theta_e_gamma_deg']:.16e}",
        row["aligned"],
        f"{row['squared_amplitude_M2']:.16e}",
        f"{row['unpolarized_trace']:.16e}",
        f"{row['unpolarized_h_out_mean']:.16e}",
        f"{row['unpolarized_lambda_mean']:.16e}",
        f"{row['unpolarized_h_lambda']:.16e}",
        f"{row['unpolarized_h_lambda_connected']:.16e}",
        f"{row['unpolarized_C13']:.16e}",
        f"{row['polarized_trace']:.16e}",
        f"{row['polarized_spin_signal_M2']:.16e}",
        f"{row['polarized_h_lambda']:.16e}",
        f"{row['polarized_delta_C13']:.16e}",
    ]


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


def _save_alignment_heatmap_pdf(alignment_scan, observable_specs, output_path, title_prefix):
    """Save one alignment heatmap PDF for a list of observables."""
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
    aligned_mask = np.asarray([row["aligned"] for row in rows], dtype=bool)
    kinematic_specs = (
        (np.asarray([row["Q2"] for row in rows]), r"$Q^2$ [GeV$^2$]"),
        (np.asarray([row["xB"] for row in rows]), r"$x_B$"),
        (np.asarray([row["t"] for row in rows]), r"$t$ [GeV$^2$]"),
        (np.asarray([row["phi"] for row in rows]), r"$\phi$ [rad]"),
    )
    angle_cut = alignment_scan["angle_max_deg"]
    theta_edges = np.linspace(0.0, angle_cut, 16)

    with PdfPages(output_path) as pdf:
        for key, z_label, title, cmap, vmin, vmax in observable_specs:
            values = np.asarray([row[key] for row in rows])
            heatmap_mask = aligned_mask & np.isfinite(values)
            for variable, y_label in kinematic_specs:
                y_edges = _bin_edges_from_values(variable[heatmap_mask])
                mean_grid, count_grid = _binned_mean_2d(
                    theta[heatmap_mask],
                    variable[heatmap_mask],
                    values[heatmap_mask],
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
                        s=12 + 6 * count_grid[occupied_y, occupied_x],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                    )
                ax.set_xlabel(r"$\theta(e', \gamma)$ [deg]")
                ax.set_ylabel(y_label)
                ax.set_xlim(0.0, angle_cut)
                ax.set_title(f"{title_prefix}: {title}")
                fig.colorbar(mesh, ax=ax, label=z_label)
                pdf.savefig(fig)
                plt.close(fig)
    return output_path


def save_unpolarized_alignment_plot(
    alignment_scan,
    output_path=ALIGNMENT_OUTPUT_DIR / "electron_photon_spin_correlation_unpolarized.pdf",
):
    """Save unpolarized alignment heatmaps."""
    observable_specs = (
        (
            "unpolarized_h_lambda",
            r"$\langle h_{\mathrm{out}}\lambda\rangle$",
            "mean final electron-photon helicity correlation",
            "coolwarm",
            -1.0,
            1.0,
        ),
        (
            "unpolarized_h_lambda_connected",
            r"$\langle h_{\mathrm{out}}\lambda\rangle - \langle h_{\mathrm{out}}\rangle\langle\lambda\rangle$",
            "mean connected helicity correlation",
            "coolwarm",
            -1.0,
            1.0,
        ),
        (
            "unpolarized_C13",
            r"$C_{13}$",
            "mean electron-photon concurrence channel",
            "viridis",
            0.0,
            1.0,
        ),
    )
    return _save_alignment_heatmap_pdf(
        alignment_scan,
        observable_specs,
        output_path,
        "Unpolarized",
    )


def save_polarized_alignment_plot(
    alignment_scan,
    output_path=ALIGNMENT_OUTPUT_DIR / "electron_photon_spin_correlation_polarized.pdf",
):
    """Save polarized alignment heatmaps."""
    observable_specs = (
        (
            "polarized_h_lambda",
            r"$\Delta_h\langle h_{\mathrm{out}}\lambda\rangle$",
            "mean polarized helicity-difference correlation",
            "coolwarm",
            -1.0,
            1.0,
        ),
        (
            "polarized_delta_C13",
            r"$\Delta_h C_{13}$",
            "mean polarized electron-photon concurrence difference",
            "coolwarm",
            -1.0,
            1.0,
        ),
    )
    return _save_alignment_heatmap_pdf(
        alignment_scan,
        observable_specs,
        output_path,
        "Polarized",
    )


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
        f"  saved unpolarized plots: {alignment_paths['unpolarized_plot']}",
        f"  saved polarized plots: {alignment_paths['polarized_plot']}",
    ])
    if alignment_scan["failures"]:
        lines.append(f"  invalid phase-space points: {len(alignment_scan['failures'])}")
    return "\n".join(lines)


def main():
    """Regenerate final electron-photon alignment scan outputs."""
    clean_alignment_outputs()
    alignment_scan = scan_final_electron_photon_alignment()
    paths = save_alignment_scan_csv_files(alignment_scan)
    paths["unpolarized_plot"] = save_unpolarized_alignment_plot(alignment_scan)
    paths["polarized_plot"] = save_polarized_alignment_plot(alignment_scan)

    log_text = build_alignment_report(alignment_scan, paths) + f"\n\nSaved log: {ALIGNMENT_LOG_PATH}\n"
    ALIGNMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
