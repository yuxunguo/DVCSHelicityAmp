from itertools import product
import os
from pathlib import Path

import numpy as np

from Algebra import HELICITIES, photon_pol
from BHHelicityAmp import bh_amplitude_core
from Kinematics import kinematics_user_from_scalar_inputs


# ============================================================
# Scan and plot settings
# ============================================================

EB = 5.0
XB = 0.36
PHI = 0.7
M = 0.938
F1 = 1.0
F2 = 0.0
AZIMUTH_INPUT = "phi_hadron"

Q2_VALUES = np.linspace(1.0, 3.0, 9)
T_VALUES = np.linspace(-0.8, -0.2, 9)

AVERAGE_INITIAL_SPINS = False
NORMALIZE_TRACE = False

OUTPUT_DIR = Path("Output") / "SpinDensityMat"


def outgoing_spin_states():
    """Return final-state labels (hOut, sOut, lambda)."""
    return tuple(product(HELICITIES, repeat=3))


def initial_spin_states():
    """Return initial-state labels (hIn, sIn)."""
    return tuple(product(HELICITIES, repeat=2))


def spin_label(state):
    h_out, s_out, lam = state
    return f"h'={h_out:+d}, s'={s_out:+d}, lam={lam:+d}"


def amplitude_table(mom, m, F1, F2):
    """
    Return A[in_state, out_state] for BH amplitudes.

    The outgoing basis is (hOut, sOut, lambda), so the final spin-density
    matrix is an 8x8 matrix in this basis.
    """
    in_states = initial_spin_states()
    out_states = outgoing_spin_states()
    photon_pols = {lam: photon_pol(mom["qout"], lam) for lam in HELICITIES}
    amplitudes = np.zeros((len(in_states), len(out_states)), dtype=complex)

    for in_index, (h_in, s_in) in enumerate(in_states):
        for out_index, (h_out, s_out, lam) in enumerate(out_states):
            amplitudes[in_index, out_index] = bh_amplitude_core(
                mom["k"], mom["kp"], mom["qout"],
                mom["p"], mom["pp"],
                photon_pols[lam],
                h_in, h_out,
                s_in, s_out,
                m, F1, F2,
            )
    return amplitudes


def spin_density_matrix_from_momenta(
    mom,
    m,
    F1,
    F2,
    average_initial=False,
    normalize_trace=False,
):
    """
    Build rho[out_i,out_j] = sum_initial A_i conj(A_j).

    If average_initial is true, rho is divided by the number of incoming
    electron/proton spin states. If normalize_trace is true, rho is divided by
    Tr(rho) after the initial-spin sum or average.
    """
    amplitudes = amplitude_table(mom, m, F1, F2)
    rho = amplitudes.T @ np.conjugate(amplitudes)
    if average_initial:
        rho /= amplitudes.shape[0]

    trace = np.trace(rho)
    if normalize_trace:
        if abs(trace) <= 1e-14:
            raise ZeroDivisionError("Cannot trace-normalize a zero density matrix.")
        rho /= trace
    return rho


def spin_density_matrix_from_scalars(
    Eb,
    Q2,
    xB,
    t,
    phi,
    m,
    F1,
    F2,
    azimuth_input=AZIMUTH_INPUT,
    average_initial=False,
    normalize_trace=False,
):
    kin = kinematics_user_from_scalar_inputs(
        Eb, Q2, xB, t, phi, m,
        azimuth_input=azimuth_input,
        label=f"Q2={Q2:.6g}, t={t:.6g}",
    )
    rho = spin_density_matrix_from_momenta(
        kin["momenta"],
        kin["m"],
        F1,
        F2,
        average_initial=average_initial,
        normalize_trace=normalize_trace,
    )
    return rho, kin


def scan_spin_density_grid(
    Q2_values,
    t_values,
    Eb=EB,
    xB=XB,
    phi=PHI,
    m=M,
    F1=F1,
    F2=F2,
    azimuth_input=AZIMUTH_INPUT,
    average_initial=AVERAGE_INITIAL_SPINS,
    normalize_trace=NORMALIZE_TRACE,
):
    out_states = outgoing_spin_states()
    rho_grid = np.full(
        (len(t_values), len(Q2_values), len(out_states), len(out_states)),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    valid = np.zeros((len(t_values), len(Q2_values)), dtype=bool)
    failures = []

    for t_index, t in enumerate(t_values):
        for Q2_index, Q2 in enumerate(Q2_values):
            try:
                rho, _kin = spin_density_matrix_from_scalars(
                    Eb, Q2, xB, t, phi, m, F1, F2,
                    azimuth_input=azimuth_input,
                    average_initial=average_initial,
                    normalize_trace=normalize_trace,
                )
            except Exception as exc:
                failures.append((Q2, t, str(exc)))
                continue
            rho_grid[t_index, Q2_index] = rho
            valid[t_index, Q2_index] = True

    return {
        "rho": rho_grid,
        "valid": valid,
        "failures": failures,
        "Q2_values": np.asarray(Q2_values, dtype=float),
        "t_values": np.asarray(t_values, dtype=float),
        "out_states": out_states,
    }


def _require_matplotlib():
    cache_dir = Path("/tmp") / "dvcs_helicity_amp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def _plot_coefficient_page(
    ax,
    data,
    Q2_values,
    t_values,
    title,
    cmap,
    vmin=None,
    vmax=None,
):
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=[Q2_values[0], Q2_values[-1], t_values[0], t_values[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(r"$Q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$t$ [GeV$^2$]")
    return image


def save_coefficient_plots(scan, output_dir=OUTPUT_DIR):
    plt, PdfPages = _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    rho = scan["rho"]
    Q2_values = scan["Q2_values"]
    t_values = scan["t_values"]
    out_states = scan["out_states"]
    norm_path = output_dir / "spin_density_norm_by_coefficient.pdf"
    phase_path = output_dir / "spin_density_phase_by_coefficient.pdf"

    with PdfPages(norm_path) as norm_pdf, PdfPages(phase_path) as phase_pdf:
        for row, row_state in enumerate(out_states):
            for col, col_state in enumerate(out_states):
                coefficient = rho[:, :, row, col]
                label = (
                    rf"$\rho_{{{row}{col}}}$: "
                    f"{spin_label(row_state)} -> {spin_label(col_state)}"
                )

                fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
                image = _plot_coefficient_page(
                    ax,
                    np.abs(coefficient),
                    Q2_values,
                    t_values,
                    label,
                    cmap="viridis",
                )
                fig.colorbar(image, ax=ax, label=r"$|\rho_{ij}|$")
                norm_pdf.savefig(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
                image = _plot_coefficient_page(
                    ax,
                    np.angle(coefficient),
                    Q2_values,
                    t_values,
                    label,
                    cmap="twilight",
                    vmin=-np.pi,
                    vmax=np.pi,
                )
                fig.colorbar(image, ax=ax, label=r"$\arg(\rho_{ij})$ [rad]")
                phase_pdf.savefig(fig)
                plt.close(fig)

    return norm_path, phase_path


def save_scan_npz(scan, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "spin_density_scan.npz"
    np.savez(
        path,
        rho=scan["rho"],
        valid=scan["valid"],
        Q2_values=scan["Q2_values"],
        t_values=scan["t_values"],
        out_states=np.asarray(scan["out_states"], dtype=int),
    )
    return path


def main():
    scan = scan_spin_density_grid(Q2_VALUES, T_VALUES)
    npz_path = save_scan_npz(scan)
    norm_path, phase_path = save_coefficient_plots(scan)

    print("Spin-density matrix scan")
    print(f"  outgoing basis size: {len(scan['out_states'])}")
    print(f"  coefficients plotted: {len(scan['out_states']) ** 2}")
    print(f"  Q2 grid: {Q2_VALUES[0]:.6g} to {Q2_VALUES[-1]:.6g}")
    print(f"  t grid: {T_VALUES[0]:.6g} to {T_VALUES[-1]:.6g}")
    print(f"  valid points: {int(scan['valid'].sum())}/{scan['valid'].size}")
    print(f"  initial spins averaged: {AVERAGE_INITIAL_SPINS}")
    print(f"  trace normalized: {NORMALIZE_TRACE}")
    print(f"  saved data: {npz_path}")
    print(f"  saved norm plots: {norm_path}")
    print(f"  saved phase plots: {phase_path}")
    if scan["failures"]:
        print("  invalid grid points:")
        for Q2, t, message in scan["failures"]:
            print(f"    Q2={Q2:.8g}, t={t:.8g}: {message}")


if __name__ == "__main__":
    main()
