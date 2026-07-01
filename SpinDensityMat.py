"""Spin-density matrix scans and three-qubit entanglement observables.

The outgoing spin basis is ordered as ``(hOut, sOut, lambda)``:

* particle 1 is the outgoing electron helicity ``hOut``;
* particle 2 is the outgoing proton spin/helicity ``sOut``;
* particle 3 is the outgoing real-photon helicity ``lambda``.

For each scalar kinematic point, this module builds the ``4 x 8`` table of
Bethe-Heitler amplitudes over incoming and outgoing spin labels, converts it
into an ``8 x 8`` final-state spin-density matrix, normalizes it by the
corresponding squared amplitude when requested, and computes the two-body and
one-to-rest concurrence observables plus ``F3`` from arXiv:2310.01477v2.

Running this file as a script regenerates the SpinDensityMat output directory:
the NPZ scan, a summary entanglement CSV/PDF, per-kinematic-point matrix
CSVs/PDFs, and ``Output/SpinDensityMat.log``.
"""

from itertools import product
import csv
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from Algebra import HELICITIES, photon_pol
from BHHelicityAmp import bh_amplitude_core
from Kinematics import kinematics_user_from_scalar_inputs


# ============================================================
# Scan and output settings
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
NORMALIZE_TRACE = True
TRACE_BENCHMARK_TOL = 1e-10
ENTANGLEMENT_INITIAL_STATE = (+1, +1)
ENTANGLEMENT_NAMES = (
    "C12",
    "C13",
    "C23",
    "C1_23",
    "C2_13",
    "C3_12",
    "F3",
    "M1",
    "M2",
    "M3",
)

BENCHMARK_KINEMATIC_INPUTS = (
    ("K1", 5.0, 2.0, 0.36, -0.4, 0.7),
    ("K2", 6.0, 1.5, 0.20, -0.25, 1.2),
    ("K3", 10.0, 4.0, 0.30, -0.8, 2.1),
)

OUTPUT_DIR = Path("Output") / "SpinDensityMat"
LOG_PATH = Path("Output") / "SpinDensityMat.log"
SCAN_POINT_DIR = OUTPUT_DIR / "SpinDensityScan"
SCAN_NPZ_PATH = OUTPUT_DIR / "spin_density_scan.npz"
ENTANGLEMENT_CSV_PATH = OUTPUT_DIR / "spin_entanglement_scan.csv"
ENTANGLEMENT_PLOT_PATH = OUTPUT_DIR / "spin_entanglement_scan.pdf"
REMOVED_COEFFICIENT_PLOTS = (
    OUTPUT_DIR / "spin_density_norm_by_coefficient.pdf",
    OUTPUT_DIR / "spin_density_phase_by_coefficient.pdf",
)


def outgoing_spin_states():
    """Return final-state labels (hOut, sOut, lambda)."""
    return tuple(product(HELICITIES, repeat=3))


def initial_spin_states():
    """Return initial-state labels (hIn, sIn)."""
    return tuple(product(HELICITIES, repeat=2))


def amplitude_table(mom, m, F1, F2):
    """Return ``A[in_state, out_state]`` for all BH helicity amplitudes.

    ``in_state`` spans incoming electron/proton labels ``(hIn, sIn)`` and
    ``out_state`` spans outgoing electron/proton/photon labels
    ``(hOut, sOut, lambda)``. The result has shape ``(4, 8)``.
    """
    in_states = initial_spin_states()
    out_states = outgoing_spin_states()
    photon_pols = {lam: photon_pol(mom["qout"], lam) for lam in HELICITIES}
    amplitudes = np.zeros((len(in_states), len(out_states)), dtype=complex)

    for in_index, (h_in, s_in) in enumerate(in_states):
        for out_index, (h_out, s_out, lam) in enumerate(out_states):
            amplitudes[in_index, out_index] = bh_amplitude_core(
                mom["k"],
                mom["kp"],
                mom["qout"],
                mom["p"],
                mom["pp"],
                photon_pols[lam],
                h_in,
                h_out,
                s_in,
                s_out,
                m,
                F1,
                F2,
            )
    return amplitudes


def density_matrix_from_amplitudes(amplitudes, average_initial=False):
    """Build the outgoing spin-density matrix from an amplitude table.

    The convention is ``rho_ij = sum_initial A_initial,i conj(A_initial,j)``.
    The returned squared amplitude is ``sum |A|^2`` with the same optional
    initial-spin averaging as ``rho``.
    """
    rho = amplitudes.T @ np.conjugate(amplitudes)
    squared_amplitude = float(np.sum(np.abs(amplitudes) ** 2))
    if average_initial:
        rho /= amplitudes.shape[0]
        squared_amplitude /= amplitudes.shape[0]
    return rho, squared_amplitude


def normalized_final_state(amplitudes, initial_state=ENTANGLEMENT_INITIAL_STATE):
    """Return the normalized outgoing pure state for one incoming spin pair.

    The concurrence formulae used here are pure-state formulae. This helper
    selects one row of the amplitude table, corresponding to
    ``initial_state=(hIn, sIn)``, and normalizes it as an eight-component
    three-qubit state in the outgoing basis.
    """
    in_states = initial_spin_states()
    if initial_state not in in_states:
        raise ValueError(f"Unknown initial spin state: {initial_state}")

    state = amplitudes[in_states.index(initial_state)]
    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    if norm <= 1e-14:
        raise ZeroDivisionError(
            f"Cannot normalize a zero final-state amplitude for initial state {initial_state}."
        )
    return state / norm


def trace_value(rho):
    """Return ``Tr(rho)`` as a real float after checking numerical hermiticity."""
    trace = np.trace(rho)
    if abs(trace.imag) > 1e-10 * max(1.0, abs(trace.real)):
        raise ValueError(f"Density-matrix trace has a non-negligible imaginary part: {trace}")
    return float(trace.real)


def normalized_density_matrix(rho):
    """Return a Hermitian density matrix normalized to unit trace."""
    rho = np.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    trace = trace_value(rho)
    if abs(trace) <= 1e-14:
        raise ZeroDivisionError("Cannot normalize a zero-trace density matrix.")
    return rho / trace


def pure_density_matrix(state):
    """Return ``|state><state|`` for an already normalized state vector."""
    state = np.asarray(state, dtype=complex)
    return np.outer(state, state.conj())


def reduced_density_matrix(rho, keep):
    """Trace out unwanted qubits from an ``8 x 8`` three-qubit density matrix.

    Parameters
    ----------
    rho : array-like
        Three-qubit density matrix in the outgoing basis
        ``(hOut, sOut, lambda)``.
    keep : iterable of int
        Subsystem indices to keep: ``0`` for particle 1, ``1`` for particle 2,
        and ``2`` for particle 3.
    """
    keep = tuple(keep)
    if any(index not in (0, 1, 2) for index in keep):
        raise ValueError("Three-qubit subsystem indices must be 0, 1, or 2.")

    tensor = normalized_density_matrix(rho).reshape((2, 2, 2, 2, 2, 2))
    for axis in sorted(set((0, 1, 2)) - set(keep), reverse=True):
        half_ndim = tensor.ndim // 2
        tensor = np.trace(tensor, axis1=axis, axis2=axis + half_ndim)
    return normalized_density_matrix(tensor.reshape((2 ** len(keep), 2 ** len(keep))))


def two_qubit_concurrence(rho2):
    """Wootters concurrence for a reduced two-qubit density matrix."""
    rho2 = normalized_density_matrix(rho2)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    spin_flip = np.kron(sigma_y, sigma_y)
    eigenvalues = np.linalg.eigvals(rho2 @ spin_flip @ rho2.conj() @ spin_flip)
    lambdas = np.sqrt(np.clip(np.real_if_close(eigenvalues, tol=1000).real, 0.0, None))
    lambdas.sort()
    return float(max(0.0, lambdas[-1] - lambdas[-2] - lambdas[-3] - lambdas[-4]))


def one_to_rest_concurrence(rho, traced_subsystem):
    """Pure-state one-to-rest concurrence from Eq. (4) of arXiv:2310.01477v2."""
    rest = [index for index in (0, 1, 2) if index != traced_subsystem]
    rho_rest = reduced_density_matrix(rho, rest)
    purity = float(np.real_if_close(np.trace(rho_rest @ rho_rest), tol=1000).real)
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - purity))))


def f3_from_one_to_rest(c1_23, c2_13, c3_12):
    """Concurrence-triangle area measure from Eq. (6) of arXiv:2310.01477v2."""
    q_value = 0.5 * (c1_23 + c2_13 + c3_12)
    area_argument = (
        (16.0 / 3.0)
        * q_value
        * (q_value - c1_23)
        * (q_value - c2_13)
        * (q_value - c3_12)
    )
    return float(np.sqrt(max(0.0, area_argument)))


def entanglement_measures_from_state(state):
    """Compute concurrence observables and monogamy residuals.

    The returned dictionary contains two-body concurrences ``C12``, ``C13``,
    ``C23``; one-to-rest concurrences ``C1_23``, ``C2_13``, ``C3_12``; the
    concurrence-triangle measure ``F3``; and CKW monogamy residuals ``M1``,
    ``M2``, and ``M3``.
    """
    rho = pure_density_matrix(state)
    c12 = two_qubit_concurrence(reduced_density_matrix(rho, (0, 1)))
    c13 = two_qubit_concurrence(reduced_density_matrix(rho, (0, 2)))
    c23 = two_qubit_concurrence(reduced_density_matrix(rho, (1, 2)))
    c1_23 = one_to_rest_concurrence(rho, 0)
    c2_13 = one_to_rest_concurrence(rho, 1)
    c3_12 = one_to_rest_concurrence(rho, 2)
    return {
        "C12": c12,
        "C13": c13,
        "C23": c23,
        "C1_23": c1_23,
        "C2_13": c2_13,
        "C3_12": c3_12,
        "F3": f3_from_one_to_rest(c1_23, c2_13, c3_12),
        "M1": c1_23**2 - c12**2 - c13**2,
        "M2": c2_13**2 - c12**2 - c23**2,
        "M3": c3_12**2 - c13**2 - c23**2,
    }


def build_scan_point(
    Eb,
    Q2,
    xB,
    t,
    phi,
    m,
    F1,
    F2,
    azimuth_input=AZIMUTH_INPUT,
    average_initial=AVERAGE_INITIAL_SPINS,
    normalize_trace=NORMALIZE_TRACE,
    entanglement_initial_state=ENTANGLEMENT_INITIAL_STATE,
):
    """Evaluate all spin-density and entanglement data for one kinematic point."""
    kin = kinematics_user_from_scalar_inputs(
        Eb,
        Q2,
        xB,
        t,
        phi,
        m,
        azimuth_input=azimuth_input,
        label=f"Q2={Q2:.6g}, t={t:.6g}",
    )
    amplitudes = amplitude_table(kin["momenta"], kin["m"], F1, F2)
    rho, squared_amplitude = density_matrix_from_amplitudes(
        amplitudes,
        average_initial=average_initial,
    )
    if normalize_trace:
        if squared_amplitude <= 1e-14:
            raise ZeroDivisionError("Cannot trace-normalize a zero density matrix.")
        rho /= squared_amplitude

    state = normalized_final_state(amplitudes, initial_state=entanglement_initial_state)
    return {
        "rho": rho,
        "squared_amplitude": squared_amplitude,
        "trace": trace_value(rho),
        "entanglement": entanglement_measures_from_state(state),
    }


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
    entanglement_initial_state=ENTANGLEMENT_INITIAL_STATE,
):
    """Scan a rectangular ``(t, Q2)`` grid of spin-density matrices.

    Returns a dictionary containing the complex density-matrix grid, per-point
    squared amplitudes and traces, concurrence/F3 grids, validity mask,
    failures, axis values, basis labels, and normalization metadata.
    """
    out_states = outgoing_spin_states()
    shape = (len(t_values), len(Q2_values))
    rho_grid = np.full((*shape, len(out_states), len(out_states)), np.nan + 1j * np.nan)
    squared_amplitude_grid = np.full(shape, np.nan, dtype=float)
    trace_grid = np.full(shape, np.nan, dtype=float)
    valid = np.zeros(shape, dtype=bool)
    entanglement_grid = {
        name: np.full(shape, np.nan, dtype=float)
        for name in ENTANGLEMENT_NAMES
    }
    failures = []

    for t_index, t in enumerate(t_values):
        for Q2_index, Q2 in enumerate(Q2_values):
            try:
                point = build_scan_point(
                    Eb,
                    Q2,
                    xB,
                    t,
                    phi,
                    m,
                    F1,
                    F2,
                    azimuth_input=azimuth_input,
                    average_initial=average_initial,
                    normalize_trace=normalize_trace,
                    entanglement_initial_state=entanglement_initial_state,
                )
            except Exception as exc:
                failures.append((Q2, t, str(exc)))
                continue

            rho_grid[t_index, Q2_index] = point["rho"]
            squared_amplitude_grid[t_index, Q2_index] = point["squared_amplitude"]
            trace_grid[t_index, Q2_index] = point["trace"]
            for name, value in point["entanglement"].items():
                entanglement_grid[name][t_index, Q2_index] = value
            valid[t_index, Q2_index] = True

    return {
        "rho": rho_grid,
        "squared_amplitude": squared_amplitude_grid,
        "trace": trace_grid,
        "entanglement": entanglement_grid,
        "entanglement_names": ENTANGLEMENT_NAMES,
        "valid": valid,
        "failures": failures,
        "Q2_values": np.asarray(Q2_values, dtype=float),
        "t_values": np.asarray(t_values, dtype=float),
        "out_states": out_states,
        "normalized_by_squared_amplitude": normalize_trace,
        "entanglement_initial_state": entanglement_initial_state,
    }


def benchmark_spin_density_trace(
    kinematic_inputs=BENCHMARK_KINEMATIC_INPUTS,
    Eb_default=EB,
    xB_default=XB,
    phi_default=PHI,
    m=M,
    F1=F1,
    F2=F2,
    azimuth_input=AZIMUTH_INPUT,
    average_initial=AVERAGE_INITIAL_SPINS,
    tol=TRACE_BENCHMARK_TOL,
):
    """Check that selected benchmark density matrices normalize to trace one."""
    rows = []
    for item in kinematic_inputs:
        if len(item) == 6:
            case_id, Eb, Q2, xB, t, phi = item
        elif len(item) == 4:
            case_id, Q2, t, phi = item
            Eb = Eb_default
            xB = xB_default
        elif len(item) == 3:
            case_id, Q2, t = item
            Eb = Eb_default
            xB = xB_default
            phi = phi_default
        else:
            raise ValueError(
                "Benchmark kinematic inputs must be "
                "(case, Eb, Q2, xB, t, phi), (case, Q2, t, phi), or (case, Q2, t)."
            )

        kin = kinematics_user_from_scalar_inputs(
            Eb,
            Q2,
            xB,
            t,
            phi,
            m,
            azimuth_input=azimuth_input,
            label=f"trace benchmark {case_id}",
        )
        amplitudes = amplitude_table(kin["momenta"], kin["m"], F1, F2)
        rho, squared_amplitude = density_matrix_from_amplitudes(
            amplitudes,
            average_initial=average_initial,
        )
        if squared_amplitude <= 1e-14:
            raise ZeroDivisionError(
                f"Cannot normalize zero density matrix for trace benchmark {case_id}."
            )

        raw_trace = trace_value(rho)
        normalized_trace = trace_value(rho / squared_amplitude)
        trace_error = abs(normalized_trace - 1.0)
        if trace_error > tol:
            raise AssertionError(
                f"Trace benchmark {case_id} failed: "
                f"Tr(rho)={normalized_trace:.16e} after normalization."
            )
        if rho.shape != (8, 8):
            raise AssertionError(
                f"Trace benchmark {case_id} produced shape {rho.shape}, expected (8, 8)."
            )

        rows.append({
            "case": case_id,
            "Eb": Eb,
            "Q2": Q2,
            "xB": xB,
            "t": t,
            "phi": phi,
            "raw_trace": raw_trace,
            "squared_amplitude": squared_amplitude,
            "normalized": not np.isclose(raw_trace, 1.0, rtol=tol, atol=tol),
            "normalized_trace": normalized_trace,
            "trace_error": trace_error,
        })
    return rows


def clean_generated_outputs():
    """Remove files generated by this script before creating fresh outputs."""
    for path in (
        LOG_PATH,
        SCAN_NPZ_PATH,
        ENTANGLEMENT_CSV_PATH,
        ENTANGLEMENT_PLOT_PATH,
        SCAN_POINT_DIR,
        *REMOVED_COEFFICIENT_PLOTS,
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


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


def _safe_float_for_filename(name, value):
    """Format a floating-point value as a filesystem-safe filename token."""
    text = f"{value:.6f}".replace("-", "m").replace(".", "p")
    return f"{name}_{text}"


def _scan_point_stem(Q2, t):
    """Return the shared filename stem for a single ``(Q2, t)`` scan point."""
    return (
        "spin_density_"
        f"{_safe_float_for_filename('Q2', Q2)}_"
        f"{_safe_float_for_filename('t', t)}"
    )


def _plot_scan_page(ax, data, Q2_values, t_values, title, cmap, vmin=None, vmax=None):
    """Draw one kinematic-grid heatmap page and return the image artist."""
    image = ax.imshow(
        np.ma.masked_invalid(data),
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


def save_entanglement_plot(scan, output_path=ENTANGLEMENT_PLOT_PATH):
    """Save heatmap pages for concurrence observables and ``F3``."""
    plt, PdfPages = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_specs = (
        ("C12", r"$C_{12}$"),
        ("C13", r"$C_{13}$"),
        ("C23", r"$C_{23}$"),
        ("C1_23", r"$C_{1(23)}$"),
        ("C2_13", r"$C_{2(13)}$"),
        ("C3_12", r"$C_{3(12)}$"),
        ("F3", r"$F_3$"),
    )

    with PdfPages(output_path) as pdf:
        for name, label in plot_specs:
            fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
            image = _plot_scan_page(
                ax,
                scan["entanglement"][name],
                scan["Q2_values"],
                scan["t_values"],
                label,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(image, ax=ax, label=label)
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def save_scan_npz(scan, path=SCAN_NPZ_PATH):
    """Persist the full scan arrays to an NPZ archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entanglement_arrays = {
        f"entanglement_{name}": values
        for name, values in scan["entanglement"].items()
    }
    np.savez(
        path,
        rho=scan["rho"],
        squared_amplitude=scan["squared_amplitude"],
        trace=scan["trace"],
        valid=scan["valid"],
        Q2_values=scan["Q2_values"],
        t_values=scan["t_values"],
        out_states=np.asarray(scan["out_states"], dtype=int),
        normalized_by_squared_amplitude=scan["normalized_by_squared_amplitude"],
        entanglement_names=np.asarray(scan["entanglement_names"], dtype=str),
        entanglement_initial_state=np.asarray(scan["entanglement_initial_state"], dtype=int),
        **entanglement_arrays,
    )
    return path


def _matrix_headers(include_matrix_indices):
    """Return CSV headers for summary or per-matrix rows."""
    headers = [
        "Q2",
        "t",
        "squared_amplitude_M2",
        "trace",
        "normalized_by_squared_amplitude",
        "entanglement_h_in",
        "entanglement_s_in",
        *ENTANGLEMENT_NAMES,
    ]
    if include_matrix_indices:
        headers += [
            "row_index",
            "row_h_out",
            "row_s_out",
            "row_lambda",
            "col_index",
            "col_h_out",
            "col_s_out",
            "col_lambda",
            "rho_real",
            "rho_imag",
            "rho_abs",
            "rho_phase",
        ]
    return headers


def _metadata_row(scan, t_index, Q2_index):
    """Return common scalar and entanglement columns for one scan point."""
    return [
        f"{scan['Q2_values'][Q2_index]:.16e}",
        f"{scan['t_values'][t_index]:.16e}",
        f"{scan['squared_amplitude'][t_index, Q2_index]:.16e}",
        f"{scan['trace'][t_index, Q2_index]:.16e}",
        scan["normalized_by_squared_amplitude"],
        *scan["entanglement_initial_state"],
        *(
            f"{scan['entanglement'][name][t_index, Q2_index]:.16e}"
            for name in scan["entanglement_names"]
        ),
    ]


def save_entanglement_csv(scan, path=ENTANGLEMENT_CSV_PATH):
    """Save one summary row per valid kinematic point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_matrix_headers(include_matrix_indices=False))
        for t_index, _t in enumerate(scan["t_values"]):
            for Q2_index, _Q2 in enumerate(scan["Q2_values"]):
                if scan["valid"][t_index, Q2_index]:
                    writer.writerow(_metadata_row(scan, t_index, Q2_index))
    return path


def save_scan_csv_files(scan, output_dir=SCAN_POINT_DIR):
    """Save one long-form ``8 x 8`` density-matrix CSV per valid point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    headers = _matrix_headers(include_matrix_indices=True)

    for t_index, t in enumerate(scan["t_values"]):
        for Q2_index, Q2 in enumerate(scan["Q2_values"]):
            if not scan["valid"][t_index, Q2_index]:
                continue

            path = output_dir / f"{_scan_point_stem(Q2, t)}.csv"
            matrix = scan["rho"][t_index, Q2_index]
            metadata = _metadata_row(scan, t_index, Q2_index)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                for row_index, row_state in enumerate(scan["out_states"]):
                    for col_index, col_state in enumerate(scan["out_states"]):
                        value = matrix[row_index, col_index]
                        writer.writerow([
                            *metadata,
                            row_index,
                            *row_state,
                            col_index,
                            *col_state,
                            f"{value.real:.16e}",
                            f"{value.imag:.16e}",
                            f"{abs(value):.16e}",
                            f"{np.angle(value):.16e}",
                        ])
            paths.append(path)
    return paths


def _state_tick_labels(out_states):
    """Return compact labels for the eight outgoing basis states."""
    return [
        f"{index}: h'={state[0]:+d}, s'={state[1]:+d}, lam={state[2]:+d}"
        for index, state in enumerate(out_states)
    ]


def _plot_matrix_heatmap(
    ax,
    matrix,
    out_states,
    title,
    cmap,
    colorbar_label,
    vmin=None,
    vmax=None,
):
    """Draw an ``8 x 8`` matrix heatmap with outgoing-state index ticks."""
    image = ax.imshow(
        matrix,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("column final state")
    ax.set_ylabel("row final state")
    ticks = np.arange(len(out_states))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.figure.colorbar(image, ax=ax, label=colorbar_label)
    return image


def save_point_matrix_plots(scan, output_dir=SCAN_POINT_DIR):
    """Save norm and phase ``8 x 8`` matrix plots for every valid point."""
    plt, _PdfPages = _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    rho_symbol = r"\rho/M^2" if scan["normalized_by_squared_amplitude"] else r"\rho"
    state_key = "\n".join(_state_tick_labels(scan["out_states"]))
    paths = []

    for t_index, t in enumerate(scan["t_values"]):
        for Q2_index, Q2 in enumerate(scan["Q2_values"]):
            if not scan["valid"][t_index, Q2_index]:
                continue

            stem = _scan_point_stem(Q2, t)
            matrix = scan["rho"][t_index, Q2_index]
            title_suffix = f"Q2={Q2:.6g}, t={t:.6g}"

            for suffix, data, title, cmap, label, vmin, vmax in (
                (
                    "matrix_norm",
                    np.abs(matrix),
                    rf"$|{rho_symbol}|$ at {title_suffix}",
                    "viridis",
                    rf"$|{rho_symbol}|$",
                    None,
                    None,
                ),
                (
                    "matrix_phase",
                    np.angle(matrix),
                    rf"$\arg({rho_symbol})$ at {title_suffix}",
                    "twilight",
                    rf"$\arg({rho_symbol})$ [rad]",
                    -np.pi,
                    np.pi,
                ),
            ):
                path = output_dir / f"{stem}_{suffix}.pdf"
                fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
                _plot_matrix_heatmap(
                    ax,
                    data,
                    scan["out_states"],
                    title,
                    cmap,
                    label,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.text(
                    0.0,
                    -0.18,
                    state_key,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.0,
                )
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)
                paths.append(path)
    return paths


def format_trace_benchmark_rows(rows):
    """Format trace benchmark rows as a fixed-width text table."""
    headers = (
        "case",
        "Q2",
        "t",
        "raw Tr(rho)",
        "|M|^2",
        "normalized",
        "Tr(rho_norm)",
        "error",
    )
    table_rows = [
        (
            row["case"],
            f"{row['Q2']:.6g}",
            f"{row['t']:.6g}",
            f"{row['raw_trace']:.8e}",
            f"{row['squared_amplitude']:.8e}",
            str(row["normalized"]),
            f"{row['normalized_trace']:.8e}",
            f"{row['trace_error']:.3e}",
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  " + "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    ]
    lines.append("  " + "  ".join("-" * width for width in widths))
    lines.extend(
        "  " + "  ".join(item.rjust(width) for item, width in zip(row, widths))
        for row in table_rows
    )
    return "\n".join(lines)


def build_report(scan, trace_benchmark_rows, paths):
    """Build the text report printed to console and written to the log file."""
    lines = [
        "Spin-density matrix scan",
        f"  outgoing basis size: {len(scan['out_states'])}",
        "  particle map: 1 = outgoing electron hOut, 2 = outgoing proton sOut, 3 = outgoing photon lambda",
        f"  entanglement observables: {', '.join(scan['entanglement_names'])}",
        f"  Q2 grid: {Q2_VALUES[0]:.6g} to {Q2_VALUES[-1]:.6g}",
        f"  t grid: {T_VALUES[0]:.6g} to {T_VALUES[-1]:.6g}",
        f"  valid points: {int(scan['valid'].sum())}/{scan['valid'].size}",
        f"  initial spins averaged: {AVERAGE_INITIAL_SPINS}",
        f"  normalized by M^2: {scan['normalized_by_squared_amplitude']}",
        (
            "  entanglement initial state: "
            f"hIn={scan['entanglement_initial_state'][0]:+d}, "
            f"sIn={scan['entanglement_initial_state'][1]:+d}"
        ),
        "  trace benchmark: passed",
        format_trace_benchmark_rows(trace_benchmark_rows),
        f"  saved data: {paths['npz']}",
        f"  saved entanglement csv: {paths['entanglement_csv']}",
        f"  saved entanglement plots: {paths['entanglement_plot']}",
        f"  saved matrix csv files: {len(paths['matrix_csv'])} in {SCAN_POINT_DIR}",
        f"  saved matrix plot files: {len(paths['matrix_plots'])} in {SCAN_POINT_DIR}",
        f"  saved log: {LOG_PATH}",
    ]
    if scan["failures"]:
        lines.append("  invalid grid points:")
        for Q2, t, message in scan["failures"]:
            lines.append(f"    Q2={Q2:.8g}, t={t:.8g}: {message}")
    return "\n".join(lines) + "\n"


def main():
    """Regenerate all SpinDensityMat outputs from the current settings."""
    clean_generated_outputs()
    trace_benchmark_rows = benchmark_spin_density_trace()
    scan = scan_spin_density_grid(Q2_VALUES, T_VALUES)
    paths = {
        "npz": save_scan_npz(scan),
        "entanglement_csv": save_entanglement_csv(scan),
        "entanglement_plot": save_entanglement_plot(scan),
        "matrix_csv": save_scan_csv_files(scan),
        "matrix_plots": save_point_matrix_plots(scan),
    }

    log_text = build_report(scan, trace_benchmark_rows, paths)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
