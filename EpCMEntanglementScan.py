"""Focused Bethe--Heitler entanglement scan with a slow recoil proton.

The construction starts in the exact lepton--proton CM frame with the proton
moving along +z and the lepton along -z.  A recoil proton carrying ``(1-z)``
of the incoming proton three-momentum fixes the exchanged photon
``q = p - p'``.  The default grid gives the virtual photon most of the incoming
proton energy while retaining a slow final proton.  Its recoil polar angle is
scanned over the complete physical range.  The
resulting ``q + lepton`` subsystem is scattered into an on-shell photon and
lepton at ``theta_cm``, then boosted back to the ep CM frame.

All external masses and four-momentum conservation are retained exactly.
Outputs are written below ``Output/EpCMEntanglementScan``.
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
    process_density_matrix_from_amplitudes,
    spin_case_display_label,
    spin_density_observables_from_amplitudes,
    w_observables_from_density_matrix,
)
from config import (
    HEAVY_LEPTON_MASS_GEV,
    NORMALIZE_TRACE,
    PROTON_MASS_GEV,
    SCAN_WORKERS,
)


# Editable scan controls.  Sample the final-proton momentum directly
# from 5 GeV down to 0.05 GeV, corresponding to z=0.90--0.999 at p=50 GeV.
BEAM_MOMENTUM_GEV = 50.0
LEPTON_MASS_GEV = HEAVY_LEPTON_MASS_GEV
FINAL_PROTON_MOMENTUM_VALUES_GEV = np.geomspace(5.0, 0.05, 13)
Z_VALUES = 1.0 - FINAL_PROTON_MOMENTUM_VALUES_GEV / BEAM_MOMENTUM_GEV
# Three-degree coverage of the complete final-state polar-angle range.
THETA_CM_VALUES = np.linspace(2.4, 2.5, 20)
# Ten-degree coverage of the complete final-proton recoil polar-angle range.
THETA_P_VALUES = np.linspace(0.0, np.pi, 19)
PHI_CM_RAD = 0.0
PHI_P_RAD = 0.0
SCAN_WORKER_COUNT = SCAN_WORKERS
EP_CM_SPIN_CASES = SPIN_CASES + (
    SPIN_CASE_MINUS_TX_PROTON,
    SPIN_CASE_MINUS_TY_PROTON,
    SPIN_CASE_L_MINUS_TX,
    SPIN_CASE_L_MINUS_TY,
)
SCAN_PLOT_WORKER_COUNT = max(1, min(SCAN_WORKERS, len(EP_CM_SPIN_CASES)))
TOP_POINTS = 10

OUTPUT_DIR = Path("Output") / "EpCMEntanglementScan"
FULL_CSV = OUTPUT_DIR / "ep_cm_entanglement_scan.csv"
TOP_CSV = OUTPUT_DIR / "ep_cm_entanglement_top.csv"
PLOT_DIR = OUTPUT_DIR / "Plots"
LOG_PATH = OUTPUT_DIR / "EpCMEntanglementScan.log"

PLOT_OBSERVABLES = COARSE_CONCURRENCE_NAMES
RANKED_OBSERVABLES = COARSE_CONCURRENCE_NAMES


def polarization_prefix(spin_case):
    """Return the exact heavy-lepton polarization label used by AlignmentScan."""
    return explicit_polarization_name(spin_case, "heavy")


def observable_column(spin_case, observable):
    """Return an AlignmentScan-compatible observable column name."""
    return (
        f"{polarization_prefix(spin_case)}_"
        f"{species_observable_name(observable, 'heavy')}"
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
    if not 0.0 <= theta_p <= np.pi:
        raise ValueError("theta_p must lie in [0, pi].")

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


def evaluate_point(task):
    """Evaluate every incoming polarization at one scan point."""
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
        "lepton": "heavy",
        "kinematic_point": (
            f"ep_cm_z_{z:.8g}_theta_cm_{theta_cm:.8g}_theta_p_{theta_p:.8g}"
        ),
        "s_regime": "sqrt_s_100_GeV",
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
    }
    for spin_case in EP_CM_SPIN_CASES:
        result = spin_density_observables_from_amplitudes(
            amplitudes,
            spin_case=spin_case,
            normalize_trace=NORMALIZE_TRACE,
            process_rho=process_rho,
        )
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
    return row


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


def write_csv(path, rows, fieldnames=None):
    """Write dictionaries with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_csv(rows):
    """Rank each entanglement observable for every incoming spin case."""
    ranked = []
    base_fields = (
        "z", "theta_cm_rad", "theta_p_rad", "theta_p_deg",
        "mu", "p_cm_GeV", "t_GeV2",
        "final_proton_momentum_GeV", "final_proton_energy_GeV",
        "proton_energy_loss_GeV", "proton_energy_loss_fraction",
        "qout_E", "qout_px", "qout_pz", "kp_E", "kp_px", "kp_pz",
        "pp_E", "pp_pz",
    )
    for spin_case in EP_CM_SPIN_CASES:
        for observable in RANKED_OBSERVABLES:
            key = observable_column(spin_case, observable)
            reverse = observable != "D_W"
            ordered = sorted(rows, key=lambda row: row[key], reverse=reverse)
            for rank, row in enumerate(ordered[:TOP_POINTS], start=1):
                ranked.append({
                    "spin_case": spin_case,
                    "spin_label": spin_case_display_label(spin_case),
                    "observable": observable,
                    "rank": rank,
                    "value": row[key],
                    **{field: row[field] for field in base_fields},
                })
    write_csv(TOP_CSV, ranked)


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
            display_name = species_observable_name(observable, "heavy")
            ax.set_title(f"|{display_name}|")
            fig.colorbar(image, ax=ax)
        for ax in axes.flat[len(PLOT_OBSERVABLES):]:
            ax.set_visible(False)
        fig.suptitle(
            f"ep-CM slow-recoil-proton scan: {spin_case_display_label(spin_case)}\n"
            f"p={BEAM_MOMENTUM_GEV:g} GeV, m_lepton={LEPTON_MASS_GEV:g} GeV, "
            f"z={Z_VALUES[0]:.3f}--{Z_VALUES[-1]:.3f}; "
            r"optimized over $0\leq\theta_p\leq\pi$"
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
    if not max_workers or max_workers <= 1 or len(EP_CM_SPIN_CASES) == 1:
        return {
            spin_case: save_spin_plot(rows, spin_case, PLOT_DIR)
            for spin_case in EP_CM_SPIN_CASES
        }
    worker_count = min(int(max_workers), len(EP_CM_SPIN_CASES))
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_plot_worker,
            initargs=(rows, PLOT_DIR),
        ) as executor:
            return dict(
                executor.map(
                    _save_spin_plot_worker,
                    EP_CM_SPIN_CASES,
                    chunksize=1,
                )
            )
    except (OSError, PermissionError):
        return {
            spin_case: save_spin_plot(rows, spin_case, PLOT_DIR)
            for spin_case in EP_CM_SPIN_CASES
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


def build_report(rows, plot_paths):
    """Summarize slow-proton anchors and the strongest lepton-photon points."""
    anchors = (
        nearest_row(rows, 0.90, 2.45),
        nearest_row(rows, 0.98, 2.45),
        nearest_row(rows, 0.999, 2.45),
    )
    lines = [
        "Focused ep-CM slow-recoil-proton entanglement scan",
        f"  points: {len(rows)} "
        f"({len(Z_VALUES)} z x {len(THETA_CM_VALUES)} theta_cm "
        f"x {len(THETA_P_VALUES)} theta_p)",
        f"  sqrt(s): {anchors[0]['sqrt_s_GeV']:.8g} GeV",
        f"  lepton mass: {LEPTON_MASS_GEV:.8g} GeV",
        f"  z range: {Z_VALUES[0]:.8g}--{Z_VALUES[-1]:.8g}",
        f"  theta_cm range: {THETA_CM_VALUES[0]:.8g}--"
        f"{THETA_CM_VALUES[-1]:.8g} rad",
        f"  theta_p range: {THETA_P_VALUES[0]:.8g}--"
        f"{THETA_P_VALUES[-1]:.8g} rad",
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
    for spin_case in EP_CM_SPIN_CASES:
        key = observable_column(spin_case, "C_e_gamma")
        best = max(rows, key=lambda row: row[key])
        lines.append(
            f"  max C_lepton_gamma ({spin_case_display_label(spin_case)}): "
            f"{best[key]:.8g} at z={best['z']:.6g}, "
            f"theta_cm={best['theta_cm_rad']:.6g}, "
            f"theta_p={best['theta_p_rad']:.6g}, mu={best['mu']:.6g}"
        )
    lines.extend((
        "",
        f"  scan workers: {SCAN_WORKER_COUNT}",
        f"  plot workers: {SCAN_PLOT_WORKER_COUNT}",
        f"  full CSV: {FULL_CSV}",
        f"  ranked CSV: {TOP_CSV}",
        f"  plot directory: {PLOT_DIR}",
    ))
    for spin_case, path in plot_paths.items():
        lines.append(f"    {spin_case}: {path}")
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
    rows = run_tasks(tasks)
    write_csv(FULL_CSV, rows)
    write_ranked_csv(rows)
    plot_paths = save_plots(rows)
    report = build_report(rows, plot_paths)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
