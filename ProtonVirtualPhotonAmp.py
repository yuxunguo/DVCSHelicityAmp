"""Proton--virtual-photon helicity-amplitude decomposition.

This standalone utility isolates the elastic proton electromagnetic current

    J_nu = ubar(p', sOut) Gamma_nu u(p, sIn)

and projects it onto transverse and longitudinal polarization vectors of the
spacelike virtual photon emitted by the proton, ``q = p - p'``.  The current
is decomposed both as ``F1 * Dirac + F2 * Pauli`` and as
``GE * electric + GM * magnetic`` using the same conventions as
``BHHelicityAmp.py``.

Edit the values in :func:`main` and run

    python ProtonVirtualPhotonAmp.py

Outputs are written below ``Output/ProtonVirtualPhotonAmp``.
"""

import csv
from pathlib import Path

import numpy as np

from Algebra import HELICITIES, cov, mdot
from BHHelicityAmp import proton_current_helicity_decomposition
from EpCMEntanglementScan import BEAM_MOMENTUM_GEV
from FormFactors import YAHL_MODEL_NAME, yahl_dirac_pauli_from_t
from PlotUtils import print_console_text, require_matplotlib
from config import PROTON_MASS_GEV


OUTPUT_DIR = Path("Output") / "ProtonVirtualPhotonAmp"
AMPLITUDE_CSV = OUTPUT_DIR / "proton_virtual_photon_amplitudes.csv"
CURRENT_CSV = OUTPUT_DIR / "proton_current_components.csv"
THETA_SCAN_CSV = OUTPUT_DIR / "proton_virtual_photon_theta_scan.csv"
Z_SCAN_CSV = OUTPUT_DIR / "proton_virtual_photon_z_scan.csv"
PLOT_PATH = OUTPUT_DIR / "proton_virtual_photon_amplitude_decomposition.pdf"
LOG_PATH = Path("Output") / "ProtonVirtualPhotonAmp.log"


def real_scalar(value, name, tolerance=1.0e-10):
    """Return a numerically real scalar or reject a significant imaginary part."""
    value = complex(value)
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > tolerance * scale:
        raise ValueError(f"{name} has a non-negligible imaginary part: {value}")
    return float(value.real)


def virtual_photon_polarizations(q):
    """Return ``T-``, ``T+``, ``L``, and gauge polarizations for spacelike ``q``.

    The transverse convention matches the real-photon convention in
    :mod:`Algebra`.  For ``q^2=-Q2<0``, the longitudinal vector is

    ``epsilon_L = (|q|, q0 * qhat) / sqrt(Q2)``,

    so ``epsilon_L.q=0`` and ``epsilon_L^2=+1``.  The gauge vector
    ``q/sqrt(Q2)`` is included only to test current conservation.
    """
    q = np.asarray(q, dtype=float)
    q3 = q[1:]
    qabs = float(np.linalg.norm(q3))
    if qabs <= 0.0:
        raise ValueError("Virtual-photon spatial momentum must be nonzero.")
    Q2 = -real_scalar(mdot(q, q), "virtual-photon q^2")
    if Q2 <= 0.0:
        raise ValueError(f"Virtual photon must be spacelike; obtained Q2={Q2}.")

    qhat = q3 / qabs
    reference = np.array((0.0, 0.0, 1.0))
    if abs(float(np.dot(qhat, reference))) > 0.9:
        reference = np.array((1.0, 0.0, 0.0))
    e1 = np.cross(reference, qhat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(qhat, e1)
    transverse_minus = np.concatenate(
        ([0.0], (e1 - 1.0j * e2) / np.sqrt(2.0))
    )
    transverse_plus = np.concatenate(
        ([0.0], (e1 + 1.0j * e2) / np.sqrt(2.0))
    )
    longitudinal = np.concatenate(
        ([qabs], q[0] * qhat)
    ) / np.sqrt(Q2)
    gauge = q / np.sqrt(Q2)
    return {
        "T-": transverse_minus,
        "T+": transverse_plus,
        "L": longitudinal.astype(complex),
        "gauge": gauge.astype(complex),
    }


def contract_current(polarization, current_lower):
    """Return ``epsilon^nu J_nu`` for one contravariant polarization."""
    return complex(np.dot(np.asarray(polarization, complex), current_lower))


def complex_fields(prefix, value):
    """Return CSV fields for one complex amplitude."""
    value = complex(value)
    phase = float(np.angle(value))
    return {
        f"{prefix}_real": float(value.real),
        f"{prefix}_imag": float(value.imag),
        f"{prefix}_abs": float(abs(value)),
        f"{prefix}_phase_rad": phase,
        f"{prefix}_phase_over_pi": phase / np.pi,
        f"{prefix}_phase_deg": float(np.degrees(phase)),
    }


def current_component_rows(decomposition):
    """Return Lorentz-component rows for every proton helicity transition."""
    rows = []
    for out_index, s_out in enumerate(HELICITIES):
        for in_index, s_in in enumerate(HELICITIES):
            for nu in range(4):
                row = {"sIn": s_in, "sOut": s_out, "nu": nu}
                for name in ("dirac", "pauli", "electric", "magnetic", "total"):
                    row.update(complex_fields(
                        name,
                        decomposition[name][out_index, in_index, nu],
                    ))
                rows.append(row)
    return rows


def virtual_photon_amplitude_rows(decomposition, polarizations):
    """Project each helicity current onto the virtual-photon basis."""
    rows = []
    physical_records = []
    for out_index, s_out in enumerate(HELICITIES):
        for in_index, s_in in enumerate(HELICITIES):
            for polarization_name, epsilon in polarizations.items():
                amplitudes = {
                    name: contract_current(
                        epsilon,
                        decomposition[name][out_index, in_index],
                    )
                    for name in ("dirac", "pauli", "electric", "magnetic", "total")
                }
                row = {
                    "sIn": s_in,
                    "sOut": s_out,
                    "virtual_photon_polarization": polarization_name,
                    "is_physical_polarization": polarization_name != "gauge",
                }
                for name, value in amplitudes.items():
                    row.update(complex_fields(name, value))
                rows.append(row)
                if polarization_name != "gauge":
                    physical_records.append(row)

    total_abs2 = sum(row["total_abs"] ** 2 for row in physical_records)
    unpolarized_abs2 = 0.5 * total_abs2
    unpolarized_amplitude = np.sqrt(unpolarized_abs2)
    if unpolarized_amplitude <= 0.0:
        raise ZeroDivisionError("The unpolarized proton amplitude is zero.")
    for row in rows:
        row["total_abs2"] = row["total_abs"] ** 2
        row["unpolarized_abs2"] = unpolarized_abs2
        row["unpolarized_amplitude"] = unpolarized_amplitude
        for name in ("dirac", "pauli", "electric", "magnetic", "total"):
            normalized = complex(
                row[f"{name}_real"],
                row[f"{name}_imag"],
            ) / unpolarized_amplitude
            row.update(complex_fields(f"{name}_normalized", normalized))
        row["normalized_abs2"] = row["total_normalized_abs"] ** 2
        row["physical_fraction"] = (
            row["total_abs2"] / total_abs2
            if row["is_physical_polarization"] and total_abs2 > 0.0
            else 0.0
        )
    return rows


def write_rows(path, rows):
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return path


def scan_point_rows(z, theta_p, phi_p, beam_momentum):
    """Return physical polarization rows at one ``(z, theta_p)`` point."""
    kin = proton_virtual_photon_momenta(
        z,
        theta_p,
        phi_p,
        beam_momentum,
    )
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    decomposition = proton_current_helicity_decomposition(
        kin["momenta"]["p"],
        kin["momenta"]["pp"],
        PROTON_MASS_GEV,
        F1,
        F2,
    )
    point_rows = virtual_photon_amplitude_rows(
        decomposition,
        virtual_photon_polarizations(kin["virtual_photon"]),
    )
    physical_by_transition = {}
    for row in point_rows:
        if row["is_physical_polarization"]:
            physical_by_transition.setdefault(
                (row["sIn"], row["sOut"]),
                {},
            )[row["virtual_photon_polarization"]] = row

    rows = []
    for row in point_rows:
        if not row["is_physical_polarization"]:
            continue
        transition_rows = physical_by_transition[(row["sIn"], row["sOut"])]
        longitudinal_abs2 = transition_rows["L"]["total_abs2"]
        transverse_abs2 = (
            transition_rows["T-"]["total_abs2"]
            + transition_rows["T+"]["total_abs2"]
        )
        longitudinal_transverse_ratio = (
            longitudinal_abs2 / transverse_abs2
            if transverse_abs2 > 1.0e-30
            else np.nan
        )
        rows.append({
            "theta_p_rad": float(theta_p),
            "theta_p_deg": float(np.degrees(theta_p)),
            "phi_p_rad": float(phi_p),
            "z": float(z),
            "beam_momentum_GeV": float(beam_momentum),
            "Q2_GeV2": float(decomposition["Q2"]),
            "F1": float(decomposition["F1"].real),
            "F2": float(decomposition["F2"].real),
            "GE": float(decomposition["GE"]),
            "GM": float(decomposition["GM"]),
            "longitudinal_abs2": float(longitudinal_abs2),
            "transverse_abs2_sum": float(transverse_abs2),
            "longitudinal_transverse_ratio": float(
                longitudinal_transverse_ratio
            ),
            **row,
        })
    return rows


def theta_scan_rows(z, theta_values, phi_p, beam_momentum):
    """Return physical polarization amplitudes over a ``theta_p`` grid."""
    rows = []
    for theta_p in np.asarray(theta_values, dtype=float):
        rows.extend(scan_point_rows(z, theta_p, phi_p, beam_momentum))
    return rows


def z_scan_rows(z_values, theta_p, phi_p, beam_momentum):
    """Return physical polarization amplitudes over a recoil-fraction grid."""
    rows = []
    for z in np.asarray(z_values, dtype=float):
        rows.extend(scan_point_rows(z, theta_p, phi_p, beam_momentum))
    return rows


def plot_scan_pages(plt, pdf, rows, x_key, x_label, scan_description):
    """Append amplitude, phase, ratio, and summary pages for one scan."""
    transitions = [(s_in, s_out) for s_in in HELICITIES for s_out in HELICITIES]
    polarization_styles = {
        "T-": ("tab:blue", "--"),
        "T+": ("tab:orange", "--"),
        "L": ("tab:green", "-"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    for ax, (s_in, s_out) in zip(axes.flat, transitions):
        for polarization, (color, linestyle) in polarization_styles.items():
            selected = sorted(
                (
                    row for row in rows
                    if row["sIn"] == s_in
                    and row["sOut"] == s_out
                    and row["virtual_photon_polarization"] == polarization
                ),
                key=lambda row: row[x_key],
            )
            ax.plot(
                [row[x_key] for row in selected],
                [row["total_normalized_abs"] for row in selected],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=polarization,
            )
        ax.set_ylabel(
            r"$|\epsilon^\nu J_\nu|/|\mathcal{A}_{\rm unpol}|$"
        )
        ax.set_xlabel(x_label)
        ax.set_title(rf"$s_{{in}}={s_in:+d}\to s_{{out}}={s_out:+d}$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(
        f"Normalized proton--virtual-photon amplitudes versus {scan_description}"
    )
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    for ax, (s_in, s_out) in zip(axes.flat, transitions):
        for polarization, (color, linestyle) in polarization_styles.items():
            selected = sorted(
                (
                    row for row in rows
                    if row["sIn"] == s_in
                    and row["sOut"] == s_out
                    and row["virtual_photon_polarization"] == polarization
                ),
                key=lambda row: row[x_key],
            )
            x_values = np.asarray([row[x_key] for row in selected])
            magnitude = np.asarray([
                row["total_normalized_abs"] for row in selected
            ])
            phase = np.asarray([
                row["total_normalized_phase_over_pi"] for row in selected
            ])
            phase[magnitude < 1.0e-12] = np.nan
            ax.plot(
                x_values,
                phase,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=polarization,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\arg(\epsilon^\nu J_\nu)/\pi$")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(rf"$s_{{in}}={s_in:+d}\to s_{{out}}={s_out:+d}$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(
        f"Proton--virtual-photon amplitude phases versus {scan_description}"
    )
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    for ax, (s_in, s_out) in zip(axes.flat, transitions):
        selected = sorted(
            (
                row for row in rows
                if row["sIn"] == s_in
                and row["sOut"] == s_out
                and row["virtual_photon_polarization"] == "L"
            ),
            key=lambda row: row[x_key],
        )
        ax.plot(
            [row[x_key] for row in selected],
            [row["longitudinal_transverse_ratio"] for row in selected],
            color="tab:purple",
            linestyle="-",
            linewidth=1.8,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(
            r"$R_{L/T}=|A_L|^2/(|A_{T-}|^2+|A_{T+}|^2)$"
        )
        ax.set_title(rf"$s_{{in}}={s_in:+d}\to s_{{out}}={s_out:+d}$")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Longitudinal-to-transverse ratio versus {scan_description}"
    )
    pdf.savefig(fig)
    plt.close(fig)

    scan_values = sorted({row[x_key] for row in rows})
    summary = []
    for scan_value in scan_values:
        row = next(row for row in rows if row[x_key] == scan_value)
        summary.append((
            scan_value,
            row["Q2_GeV2"],
            row["unpolarized_amplitude"],
        ))
    summary = np.asarray(summary)
    fig, left = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    right = left.twinx()
    left.plot(summary[:, 0], summary[:, 1], color="tab:purple", linewidth=2.0)
    right.plot(summary[:, 0], summary[:, 2], color="tab:red", linewidth=2.0)
    left.set_xlabel(x_label)
    left.set_ylabel(r"$Q^2=-q^2$ [GeV$^2$]", color="tab:purple")
    right.set_ylabel(r"$|\mathcal{A}_{\rm unpol}|$", color="tab:red")
    left.tick_params(axis="y", colors="tab:purple")
    right.tick_params(axis="y", colors="tab:red")
    left.grid(alpha=0.25)
    left.set_title(
        f"Virtuality and unpolarized normalization versus {scan_description}"
    )
    pdf.savefig(fig)
    plt.close(fig)


def save_plot(theta_rows, z_rows, output_path=PLOT_PATH):
    """Plot the ``theta_p`` and ``z`` scans in one multipage PDF."""
    plt, PdfPages = require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        plot_scan_pages(
            plt,
            pdf,
            theta_rows,
            "theta_p_rad",
            r"$\theta_p$ [rad]",
            "final-proton angle",
        )
        plot_scan_pages(
            plt,
            pdf,
            z_rows,
            "z",
            r"$z$",
            "proton recoil fraction",
        )
    return output_path


def proton_virtual_photon_momenta(
    z,
    theta_p,
    phi_p=0.0,
    beam_momentum=BEAM_MOMENTUM_GEV,
):
    """Return on-shell ``p``, tilted ``p'``, and ``q=p-p'``.

    The incoming proton points along ``+z``. The outgoing proton has magnitude
    ``(1-z) * beam_momentum`` and direction ``(theta_p, phi_p)``. This
    generalizes the collinear focused ep-CM proton recoil while leaving the
    proton-current calculation independent of the final lepton-photon state.
    """
    z = float(z)
    theta_p = float(theta_p)
    phi_p = float(phi_p)
    beam_momentum = float(beam_momentum)
    if not 0.0 < z < 1.0:
        raise ValueError("z must lie strictly between zero and one.")
    if not 0.0 <= theta_p <= np.pi:
        raise ValueError("theta_p must lie in [0, pi].")
    if beam_momentum <= 0.0:
        raise ValueError("beam_momentum must be positive.")

    recoil_momentum = (1.0 - z) * beam_momentum
    p = np.array((
        np.sqrt(beam_momentum**2 + PROTON_MASS_GEV**2),
        0.0,
        0.0,
        beam_momentum,
    ))
    pp_direction = np.array((
        np.sin(theta_p) * np.cos(phi_p),
        np.sin(theta_p) * np.sin(phi_p),
        np.cos(theta_p),
    ))
    pp = np.concatenate((
        [np.sqrt(recoil_momentum**2 + PROTON_MASS_GEV**2)],
        recoil_momentum * pp_direction,
    ))
    q = p - pp
    t = real_scalar(mdot(q, q), "proton momentum transfer squared")
    if t >= 0.0:
        raise ValueError(
            f"The proton momentum transfer must be spacelike; obtained t={t}."
        )
    return {
        "momenta": {"p": p, "pp": pp},
        "virtual_photon": q,
        "t": t,
        "beam_momentum": beam_momentum,
        "recoil_momentum": recoil_momentum,
        "z": z,
        "theta_p": theta_p,
        "phi_p": phi_p,
    }


def build_report(
    kin,
    decomposition,
    polarizations,
    amplitude_rows,
):
    """Return a compact calculation and Ward-identity report."""
    q = kin["virtual_photon"]
    gauge_rows = [
        row for row in amplitude_rows
        if row["virtual_photon_polarization"] == "gauge"
    ]
    physical_rows = [
        row for row in amplitude_rows
        if row["is_physical_polarization"]
    ]
    maximum_physical = max(physical_rows, key=lambda row: row["total_abs"])
    unpolarized_abs2 = physical_rows[0]["unpolarized_abs2"]
    unpolarized_amplitude = physical_rows[0]["unpolarized_amplitude"]
    return "\n".join([
        "Proton coupled to a spacelike virtual photon",
        f"  beam momentum: {kin['beam_momentum']:.10g} GeV",
        f"  proton recoil fraction: z={kin['z']:.10g}",
        f"  final-proton theta_p: {kin['theta_p']:.10g} rad",
        f"  final-proton phi_p: {kin['phi_p']:.10g} rad",
        f"  proton mass: {PROTON_MASS_GEV:.12g} GeV",
        f"  form factors: {YAHL_MODEL_NAME}",
        f"  Q2=-q^2: {decomposition['Q2']:.12g} GeV^2",
        f"  F1={decomposition['F1'].real:.12g}",
        f"  F2={decomposition['F2'].real:.12g}",
        f"  GE={decomposition['GE']:.12g}",
        f"  GM={decomposition['GM']:.12g}",
        f"  p = {kin['momenta']['p']}",
        f"  p' = {kin['momenta']['pp']}",
        f"  q=p-p' = {q}",
        "  polarization checks:",
        *[
            f"    {name}: epsilon.q={mdot(epsilon, q):.4e}, "
            f"epsilon^2={mdot(np.conjugate(epsilon), epsilon):.4e}"
            for name, epsilon in polarizations.items()
        ],
        "  Ward check max |(q/sqrt(Q2)).J|: "
        f"{max(row['total_abs'] for row in gauge_rows):.6e}",
        "  unpolarized squared amplitude "
        "(1/2 sum_sIn,sOut,lambda |A|^2): "
        f"{unpolarized_abs2:.12g}",
        f"  unpolarized amplitude sqrt(|A|^2_unpol): {unpolarized_amplitude:.12g}",
        "  largest physical component: "
        f"sIn={maximum_physical['sIn']:+d}, "
        f"sOut={maximum_physical['sOut']:+d}, "
        f"lambda={maximum_physical['virtual_photon_polarization']}, "
        f"|A|={maximum_physical['total_abs']:.8g}, "
        f"|A/A_unpol|={maximum_physical['total_normalized_abs']:.8g}, "
        f"phase={maximum_physical['total_phase_over_pi']:+.6g} pi",
        f"  current components CSV: {CURRENT_CSV}",
        f"  polarization amplitudes CSV: {AMPLITUDE_CSV}",
        f"  theta scan CSV: {THETA_SCAN_CSV}",
        f"  z scan CSV: {Z_SCAN_CSV}",
        f"  amplitude plot: {PLOT_PATH}",
    ]) + "\n"


def main():
    """Calculate one editable proton--virtual-photon decomposition."""
    z = 0.90
    theta_p = 0.01
    phi_p = 0.0
    beam_momentum = BEAM_MOMENTUM_GEV
    theta_p_values = np.linspace(0.0, 0.02, 101)
    z_values = np.linspace(0.01, 0.80, 101)

    kin = proton_virtual_photon_momenta(
        z,
        theta_p,
        phi_p,
        beam_momentum,
    )
    p = kin["momenta"]["p"]
    pp = kin["momenta"]["pp"]
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    decomposition = proton_current_helicity_decomposition(
        p,
        pp,
        PROTON_MASS_GEV,
        F1,
        F2,
    )
    polarizations = virtual_photon_polarizations(kin["virtual_photon"])
    current_rows = current_component_rows(decomposition)
    amplitude_rows = virtual_photon_amplitude_rows(
        decomposition,
        polarizations,
    )
    write_rows(CURRENT_CSV, current_rows)
    write_rows(AMPLITUDE_CSV, amplitude_rows)
    theta_rows = theta_scan_rows(
        z,
        theta_p_values,
        phi_p,
        beam_momentum,
    )
    z_rows = z_scan_rows(
        z_values,
        theta_p,
        phi_p,
        beam_momentum,
    )
    write_rows(THETA_SCAN_CSV, theta_rows)
    write_rows(Z_SCAN_CSV, z_rows)
    save_plot(theta_rows, z_rows)
    report = build_report(
        kin,
        decomposition,
        polarizations,
        amplitude_rows,
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
