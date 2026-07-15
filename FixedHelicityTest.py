"""Small prepared-spin example using the same physics path as ConfigGen.

Edit the input block below and run ``python3 FixedHelicityTest.py``. Fixed
helicities, longitudinal/transverse pure states, and unpolarized helicity
averages are supported for both incoming particles.
"""

import csv
from pathlib import Path

import numpy as np

from Algebra import HELICITIES
from config import ELECTRON_MASS_GEV, PROTON_MASS_GEV
from FormFactors import yahl_dirac_pauli_from_t
from Kinematics import kinematics_user_from_independent
from PlotUtils import require_matplotlib
from SpinDensityMat import (
    amplitude_table,
    entanglement_measures_from_density_matrix,
    entanglement_measures_from_state,
    initial_spin_states,
    outgoing_spin_states,
    prepared_spin_coefficients,
)


# ---------------------------------------------------------------------------
# Each state may be -1, +1, "U", "L", "Tx", "-Tx", "Ty", or "-Ty".
# "U" (or "unpolarized") is the equal incoherent average over helicities.
# ---------------------------------------------------------------------------
# Near-maximal proton-photon concurrence from the transverse-proton scan:
# C_p_gamma = 0.9999305343 at phi_in = pi/2.
S = 21.515844
THETA_IN = 1.570795
PHI_IN = np.pi / 2.0
QOUT = 0.50
PHIOUT = np.pi / 2.0
ELECTRON_STATE = "U"
PROTON_STATE = "Tx"

OUTPUT_DIR = Path("Output") / "FixedHelicityTest"


def _state_coefficients(state, particle):
    """Return helicity-basis coefficients for one pure incoming state."""
    if state in HELICITIES:
        return {int(state): 1.0 + 0.0j}
    if isinstance(state, str) and state in {"L", "Tx", "-Tx", "Ty", "-Ty"}:
        return prepared_spin_coefficients(state)
    raise ValueError(
        f"{particle}_state must be -1, +1, 'U', 'unpolarized', 'L', "
        "'Tx', '-Tx', 'Ty', or '-Ty'."
    )


def _state_ensemble(state, particle):
    """Return ``(weight, coefficients, label)`` terms for one particle."""
    if isinstance(state, str) and state.lower() in {"u", "unpolarized"}:
        return [
            (0.5, {helicity: 1.0 + 0.0j}, f"{particle} h={helicity:+d}")
            for helicity in HELICITIES
        ]
    return [(1.0, _state_coefficients(state, particle), f"{particle} {state}")]


def evaluate_prepared_spin_configuration(
    s,
    theta_in,
    phi_in,
    qOut,
    phiOut,
    electron_state,
    proton_state,
):
    """Return momenta and entanglement for a prepared initial spin ensemble."""
    electron_ensemble = _state_ensemble(electron_state, "electron")
    proton_ensemble = _state_ensemble(proton_state, "proton")

    kin = kinematics_user_from_independent(
        s,
        theta_in,
        phi_in,
        qOut,
        phiOut,
        PROTON_MASS_GEV,
        electron_mass=ELECTRON_MASS_GEV,
        label=f"electron {electron_state}, proton {proton_state}",
    )
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    amplitudes = amplitude_table(
        kin["momenta"],
        PROTON_MASS_GEV,
        F1,
        F2,
        electron_mass=ELECTRON_MASS_GEV,
    )
    outgoing_components = []
    outgoing_rho = np.zeros(
        (len(outgoing_spin_states()), len(outgoing_spin_states())), dtype=complex
    )
    for electron_weight, electron_coefficients, electron_label in electron_ensemble:
        for proton_weight, proton_coefficients, proton_label in proton_ensemble:
            outgoing_state = np.zeros(len(outgoing_spin_states()), dtype=complex)
            for h_in, electron_coefficient in electron_coefficients.items():
                for s_in, proton_coefficient in proton_coefficients.items():
                    incoming_index = initial_spin_states().index((h_in, s_in))
                    outgoing_state += (
                        electron_coefficient
                        * proton_coefficient
                        * amplitudes[incoming_index]
                    )
            weight = electron_weight * proton_weight
            outgoing_components.append({
                "weight": weight,
                "label": f"{electron_label}, {proton_label}",
                "state": outgoing_state,
            })
            outgoing_rho += weight * np.outer(outgoing_state, outgoing_state.conj())

    norm = float(np.trace(outgoing_rho).real)
    if norm <= 0.0:
        raise ZeroDivisionError("The selected outgoing state has zero norm.")
    normalized_rho = outgoing_rho / norm
    is_pure_input = len(outgoing_components) == 1
    outgoing_state = outgoing_components[0]["state"] if is_pure_input else None
    normalized_state = outgoing_state / np.sqrt(norm) if is_pure_input else None
    entanglement = (
        entanglement_measures_from_state(normalized_state)
        if is_pure_input
        else entanglement_measures_from_density_matrix(normalized_rho)
    )

    return {
        "kinematics": kin,
        "F1": F1,
        "F2": F2,
        "electron_state": electron_state,
        "proton_state": proton_state,
        "amplitudes": outgoing_state,
        "normalized_state": normalized_state,
        "outgoing_components": outgoing_components,
        "density_matrix": normalized_rho,
        "purity": float(np.trace(normalized_rho @ normalized_rho).real),
        "probabilities": np.real(np.diag(normalized_rho)),
        "squared_amplitude": norm,
        "entanglement": entanglement,
    }


def write_test_outputs(result, output_dir=OUTPUT_DIR):
    """Write compact ConfigGen-style CSVs and a one-page summary PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kin = result["kinematics"]

    momentum_path = output_dir / "momentum_configuration.csv"
    with momentum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("momentum", "E", "px", "py", "pz", "mass_shell"))
        for name in ("k", "p", "kp", "pp", "qout"):
            vector = kin["momenta"][name]
            mass_shell = vector[0] ** 2 - np.dot(vector[1:4], vector[1:4])
            writer.writerow((name, *vector, mass_shell))

    amplitude_path = output_dir / "outgoing_amplitudes.csv"
    with amplitude_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "initial_component", "ensemble_weight", "hOut", "sOut", "lambda",
            "amplitude_real", "amplitude_imag", "amplitude_abs",
            "amplitude_phase_rad", "normalized_weighted_probability",
        ))
        for component in result["outgoing_components"]:
            for labels, amplitude in zip(outgoing_spin_states(), component["state"]):
                writer.writerow((
                    component["label"], component["weight"], *labels,
                    amplitude.real, amplitude.imag, abs(amplitude),
                    np.angle(amplitude),
                    component["weight"] * abs(amplitude) ** 2
                    / result["squared_amplitude"],
                ))

    entanglement_path = output_dir / "entanglement_measurements.csv"
    with entanglement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("measurement", "value"))
        writer.writerow(("electron_state", result["electron_state"]))
        writer.writerow(("proton_state", result["proton_state"]))
        writer.writerow(("outgoing_density_purity", result["purity"]))
        writer.writerow(("squared_amplitude", result["squared_amplitude"]))
        writer.writerow(("F1", result["F1"]))
        writer.writerow(("F2", result["F2"]))
        writer.writerows(result["entanglement"].items())

    pdf_path = write_summary_pdf(result, output_dir)
    return momentum_path, amplitude_path, entanglement_path, pdf_path


def _plot_momentum_panel(ax, momenta, dimensions, title):
    """Plot momentum arrows in either two or three spatial dimensions."""
    colors = {"k": "tab:blue", "p": "tab:orange", "kp": "tab:cyan",
              "pp": "tab:red", "qout": "tab:green"}
    labels = {"k": r"$\ell$", "p": r"$P$", "kp": r"$\ell'$",
              "pp": r"$P'$", "qout": r"$q_\gamma$"}
    components = [1, 2] if dimensions == 2 else [1, 2, 3]
    scale = max(
        1.0,
        1.2 * max(np.linalg.norm(momenta[name][components]) for name in labels),
    )
    for name in labels:
        end = momenta[name][components]
        if dimensions == 2:
            ax.quiver(0, 0, *end, angles="xy", scale_units="xy", scale=1,
                      color=colors[name], label=labels[name])
        else:
            ax.quiver(0, 0, 0, *end, color=colors[name],
                      arrow_length_ratio=0.12, label=labels[name])
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    if dimensions == 3:
        ax.set_zlim(-scale, scale)
        ax.set_zlabel(r"$p_z$ [GeV]")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$ [GeV]")
    ax.set_ylabel(r"$p_y$ [GeV]")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", ncol=2)


def write_summary_pdf(result, output_dir=OUTPUT_DIR):
    """Write a ConfigGen-style visual summary for the selected test point."""
    plt, PdfPages = require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "configuration_summary.pdf"
    kin = result["kinematics"]

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.0, 8.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.25))
        ax3d = fig.add_subplot(grid[0, 0], projection="3d")
        ax2d = fig.add_subplot(grid[1, 0])
        _plot_momentum_panel(ax3d, kin["momenta"], 3, "3D momenta")
        _plot_momentum_panel(ax2d, kin["momenta"], 2, "Transverse momenta")

        ax_text = fig.add_subplot(grid[0, 1])
        ax_text.axis("off")
        lines = [
            "Prepared/unpolarized-spin Bethe-Heitler test",
            (
                f"electron={result['electron_state']}, "
                f"proton={result['proton_state']}"
            ),
            f"m_e={ELECTRON_MASS_GEV:.9g} GeV, m_p={PROTON_MASS_GEV:.6g} GeV",
            "",
            f"s={kin['s']:.6g} GeV^2, sqrt(s)={kin['sqrt_s']:.6g} GeV",
            f"theta_in={kin['theta_in']:.6g}, phi_in={kin['phi_in']:.6g}",
            f"E_gamma={kin['qOut']:.6g} GeV, phi_gamma={kin['phiOut']:.6g}",
            f"Q2={kin['Q2']:.6g}, xB={kin['xB']:.6g}, t={kin['t']:.6g}",
            f"F1={result['F1']:.6g}, F2={result['F2']:.6g}",
            f"|M|^2={result['squared_amplitude']:.6g}",
            f"outgoing density purity={result['purity']:.7g}",
            "",
            "Entanglement measurements",
        ]
        lines.extend(
            f"{name:12s} {value:.7g}"
            for name, value in result["entanglement"].items()
        )
        ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                     family="monospace", fontsize=9.5)

        ax_amp = fig.add_subplot(grid[1, 1])
        records = []
        for component in result["outgoing_components"]:
            for labels, amplitude in zip(outgoing_spin_states(), component["state"]):
                fraction = (
                    component["weight"] * abs(amplitude) ** 2
                    / result["squared_amplitude"]
                )
                records.append((fraction, np.angle(amplitude), labels, component["label"]))
        records.sort(reverse=True, key=lambda record: record[0])
        records = [record for record in records if record[0] >= 1.0e-8][:8]
        fractions = np.asarray([record[0] for record in records])
        labels = [
            rf"$h_e={h:+d}, h_p={s:+d}, h_\gamma={lam:+d}$" + "\n" + initial
            for _, _, (h, s, lam), initial in records
        ]
        y_positions = np.arange(len(records))
        bars = ax_amp.barh(y_positions, fractions, color="tab:blue", alpha=0.75)
        ax_amp.set_yticks(y_positions, labels)
        ax_amp.invert_yaxis()
        ax_amp.set_xlabel(r"Ensemble-weighted $|A|^2$ fraction")
        ax_amp.set_xlim(0.0, max(0.08, 1.30 * fractions.max()))
        ax_amp.set_title("Leading outgoing components and amplitude phases")
        for bar, (_, phase, _, _) in zip(bars, records):
            ax_amp.text(
                bar.get_width(), bar.get_y() + 0.5 * bar.get_height(),
                rf"  phase={phase:.3f} rad", va="center", fontsize=8,
            )
        ax_amp.tick_params(axis="y", labelsize=7)
        ax_amp.grid(axis="x", alpha=0.25)

        fig.suptitle("FixedHelicityTest configuration summary", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def main():
    """Evaluate the editable test point and write CSV and PDF summaries."""
    result = evaluate_prepared_spin_configuration(
        S,
        THETA_IN,
        PHI_IN,
        QOUT,
        PHIOUT,
        ELECTRON_STATE,
        PROTON_STATE,
    )
    paths = write_test_outputs(result)
    print(
        f"prepared incoming state: electron={ELECTRON_STATE}, "
        f"proton={PROTON_STATE}"
    )
    for name, value in result["entanglement"].items():
        print(f"{name:14s} = {value:.12g}")
    for path in paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
