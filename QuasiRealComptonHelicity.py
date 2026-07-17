"""Helicity decomposition for quasi-real ``gamma* + lepton -> gamma + lepton``.

The calculation is performed in the incoming virtual-photon--lepton CM frame.
The incoming photon is spacelike, ``q_in^2 = -Q2``, and is resolved into two
transverse helicities, their normalized coherent sum
``(|+> + |->)/sqrt(2)``, and a longitudinal basis vector.  The outgoing photon
is real and has helicity ``+/-1``.  The tree-level reduced Compton amplitude
uses the same two-diagram lepton kernel and spinor conventions as the
Bethe--Heitler calculation.

The default points use a nominal 40 GeV energy per incoming particle:

    m_lepton = 1 GeV
    W(gamma* lepton) = 80 GeV
    Q2 = 1, 10, 100 GeV^2
    theta_cm = 2.25--2.65 rad

The reported amplitudes omit the overall QED coupling ``e^2``.  Longitudinal
entries are polarization-basis response amplitudes for an internal spacelike
photon, not probabilities for an asymptotic photon.
"""

import csv
from pathlib import Path

import numpy as np

from Algebra import (
    HELICITIES,
    cov,
    electron_spinor,
    mdot,
    photon_pol,
    spinor_bar,
)
from BHHelicityAmp import lepton_kernel
from PlotUtils import print_console_text, require_matplotlib
from config import HEAVY_LEPTON_MASS_GEV


# Editable calculation controls.
LEPTON_MASS_GEV = HEAVY_LEPTON_MASS_GEV
NOMINAL_INCOMING_ENERGY_GEV = 40.0
COM_INVARIANT_MASS_GEV = 2.0 * NOMINAL_INCOMING_ENERGY_GEV
Q2_VALUES_GEV2 = (1.0, 10.0, 40.0)
THETA_CM_VALUES = np.linspace(1.57, 3.14, 81)
PHI_CM_RAD = 0.0
WARD_RELATIVE_TOL = 1.0e-10
FIXED_INCOMING_LEPTON_HELICITY = +1

OUTPUT_DIR = Path("Output") / "QuasiRealComptonHelicity"
COMPONENT_CSV = OUTPUT_DIR / "compton_helicity_components.csv"
RESPONSE_CSV = OUTPUT_DIR / "compton_polarization_responses.csv"
PLOT_PDF = OUTPUT_DIR / "compton_helicity_components.pdf"
FIXED_COMPONENT_CSV = (
    OUTPUT_DIR / "compton_fixed_h_l_plus_helicity_components.csv"
)
FIXED_PLOT_PDF = OUTPUT_DIR / "compton_fixed_h_l_plus_amplitudes.pdf"
LOG_PATH = OUTPUT_DIR / "QuasiRealComptonHelicity.log"

TRANSVERSE_HELICITY_POLARIZATIONS = (-1, +1)
COHERENT_TRANSVERSE_SUM = "T_sum"
VIRTUAL_PHOTON_POLARIZATIONS = (
    *TRANSVERSE_HELICITY_POLARIZATIONS,
    COHERENT_TRANSVERSE_SUM,
    "L",
)
COHERENT_TRANSVERSE_SUM_LABEL = "(|+>+|->)/sqrt(2)"


def real_scalar(value, label):
    """Return a real scalar after rejecting non-negligible imaginary residue."""
    value = complex(value)
    if abs(value.imag) > 1.0e-10 * max(1.0, abs(value.real)):
        raise ValueError(f"{label} has a non-negligible imaginary part: {value}")
    return float(value.real)


def compton_cm_momenta(W, Q2, theta_cm, phi_cm=PHI_CM_RAD):
    """Return exact ``gamma* lepton -> gamma lepton`` CM four-momenta."""
    W = float(W)
    Q2 = float(Q2)
    theta_cm = float(theta_cm)
    phi_cm = float(phi_cm)
    mass = LEPTON_MASS_GEV
    if W <= mass:
        raise ValueError("COM_INVARIANT_MASS_GEV must exceed the lepton mass.")
    if Q2 <= 0.0:
        raise ValueError("Q2 must be positive for a spacelike incoming photon.")
    if not 0.0 <= theta_cm <= np.pi:
        raise ValueError("theta_cm must lie in [0, pi].")

    # Initial two-body kinematics with m_gamma*^2 = -Q2.
    kallen = (
        W**4 + Q2**2 + mass**4
        + 2.0 * W**2 * Q2
        - 2.0 * W**2 * mass**2
        + 2.0 * Q2 * mass**2
    )
    incoming_momentum = np.sqrt(max(0.0, kallen)) / (2.0 * W)
    virtual_energy = (W**2 - mass**2 - Q2) / (2.0 * W)
    lepton_energy = (W**2 + mass**2 + Q2) / (2.0 * W)
    if virtual_energy <= 0.0:
        raise ValueError("The selected W and Q2 give a nonpositive photon energy.")

    # Final two-body kinematics with a real photon.
    outgoing_momentum = (W**2 - mass**2) / (2.0 * W)
    outgoing_lepton_energy = (W**2 + mass**2) / (2.0 * W)
    direction = np.array((
        np.sin(theta_cm) * np.cos(phi_cm),
        np.sin(theta_cm) * np.sin(phi_cm),
        np.cos(theta_cm),
    ))
    q_in = np.array((virtual_energy, 0.0, 0.0, incoming_momentum))
    k_in = np.array((lepton_energy, 0.0, 0.0, -incoming_momentum))
    q_out = np.concatenate(([outgoing_momentum], outgoing_momentum * direction))
    k_out = np.concatenate((
        [outgoing_lepton_energy],
        -outgoing_momentum * direction,
    ))

    residual = q_in + k_in - q_out - k_out
    shells = {
        "q_in": abs(real_scalar(mdot(q_in, q_in), "q_in^2") + Q2),
        "k_in": abs(real_scalar(mdot(k_in, k_in), "k_in^2") - mass**2),
        "q_out": abs(real_scalar(mdot(q_out, q_out), "q_out^2")),
        "k_out": abs(real_scalar(mdot(k_out, k_out), "k_out^2") - mass**2),
    }
    if np.max(np.abs(residual)) > 1.0e-10 or max(shells.values()) > 1.0e-9:
        raise ValueError("Compton momenta failed conservation or on-shell checks.")
    return {
        "q_in": q_in,
        "k_in": k_in,
        "q_out": q_out,
        "k_out": k_out,
        "W": W,
        "Q2": Q2,
        "theta_cm": theta_cm,
        "phi_cm": phi_cm,
        "incoming_momentum": incoming_momentum,
        "outgoing_momentum": outgoing_momentum,
        "virtual_energy": virtual_energy,
        "conservation_error": float(np.max(np.abs(residual))),
        "mass_shell_error": float(max(shells.values())),
    }


def virtual_photon_polarization(q_in, polarization):
    """Return transverse or longitudinal incoming virtual-photon polarization."""
    if polarization in HELICITIES:
        return photon_pol(q_in, polarization)
    if polarization == COHERENT_TRANSVERSE_SUM:
        return (
            photon_pol(q_in, +1) + photon_pol(q_in, -1)
        ) / np.sqrt(2.0)
    if polarization != "L":
        raise ValueError(
            "Virtual-photon polarization must be -1, +1, 'T_sum', or 'L'."
        )
    Q2 = -real_scalar(mdot(q_in, q_in), "q_in^2")
    Q = np.sqrt(Q2)
    spatial = q_in[1:]
    momentum = np.linalg.norm(spatial)
    if momentum <= 0.0:
        raise ValueError("Longitudinal polarization requires nonzero momentum.")
    direction = spatial / momentum
    return np.concatenate((
        [momentum / Q],
        q_in[0] * direction / Q,
    )).astype(complex)


def reduced_compton_amplitude(
    momenta,
    incoming_polarization,
    h_in,
    h_out,
    outgoing_photon_helicity,
):
    """Return the reduced tree-level helicity amplitude without overall ``e^2``."""
    k_in = momenta["k_in"]
    k_out = momenta["k_out"]
    q_out = momenta["q_out"]
    eps_in = virtual_photon_polarization(
        momenta["q_in"], incoming_polarization
    )
    eps_out = photon_pol(q_out, outgoing_photon_helicity)
    incoming_spinor = electron_spinor(
        k_in, h_in, electron_mass=LEPTON_MASS_GEV
    )
    outgoing_bar = spinor_bar(electron_spinor(
        k_out, h_out, electron_mass=LEPTON_MASS_GEV
    ))
    eps_in_cov = cov(eps_in)
    eps_out_cov_star = cov(np.conjugate(eps_out))
    amplitude = 0.0 + 0.0j
    for mu in range(4):
        for nu in range(4):
            kernel = lepton_kernel(
                mu,
                nu,
                k_in,
                k_out,
                q_out,
                electron_mass=LEPTON_MASS_GEV,
            )
            amplitude += (
                eps_out_cov_star[mu]
                * (outgoing_bar @ kernel @ incoming_spinor)
                * eps_in_cov[nu]
            )
    return complex(amplitude)


def amplitude_with_vectors(momenta, eps_in, eps_out, h_in, h_out):
    """Return an amplitude with caller-supplied vectors for Ward checks."""
    incoming_spinor = electron_spinor(
        momenta["k_in"], h_in, electron_mass=LEPTON_MASS_GEV
    )
    outgoing_bar = spinor_bar(electron_spinor(
        momenta["k_out"], h_out, electron_mass=LEPTON_MASS_GEV
    ))
    eps_in_cov = cov(eps_in)
    eps_out_cov_star = cov(np.conjugate(eps_out))
    amplitude = 0.0 + 0.0j
    for mu in range(4):
        for nu in range(4):
            amplitude += (
                eps_out_cov_star[mu]
                * (
                    outgoing_bar
                    @ lepton_kernel(
                        mu,
                        nu,
                        momenta["k_in"],
                        momenta["k_out"],
                        momenta["q_out"],
                        electron_mass=LEPTON_MASS_GEV,
                    )
                    @ incoming_spinor
                )
                * eps_in_cov[nu]
            )
    return complex(amplitude)


def polarization_label(polarization):
    if polarization == "L":
        return "L"
    if polarization == COHERENT_TRANSVERSE_SUM:
        return COHERENT_TRANSVERSE_SUM_LABEL
    return f"{int(polarization):+d}"


def calculate_components():
    """Calculate all helicity components and polarization response summaries."""
    component_rows = []
    response_rows = []
    for Q2 in Q2_VALUES_GEV2:
        for theta_index, theta_cm in enumerate(THETA_CM_VALUES):
            momenta = compton_cm_momenta(
                COM_INVARIANT_MASS_GEV, Q2, theta_cm
            )
            amplitudes = {}
            for polarization in VIRTUAL_PHOTON_POLARIZATIONS:
                for h_in in HELICITIES:
                    for h_out in HELICITIES:
                        for h_gamma in HELICITIES:
                            key = (polarization, h_in, h_out, h_gamma)
                            amplitudes[key] = reduced_compton_amplitude(
                                momenta, polarization, h_in, h_out, h_gamma
                            )

            # The coherent sum is a prepared state made from the transverse
            # basis and must not be counted as an additional independent basis
            # component in normalization denominators.
            basis_amplitudes = {
                key: value for key, value in amplitudes.items()
                if key[0] != COHERENT_TRANSVERSE_SUM
            }
            total_response = sum(
                abs(value) ** 2 for value in basis_amplitudes.values()
            )
            transverse_response = sum(
                abs(value) ** 2
                for key, value in basis_amplitudes.items()
                if key[0] in HELICITIES
            )
            longitudinal_response = sum(
                abs(value) ** 2
                for key, value in basis_amplitudes.items()
                if key[0] == "L"
            )
            initial_state_responses = {
                (polarization, h_in): sum(
                    abs(value) ** 2
                    for key, value in amplitudes.items()
                    if key[0] == polarization and key[1] == h_in
                )
                for polarization in VIRTUAL_PHOTON_POLARIZATIONS
                for h_in in HELICITIES
            }
            final_state_ranks = {}
            for initial_state in initial_state_responses:
                polarization, h_in = initial_state
                ordered = sorted(
                    (
                        key for key in amplitudes
                        if key[0] == polarization and key[1] == h_in
                    ),
                    key=lambda key: abs(amplitudes[key]) ** 2,
                    reverse=True,
                )
                for rank, key in enumerate(ordered, start=1):
                    final_state_ranks[key] = rank

            # Ward checks over all lepton-helicity combinations.
            incoming_ward = 0.0
            outgoing_ward = 0.0
            amplitude_scale = max(
                max(abs(value) for value in amplitudes.values()), 1.0e-300
            )
            for h_in in HELICITIES:
                for h_out in HELICITIES:
                    incoming_ward = max(
                        incoming_ward,
                        abs(amplitude_with_vectors(
                            momenta,
                            momenta["q_in"],
                            photon_pol(momenta["q_out"], +1),
                            h_in,
                            h_out,
                        )),
                    )
                    outgoing_ward = max(
                        outgoing_ward,
                        abs(amplitude_with_vectors(
                            momenta,
                            virtual_photon_polarization(
                                momenta["q_in"], +1
                            ),
                            momenta["q_out"],
                            h_in,
                            h_out,
                        )),
                    )

            for key, amplitude in amplitudes.items():
                polarization, h_in, h_out, h_gamma = key
                response = abs(amplitude) ** 2
                component_rows.append({
                    "Q2_GeV2": Q2,
                    "Q_GeV": np.sqrt(Q2),
                    "W_GeV": momenta["W"],
                    "theta_index": theta_index,
                    "theta_cm_rad": theta_cm,
                    "phi_cm_rad": PHI_CM_RAD,
                    "incoming_virtual_photon_polarization": polarization_label(
                        polarization
                    ),
                    "h_l_in": h_in,
                    "h_l_out": h_out,
                    "h_gamma_out": h_gamma,
                    "amplitude_real": amplitude.real,
                    "amplitude_imag": amplitude.imag,
                    "amplitude_abs": abs(amplitude),
                    "amplitude_phase_rad": np.angle(amplitude),
                    "amplitude_abs2": response,
                    "fraction_all_basis_components": response / total_response,
                    "initial_state_response": initial_state_responses[
                        (polarization, h_in)
                    ],
                    "fraction_within_initial_state": (
                        response / initial_state_responses[(polarization, h_in)]
                    ),
                    "final_state_rank_within_initial_state": final_state_ranks[key],
                    "incoming_photon_energy_GeV": momenta["virtual_energy"],
                    "incoming_momentum_GeV": momenta["incoming_momentum"],
                    "outgoing_momentum_GeV": momenta["outgoing_momentum"],
                })

            for polarization in VIRTUAL_PHOTON_POLARIZATIONS:
                response = sum(
                    abs(value) ** 2
                    for key, value in amplitudes.items()
                    if key[0] == polarization
                )
                response_rows.append({
                    "Q2_GeV2": Q2,
                    "Q_GeV": np.sqrt(Q2),
                    "W_GeV": momenta["W"],
                    "theta_cm_rad": theta_cm,
                    "incoming_virtual_photon_polarization": polarization_label(
                        polarization
                    ),
                    "helicity_summed_response": response,
                    "fraction_all_basis_responses": response / total_response,
                    "transverse_response": transverse_response,
                    "longitudinal_response": longitudinal_response,
                    "longitudinal_to_transverse_ratio": (
                        longitudinal_response / transverse_response
                    ),
                    "Q_over_photon_energy": (
                        np.sqrt(Q2) / momenta["virtual_energy"]
                    ),
                    "incoming_ward_abs": incoming_ward,
                    "outgoing_ward_abs": outgoing_ward,
                    "incoming_ward_relative": incoming_ward / amplitude_scale,
                    "outgoing_ward_relative": outgoing_ward / amplitude_scale,
                })
            if (
                incoming_ward / amplitude_scale > WARD_RELATIVE_TOL
                or outgoing_ward / amplitude_scale > WARD_RELATIVE_TOL
            ):
                raise ValueError(
                    "Compton Ward-identity check failed at "
                    f"Q2={Q2:.8g}, theta_cm={theta_cm:.8g}."
                )
    return component_rows, response_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_plots(component_rows):
    """Plot all final helicity components for each incoming photon polarization."""
    plt, PdfPages = require_matplotlib()
    with PdfPages(PLOT_PDF) as pdf:
        for Q2 in Q2_VALUES_GEV2:
            for polarization in VIRTUAL_PHOTON_POLARIZATIONS:
                fig, axes = plt.subplots(
                    1, 2, figsize=(12.0, 4.8), constrained_layout=True
                )
                for ax, h_in in zip(axes, HELICITIES):
                    for h_out in HELICITIES:
                        for h_gamma in HELICITIES:
                            selected = [
                                row for row in component_rows
                                if np.isclose(row["Q2_GeV2"], Q2)
                                and row["incoming_virtual_photon_polarization"]
                                == polarization_label(polarization)
                                and row["h_l_in"] == h_in
                                and row["h_l_out"] == h_out
                                and row["h_gamma_out"] == h_gamma
                            ]
                            selected.sort(key=lambda row: row["theta_cm_rad"])
                            ax.plot(
                                [row["theta_cm_rad"] for row in selected],
                                [row["amplitude_abs"] for row in selected],
                                label=(
                                    rf"$h_\ell'={h_out:+d},"
                                    rf"\ h_\gamma={h_gamma:+d}$"
                                ),
                            )
                    ax.set_yscale("log")
                    ax.set_xlabel(r"$\theta_{\rm cm}$ [rad]")
                    ax.set_ylabel(r"$|\mathcal{M}/e^2|$")
                    ax.set_title(rf"$h_\ell={h_in:+d}$")
                    ax.grid(alpha=0.25)
                    ax.legend(fontsize=8)
                fig.suptitle(
                    r"$\gamma^*\ell\to\gamma\ell$: "
                    f"incoming polarization {polarization_label(polarization)}, "
                    rf"$Q^2={Q2:.6g}\ {{\rm GeV}}^2$"
                )
                pdf.savefig(fig)
                plt.close(fig)


def fixed_incoming_lepton_rows(component_rows):
    """Return components with the configured incoming lepton helicity."""
    return [
        row for row in component_rows
        if row["h_l_in"] == FIXED_INCOMING_LEPTON_HELICITY
    ]


def save_fixed_incoming_lepton_plots(component_rows):
    """For each initial photon state, compare all final helicity channels."""
    plt, PdfPages = require_matplotlib()
    final_styles = {
        (-1, -1): ("tab:blue", "-"),
        (-1, +1): ("tab:orange", "--"),
        (+1, -1): ("tab:green", "-."),
        (+1, +1): ("tab:red", ":"),
    }
    fixed_rows = fixed_incoming_lepton_rows(component_rows)
    with PdfPages(FIXED_PLOT_PDF) as pdf:
        for Q2 in Q2_VALUES_GEV2:
            fig, axes = plt.subplots(
                2, 2, figsize=(12.5, 9.2), constrained_layout=True,
                sharey=True,
            )
            for ax, polarization in zip(
                axes.flat,
                ("-1", "+1", COHERENT_TRANSVERSE_SUM_LABEL, "L"),
            ):
                for (h_out, h_gamma), (color, linestyle) in final_styles.items():
                    selected = [
                        row for row in fixed_rows
                        if np.isclose(row["Q2_GeV2"], Q2)
                        and row["incoming_virtual_photon_polarization"]
                        == polarization
                        and row["h_l_out"] == h_out
                        and row["h_gamma_out"] == h_gamma
                    ]
                    selected.sort(key=lambda row: row["theta_cm_rad"])
                    ax.plot(
                        [row["theta_cm_rad"] for row in selected],
                        [row["amplitude_abs"] for row in selected],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.7,
                        label=(
                            rf"$h_\ell'={h_out:+d},"
                            rf"\ h_\gamma={h_gamma:+d}$"
                        ),
                    )
                ax.set_yscale("log")
                ax.set_xlabel(r"$\theta_{\rm cm}$ [rad]")
                ax.set_ylabel(r"$|\mathcal{M}/e^2|$")
                ax.set_title(
                    rf"$h_\ell={FIXED_INCOMING_LEPTON_HELICITY:+d}$, "
                    + (
                        r"$|\gamma^*\rangle=(|+\rangle+|-\rangle)/\sqrt{2}$"
                        if polarization == COHERENT_TRANSVERSE_SUM_LABEL
                        else rf"$\lambda_{{\gamma^*}}={polarization}$"
                    )
                )
                ax.grid(alpha=0.25)
                ax.legend(fontsize=9)
            fig.suptitle(
                r"$\gamma^*\ell\to\gamma\ell$, "
                "final-state comparison for each initial state: "
                rf"$Q^2={Q2:.6g}\ {{\rm GeV}}^2$"
            )
            pdf.savefig(fig)
            plt.close(fig)


def nearest_response_rows(response_rows, theta, Q2):
    return [
        min(
            (
                row for row in response_rows
                if row["incoming_virtual_photon_polarization"] == label
                and np.isclose(row["Q2_GeV2"], Q2)
            ),
            key=lambda row: abs(row["theta_cm_rad"] - theta),
        )
        for label in ("-1", "+1", "L")
    ]


def build_report(component_rows, response_rows):
    """Return a concise report at the center of the configured angular scan."""
    theta = float(THETA_CM_VALUES[len(THETA_CM_VALUES) // 2])
    lines = [
        "Quasi-real gamma* + lepton Compton helicity decomposition",
        f"  lepton mass: {LEPTON_MASS_GEV:.12g} GeV",
        f"  W(gamma* lepton): {COM_INVARIANT_MASS_GEV:.12g} GeV",
        f"  Q2 values: {Q2_VALUES_GEV2}",
        (
            f"  theta_cm range: {THETA_CM_VALUES[0]:.8g}.."
            f"{THETA_CM_VALUES[-1]:.8g} rad ({len(THETA_CM_VALUES)} points)"
        ),
        "  amplitudes omit the overall e^2 coupling",
        "  longitudinal response is a spacelike basis diagnostic",
        (
            "  coherent transverse state: "
            "|gamma*>=(|+>+|->)/sqrt(2), combined at amplitude level"
        ),
        "",
    ]
    for Q2 in Q2_VALUES_GEV2:
        selected = nearest_response_rows(response_rows, theta, Q2)
        reference = selected[0]
        component_reference = next(
            row for row in component_rows
            if np.isclose(row["Q2_GeV2"], Q2)
            and np.isclose(row["theta_cm_rad"], theta)
        )
        incoming_photon_energy = component_reference[
            "incoming_photon_energy_GeV"
        ]
        incoming_lepton_energy = (
            COM_INVARIANT_MASS_GEV - incoming_photon_energy
        )
        outgoing_photon_energy = component_reference["outgoing_momentum_GeV"]
        outgoing_lepton_energy = (
            COM_INVARIANT_MASS_GEV - outgoing_photon_energy
        )
        lines.extend([
            f"Q2={Q2:.8g} GeV^2 at theta_cm={theta:.8g} rad:",
            (
                f"  energies in: E_gamma*={incoming_photon_energy:.10g}, "
                f"E_l={incoming_lepton_energy:.10g} GeV"
            ),
            (
                f"  energies out: E_gamma={outgoing_photon_energy:.10g}, "
                f"E_l'={outgoing_lepton_energy:.10g} GeV"
            ),
        ])
        coherent_components = [
            row for row in component_rows
            if np.isclose(row["Q2_GeV2"], Q2)
            and np.isclose(row["theta_cm_rad"], theta)
            and row["h_l_in"] == FIXED_INCOMING_LEPTON_HELICITY
            and row["incoming_virtual_photon_polarization"]
            == COHERENT_TRANSVERSE_SUM_LABEL
        ]
        coherent_response = sum(
            row["amplitude_abs2"] for row in coherent_components
        )
        lines.append(
            "  coherent transverse response for fixed h_l="
            f"{FIXED_INCOMING_LEPTON_HELICITY:+d}: {coherent_response:.12g}"
        )
        for row in selected:
            lines.append(
                f"  incoming gamma* {row['incoming_virtual_photon_polarization']}: "
                f"summed response={row['helicity_summed_response']:.12g}, "
                f"basis fraction={row['fraction_all_basis_responses']:.12g}"
            )
        lines.extend([
            (
                "  longitudinal/transverse response ratio: "
                f"{reference['longitudinal_to_transverse_ratio']:.12g}"
            ),
            f"  Q/omega_gamma*: {reference['Q_over_photon_energy']:.12g}",
            (
                "  Ward relative residuals (in,out): "
                f"({reference['incoming_ward_relative']:.3e}, "
                f"{reference['outgoing_ward_relative']:.3e})"
            ),
            "",
        ])
    lines.append(
        "Fixed-h_l=+1 channel amplitudes are tabulated in the dedicated CSV."
    )
    lines.extend([
        "",
        f"  component CSV: {COMPONENT_CSV}",
        f"  response CSV: {RESPONSE_CSV}",
        f"  plot PDF: {PLOT_PDF}",
        f"  fixed-h_l component CSV: {FIXED_COMPONENT_CSV}",
        f"  fixed-h_l comparison PDF: {FIXED_PLOT_PDF}",
    ])
    return "\n".join(lines) + "\n"


def validate_settings():
    if LEPTON_MASS_GEV <= 0.0:
        raise ValueError("LEPTON_MASS_GEV must be positive.")
    if COM_INVARIANT_MASS_GEV <= LEPTON_MASS_GEV:
        raise ValueError("COM_INVARIANT_MASS_GEV must exceed the lepton mass.")
    if not Q2_VALUES_GEV2 or any(value <= 0.0 for value in Q2_VALUES_GEV2):
        raise ValueError("Q2_VALUES_GEV2 must contain positive values.")
    if len(THETA_CM_VALUES) < 2 or np.any(np.diff(THETA_CM_VALUES) <= 0.0):
        raise ValueError("THETA_CM_VALUES must be strictly increasing.")


def main():
    validate_settings()
    component_rows, response_rows = calculate_components()
    write_csv(COMPONENT_CSV, component_rows)
    write_csv(RESPONSE_CSV, response_rows)
    write_csv(FIXED_COMPONENT_CSV, fixed_incoming_lepton_rows(component_rows))
    save_plots(component_rows)
    save_fixed_incoming_lepton_plots(component_rows)
    report = build_report(component_rows, response_rows)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
