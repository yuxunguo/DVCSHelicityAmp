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
from PlotUtils import (
    configure_polar_angle_axis,
    print_console_text,
    require_matplotlib,
)
from ProtonVirtualPhotonAmp import (
    EPCM_W_ELECTRON_MIXING_ANGLE_RAD,
    EPCM_W_HELICITY_ORDER,
    EPCM_W_PROTON_MIXING_ANGLE_RAD,
    EPCM_W_VIRTUAL_POLARIZATION_ORDER,
    epcm_w_proton_results,
)
from SpinDensityMat import (
    amplitude_table,
    initial_spin_states,
    outgoing_spin_states,
)
from config import (
    ELECTRON_MASS_GEV,
    HEAVY_LEPTON_MASS_GEV,
    PROTON_MASS_GEV,
)


# Editable calculation controls.
LEPTON_MASS_GEV = HEAVY_LEPTON_MASS_GEV
NOMINAL_INCOMING_ENERGY_GEV = 40.0
COM_INVARIANT_MASS_GEV = 2.0 * NOMINAL_INCOMING_ENERGY_GEV
Q2_VALUES_GEV2 = (1.0, 10.0, 40.0)
THETA_CM_VALUES = np.linspace(1.57, 3.14, 81)
PHI_CM_RAD = 0.0
WARD_RELATIVE_TOL = 1.0e-10
FIXED_INCOMING_LEPTON_HELICITY = +1

OUTPUT_DIR = Path("Output_local") / "QuasiRealComptonHelicity"
COMPONENT_CSV = OUTPUT_DIR / "compton_helicity_components.csv"
RESPONSE_CSV = OUTPUT_DIR / "compton_polarization_responses.csv"
PLOT_PDF = OUTPUT_DIR / "compton_helicity_components.pdf"
FIXED_COMPONENT_CSV = (
    OUTPUT_DIR / "compton_fixed_h_l_plus_helicity_components.csv"
)
TRANSVERSE_LEPTON_COMPONENT_CSV = (
    OUTPUT_DIR / "compton_transverse_initial_lepton_components.csv"
)
FIXED_PLOT_PDF = (
    OUTPUT_DIR / "compton_initial_lepton_polarization_amplitudes.pdf"
)
LOG_PATH = OUTPUT_DIR / "QuasiRealComptonHelicity.log"
EPCM_W_COMPTON_MATRIX_CSV = OUTPUT_DIR / "epcm_w_compton_matrix.csv"
EPCM_W_FULL_MATRIX_CSV = OUTPUT_DIR / "epcm_w_full_scattering_matrix.csv"
EPCM_W_FINAL_STATE_CSV = OUTPUT_DIR / "epcm_w_final_state.csv"
EPCM_W_LOG_PATH = OUTPUT_DIR / "epcm_w_matrix_chain.log"
EPCM_W_MATRIX_RECONSTRUCTION_TOL = 1.0e-10

TRANSVERSE_HELICITY_POLARIZATIONS = (-1, +1)
COHERENT_TRANSVERSE_SUM = "T_sum"
VIRTUAL_PHOTON_POLARIZATIONS = (
    *TRANSVERSE_HELICITY_POLARIZATIONS,
    COHERENT_TRANSVERSE_SUM,
    "L",
)
PLOTTED_VIRTUAL_PHOTON_POLARIZATIONS = (
    *TRANSVERSE_HELICITY_POLARIZATIONS,
    "L",
)
COHERENT_TRANSVERSE_SUM_LABEL = "(|+>+|->)/sqrt(2)"
TRANSVERSE_LEPTON_LABEL = "(|+>+|->)/sqrt(2)"


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


def amplitude_with_vectors(
    momenta,
    eps_in,
    eps_out,
    h_in,
    h_out,
    electron_mass=None,
):
    """Return an amplitude with caller-supplied vectors for Ward checks."""
    if electron_mass is None:
        electron_mass = LEPTON_MASS_GEV
    electron_mass = float(electron_mass)
    incoming_spinor = electron_spinor(
        momenta["k_in"], h_in, electron_mass=electron_mass
    )
    outgoing_bar = spinor_bar(electron_spinor(
        momenta["k_out"], h_out, electron_mass=electron_mass
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
                        electron_mass=electron_mass,
                    )
                    @ incoming_spinor
                )
                * eps_in_cov[nu]
            )
    return complex(amplitude)


def _complex_output_fields(prefix, value):
    """Return real, imaginary, magnitude, and phase fields for one value."""
    value = complex(value)
    return {
        f"{prefix}_real": float(value.real),
        f"{prefix}_imag": float(value.imag),
        f"{prefix}_abs": float(abs(value)),
        f"{prefix}_phase_rad": float(np.angle(value)),
    }


def epcm_w_compton_matrix(proton_results=None):
    """Return the W-benchmark Compton matrix in the shared gamma* basis.

    Rows are ``(gamma+,e+), (gamma+,e-), (gamma-,e+), (gamma-,e-)``.
    Columns are ``(T-,e+), (T-,e-), (T+,e+), (T+,e-), (L,e+), (L,e-)``.
    """
    if proton_results is None:
        proton_results = epcm_w_proton_results()
    kin = proton_results["kinematics"]
    polarizations = proton_results["polarizations"]
    momenta = {
        "q_in": kin["virtual_photon"],
        "k_in": kin["momenta"]["k"],
        "q_out": kin["momenta"]["qout"],
        "k_out": kin["momenta"]["kp"],
    }
    matrix = np.zeros((4, 6), dtype=complex)
    for photon_index, photon_helicity in enumerate(EPCM_W_HELICITY_ORDER):
        eps_out = photon_pol(kin["momenta"]["qout"], photon_helicity)
        for out_index, h_out in enumerate(EPCM_W_HELICITY_ORDER):
            row = 2 * photon_index + out_index
            for polarization_index, name in enumerate(
                EPCM_W_VIRTUAL_POLARIZATION_ORDER
            ):
                for in_index, h_in in enumerate(EPCM_W_HELICITY_ORDER):
                    column = 2 * polarization_index + in_index
                    matrix[row, column] = amplitude_with_vectors(
                        momenta,
                        polarizations[name],
                        eps_out,
                        h_in,
                        h_out,
                        electron_mass=ELECTRON_MASS_GEV,
                    )
    return matrix


def _direct_epcm_w_scattering_matrix(proton_results):
    """Return the directly evaluated 8 x 4 BH matrix in p,gamma,e ordering."""
    kin = proton_results["kinematics"]
    decomposition = proton_results["decomposition"]
    amplitudes = amplitude_table(
        kin["momenta"],
        PROTON_MASS_GEV,
        decomposition["F1"],
        decomposition["F2"],
        electron_mass=ELECTRON_MASS_GEV,
    )
    matrix = np.zeros((8, 4), dtype=complex)
    for proton_out_index, s_out in enumerate(EPCM_W_HELICITY_ORDER):
        for photon_index, photon_helicity in enumerate(EPCM_W_HELICITY_ORDER):
            for lepton_out_index, h_out in enumerate(EPCM_W_HELICITY_ORDER):
                row = (
                    4 * proton_out_index
                    + 2 * photon_index
                    + lepton_out_index
                )
                source_out = outgoing_spin_states().index(
                    (h_out, s_out, photon_helicity)
                )
                for proton_in_index, s_in in enumerate(EPCM_W_HELICITY_ORDER):
                    for lepton_in_index, h_in in enumerate(
                        EPCM_W_HELICITY_ORDER
                    ):
                        column = 2 * proton_in_index + lepton_in_index
                        source_in = initial_spin_states().index((h_in, s_in))
                        matrix[row, column] = amplitudes[source_in, source_out]
    return matrix


def epcm_w_matrix_chain():
    """Assemble H, the polarization propagator, C, and the final W state."""
    proton_results = epcm_w_proton_results()
    proton_matrix = proton_results["matrix"]
    compton_matrix = epcm_w_compton_matrix(proton_results)
    t = proton_results["kinematics"]["t"]
    polarization_propagator = np.array((
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ), dtype=complex) / t
    proton_state = np.array((
        np.cos(EPCM_W_PROTON_MIXING_ANGLE_RAD),
        np.sin(EPCM_W_PROTON_MIXING_ANGLE_RAD),
    ))
    electron_state = np.array((
        np.cos(EPCM_W_ELECTRON_MIXING_ANGLE_RAD),
        np.sin(EPCM_W_ELECTRON_MIXING_ANGLE_RAD),
    ))
    initial_state = np.kron(proton_state, electron_state)
    emission_operator = np.kron(proton_matrix, np.eye(2))
    propagator_operator = np.kron(
        np.eye(2),
        np.kron(polarization_propagator, np.eye(2)),
    )
    compton_operator = np.kron(np.eye(2), compton_matrix)
    scattering_matrix = (
        compton_operator @ propagator_operator @ emission_operator
    )
    two_branch_proton_matrix = proton_matrix.copy()
    two_branch_proton_matrix[[0, 1, 4, 5], :] = 0.0
    two_branch_matrix = (
        compton_operator
        @ propagator_operator
        @ np.kron(two_branch_proton_matrix, np.eye(2))
    )
    direct_matrix = _direct_epcm_w_scattering_matrix(proton_results)
    matrix_error = float(np.max(np.abs(scattering_matrix - direct_matrix)))
    if matrix_error > EPCM_W_MATRIX_RECONSTRUCTION_TOL:
        raise ValueError(
            "The factorized ep-CM matrix does not reproduce the direct "
            f"Bethe--Heitler matrix: error={matrix_error:.6e}."
        )
    final_state = scattering_matrix @ initial_state
    norm = float(np.linalg.norm(final_state))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ZeroDivisionError("The ep-CM W-state amplitude has zero norm.")
    normalized_state = final_state / norm
    two_branch_final_state = two_branch_matrix @ initial_state
    two_branch_norm = float(np.linalg.norm(two_branch_final_state))
    if two_branch_norm <= 0.0 or not np.isfinite(two_branch_norm):
        raise ZeroDivisionError("The two-branch W-state amplitude has zero norm.")
    two_branch_normalized_state = two_branch_final_state / two_branch_norm
    wbar = np.zeros(8, dtype=complex)
    wbar[[1, 2, 4]] = 1.0 / np.sqrt(3.0)
    w_fidelity = float(abs(np.vdot(wbar, normalized_state)) ** 2)
    two_branch_w_fidelity = float(
        abs(np.vdot(wbar, two_branch_normalized_state)) ** 2
    )
    return {
        "proton_results": proton_results,
        "H": proton_matrix,
        "G": polarization_propagator,
        "C": compton_matrix,
        "initial_state": initial_state,
        "scattering_matrix": scattering_matrix,
        "two_branch_scattering_matrix": two_branch_matrix,
        "direct_matrix": direct_matrix,
        "matrix_reconstruction_error": matrix_error,
        "final_state": final_state,
        "final_state_norm": norm,
        "normalized_state": normalized_state,
        "W_fidelity": w_fidelity,
        "two_branch_final_state": two_branch_final_state,
        "two_branch_final_state_norm": two_branch_norm,
        "two_branch_normalized_state": two_branch_normalized_state,
        "two_branch_W_fidelity": two_branch_w_fidelity,
    }


def epcm_w_output_rows(results):
    """Return Compton, full-matrix, and final-state rows for the benchmark."""
    compton_rows = []
    for photon_index, photon_helicity in enumerate(EPCM_W_HELICITY_ORDER):
        for out_index, h_out in enumerate(EPCM_W_HELICITY_ORDER):
            row = 2 * photon_index + out_index
            for polarization_index, name in enumerate(
                EPCM_W_VIRTUAL_POLARIZATION_ORDER
            ):
                for in_index, h_in in enumerate(EPCM_W_HELICITY_ORDER):
                    column = 2 * polarization_index + in_index
                    compton_rows.append({
                        "row_index": row,
                        "column_index": column,
                        "h_gamma_out": photon_helicity,
                        "h_e_out": h_out,
                        "virtual_photon_polarization": name,
                        "h_e_in": h_in,
                        **_complex_output_fields(
                            "C", results["C"][row, column]
                        ),
                    })
    matrix_rows = []
    for row, final_labels in enumerate(
        (
            "+++",
            "++-",
            "+-+",
            "+--",
            "-++",
            "-+-",
            "--+",
            "---",
        )
    ):
        for column, initial_labels in enumerate(("++", "+-", "-+", "--")):
            matrix_rows.append({
                "row_index": row,
                "column_index": column,
                "final_state_order": "p_gamma_e",
                "final_ket": f"|{final_labels}>",
                "initial_state_order": "p_e",
                "initial_ket": f"|{initial_labels}>",
                **_complex_output_fields(
                    "M", results["scattering_matrix"][row, column]
                ),
                **_complex_output_fields(
                    "M_direct", results["direct_matrix"][row, column]
                ),
                **_complex_output_fields(
                    "M_two_branch",
                    results["two_branch_scattering_matrix"][row, column],
                ),
                "factorization_abs_error": float(abs(
                    results["scattering_matrix"][row, column]
                    - results["direct_matrix"][row, column]
                )),
            })
    final_rows = []
    for index, labels in enumerate(
        ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
    ):
        amplitude = results["final_state"][index]
        normalized = results["normalized_state"][index]
        two_branch_amplitude = results["two_branch_final_state"][index]
        two_branch_normalized = results["two_branch_normalized_state"][index]
        final_rows.append({
            "out_index": index,
            "final_state_order": "p_gamma_e",
            "final_ket": f"|{labels}>",
            **_complex_output_fields("amplitude", amplitude),
            **_complex_output_fields("normalized_amplitude", normalized),
            "probability": float(abs(normalized) ** 2),
            **_complex_output_fields(
                "two_branch_amplitude", two_branch_amplitude
            ),
            **_complex_output_fields(
                "two_branch_normalized_amplitude", two_branch_normalized
            ),
            "two_branch_probability": float(abs(two_branch_normalized) ** 2),
        })
    return compton_rows, matrix_rows, final_rows


def epcm_w_matrix_report(results):
    """Return the numerical matrix-chain and W-state summary."""
    final_labels = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
    leading = sorted(
        zip(final_labels, results["normalized_state"]),
        key=lambda item: -abs(item[1]) ** 2,
    )
    two_branch_leading = sorted(
        zip(final_labels, results["two_branch_normalized_state"]),
        key=lambda item: -abs(item[1]) ** 2,
    )
    return "\n".join([
        "ep-CM W-state matrix chain",
        "  v_out = (I_p x C)(I_p x G x I_e)(H x I_e) v_in",
        "  basis in: (++,+-,-+,--)_(p,e)",
        "  basis gamma*: (T-,T+,L)",
        "  basis out: (+++,++-,+-+,+--,-++,-+-,--+,---)_(p,gamma,e)",
        (
            f"  maximum |M_factorized-M_direct|: "
            f"{results['matrix_reconstruction_error']:.3e}"
        ),
        f"  unnormalized final-state norm: {results['final_state_norm']:.12g}",
        f"  canonical Wbar fidelity: {results['W_fidelity']:.12g}",
        (
            "  two-branch fidelity [only p'(+1)L and p'(-1)T-]: "
            f"{results['two_branch_W_fidelity']:.12g}"
        ),
        "  leading normalized final amplitudes:",
        *[
            f"    |{labels}>: {amplitude.real:+.9g}{amplitude.imag:+.9g}i, "
            f"P={abs(amplitude) ** 2:.9g}"
            for labels, amplitude in leading
        ],
        "  leading two-branch normalized amplitudes:",
        *[
            f"    |{labels}>: {amplitude.real:+.9g}{amplitude.imag:+.9g}i, "
            f"P={abs(amplitude) ** 2:.9g}"
            for labels, amplitude in two_branch_leading[:3]
        ],
        f"  Compton matrix CSV: {EPCM_W_COMPTON_MATRIX_CSV}",
        f"  full scattering matrix CSV: {EPCM_W_FULL_MATRIX_CSV}",
        f"  final state CSV: {EPCM_W_FINAL_STATE_CSV}",
    ]) + "\n"


def write_epcm_w_outputs(results=None):
    """Write all exact ep-CM W-state matrix-chain products."""
    if results is None:
        results = epcm_w_matrix_chain()
    compton_rows, matrix_rows, final_rows = epcm_w_output_rows(results)
    write_csv(EPCM_W_COMPTON_MATRIX_CSV, compton_rows)
    write_csv(EPCM_W_FULL_MATRIX_CSV, matrix_rows)
    write_csv(EPCM_W_FINAL_STATE_CSV, final_rows)
    report = epcm_w_matrix_report(results)
    EPCM_W_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    EPCM_W_LOG_PATH.write_text(report, encoding="utf-8")
    return results, report


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
            for polarization in PLOTTED_VIRTUAL_PHOTON_POLARIZATIONS:
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
                    configure_polar_angle_axis(ax, "x")
                    ax.set_ylabel(r"$|\mathcal{M}/e^2|$")
                    ax.set_title(rf"$h_\ell={h_in:+d}$")
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


def transverse_incoming_lepton_rows(component_rows):
    """Coherently combine ``h=+/-1`` into the transverse lepton state.

    The preparation is ``(|+> + |->) / sqrt(2)``.  Only the physical
    ``T-``, ``T+``, and ``L`` virtual-photon basis states are retained.
    """
    grouped = {}
    plotted_labels = {
        polarization_label(value)
        for value in PLOTTED_VIRTUAL_PHOTON_POLARIZATIONS
    }
    for row in component_rows:
        if row["incoming_virtual_photon_polarization"] not in plotted_labels:
            continue
        key = (
            row["Q2_GeV2"],
            row["theta_index"],
            row["incoming_virtual_photon_polarization"],
            row["h_l_out"],
            row["h_gamma_out"],
        )
        grouped.setdefault(key, {})[row["h_l_in"]] = row

    rows = []
    for key, helicity_rows in grouped.items():
        if any(helicity not in helicity_rows for helicity in HELICITIES):
            raise ValueError(
                "Both incoming-lepton helicities are required for the "
                "transverse coherent state."
            )
        reference = helicity_rows[FIXED_INCOMING_LEPTON_HELICITY]
        amplitude = sum(
            complex(
                helicity_rows[helicity]["amplitude_real"],
                helicity_rows[helicity]["amplitude_imag"],
            )
            for helicity in HELICITIES
        ) / np.sqrt(2.0)
        rows.append({
            "Q2_GeV2": reference["Q2_GeV2"],
            "Q_GeV": reference["Q_GeV"],
            "W_GeV": reference["W_GeV"],
            "theta_index": reference["theta_index"],
            "theta_cm_rad": reference["theta_cm_rad"],
            "phi_cm_rad": reference["phi_cm_rad"],
            "incoming_virtual_photon_polarization": (
                reference["incoming_virtual_photon_polarization"]
            ),
            "initial_lepton_polarization": "Tx",
            "initial_lepton_state": TRANSVERSE_LEPTON_LABEL,
            "h_l_out": reference["h_l_out"],
            "h_gamma_out": reference["h_gamma_out"],
            "amplitude_real": float(amplitude.real),
            "amplitude_imag": float(amplitude.imag),
            "amplitude_abs": float(abs(amplitude)),
            "amplitude_phase_rad": float(np.angle(amplitude)),
            "amplitude_abs2": float(abs(amplitude) ** 2),
        })
    rows.sort(key=lambda row: (
        row["Q2_GeV2"],
        row["incoming_virtual_photon_polarization"],
        row["theta_index"],
        row["h_l_out"],
        row["h_gamma_out"],
    ))
    return rows


def save_fixed_incoming_lepton_plots(component_rows):
    """Compare longitudinal and transverse initial-lepton preparations.

    Columns are the physical virtual-photon basis states ``T-``, ``T+``, and
    ``L``.  The top row uses the configured fixed lepton helicity, while the
    bottom row uses ``(|+>+|->)/sqrt(2)`` coherently at amplitude level.
    """
    plt, PdfPages = require_matplotlib()
    final_styles = {
        (-1, -1): ("tab:blue", "-"),
        (-1, +1): ("tab:orange", "--"),
        (+1, -1): ("tab:green", "-."),
        (+1, +1): ("tab:red", ":"),
    }
    fixed_rows = fixed_incoming_lepton_rows(component_rows)
    transverse_rows = transverse_incoming_lepton_rows(component_rows)
    with PdfPages(FIXED_PLOT_PDF) as pdf:
        for Q2 in Q2_VALUES_GEV2:
            fig, axes = plt.subplots(
                2, 3, figsize=(15.0, 8.8), constrained_layout=True,
                sharey=True,
            )
            for lepton_row, (source_rows, lepton_title) in enumerate((
                (
                    fixed_rows,
                    rf"$h_\ell={FIXED_INCOMING_LEPTON_HELICITY:+d}$",
                ),
                (
                    transverse_rows,
                    r"$|\ell_{\rm in}\rangle=(|+\rangle+|-\rangle)/\sqrt{2}$",
                ),
            )):
                for column, polarization in enumerate(("-1", "+1", "L")):
                    ax = axes[lepton_row, column]
                    for (
                        h_out,
                        h_gamma,
                    ), (color, linestyle) in final_styles.items():
                        selected = [
                            row for row in source_rows
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
                    configure_polar_angle_axis(ax, "x")
                    ax.set_ylabel(r"$|\mathcal{M}/e^2|$")
                    ax.set_title(
                        lepton_title
                        + ", "
                        + rf"$\lambda_{{\gamma^*}}={polarization}$"
                    )
                    ax.legend(fontsize=8)
            fig.suptitle(
                r"$\gamma^*\ell\to\gamma\ell$, "
                "initial-lepton polarization comparison: "
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
            "  coherent transverse photon response is retained in CSV only; "
            "it is excluded from plots"
        ),
        (
            "  plotted transverse lepton: "
            "|l_in>=(|+>+|->)/sqrt(2), combined at amplitude level"
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
        "Fixed-h_l=+1 and transverse-lepton amplitudes are tabulated separately."
    )
    lines.extend([
        "",
        f"  component CSV: {COMPONENT_CSV}",
        f"  response CSV: {RESPONSE_CSV}",
        f"  plot PDF: {PLOT_PDF}",
        f"  fixed-h_l component CSV: {FIXED_COMPONENT_CSV}",
        f"  transverse-lepton component CSV: {TRANSVERSE_LEPTON_COMPONENT_CSV}",
        f"  initial-lepton comparison PDF: {FIXED_PLOT_PDF}",
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
    write_csv(
        TRANSVERSE_LEPTON_COMPONENT_CSV,
        transverse_incoming_lepton_rows(component_rows),
    )
    save_plots(component_rows)
    save_fixed_incoming_lepton_plots(component_rows)
    _, epcm_report = write_epcm_w_outputs()
    report = build_report(component_rows, response_rows) + "\n" + epcm_report
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
