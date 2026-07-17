"""Spin-density matrix scans and three-qubit entanglement observables.

The outgoing spin basis is ordered as ``(hOut, sOut, lambda)``:

* particle 1 is the outgoing electron helicity ``hOut``;
* particle 2 is the outgoing proton spin/helicity ``sOut``;
* particle 3 is the outgoing real-photon helicity ``lambda``.

For each independent user-frame kinematic point, this module builds the ``4 x 8`` table of
Bethe-Heitler amplitudes over incoming and outgoing spin labels, converts it
into an ``8 x 8`` final-state spin-density matrix, normalizes it by the
corresponding squared amplitude when requested, and computes the two-body and
one-to-rest concurrence observables plus ``F3`` from arXiv:2310.01477v2.

Running this file as a script regenerates the SpinDensityMat output directory:
unpolarized, incoming-electron-polarized, and transverse incoming-electron
scan folders with NPZ scans, summary entanglement CSV/PDFs where defined,
per-kinematic-point matrix CSVs/PDFs, and ``Output/SpinDensityMat.log``.
"""

from itertools import product
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import shutil

import numpy as np

from Algebra import HELICITIES
from BHHelicityAmp import bh_amplitude_table
from FormFactors import YAHL_MODEL_NAME, yahl_dirac_pauli_from_t
from Kinematics import kinematics_user_from_independent
from config import NORMALIZE_TRACE, PROTON_MASS_GEV as M, SCAN_WORKERS


# ============================================================
# Scan and output settings
# ============================================================

USER_BEAM_ENERGY_REFERENCE = 11.0
USER_S_CENTER = M**2 + 2.0 * M * USER_BEAM_ENERGY_REFERENCE
USER_S_VALUES = np.linspace(0.72 * USER_S_CENTER, 1.20 * USER_S_CENTER, 9)
USER_QOUT_VALUES = np.linspace(0.30, 1.55, 9)
USER_THETA_IN_VALUES = np.linspace(0.35, 2.80, 9)
USER_PHI_OUT_VALUES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
USER_FIXED_S = USER_S_CENTER
USER_FIXED_THETA_IN = 1.30
USER_FIXED_PHI_IN = 0.0
USER_FIXED_QOUT = 0.85
USER_FIXED_PHI_OUT = np.pi

TRACE_BENCHMARK_TOL = 1e-10
SPIN_CASE_UNPOLARIZED = "unpolarized"
SPIN_CASE_L_PROTON = "L_proton"
SPIN_CASE_L_LEPTON = "L_lepton"
SPIN_CASE_TX_PROTON = "Tx_proton"
SPIN_CASE_TY_PROTON = "Ty_proton"
SPIN_CASE_MINUS_TX_PROTON = "minus_Tx_proton"
SPIN_CASE_MINUS_TY_PROTON = "minus_Ty_proton"
SPIN_CASE_TX_LEPTON = "Tx_lepton"
SPIN_CASE_TY_LEPTON = "Ty_lepton"
SPIN_CASE_LL = "LL"
SPIN_CASE_LANTI = "Lanti"
SPIN_CASE_LTX = "LTx"
SPIN_CASE_LTY = "LTy"
SPIN_CASE_L_MINUS_TX = "L_minus_Tx"
SPIN_CASE_L_MINUS_TY = "L_minus_Ty"
SPIN_CASE_TXTX = "TxTx"
SPIN_CASE_TXTY = "TxTy"

SPIN_CASES = (
    SPIN_CASE_UNPOLARIZED,
    SPIN_CASE_L_PROTON,
    SPIN_CASE_L_LEPTON,
    SPIN_CASE_TX_PROTON,
    SPIN_CASE_TY_PROTON,
    SPIN_CASE_TX_LEPTON,
    SPIN_CASE_TY_LEPTON,
    SPIN_CASE_LL,
    SPIN_CASE_LANTI,
    SPIN_CASE_LTX,
    SPIN_CASE_LTY,
    SPIN_CASE_TXTX,
    SPIN_CASE_TXTY,
)

SPIN_CASE_DISPLAY_LABELS = {
    SPIN_CASE_UNPOLARIZED: "Unpolarized",
    SPIN_CASE_L_PROTON: "L proton",
    SPIN_CASE_L_LEPTON: "L lepton",
    SPIN_CASE_TX_PROTON: "Tx proton",
    SPIN_CASE_TY_PROTON: "Ty proton",
    SPIN_CASE_MINUS_TX_PROTON: "-Tx proton",
    SPIN_CASE_MINUS_TY_PROTON: "-Ty proton",
    SPIN_CASE_TX_LEPTON: "Tx lepton",
    SPIN_CASE_TY_LEPTON: "Ty lepton",
    SPIN_CASE_LL: "L lepton + L proton",
    SPIN_CASE_LANTI: "L+ lepton + L- proton",
    SPIN_CASE_LTX: "L lepton + Tx proton",
    SPIN_CASE_LTY: "L lepton + Ty proton",
    SPIN_CASE_L_MINUS_TX: "L lepton + -Tx proton",
    SPIN_CASE_L_MINUS_TY: "L lepton + -Ty proton",
    SPIN_CASE_TXTX: "Tx lepton + Tx proton",
    SPIN_CASE_TXTY: "Tx lepton + Ty proton",
}
ENTANGLEMENT_NAMES = (
    "C_e_p",
    "C_e_gamma",
    "C_p_gamma",
    "C_e_rest",
    "C_p_rest",
    "C_gamma_rest",
    "D_W",
    "F3",
    "M_e",
    "M_p",
    "M_gamma",
)

PAULI_SINGLE_QUBIT = (
    np.eye(2, dtype=complex),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
)
THREE_QUBIT_PAULI_STRINGS = np.stack(
    [
        np.kron(np.kron(first, second), third)
        for first, second, third in product(PAULI_SINGLE_QUBIT, repeat=3)
    ]
)


def w_concurrence_distance(c_e_p, c_e_gamma, c_p_gamma):
    """Return Euclidean distance from the ideal W pairwise concurrences."""
    target = 2.0 / 3.0
    return float(np.sqrt(
        (c_e_p - target) ** 2
        + (c_e_gamma - target) ** 2
        + (c_p_gamma - target) ** 2
    ))

BENCHMARK_USER_KINEMATIC_INPUTS = (
    ("U1", USER_S_CENTER, 1.30, 0.0, 0.85, np.pi),
    ("U2", 0.90 * USER_S_CENTER, 0.85, 0.5 * np.pi, 0.60, 0.5 * np.pi),
    ("U3", 1.15 * USER_S_CENTER, 2.20, np.pi, 1.10, 0.0),
)

OUTPUT_DIR = Path("Output") / "SpinDensityMat"
LOG_PATH = Path("Output") / "SpinDensityMat.log"



def outgoing_spin_states():
    """Return final-state labels (hOut, sOut, lambda)."""
    return tuple(product(HELICITIES, repeat=3))


def initial_spin_states():
    """Return initial-state labels (hIn, sIn)."""
    return tuple(product(HELICITIES, repeat=2))


def prepared_spin_coefficients(axis):
    """Return helicity coefficients for a longitudinal/transverse preparation."""
    coefficient = 1.0 / np.sqrt(2.0)
    if axis == "L":
        return {+1: 1.0 + 0.0j}
    if axis == "-L":
        return {-1: 1.0 + 0.0j}
    if axis == "Tx":
        return {-1: coefficient, +1: coefficient}
    if axis == "-Tx":
        return {-1: -coefficient, +1: coefficient}
    if axis == "Ty":
        return {-1: 1.0j * coefficient, +1: coefficient}
    if axis == "-Ty":
        return {-1: -1.0j * coefficient, +1: coefficient}
    raise ValueError(f"Unknown prepared spin axis: {axis}")


def _unpolarized_particle_ensemble():
    """Return the equal incoherent helicity ensemble for one particle."""
    return [
        (0.5, {helicity: 1.0 + 0.0j}, f"h={helicity:+d}")
        for helicity in HELICITIES
    ]


def _prepared_particle_ensemble(axis):
    """Return one coherently prepared particle state."""
    return [(1.0, prepared_spin_coefficients(axis), axis)]


def spin_case_axes(spin_case):
    """Return ``(electron_axis, proton_axis)``; ``None`` means unpolarized."""
    cases = {
        SPIN_CASE_UNPOLARIZED: (None, None),
        SPIN_CASE_L_PROTON: (None, "L"),
        SPIN_CASE_L_LEPTON: ("L", None),
        SPIN_CASE_TX_PROTON: (None, "Tx"),
        SPIN_CASE_TY_PROTON: (None, "Ty"),
        SPIN_CASE_MINUS_TX_PROTON: (None, "-Tx"),
        SPIN_CASE_MINUS_TY_PROTON: (None, "-Ty"),
        SPIN_CASE_TX_LEPTON: ("Tx", None),
        SPIN_CASE_TY_LEPTON: ("Ty", None),
        SPIN_CASE_LL: ("L", "L"),
        SPIN_CASE_LANTI: ("L", "-L"),
        SPIN_CASE_LTX: ("L", "Tx"),
        SPIN_CASE_LTY: ("L", "Ty"),
        SPIN_CASE_L_MINUS_TX: ("L", "-Tx"),
        SPIN_CASE_L_MINUS_TY: ("L", "-Ty"),
        SPIN_CASE_TXTX: ("Tx", "Tx"),
        SPIN_CASE_TXTY: ("Tx", "Ty"),
    }
    if spin_case not in cases:
        raise ValueError(f"Unknown spin density case: {spin_case}")
    return cases[spin_case]


def spin_case_display_label(spin_case):
    """Return an explicit user-facing electron/proton polarization label."""
    spin_case_axes(spin_case)
    return SPIN_CASE_DISPLAY_LABELS[spin_case]


def initial_state_ensemble(spin_case):
    """Return the weighted incoming electron-proton preparation ensemble.

    The first axis in a double-polarization label belongs to the electron and
    the second to the proton.  A particle absent from the label is summed
    incoherently over its two helicities.
    """
    electron_axis, proton_axis = spin_case_axes(spin_case)
    electron = (
        _unpolarized_particle_ensemble()
        if electron_axis is None
        else _prepared_particle_ensemble(electron_axis)
    )
    proton = (
        _unpolarized_particle_ensemble()
        if proton_axis is None
        else _prepared_particle_ensemble(proton_axis)
    )
    return [
        {
            "weight": electron_weight * proton_weight,
            "electron_coefficients": electron_coefficients,
            "proton_coefficients": proton_coefficients,
            "label": f"electron {electron_label}, proton {proton_label}",
        }
        for electron_weight, electron_coefficients, electron_label in electron
        for proton_weight, proton_coefficients, proton_label in proton
    ]


def final_state_ensemble(amplitudes, spin_case):
    """Apply each incoming ensemble component to the amplitude table."""
    in_states = initial_spin_states()
    final_states = []
    for component in initial_state_ensemble(spin_case):
        state = np.zeros(amplitudes.shape[1], dtype=complex)
        for h_in, electron_coefficient in component["electron_coefficients"].items():
            for s_in, proton_coefficient in component["proton_coefficients"].items():
                state += (
                    electron_coefficient
                    * proton_coefficient
                    * amplitudes[in_states.index((h_in, s_in))]
                )
        final_states.append({**component, "state": state})
    return final_states


def single_particle_spin_density(axis):
    """Return a normalized incoming one-qubit density matrix.

    ``axis=None`` represents an unpolarized particle, ``I_2/2``. Prepared
    ``L``, ``-L``, ``Tx``, ``-Tx``, ``Ty``, and ``-Ty`` states are rank-one
    projectors in the helicity basis ordered as ``(-1, +1)``.
    """
    if axis is None:
        return 0.5 * np.eye(2, dtype=complex)
    coefficients = prepared_spin_coefficients(axis)
    state = np.asarray(
        [coefficients.get(helicity, 0.0) for helicity in HELICITIES],
        dtype=complex,
    )
    return np.outer(state, state.conj())


def initial_spin_density_matrix(spin_case):
    """Return the normalized 4 x 4 electron-proton incoming density matrix."""
    electron_axis, proton_axis = spin_case_axes(spin_case)
    return np.kron(
        single_particle_spin_density(electron_axis),
        single_particle_spin_density(proton_axis),
    )


def process_density_matrix_from_amplitudes(amplitudes):
    """Return the full 32 x 32 five-particle process density matrix.

    The flattened basis is ``(hIn, sIn, hOut, sOut, lambda)`` and the matrix
    is ``R = |M><M|`` for the complete ``4 x 8`` amplitude table.
    """
    amplitudes = np.asarray(amplitudes, dtype=complex)
    if amplitudes.shape != (4, 8):
        raise ValueError("The five-particle amplitude table must have shape (4, 8).")
    process_state = amplitudes.reshape(32)
    return np.outer(process_state, process_state.conj())


def contract_initial_state(process_rho, spin_case):
    """Contract the incoming preparation and return the outgoing 8 x 8 matrix.

    Unpolarized incoming particles are traced with ``I_2/2``; hence their
    helicities are summed incoherently and averaged. Polarized particles are
    contracted with the corresponding pure-state projector.
    """
    process_rho = np.asarray(process_rho, dtype=complex)
    if process_rho.shape != (32, 32):
        raise ValueError("The five-particle process density matrix must be 32 x 32.")
    electron_axis, proton_axis = spin_case_axes(spin_case)
    rho_e = single_particle_spin_density(electron_axis)
    rho_p = single_particle_spin_density(proton_axis)
    tensor = process_rho.reshape(2, 2, 8, 2, 2, 8)
    return np.einsum("abicdj,ac,bd->ij", tensor, rho_e, rho_p, optimize=True)


def incoming_spin_weights(spin_case=SPIN_CASE_UNPOLARIZED):
    """Return initial-state diagonal weights for a configured spin scan.

    The polarized case is the incoming-electron helicity difference
    ``hIn=+1`` minus ``hIn=-1``, with the incoming proton spin summed in both
    terms. The transverse case stores only the diagonal populations here; its
    coherent interference terms are handled by
    :func:`density_matrix_from_amplitudes`.
    """
    populations = []
    for h_in, s_in in initial_spin_states():
        population = 0.0
        for component in initial_state_ensemble(spin_case):
            electron = component["electron_coefficients"].get(h_in, 0.0)
            proton = component["proton_coefficients"].get(s_in, 0.0)
            population += component["weight"] * abs(electron * proton) ** 2
        populations.append(population)
    return np.asarray(populations, dtype=float)


def initial_spin_average_divisor(spin_case):
    """Return the physical initial-spin averaging divisor for one spin case.

    The unpolarized case averages over both electron and proton helicities.
    A single polarized electron state averages only over the unobserved proton
    helicity.  The double-transverse state is one coherent preparation and is
    therefore not spin averaged.
    """
    ensemble_size = len(initial_state_ensemble(spin_case))
    return float(ensemble_size)


def amplitude_table(mom, m, F1, F2, electron_mass=0.0):
    """Return ``A[in_state, out_state]`` for all BH helicity amplitudes.

    ``in_state`` spans incoming electron/proton labels ``(hIn, sIn)`` and
    ``out_state`` spans outgoing electron/proton/photon labels
    ``(hOut, sOut, lambda)``. The result has shape ``(4, 8)``.
    """
    return bh_amplitude_table(
        mom,
        m,
        F1,
        F2,
        initial_states=initial_spin_states(),
        outgoing_states=outgoing_spin_states(),
        electron_mass=electron_mass,
    )


def density_matrix_from_amplitudes(
    amplitudes,
    spin_case=SPIN_CASE_UNPOLARIZED,
    process_rho=None,
):
    """Build the outgoing 8 x 8 matrix by contracting the full 32 x 32 matrix."""
    in_states = initial_spin_states()
    if amplitudes.shape[0] != len(in_states):
        raise ValueError(
            "Amplitude table first axis does not match the incoming spin basis."
        )

    if process_rho is None:
        process_rho = process_density_matrix_from_amplitudes(amplitudes)
    rho = contract_initial_state(process_rho, spin_case)
    spin_signal = trace_value(rho)
    unpolarized_rho = contract_initial_state(process_rho, SPIN_CASE_UNPOLARIZED)
    squared_amplitude = trace_value(unpolarized_rho)
    return rho, spin_signal, squared_amplitude


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
    if not np.isfinite(trace) or trace <= 0.0:
        raise ZeroDivisionError("Cannot normalize a nonpositive or nonfinite trace.")
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


def second_stabilizer_renyi_entropy(rho):
    """Return the three-qubit second stabilizer Renyi entropy ``M2``.

    The purity-normalized definition is
    ``M2 = -ln[(1/8) sum_P Tr(P rho)^4 / Tr(rho^2)^2]``, where ``P`` runs over
    all 64 three-qubit Pauli strings. For a pure state the purity denominator
    is one.
    """
    rho = normalized_density_matrix(rho)
    purity = float(np.real_if_close(np.trace(rho @ rho), tol=1000).real)
    if purity <= 0.0:
        raise ValueError("Density-matrix purity must be positive.")
    expectations = np.real_if_close(
        np.einsum("pij,ji->p", THREE_QUBIT_PAULI_STRINGS, rho),
        tol=1000,
    ).real
    stabilizer_moment = np.sum(expectations**4) / (8.0 * purity**2)
    if stabilizer_moment <= 0.0:
        raise ValueError("The Pauli fourth moment must be positive.")
    entropy = float(-np.log(stabilizer_moment))
    return 0.0 if np.isclose(entropy, 0.0, atol=1e-14) else entropy


def f3_from_one_to_rest(c_e_rest, c_p_rest, c_gamma_rest):
    """Concurrence-triangle area measure from Eq. (6) of arXiv:2310.01477v2."""
    q_value = 0.5 * (c_e_rest + c_p_rest + c_gamma_rest)
    area_argument = (
        (16.0 / 3.0)
        * q_value
        * (q_value - c_e_rest)
        * (q_value - c_p_rest)
        * (q_value - c_gamma_rest)
    )
    return float(np.sqrt(max(0.0, area_argument)))


def entanglement_measures_from_state(state):
    """Compute concurrence observables and monogamy residuals.

    The returned dictionary contains two-body concurrences ``C_e_p``,
    ``C_e_gamma``, ``C_p_gamma``; one-to-rest concurrences ``C_e_rest``,
    ``C_p_rest``, ``C_gamma_rest``; the concurrence-triangle measure ``F3``;
    and CKW monogamy residuals ``M_e``, ``M_p``, and ``M_gamma``.
    """
    rho = pure_density_matrix(state)
    c_e_p = two_qubit_concurrence(reduced_density_matrix(rho, (0, 1)))
    c_e_gamma = two_qubit_concurrence(reduced_density_matrix(rho, (0, 2)))
    c_p_gamma = two_qubit_concurrence(reduced_density_matrix(rho, (1, 2)))
    c_e_rest = one_to_rest_concurrence(rho, 0)
    c_p_rest = one_to_rest_concurrence(rho, 1)
    c_gamma_rest = one_to_rest_concurrence(rho, 2)
    return {
        "C_e_p": c_e_p,
        "C_e_gamma": c_e_gamma,
        "C_p_gamma": c_p_gamma,
        "C_e_rest": c_e_rest,
        "C_p_rest": c_p_rest,
        "C_gamma_rest": c_gamma_rest,
        "D_W": w_concurrence_distance(c_e_p, c_e_gamma, c_p_gamma),
        "F3": f3_from_one_to_rest(c_e_rest, c_p_rest, c_gamma_rest),
        "M_e": c_e_rest**2 - c_e_p**2 - c_e_gamma**2,
        "M_p": c_p_rest**2 - c_e_p**2 - c_p_gamma**2,
        "M_gamma": c_gamma_rest**2 - c_e_gamma**2 - c_p_gamma**2,
    }


def entanglement_measures_from_density_matrix(rho):
    """Compute entanglement observables from an outgoing density matrix.

    Wootters two-qubit concurrence is valid for both pure and mixed reduced
    states.  The one-to-rest concurrence, F3, and CKW residual formulas used
    by this project are pure-three-qubit formulas; they are set to zero when
    the contracted outgoing state is mixed, reflecting that the pure-state
    decomposition does not apply.
    """
    rho = normalized_density_matrix(rho)
    c_e_p = two_qubit_concurrence(reduced_density_matrix(rho, (0, 1)))
    c_e_gamma = two_qubit_concurrence(reduced_density_matrix(rho, (0, 2)))
    c_p_gamma = two_qubit_concurrence(reduced_density_matrix(rho, (1, 2)))
    purity = float(np.real_if_close(np.trace(rho @ rho), tol=1000).real)
    if not np.isclose(purity, 1.0, rtol=1e-9, atol=1e-10):
        return {
            "C_e_p": c_e_p,
            "C_e_gamma": c_e_gamma,
            "C_p_gamma": c_p_gamma,
            "C_e_rest": 0.0,
            "C_p_rest": 0.0,
            "C_gamma_rest": 0.0,
            "D_W": w_concurrence_distance(c_e_p, c_e_gamma, c_p_gamma),
            "F3": 0.0,
            "M_e": -(c_e_p**2 + c_e_gamma**2),
            "M_p": -(c_e_p**2 + c_p_gamma**2),
            "M_gamma": -(c_e_gamma**2 + c_p_gamma**2),
        }
    c_e_rest = one_to_rest_concurrence(rho, 0)
    c_p_rest = one_to_rest_concurrence(rho, 1)
    c_gamma_rest = one_to_rest_concurrence(rho, 2)
    return {
        "C_e_p": c_e_p,
        "C_e_gamma": c_e_gamma,
        "C_p_gamma": c_p_gamma,
        "C_e_rest": c_e_rest,
        "C_p_rest": c_p_rest,
        "C_gamma_rest": c_gamma_rest,
        "D_W": w_concurrence_distance(c_e_p, c_e_gamma, c_p_gamma),
        "F3": f3_from_one_to_rest(c_e_rest, c_p_rest, c_gamma_rest),
        "M_e": c_e_rest**2 - c_e_p**2 - c_e_gamma**2,
        "M_p": c_p_rest**2 - c_e_p**2 - c_p_gamma**2,
        "M_gamma": c_gamma_rest**2 - c_e_gamma**2 - c_p_gamma**2,
    }


def ghz_observables_from_density_matrix(rho):
    """Return observables for ``(|---\u27e9 + |+++\u27e9) / sqrt(2)``.

    ``phase_fidelity`` maximizes the overlap over the relative phase between
    the two GHZ basis components.  It therefore distinguishes a harmless
    phase convention from leakage outside the GHZ subspace.
    """
    rho = normalized_density_matrix(rho)
    minus_index = outgoing_spin_states().index((-1, -1, -1))
    plus_index = outgoing_spin_states().index((+1, +1, +1))
    p_minus = float(np.real(rho[minus_index, minus_index]))
    p_plus = float(np.real(rho[plus_index, plus_index]))
    coherence = complex(rho[minus_index, plus_index])
    population = p_minus + p_plus
    plus_fidelity = 0.5 * population + float(np.real(coherence))
    minus_fidelity = 0.5 * population - float(np.real(coherence))
    phase_fidelity = 0.5 * population + abs(coherence)
    visibility = 0.0 if population <= 0.0 else 2.0 * abs(coherence) / population
    return {
        "GHZ_plus_fidelity": float(np.clip(plus_fidelity, 0.0, 1.0)),
        "GHZ_minus_fidelity": float(np.clip(minus_fidelity, 0.0, 1.0)),
        "GHZ_phase_fidelity": float(np.clip(phase_fidelity, 0.0, 1.0)),
        "GHZ_subspace_population": float(np.clip(population, 0.0, 1.0)),
        "GHZ_coherence_abs": float(abs(coherence)),
        "GHZ_coherence_phase_rad": float(np.angle(coherence)),
        "GHZ_coherence_visibility": float(np.clip(visibility, 0.0, 1.0)),
    }


def w_observables_from_density_matrix(rho):
    """Return projections onto canonical W and opposite-helicity W states.

    The canonical state is ``(|+--\u27e9 + |-+-\u27e9 + |--+\u27e9) / sqrt(3)``.
    ``W_subspace_max_fidelity`` is the largest possible overlap with any
    normalized state in that three-dimensional single-excitation subspace.
    """
    rho = normalized_density_matrix(rho)
    states = outgoing_spin_states()
    w_indices = [states.index(labels) for labels in (
        (+1, -1, -1), (-1, +1, -1), (-1, -1, +1),
    )]
    wbar_indices = [states.index(labels) for labels in (
        (-1, +1, +1), (+1, -1, +1), (+1, +1, -1),
    )]

    def subspace_observables(indices):
        block = rho[np.ix_(indices, indices)]
        equal_state = np.ones(3, dtype=complex) / np.sqrt(3.0)
        fidelity = float(np.real(np.vdot(equal_state, block @ equal_state)))
        population = float(np.real(np.trace(block)))
        eigenvalues = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
        maximum = float(max(0.0, eigenvalues[-1]))
        return fidelity, population, maximum

    w_fidelity, w_population, w_maximum = subspace_observables(w_indices)
    wbar_fidelity, wbar_population, wbar_maximum = subspace_observables(wbar_indices)
    return {
        "W_fidelity": float(np.clip(w_fidelity, 0.0, 1.0)),
        "W_subspace_population": float(np.clip(w_population, 0.0, 1.0)),
        "W_subspace_max_fidelity": float(np.clip(w_maximum, 0.0, 1.0)),
        "Wbar_fidelity": float(np.clip(wbar_fidelity, 0.0, 1.0)),
        "Wbar_subspace_population": float(np.clip(wbar_population, 0.0, 1.0)),
        "Wbar_subspace_max_fidelity": float(np.clip(wbar_maximum, 0.0, 1.0)),
    }


def density_matrix_for_spin_case(
    amplitudes,
    spin_case=SPIN_CASE_UNPOLARIZED,
    process_rho=None,
):
    """Return ``rho``, spin signal, and unpolarized ``|M|^2`` for one spin case."""
    return density_matrix_from_amplitudes(
        amplitudes,
        spin_case=spin_case,
        process_rho=process_rho,
    )


def spin_density_observables_from_amplitudes(
    amplitudes,
    spin_case=SPIN_CASE_UNPOLARIZED,
    normalize_trace=NORMALIZE_TRACE,
    process_rho=None,
):
    """Return density-matrix and entanglement observables for one spin case."""
    if process_rho is None:
        process_rho = process_density_matrix_from_amplitudes(amplitudes)
    rho, spin_signal, squared_amplitude = density_matrix_for_spin_case(
        amplitudes,
        spin_case=spin_case,
        process_rho=process_rho,
    )
    state_rho = normalized_density_matrix(rho)
    if normalize_trace:
        rho = state_rho
    purity = float(np.real_if_close(np.trace(state_rho @ state_rho), tol=1000).real)
    return {
        "rho": rho,
        "squared_amplitude": squared_amplitude,
        "spin_signal": spin_signal,
        "trace": trace_value(rho),
        "cross_section_ratio": spin_signal / squared_amplitude,
        "purity": purity,
        "M2_magic": second_stabilizer_renyi_entropy(state_rho),
        "entanglement": entanglement_measures_from_density_matrix(state_rho),
    }


def build_user_scan_point(
    s,
    theta_in,
    phi_in,
    qOut,
    phiOut,
    m,
    normalize_trace=NORMALIZE_TRACE,
    spin_case=SPIN_CASE_UNPOLARIZED,
):
    """Evaluate spin-density data at one independent user-frame point."""
    kin = kinematics_user_from_independent(
        s,
        theta_in,
        phi_in,
        qOut,
        phiOut,
        m,
        label=f"user s={s:.6g}, theta={theta_in:.6g}, qOut={qOut:.6g}",
    )
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], kin["m"])
    amplitudes = amplitude_table(kin["momenta"], kin["m"], F1, F2)
    spin_data = spin_density_observables_from_amplitudes(
        amplitudes,
        spin_case=spin_case,
        normalize_trace=normalize_trace,
    )

    return {
        **spin_data,
        "kinematics": kin,
        "form_factors": {"model": YAHL_MODEL_NAME, "F1": F1, "F2": F2},
    }


def _scan_spin_density_user_grid_task(task):
    """Evaluate one independent user-frame spin-density grid point."""
    y_index, x_index, user_vars, settings = task
    point = build_user_scan_point(
        user_vars["s"],
        user_vars["theta_in"],
        user_vars["phi_in"],
        user_vars["qOut"],
        user_vars["phiOut"],
        settings["m"],
        normalize_trace=settings["normalize_trace"],
        spin_case=settings["spin_case"],
    )
    return {
        "ok": True,
        "y_index": y_index,
        "x_index": x_index,
        "user_vars": user_vars,
        "point": point,
    }


def user_vars_for_scan_point(x_name, x_value, y_name, y_value, fixed_user):
    """Return the independent user variables for one 2D scan point."""
    user_vars = dict(fixed_user)
    user_vars[x_name] = float(x_value)
    user_vars[y_name] = float(y_value)
    return user_vars


def scan_spin_density_user_grid(
    x_values,
    y_values,
    x_name,
    y_name,
    fixed_user,
    m=M,
    normalize_trace=NORMALIZE_TRACE,
    spin_case=SPIN_CASE_UNPOLARIZED,
    max_workers=SCAN_WORKERS,
):
    """Scan a 2D grid of independent user-frame kinematic variables."""
    allowed = {"s", "theta_in", "phi_in", "qOut", "phiOut"}
    if x_name not in allowed or y_name not in allowed:
        raise ValueError(f"x_name and y_name must be in {sorted(allowed)}.")
    if x_name == y_name:
        raise ValueError("x_name and y_name must be different.")

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    out_states = outgoing_spin_states()
    shape = (len(y_values), len(x_values))
    rho_grid = np.full((*shape, len(out_states), len(out_states)), np.nan + 1j * np.nan)
    squared_amplitude_grid = np.full(shape, np.nan, dtype=float)
    spin_signal_grid = np.full(shape, np.nan, dtype=float)
    cross_section_ratio_grid = np.full(shape, np.nan, dtype=float)
    purity_grid = np.full(shape, np.nan, dtype=float)
    trace_grid = np.full(shape, np.nan, dtype=float)
    valid = np.zeros(shape, dtype=bool)
    entanglement_grid = {
        name: np.full(shape, np.nan, dtype=float)
        for name in ENTANGLEMENT_NAMES
    }
    kinematic_grids = {
        key: np.full(shape, np.nan, dtype=float)
        for key in (
            "s",
            "sqrt_s",
            "pIn",
            "pOut",
            "qOut",
            "theta_in",
            "phi_in",
            "phiOut",
            "Q2",
            "xB",
            "t",
            "F1",
            "F2",
            "W2",
            "y",
        )
    }
    failures = []

    settings = {
        "m": m,
        "normalize_trace": normalize_trace,
        "spin_case": spin_case,
    }
    tasks = []
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            user_vars = user_vars_for_scan_point(
                x_name,
                x_value,
                y_name,
                y_value,
                fixed_user,
            )
            tasks.append((y_index, x_index, user_vars, settings))

    if max_workers and max_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_scan_spin_density_user_grid_task, tasks))
    else:
        results = [_scan_spin_density_user_grid_task(task) for task in tasks]

    for result in results:
        if not result["ok"]:
            failures.append((
                result.get("s", np.nan),
                result.get("theta_in", np.nan),
                result.get("phi_in", np.nan),
                result.get("qOut", np.nan),
                result.get("phiOut", np.nan),
                result["error"],
            ))
            continue

        y_index = result["y_index"]
        x_index = result["x_index"]
        point = result["point"]
        kin = point["kinematics"]
        rho_grid[y_index, x_index] = point["rho"]
        squared_amplitude_grid[y_index, x_index] = point["squared_amplitude"]
        spin_signal_grid[y_index, x_index] = point["spin_signal"]
        cross_section_ratio_grid[y_index, x_index] = point["cross_section_ratio"]
        purity_grid[y_index, x_index] = point["purity"]
        trace_grid[y_index, x_index] = point["trace"]
        for name, value in point["entanglement"].items():
            entanglement_grid[name][y_index, x_index] = value
        for key in ("s", "theta_in", "phi_in", "qOut", "phiOut"):
            kinematic_grids[key][y_index, x_index] = kin[key]
        kinematic_grids["sqrt_s"][y_index, x_index] = kin["sqrt_s"]
        kinematic_grids["pIn"][y_index, x_index] = kin["pIn"]
        kinematic_grids["pOut"][y_index, x_index] = kin["pOut"]
        for key in ("Q2", "xB", "t", "W2", "y"):
            kinematic_grids[key][y_index, x_index] = kin[key]
        kinematic_grids["F1"][y_index, x_index] = point["form_factors"]["F1"]
        kinematic_grids["F2"][y_index, x_index] = point["form_factors"]["F2"]
        valid[y_index, x_index] = True

    return {
        "rho": rho_grid,
        "squared_amplitude": squared_amplitude_grid,
        "spin_signal": spin_signal_grid,
        "cross_section_ratio": cross_section_ratio_grid,
        "purity": purity_grid,
        "trace": trace_grid,
        "kinematic_grids": kinematic_grids,
        "entanglement": entanglement_grid,
        "entanglement_names": ENTANGLEMENT_NAMES,
        "valid": valid,
        "failures": failures,
        "label": f"user_{x_name}_{y_name}",
        "x_name": x_name,
        "y_name": y_name,
        "x_label": user_axis_label(x_name),
        "y_label": user_axis_label(y_name),
        "y_values": y_values,
        "x_values": x_values,
        "fixed_user": dict(fixed_user),
        "out_states": out_states,
        "initial_states": initial_spin_states(),
        "incoming_spin_weights": incoming_spin_weights(spin_case),
        "normalized_to_unit_trace": normalize_trace,
        "form_factor_model": YAHL_MODEL_NAME,
        "entanglement_defined": True,
        "entanglement_mode": entanglement_mode(spin_case),
        "spin_case": spin_case,
    }


def user_axis_label(name):
    """Return a plot/report label for one independent user-frame variable."""
    labels = {
        "s": r"$s$ [GeV$^2$]",
        "theta_in": r"$\theta_{\rm in}$ [rad]",
        "phi_in": r"$\phi_{\rm in}$ [rad]",
        "qOut": r"$E_{\gamma}'$ [GeV]",
        "phiOut": r"$\phi_{\gamma}'$ [rad]",
    }
    return labels.get(name, name)


def entanglement_mode(spin_case):
    """Return a stable label for the scan entanglement convention."""
    initial_state_ensemble(spin_case)
    return f"prepared_state_ensemble_{spin_case}"


def benchmark_spin_density_trace(
    kinematic_inputs=BENCHMARK_USER_KINEMATIC_INPUTS,
    m=M,
    tol=TRACE_BENCHMARK_TOL,
):
    """Check that selected benchmark density matrices normalize to trace one."""
    rows = []
    for case_id, s, theta_in, phi_in, qOut, phiOut in kinematic_inputs:
        kin = kinematics_user_from_independent(
            s,
            theta_in,
            phi_in,
            qOut,
            phiOut,
            m,
            label=f"trace benchmark {case_id}",
        )
        F1, F2 = yahl_dirac_pauli_from_t(kin["t"], kin["m"])
        amplitudes = amplitude_table(kin["momenta"], kin["m"], F1, F2)
        rho, _spin_signal, squared_amplitude = density_matrix_from_amplitudes(
            amplitudes,
            spin_case=SPIN_CASE_UNPOLARIZED,
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
            "s": s,
            "theta_in": theta_in,
            "phi_in": phi_in,
            "qOut": qOut,
            "phiOut": phiOut,
            "Q2": kin["Q2"],
            "xB": kin["xB"],
            "t": kin["t"],
            "F1": F1,
            "F2": F2,
            "raw_trace": raw_trace,
            "squared_amplitude": squared_amplitude,
            "normalized": not np.isclose(raw_trace, 1.0, rtol=tol, atol=tol),
            "normalized_trace": normalized_trace,
            "trace_error": trace_error,
        })
    return rows


def clean_generated_outputs():
    """Remove files generated by this script before creating fresh outputs."""
    generated_paths = (LOG_PATH, OUTPUT_DIR)
    for path in generated_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


from PlotUtils import require_matplotlib as _require_matplotlib


def _safe_float_for_filename(name, value):
    """Format a floating-point value as a filesystem-safe filename token."""
    text = f"{value:.6f}".replace("-", "m").replace(".", "p")
    return f"{name}_{text}"


def scan_output_dir(scan):
    """Return the output directory for a scan dictionary."""
    return OUTPUT_DIR / scan["spin_case"] / scan["label"]


def scan_point_dir(scan):
    """Return the per-point output directory for a scan dictionary."""
    return scan_output_dir(scan) / "SpinDensityScan"


def spin_case_filename_label(spin_case):
    """Return the spin-case label used in generated filenames."""
    initial_state_ensemble(spin_case)
    return spin_case


def _scan_point_stem_from_indices(scan, y_index, x_index):
    """Return a filename stem identifying one scan point."""
    x_value = scan["x_values"][x_index]
    y_value = scan["y_values"][y_index]
    spin_label = spin_case_filename_label(scan["spin_case"])
    return (
        f"spin_density_{spin_label}_"
        f"{_safe_float_for_filename(scan['x_name'], x_value)}_"
        f"{_safe_float_for_filename(scan['y_name'], y_value)}"
    )


def _plot_scan_page(
    ax,
    data,
    x_values,
    y_values,
    x_label,
    y_label,
    title,
    cmap,
    vmin=None,
    vmax=None,
):
    """Draw one kinematic-grid heatmap page and return the image artist."""
    image = ax.imshow(
        np.ma.masked_invalid(data),
        origin="lower",
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(labelsize=10)
    return image


def save_entanglement_plot(scan, output_path=None):
    """Save heatmap pages for concurrence observables and ``F3``."""
    plt, PdfPages = _require_matplotlib()
    if output_path is None:
        spin_label = spin_case_filename_label(scan["spin_case"])
        output_path = (
            scan_output_dir(scan)
            / f"spin_entanglement_scan_{spin_label}_{scan['label']}.pdf"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_specs = (
        ("C_e_p", r"$C_{ep}$"),
        ("C_e_gamma", r"$C_{e\gamma}$"),
        ("C_p_gamma", r"$C_{p\gamma}$"),
        ("C_e_rest", r"$C_{e|p\gamma}$"),
        ("C_p_rest", r"$C_{p|e\gamma}$"),
        ("C_gamma_rest", r"$C_{\gamma|ep}$"),
        ("F3", r"$F_3$"),
    )
    is_polarized_difference = False
    cmap = "coolwarm" if is_polarized_difference else "viridis"
    vmin = -1.0 if is_polarized_difference else 0.0
    vmax = 1.0
    title_prefix = r"$\Delta_h$ " if is_polarized_difference else ""

    with PdfPages(output_path) as pdf:
        for name, label in plot_specs:
            fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
            image = _plot_scan_page(
                ax,
                scan["entanglement"][name],
                scan["x_values"],
                scan["y_values"],
                scan["x_label"],
                scan["y_label"],
                f"{title_prefix}{label}",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            colorbar = fig.colorbar(image, ax=ax, label=f"{title_prefix}{label}")
            colorbar.set_label(f"{title_prefix}{label}", fontsize=12)
            colorbar.ax.tick_params(labelsize=10)
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def save_scan_npz(scan, path=None):
    """Persist the full scan arrays to an NPZ archive."""
    if path is None:
        spin_label = spin_case_filename_label(scan["spin_case"])
        path = scan_output_dir(scan) / f"spin_density_scan_{spin_label}_{scan['label']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    entanglement_arrays = {
        f"entanglement_{name}": values
        for name, values in scan["entanglement"].items()
    }
    np.savez(
        path,
        rho=scan["rho"],
        squared_amplitude=scan["squared_amplitude"],
        spin_signal=scan["spin_signal"],
        cross_section_ratio=scan["cross_section_ratio"],
        purity=scan["purity"],
        trace=scan["trace"],
        valid=scan["valid"],
        x_values=scan["x_values"],
        y_values=scan["y_values"],
        **{
            f"grid_{name}": values
            for name, values in scan.get("kinematic_grids", {}).items()
        },
        scan_label=scan["label"],
        x_name=scan["x_name"],
        y_name=scan["y_name"],
        out_states=np.asarray(scan["out_states"], dtype=int),
        initial_states=np.asarray(scan["initial_states"], dtype=int),
        incoming_spin_weights=scan["incoming_spin_weights"],
        initial_spin_density_matrix=initial_spin_density_matrix(scan["spin_case"]),
        normalized_to_unit_trace=scan["normalized_to_unit_trace"],
        process_density_shape=np.asarray([32, 32], dtype=int),
        process_basis="hIn,sIn,hOut,sOut,lambda",
        entanglement_names=np.asarray(scan["entanglement_names"], dtype=str),
        entanglement_defined=scan["entanglement_defined"],
        entanglement_mode=scan["entanglement_mode"],
        spin_case=scan["spin_case"],
        form_factor_model=scan["form_factor_model"],
        **entanglement_arrays,
    )
    return path


def _matrix_headers(include_matrix_indices):
    """Return CSV headers for summary or per-matrix rows."""
    headers = [
        "spin_case",
        "entanglement_mode",
        "scan_x_name",
        "scan_x_value",
        "scan_y_name",
        "scan_y_value",
        "s",
        "sqrt_s",
        "pIn",
        "pOut",
        "qOut",
        "theta_in",
        "phi_in",
        "phiOut",
        "Q2",
        "xB",
        "t",
        "F1",
        "F2",
        "W2",
        "y",
        "squared_amplitude_M2",
        "spin_signal_M2",
        "cross_section_ratio",
        "purity",
        "trace",
        "normalized_to_unit_trace",
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


def _metadata_row(scan, y_index, x_index):
    """Return common user-frame kinematic and entanglement columns for one point."""
    grids = scan.get("kinematic_grids", {})

    def grid_value(name, default=np.nan):
        if name in grids:
            return grids[name][y_index, x_index]
        return default

    return [
        scan["spin_case"],
        scan["entanglement_mode"],
        scan["x_name"],
        f"{scan['x_values'][x_index]:.16e}",
        scan["y_name"],
        f"{scan['y_values'][y_index]:.16e}",
        f"{grid_value('s'):.16e}",
        f"{grid_value('sqrt_s'):.16e}",
        f"{grid_value('pIn'):.16e}",
        f"{grid_value('pOut'):.16e}",
        f"{grid_value('qOut'):.16e}",
        f"{grid_value('theta_in'):.16e}",
        f"{grid_value('phi_in'):.16e}",
        f"{grid_value('phiOut'):.16e}",
        f"{grid_value('Q2'):.16e}",
        f"{grid_value('xB'):.16e}",
        f"{grid_value('t'):.16e}",
        f"{grid_value('F1'):.16e}",
        f"{grid_value('F2'):.16e}",
        f"{grid_value('W2'):.16e}",
        f"{grid_value('y'):.16e}",
        f"{scan['squared_amplitude'][y_index, x_index]:.16e}",
        f"{scan['spin_signal'][y_index, x_index]:.16e}",
        f"{scan['cross_section_ratio'][y_index, x_index]:.16e}",
        f"{scan['purity'][y_index, x_index]:.16e}",
        f"{scan['trace'][y_index, x_index]:.16e}",
        scan["normalized_to_unit_trace"],
        *(
            f"{scan['entanglement'][name][y_index, x_index]:.16e}"
            for name in scan["entanglement_names"]
        ),
    ]


def save_entanglement_csv(scan, path=None):
    """Save one summary row per valid kinematic point."""
    if path is None:
        spin_label = spin_case_filename_label(scan["spin_case"])
        path = (
            scan_output_dir(scan)
            / f"spin_entanglement_scan_{spin_label}_{scan['label']}.csv"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_matrix_headers(include_matrix_indices=False))
        for y_index, _y in enumerate(scan["y_values"]):
            for x_index, _x in enumerate(scan["x_values"]):
                if scan["valid"][y_index, x_index]:
                    writer.writerow(_metadata_row(scan, y_index, x_index))
    return path


def save_scan_csv_files(scan, output_dir=None):
    """Save one long-form ``8 x 8`` density-matrix CSV per valid point."""
    if output_dir is None:
        output_dir = scan_point_dir(scan)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    headers = _matrix_headers(include_matrix_indices=True)

    for y_index, _y in enumerate(scan["y_values"]):
        for x_index, _x in enumerate(scan["x_values"]):
            if not scan["valid"][y_index, x_index]:
                continue

            path = output_dir / f"{_scan_point_stem_from_indices(scan, y_index, x_index)}.csv"
            matrix = scan["rho"][y_index, x_index]
            metadata = _metadata_row(scan, y_index, x_index)
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


def save_point_matrix_plots(scan, output_dir=None):
    """Save norm and phase ``8 x 8`` matrix plots for every valid point."""
    plt, _PdfPages = _require_matplotlib()
    if output_dir is None:
        output_dir = scan_point_dir(scan)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_label = spin_case_display_label(scan["spin_case"])
    rho_symbol = rf"\rho_{{\mathrm{{{display_label}}}}}"
    state_key = "\n".join(_state_tick_labels(scan["out_states"]))
    paths = []

    for y_index, y_value in enumerate(scan["y_values"]):
        for x_index, x_value in enumerate(scan["x_values"]):
            if not scan["valid"][y_index, x_index]:
                continue

            stem = _scan_point_stem_from_indices(scan, y_index, x_index)
            matrix = scan["rho"][y_index, x_index]
            title_suffix = (
                f"{display_label}, {scan['x_name']}={x_value:.6g}, "
                f"{scan['y_name']}={y_value:.6g}"
            )

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
        "s",
        "theta_in",
        "qOut",
        "F1",
        "F2",
        "raw Tr(rho)",
        "|M|^2",
        "normalized",
        "Tr(rho_norm)",
        "error",
    )
    table_rows = [
        (
            row["case"],
            f"{row['s']:.6g}",
            f"{row['theta_in']:.6g}",
            f"{row['qOut']:.6g}",
            f"{row['F1']:.6g}",
            f"{row['F2']:.6g}",
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


def build_scan_report(scan, paths):
    """Build the report block for one completed scan."""
    y_values = scan["y_values"]
    fixed_user = scan["fixed_user"]
    fixed_line = (
        "  fixed user vars: "
        + ", ".join(
            f"{name}={value:.6g}"
            for name, value in fixed_user.items()
            if name not in (scan["x_name"], scan["y_name"])
        )
    )
    point_dir = scan_point_dir(scan)
    lines = [
        f"Scan {spin_case_display_label(scan['spin_case'])} "
        f"[{scan['spin_case']}]/{scan['label']}",
        f"  outgoing basis size: {len(scan['out_states'])}",
        "  particle map: 1 = outgoing electron hOut, 2 = outgoing proton sOut, 3 = outgoing photon lambda",
        f"  entanglement observables: {', '.join(scan['entanglement_names'])}",
        f"  entanglement mode: {scan['entanglement_mode']}",
        f"  form factor model: {scan['form_factor_model']} with F1(t), F2(t)",
        f"  {scan['x_name']} grid: {scan['x_values'][0]:.6g} to {scan['x_values'][-1]:.6g}",
        f"  {scan['y_name']} grid: {y_values[0]:.6g} to {y_values[-1]:.6g}",
        fixed_line,
        f"  valid points: {int(scan['valid'].sum())}/{scan['valid'].size}",
        f"  density-numerator averaging divisor: {initial_spin_average_divisor(scan['spin_case']):.0f}",
        "  unpolarized M^2 averaging divisor: 4",
        f"  normalized to unit trace: {scan['normalized_to_unit_trace']}",
        "  full process density matrix: 32 x 32 in (hIn,sIn,hOut,sOut,lambda)",
        f"  incoming spin weights: {scan['incoming_spin_weights'].tolist()} for {scan['initial_states']}",
    ]
    lines.append("  prepared initial-state ensemble:")
    for component in initial_state_ensemble(scan["spin_case"]):
        lines.append(
            f"    weight={component['weight']:.6g}: {component['label']}"
        )
    lines.append(
        "  entanglement convention: weighted average of the normalized "
        "pure-state observable for each incoherent ensemble component"
    )
    lines.extend([
        f"  saved data: {paths['npz']}",
        (
            f"  saved entanglement csv: {paths['entanglement_csv']}"
            if paths["entanglement_csv"] is not None
            else "  saved entanglement csv: not generated"
        ),
        (
            f"  saved entanglement plots: {paths['entanglement_plot']}"
            if paths["entanglement_plot"] is not None
            else "  saved entanglement plots: not generated"
        ),
        f"  saved matrix csv files: {len(paths['matrix_csv'])} in {point_dir}",
        f"  saved matrix plot files: {len(paths['matrix_plots'])} in {point_dir}",
    ])
    if scan["failures"]:
        lines.append("  invalid grid points:")
        for s, theta_in, phi_in, qOut, phiOut, message in scan["failures"]:
            lines.append(
                f"    s={s:.8g}, theta_in={theta_in:.8g}, phi_in={phi_in:.8g}, "
                f"qOut={qOut:.8g}, phiOut={phiOut:.8g}: {message}"
            )
    return "\n".join(lines)


def build_report(scan_results, trace_benchmark_rows):
    """Build the text report printed to console and written to the log file."""
    lines = [
        "Spin-density matrix scans",
        "  trace benchmark: passed",
        format_trace_benchmark_rows(trace_benchmark_rows),
        "",
    ]
    for scan, paths in scan_results:
        lines.append(build_scan_report(scan, paths))
        lines.append("")
    lines.append(f"Saved log: {LOG_PATH}")
    return "\n".join(lines).rstrip() + "\n"


def main():
    """Regenerate all SpinDensityMat outputs from the current settings."""
    clean_generated_outputs()
    trace_benchmark_rows = benchmark_spin_density_trace()
    fixed_user = {
        "s": USER_FIXED_S,
        "theta_in": USER_FIXED_THETA_IN,
        "phi_in": USER_FIXED_PHI_IN,
        "qOut": USER_FIXED_QOUT,
        "phiOut": USER_FIXED_PHI_OUT,
    }
    scans = []
    for spin_case in SPIN_CASES:
        scans.extend([
            scan_spin_density_user_grid(
                USER_S_VALUES,
                USER_QOUT_VALUES,
                x_name="s",
                y_name="qOut",
                fixed_user=fixed_user,
                spin_case=spin_case,
            ),
            scan_spin_density_user_grid(
                USER_THETA_IN_VALUES,
                USER_PHI_OUT_VALUES,
                x_name="theta_in",
                y_name="phiOut",
                fixed_user=fixed_user,
                spin_case=spin_case,
            ),
        ])
    scan_results = []
    for scan in scans:
        paths = {
            "npz": save_scan_npz(scan),
            "entanglement_csv": save_entanglement_csv(scan),
            "entanglement_plot": save_entanglement_plot(scan),
            "matrix_csv": save_scan_csv_files(scan),
            "matrix_plots": save_point_matrix_plots(scan),
        }
        scan_results.append((scan, paths))

    log_text = build_report(scan_results, trace_benchmark_rows)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
