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
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from Algebra import HELICITIES
from BHHelicityAmp import bh_amplitude_table
from Kinematics import kinematics_user_from_independent


# ============================================================
# Scan and output settings
# ============================================================

M = 0.938
F1 = 1.0
F2 = 0.0

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

AVERAGE_INITIAL_SPINS = False
NORMALIZE_TRACE = True
TRACE_BENCHMARK_TOL = 1e-10
SCAN_WORKERS = max(1, min(os.cpu_count() or 1, 8))
SPIN_CASE_UNPOLARIZED = "unpolarized"
SPIN_CASE_POLARIZED = "polarized"
SPIN_CASE_TRANSVERSE_TX = "transverse_Tx"
SPIN_CASE_TRANSVERSE_TY = "transverse_Ty"
LEGACY_SPIN_CASE_TRANSVERSE = "transverse"
SPIN_CASE_TRANSVERSE = SPIN_CASE_TRANSVERSE_TX
TRANSVERSE_SPIN_CASES = (
    SPIN_CASE_TRANSVERSE_TX,
    SPIN_CASE_TRANSVERSE_TY,
    LEGACY_SPIN_CASE_TRANSVERSE,
)
SPIN_CASES = (
    SPIN_CASE_UNPOLARIZED,
    SPIN_CASE_POLARIZED,
    SPIN_CASE_TRANSVERSE_TX,
    SPIN_CASE_TRANSVERSE_TY,
)
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


def incoming_spin_weights(spin_case=SPIN_CASE_UNPOLARIZED):
    """Return initial-state diagonal weights for a configured spin scan.

    The polarized case is the incoming-electron helicity difference
    ``hIn=+1`` minus ``hIn=-1``, with the incoming proton spin summed in both
    terms. The transverse case stores only the diagonal populations here; its
    coherent interference terms are handled by
    :func:`density_matrix_from_amplitudes`.
    """
    if spin_case == SPIN_CASE_UNPOLARIZED:
        return np.ones(len(initial_spin_states()), dtype=float)
    if spin_case == SPIN_CASE_POLARIZED:
        return np.asarray(
            [1.0 if h_in == 1 else -1.0 for h_in, _s_in in initial_spin_states()],
            dtype=float,
        )
    if is_transverse_spin_case(spin_case):
        return np.full(len(initial_spin_states()), 0.5, dtype=float)
    raise ValueError(f"Unknown spin density case: {spin_case}")


def is_transverse_spin_case(spin_case):
    """Return whether ``spin_case`` is a coherent transverse electron state."""
    return spin_case in TRANSVERSE_SPIN_CASES


def transverse_electron_coefficients(spin_case=SPIN_CASE_TRANSVERSE_TX):
    """Return incoming-electron coefficients for a transverse spin case."""
    coefficient = 1.0 / np.sqrt(2.0)
    if spin_case in (SPIN_CASE_TRANSVERSE_TX, LEGACY_SPIN_CASE_TRANSVERSE):
        return {-1: coefficient, +1: coefficient}
    if spin_case == SPIN_CASE_TRANSVERSE_TY:
        return {-1: 1.0j * coefficient, +1: coefficient}
    raise ValueError(f"Unknown transverse spin density case: {spin_case}")


def amplitude_table(mom, m, F1, F2):
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
    )


def density_matrix_from_amplitudes(
    amplitudes,
    average_initial=False,
    spin_case=SPIN_CASE_UNPOLARIZED,
):
    """Build the outgoing spin-density matrix from an amplitude table.

    For the unpolarized case, the convention is
    ``rho_ij = sum_initial A_initial,i conj(A_initial,j)``. For the polarized
    case, the incoming electron helicity weights are ``+1`` for ``hIn=+1`` and
    ``-1`` for ``hIn=-1``. For transverse cases, the incoming electron is
    coherent and the incoming proton spin is summed. ``Tx`` is
    ``(|h=+1> + |h=-1>)/sqrt(2)``; ``Ty`` is
    ``(|h=+1> + i |h=-1>)/sqrt(2)``.

    The returned tuple is ``(rho, spin_signal, squared_amplitude)``. The
    ``spin_signal`` is the weighted trace numerator, while
    ``squared_amplitude`` is the unpolarized ``sum |A|^2`` normalization
    denominator.
    """
    in_states = initial_spin_states()
    if amplitudes.shape[0] != len(in_states):
        raise ValueError(
            "Amplitude table first axis does not match the incoming spin basis."
        )

    amplitude_norms = np.sum(np.abs(amplitudes) ** 2, axis=1)
    squared_amplitude = float(np.sum(amplitude_norms))
    if is_transverse_spin_case(spin_case):
        coefficients = transverse_electron_coefficients(spin_case)
        rho = np.zeros((amplitudes.shape[1], amplitudes.shape[1]), dtype=complex)
        spin_signal = 0.0
        for s_in in HELICITIES:
            state = sum(
                coefficients[h_in] * amplitudes[in_states.index((h_in, s_in))]
                for h_in in HELICITIES
            )
            rho += np.outer(state, state.conj())
            spin_signal += float(np.sum(np.abs(state) ** 2))
    else:
        weights = incoming_spin_weights(spin_case)
        weighted_amplitudes = weights[:, np.newaxis] * amplitudes
        rho = weighted_amplitudes.T @ np.conjugate(amplitudes)
        spin_signal = float(np.sum(weights * amplitude_norms))
    if average_initial:
        rho /= amplitudes.shape[0]
        spin_signal /= amplitudes.shape[0]
        squared_amplitude /= amplitudes.shape[0]
    return rho, spin_signal, squared_amplitude


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


def normalized_transverse_final_state(
    amplitudes,
    proton_spin,
    spin_case=SPIN_CASE_TRANSVERSE_TX,
):
    """Return the normalized final state for transverse incoming electron spin."""
    in_states = initial_spin_states()
    if proton_spin not in HELICITIES:
        raise ValueError(f"Unknown incoming proton spin: {proton_spin}")

    coefficients = transverse_electron_coefficients(spin_case)
    state = sum(
        coefficients[h_in] * amplitudes[in_states.index((h_in, proton_spin))]
        for h_in in HELICITIES
    )
    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    if norm <= 1e-14:
        raise ZeroDivisionError(
            "Cannot normalize a zero final-state amplitude for "
            f"{spin_case} incoming electron and sIn={proton_spin}."
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


def entanglement_measures_from_amplitudes(amplitudes, initial_state):
    """Compute entanglement observables for one incoming spin state."""
    state = normalized_final_state(amplitudes, initial_state=initial_state)
    return entanglement_measures_from_state(state)


def polarized_entanglement_difference(amplitudes, proton_spin):
    """Return hIn=+1 minus hIn=-1 entanglement observables at fixed sIn."""
    plus = entanglement_measures_from_amplitudes(amplitudes, (+1, proton_spin))
    minus = entanglement_measures_from_amplitudes(amplitudes, (-1, proton_spin))
    return {name: plus[name] - minus[name] for name in ENTANGLEMENT_NAMES}


def transverse_entanglement_measures(
    amplitudes,
    proton_spin,
    spin_case=SPIN_CASE_TRANSVERSE_TX,
):
    """Return entanglement observables for the transverse incoming electron state."""
    state = normalized_transverse_final_state(amplitudes, proton_spin, spin_case)
    return entanglement_measures_from_state(state)


def build_user_scan_point(
    s,
    theta_in,
    phi_in,
    qOut,
    phiOut,
    m,
    F1,
    F2,
    average_initial=AVERAGE_INITIAL_SPINS,
    normalize_trace=NORMALIZE_TRACE,
    entanglement_initial_state=ENTANGLEMENT_INITIAL_STATE,
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
    amplitudes = amplitude_table(kin["momenta"], kin["m"], F1, F2)
    rho, spin_signal, squared_amplitude = density_matrix_from_amplitudes(
        amplitudes,
        average_initial=average_initial,
        spin_case=spin_case,
    )
    if normalize_trace:
        if squared_amplitude <= 1e-14:
            raise ZeroDivisionError("Cannot trace-normalize a zero density matrix.")
        rho /= squared_amplitude

    if spin_case == SPIN_CASE_UNPOLARIZED:
        entanglement = entanglement_measures_from_amplitudes(
            amplitudes,
            entanglement_initial_state,
        )
    elif spin_case == SPIN_CASE_POLARIZED:
        entanglement = polarized_entanglement_difference(
            amplitudes,
            entanglement_initial_state[1],
        )
    elif is_transverse_spin_case(spin_case):
        entanglement = transverse_entanglement_measures(
            amplitudes,
            entanglement_initial_state[1],
            spin_case,
        )
    else:
        raise ValueError(f"Unknown spin density case: {spin_case}")

    return {
        "rho": rho,
        "squared_amplitude": squared_amplitude,
        "spin_signal": spin_signal,
        "trace": trace_value(rho),
        "entanglement": entanglement,
        "kinematics": kin,
    }


def _scan_spin_density_user_grid_task(task):
    """Evaluate one independent user-frame spin-density grid point."""
    y_index, x_index, user_vars, settings = task
    try:
        point = build_user_scan_point(
            user_vars["s"],
            user_vars["theta_in"],
            user_vars["phi_in"],
            user_vars["qOut"],
            user_vars["phiOut"],
            settings["m"],
            settings["F1"],
            settings["F2"],
            average_initial=settings["average_initial"],
            normalize_trace=settings["normalize_trace"],
            entanglement_initial_state=settings["entanglement_initial_state"],
            spin_case=settings["spin_case"],
        )
    except Exception as exc:
        return {"ok": False, **user_vars, "error": str(exc)}
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
    F1=F1,
    F2=F2,
    average_initial=AVERAGE_INITIAL_SPINS,
    normalize_trace=NORMALIZE_TRACE,
    entanglement_initial_state=ENTANGLEMENT_INITIAL_STATE,
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
            "W2",
            "y",
        )
    }
    failures = []

    settings = {
        "m": m,
        "F1": F1,
        "F2": F2,
        "average_initial": average_initial,
        "normalize_trace": normalize_trace,
        "entanglement_initial_state": entanglement_initial_state,
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
        user_params = kin["user_params"]
        user_independent = kin["user_independent"]
        rho_grid[y_index, x_index] = point["rho"]
        squared_amplitude_grid[y_index, x_index] = point["squared_amplitude"]
        spin_signal_grid[y_index, x_index] = point["spin_signal"]
        trace_grid[y_index, x_index] = point["trace"]
        for name, value in point["entanglement"].items():
            entanglement_grid[name][y_index, x_index] = value
        for key in ("s", "theta_in", "phi_in", "qOut", "phiOut"):
            kinematic_grids[key][y_index, x_index] = user_independent[key]
        kinematic_grids["sqrt_s"][y_index, x_index] = kin["sqrt_s"]
        kinematic_grids["pIn"][y_index, x_index] = user_params["pIn"]
        kinematic_grids["pOut"][y_index, x_index] = user_params["pOut"]
        for key in ("Q2", "xB", "t", "W2", "y"):
            kinematic_grids[key][y_index, x_index] = kin[key]
        valid[y_index, x_index] = True

    return {
        "rho": rho_grid,
        "squared_amplitude": squared_amplitude_grid,
        "spin_signal": spin_signal_grid,
        "trace": trace_grid,
        "t_grid": kinematic_grids["t"],
        "phi_grid": kinematic_grids["phiOut"],
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
        "normalized_by_squared_amplitude": normalize_trace,
        "entanglement_initial_state": entanglement_initial_state,
        "entanglement_defined": True,
        "entanglement_mode": entanglement_mode(spin_case),
        "spin_case": spin_case,
        "scan_parameterization": "user_frame_independent",
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
    if spin_case == SPIN_CASE_UNPOLARIZED:
        return "pure_initial_state"
    if spin_case == SPIN_CASE_POLARIZED:
        return "h_in_plus_minus_h_in_minus"
    if spin_case in (SPIN_CASE_TRANSVERSE_TX, LEGACY_SPIN_CASE_TRANSVERSE):
        return "Tx_h_in_plus_plus_h_in_minus_over_sqrt2"
    if spin_case == SPIN_CASE_TRANSVERSE_TY:
        return "Ty_h_in_plus_plus_i_h_in_minus_over_sqrt2"
    raise ValueError(f"Unknown spin density case: {spin_case}")


def benchmark_spin_density_trace(
    kinematic_inputs=BENCHMARK_USER_KINEMATIC_INPUTS,
    m=M,
    F1=F1,
    F2=F2,
    average_initial=AVERAGE_INITIAL_SPINS,
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
        amplitudes = amplitude_table(kin["momenta"], kin["m"], F1, F2)
        rho, _spin_signal, squared_amplitude = density_matrix_from_amplitudes(
            amplitudes,
            average_initial=average_initial,
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
            "raw_trace": raw_trace,
            "squared_amplitude": squared_amplitude,
            "normalized": not np.isclose(raw_trace, 1.0, rtol=tol, atol=tol),
            "normalized_trace": normalized_trace,
            "trace_error": trace_error,
        })
    return rows


def clean_generated_outputs():
    """Remove files generated by this script before creating fresh outputs."""
    generated_paths = (
        LOG_PATH,
        OUTPUT_DIR / SPIN_CASE_UNPOLARIZED,
        OUTPUT_DIR / SPIN_CASE_POLARIZED,
        OUTPUT_DIR / LEGACY_SPIN_CASE_TRANSVERSE,
        OUTPUT_DIR / SPIN_CASE_TRANSVERSE_TX,
        OUTPUT_DIR / SPIN_CASE_TRANSVERSE_TY,
    )
    for path in generated_paths:
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


def scan_output_dir(scan):
    """Return the output directory for a scan dictionary."""
    return OUTPUT_DIR / scan["spin_case"] / scan["label"]


def scan_point_dir(scan):
    """Return the per-point output directory for a scan dictionary."""
    return scan_output_dir(scan) / "SpinDensityScan"


def spin_case_filename_label(spin_case):
    """Return the spin-case label used in generated filenames."""
    if spin_case == SPIN_CASE_UNPOLARIZED:
        return "unpolarized"
    if spin_case == SPIN_CASE_POLARIZED:
        return "longitudinal_polarized"
    if spin_case in (SPIN_CASE_TRANSVERSE_TX, LEGACY_SPIN_CASE_TRANSVERSE):
        return "transverse_Tx"
    if spin_case == SPIN_CASE_TRANSVERSE_TY:
        return "transverse_Ty"
    raise ValueError(f"Unknown spin density case: {spin_case}")


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
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
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
        ("C12", r"$C_{12}$"),
        ("C13", r"$C_{13}$"),
        ("C23", r"$C_{23}$"),
        ("C1_23", r"$C_{1(23)}$"),
        ("C2_13", r"$C_{2(13)}$"),
        ("C3_12", r"$C_{3(12)}$"),
        ("F3", r"$F_3$"),
    )
    is_polarized_difference = scan["entanglement_mode"] == "h_in_plus_minus_h_in_minus"
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
            fig.colorbar(image, ax=ax, label=f"{title_prefix}{label}")
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
        trace=scan["trace"],
        valid=scan["valid"],
        x_values=scan["x_values"],
        y_values=scan["y_values"],
        t_grid=scan["t_grid"],
        phi_grid=scan["phi_grid"],
        **{
            f"grid_{name}": values
            for name, values in scan.get("kinematic_grids", {}).items()
        },
        scan_label=scan["label"],
        x_name=scan["x_name"],
        y_name=scan["y_name"],
        scan_parameterization=scan["scan_parameterization"],
        out_states=np.asarray(scan["out_states"], dtype=int),
        initial_states=np.asarray(scan["initial_states"], dtype=int),
        incoming_spin_weights=scan["incoming_spin_weights"],
        transverse_electron_coefficients=np.asarray(
            [
                transverse_electron_coefficients(scan["spin_case"])[h_in]
                for h_in in HELICITIES
            ],
            dtype=complex,
        ),
        normalized_by_squared_amplitude=scan["normalized_by_squared_amplitude"],
        entanglement_names=np.asarray(scan["entanglement_names"], dtype=str),
        entanglement_initial_state=np.asarray(scan["entanglement_initial_state"], dtype=int),
        entanglement_defined=scan["entanglement_defined"],
        entanglement_mode=scan["entanglement_mode"],
        spin_case=scan["spin_case"],
        **entanglement_arrays,
    )
    return path


def _matrix_headers(include_matrix_indices):
    """Return CSV headers for summary or per-matrix rows."""
    headers = [
        "spin_case",
        "entanglement_mode",
        "scan_parameterization",
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
        "W2",
        "y",
        "squared_amplitude_M2",
        "spin_signal_M2",
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
        scan["scan_parameterization"],
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
        f"{grid_value('W2'):.16e}",
        f"{grid_value('y'):.16e}",
        f"{scan['squared_amplitude'][y_index, x_index]:.16e}",
        f"{scan['spin_signal'][y_index, x_index]:.16e}",
        f"{scan['trace'][y_index, x_index]:.16e}",
        scan["normalized_by_squared_amplitude"],
        *scan["entanglement_initial_state"],
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

    if scan["spin_case"] == SPIN_CASE_POLARIZED:
        rho_symbol = r"\Delta\rho_h/M^2_{\rm unpol}" if scan["normalized_by_squared_amplitude"] else r"\Delta\rho_h"
    elif scan["spin_case"] in (SPIN_CASE_TRANSVERSE_TX, LEGACY_SPIN_CASE_TRANSVERSE):
        rho_symbol = r"\rho_{T_x}/M^2_{\rm unpol}" if scan["normalized_by_squared_amplitude"] else r"\rho_{T_x}"
    elif scan["spin_case"] == SPIN_CASE_TRANSVERSE_TY:
        rho_symbol = r"\rho_{T_y}/M^2_{\rm unpol}" if scan["normalized_by_squared_amplitude"] else r"\rho_{T_y}"
    else:
        rho_symbol = r"\rho/M^2" if scan["normalized_by_squared_amplitude"] else r"\rho"
    state_key = "\n".join(_state_tick_labels(scan["out_states"]))
    paths = []

    for y_index, y_value in enumerate(scan["y_values"]):
        for x_index, x_value in enumerate(scan["x_values"]):
            if not scan["valid"][y_index, x_index]:
                continue

            stem = _scan_point_stem_from_indices(scan, y_index, x_index)
            matrix = scan["rho"][y_index, x_index]
            title_suffix = (
                f"{scan['spin_case']}, {scan['x_name']}={x_value:.6g}, "
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
        f"Scan {scan['spin_case']}/{scan['label']}",
        f"  outgoing basis size: {len(scan['out_states'])}",
        "  particle map: 1 = outgoing electron hOut, 2 = outgoing proton sOut, 3 = outgoing photon lambda",
        f"  entanglement observables: {', '.join(scan['entanglement_names'])}",
        f"  entanglement mode: {scan['entanglement_mode']}",
        f"  scan parameterization: {scan['scan_parameterization']}",
        f"  {scan['x_name']} grid: {scan['x_values'][0]:.6g} to {scan['x_values'][-1]:.6g}",
        f"  {scan['y_name']} grid: {y_values[0]:.6g} to {y_values[-1]:.6g}",
        fixed_line,
        f"  valid points: {int(scan['valid'].sum())}/{scan['valid'].size}",
        f"  initial spins averaged: {AVERAGE_INITIAL_SPINS}",
        f"  normalized by M^2: {scan['normalized_by_squared_amplitude']}",
        f"  incoming spin weights: {scan['incoming_spin_weights'].tolist()} for {scan['initial_states']}",
    ]
    if scan["spin_case"] == SPIN_CASE_UNPOLARIZED:
        lines.append(
            "  entanglement initial state: "
            f"hIn={scan['entanglement_initial_state'][0]:+d}, "
            f"sIn={scan['entanglement_initial_state'][1]:+d}"
        )
    elif scan["spin_case"] == SPIN_CASE_POLARIZED:
        lines.append(
            "  polarized convention: sum_sIn rho(hIn=+1,sIn) - "
            "sum_sIn rho(hIn=-1,sIn)"
        )
        lines.append(
            "  polarized entanglement: "
            f"E(hIn=+1, sIn={scan['entanglement_initial_state'][1]:+d}) - "
            f"E(hIn=-1, sIn={scan['entanglement_initial_state'][1]:+d})"
        )
    elif scan["spin_case"] in (SPIN_CASE_TRANSVERSE_TX, LEGACY_SPIN_CASE_TRANSVERSE):
        lines.append(
            "  transverse Tx convention: sum_sIn rho((hIn=+1 + hIn=-1)/sqrt(2), sIn)"
        )
        lines.append(
            "  transverse Tx entanglement: "
            "E((hIn=+1 + hIn=-1)/sqrt(2), "
            f"sIn={scan['entanglement_initial_state'][1]:+d})"
        )
    elif scan["spin_case"] == SPIN_CASE_TRANSVERSE_TY:
        lines.append(
            "  transverse Ty convention: sum_sIn rho((hIn=+1 + i hIn=-1)/sqrt(2), sIn)"
        )
        lines.append(
            "  transverse Ty entanglement: "
            "E((hIn=+1 + i hIn=-1)/sqrt(2), "
            f"sIn={scan['entanglement_initial_state'][1]:+d})"
        )
    else:
        raise ValueError(f"Unknown spin density case: {scan['spin_case']}")
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
