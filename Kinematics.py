"""User-frame kinematic builders and validation checks.

The repository uses a direct COM-frame backend specified by ``pIn``, ``pOut``,
``qOut``, ``th``, ``ph``, and ``phOut``.  Scan scripts use the independent
user-frame set ``(s, theta_in, phi_in, qOut, phiOut)`` and solve ``pOut`` from
energy conservation.

All four-vectors are contravariant arrays in ``[E, px, py, pz]`` order. The
massless external electron and photon are placed on shell by construction, and
the proton mass is supplied explicitly as ``m``.
"""

import numpy as np

from Algebra import (
    DEFAULT_TOL,
    _validate_nonnegative_scalar,
    _validate_positive_scalar,
    _validate_scalar,
    mdot,
)


# ============================================================
# User kinematics
#
# p' = (sqrt(pOut^2+m^2), 0, pOut, 0)
# q' = qOut (1, cos phOut, sin phOut, 0)
# k' = (sqrt(pOut^2+qOut^2+2 pOut qOut sin phOut),
#       -qOut cos phOut, -pOut - qOut sin phOut, 0)
#
# p = (sqrt(pIn^2+m^2),
#      pIn sin th cos ph, pIn sin th sin ph, pIn cos th)
# k = pIn (1, -sin th cos ph, -sin th sin ph, -cos th)
# ============================================================

def k_user(pIn, th, ph):
    """Return the incoming electron four-momentum in the user frame.

    ``pIn`` is the common incoming three-momentum magnitude in the COM frame.
    The electron is massless and points opposite to the incoming proton
    direction defined by polar angle ``th`` and azimuth ``ph``.
    """
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    return pIn * np.array([
        1.0,
        -np.sin(th) * np.cos(ph),
        -np.sin(th) * np.sin(ph),
        -np.cos(th),
    ])


def p_user(pIn, th, ph, m):
    """Return the incoming proton four-momentum in the user frame."""
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    m = _validate_positive_scalar(m, "m")
    return np.array([
        np.sqrt(pIn**2 + m**2),
        pIn * np.sin(th) * np.cos(ph),
        pIn * np.sin(th) * np.sin(ph),
        pIn * np.cos(th),
    ])


def pp_user(pOut, m):
    """Return the outgoing proton four-momentum in the user frame.

    The user frame fixes the outgoing proton spatial momentum along the
    positive y axis, ``pp = (E, 0, pOut, 0)``.
    """
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    m = _validate_positive_scalar(m, "m")
    return np.array([
        np.sqrt(pOut**2 + m**2),
        0.0,
        pOut,
        0.0,
    ])


def qout_user(qOut, phOut):
    """Return the outgoing real-photon four-momentum in the user frame."""
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    return qOut * np.array([1.0, np.cos(phOut), np.sin(phOut), 0.0])


def kp_user(pOut, qOut, phOut):
    """Return the outgoing electron four-momentum from momentum conservation.

    The spatial momentum is fixed by ``k + p = kp + pp + qout`` in the user
    frame, and the energy is the norm of the massless electron three-momentum.
    """
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    kp3 = np.array([
        -qOut * np.cos(phOut),
        -pOut - qOut * np.sin(phOut),
        0.0,
    ])
    return np.concatenate([[np.linalg.norm(kp3)], kp3])


def momenta_user(pIn, pOut, qOut, th, ph, phOut, m):
    """Return all user-frame external momenta as a dictionary.

    The returned keys are ``k`` (incoming electron), ``p`` (incoming proton),
    ``kp`` (outgoing electron), ``pp`` (outgoing proton), and ``qout``
    (outgoing real photon).
    """
    return {
        "k": k_user(pIn, th, ph),
        "p": p_user(pIn, th, ph, m),
        "kp": kp_user(pOut, qOut, phOut),
        "pp": pp_user(pOut, m),
        "qout": qout_user(qOut, phOut),
    }


def _normalize_angle(angle):
    """Normalize an angle to the interval ``[0, 2*pi)``."""
    return float(angle % (2.0 * np.pi))


def p_in_from_s(s, m):
    """Return the incoming COM momentum magnitude from invariant ``s``."""
    s = _validate_positive_scalar(s, "s")
    m = _validate_positive_scalar(m, "m")
    if s <= m**2:
        raise ValueError("s must be larger than m^2 for a massless electron plus proton.")
    return (s - m**2) / (2.0 * np.sqrt(s))


def _user_energy_residual_for_pout(pOut, sqrt_s, qOut, phOut, m):
    """Return final energy minus ``sqrt_s`` for a user-frame ``pOut`` trial."""
    proton_energy = np.sqrt(pOut**2 + m**2)
    electron_energy = np.sqrt(
        pOut**2 + qOut**2 + 2.0 * pOut * qOut * np.sin(phOut)
    )
    return proton_energy + electron_energy + qOut - sqrt_s


def solve_pout_from_user_independent(s, qOut, phOut, m, tol=1.0e-12):
    """Solve outgoing proton momentum from ``s``, photon energy and ``phOut``.

    The direct user frame conserves three-momentum by construction.  Energy
    conservation then fixes ``pOut`` for the independent set
    ``(s, theta_in, phi_in, qOut, phOut)``.
    """
    s = _validate_positive_scalar(s, "s")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    phOut = _validate_scalar(phOut, "phOut")
    m = _validate_positive_scalar(m, "m")
    sqrt_s = np.sqrt(s)

    low = 0.0
    low_value = _user_energy_residual_for_pout(low, sqrt_s, qOut, phOut, m)
    if low_value > tol:
        raise ValueError(
            "No physical pOut: photon energy is too large for this s and phOut."
        )
    if abs(low_value) <= tol:
        return 0.0

    high = max(1.0, sqrt_s)
    high_value = _user_energy_residual_for_pout(high, sqrt_s, qOut, phOut, m)
    for _iteration in range(80):
        if high_value > 0.0:
            break
        high *= 2.0
        high_value = _user_energy_residual_for_pout(high, sqrt_s, qOut, phOut, m)
    else:
        raise ValueError("Could not bracket a physical pOut solution.")

    for _iteration in range(100):
        mid = 0.5 * (low + high)
        mid_value = _user_energy_residual_for_pout(mid, sqrt_s, qOut, phOut, m)
        if abs(mid_value) <= tol:
            return mid
        if mid_value > 0.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def invariant_q2_xb_t(mom, m):
    """Return derived ``Q2``, ``xB``, and ``t`` from a momentum dictionary."""
    def real_scalar(value):
        value = np.real_if_close(value, tol=1000)
        return float(np.real(value))

    q = mom["k"] - mom["kp"]
    delta = mom["p"] - mom["pp"]
    Q2 = -real_scalar(mdot(q, q))
    p_dot_q = real_scalar(mdot(mom["p"], q))
    if abs(p_dot_q) <= DEFAULT_TOL:
        xB = np.nan
    else:
        xB = Q2 / (2.0 * p_dot_q)
    t = real_scalar(mdot(delta, delta))
    s = real_scalar(mdot(mom["k"] + mom["p"], mom["k"] + mom["p"]))
    return {
        "s": s,
        "sqrt_s": np.sqrt(max(0.0, s)),
        "Q2": Q2,
        "xB": xB,
        "t": t,
        "q": q,
        "W2": real_scalar(mdot(mom["p"] + q, mom["p"] + q)),
        "y": p_dot_q / real_scalar(mdot(mom["p"], mom["k"])),
    }


def kinematics_user_from_independent(s, theta_in, phi_in, qOut, phiOut, m, label=None):
    """Build user-frame COM kinematics from independent user variables.

    Independent variables are ``s``, the incoming proton direction
    ``theta_in``/``phi_in``, the outgoing photon energy ``qOut``, and the
    outgoing photon azimuth ``phiOut``.  The outgoing proton momentum ``pOut``
    is solved from energy conservation.
    """
    s = _validate_positive_scalar(s, "s")
    theta_in = _validate_scalar(theta_in, "theta_in")
    phi_in = _normalize_angle(_validate_scalar(phi_in, "phi_in"))
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    phiOut = _normalize_angle(_validate_scalar(phiOut, "phiOut"))
    m = _validate_positive_scalar(m, "m")

    pIn = p_in_from_s(s, m)
    pOut = solve_pout_from_user_independent(s, qOut, phiOut, m)
    mom = momenta_user(pIn, pOut, qOut, theta_in, phi_in, phiOut, m)
    mom["q"] = mom["k"] - mom["kp"]
    derived = invariant_q2_xb_t(mom, m)
    return {
        "frame": "user_kinematics_com",
        "label": label,
        "m": m,
        "momenta": mom,
        **{key: value for key, value in derived.items() if key != "q"},
        "user_params": {
            "pIn": pIn,
            "pOut": pOut,
            "qOut": qOut,
            "th": _normalize_angle(theta_in),
            "ph": phi_in,
            "phOut": phiOut,
        },
        "user_independent": {
            "s": s,
            "theta_in": _normalize_angle(theta_in),
            "phi_in": phi_in,
            "qOut": qOut,
            "phiOut": phiOut,
        },
        "user_rebuild_residual": abs(
            _user_energy_residual_for_pout(pOut, np.sqrt(s), qOut, phiOut, m)
        ),
    }


def energy_balance(pIn, pOut, qOut, th, ph, phOut, m):
    """Return E_initial - E_final; energy conservation means this is zero."""
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return mom["k"][0] + mom["p"][0] - mom["kp"][0] - mom["pp"][0] - mom["qout"][0]


def momentum_conservation_check(pIn, pOut, qOut, th, ph, phOut, m):
    """Return the residual three-momentum balance vector."""
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return (
        mom["k"][1:4] + mom["p"][1:4]
        - mom["kp"][1:4] - mom["pp"][1:4] - mom["qout"][1:4]
    )


def onshell_check(pIn, pOut, qOut, th, ph, phOut, m):
    """Return mass-shell values for ``k``, ``kp``, ``qout``, ``p``, and ``pp``."""
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return [
        mdot(mom["k"], mom["k"]),
        mdot(mom["kp"], mom["kp"]),
        mdot(mom["qout"], mom["qout"]),
        mdot(mom["p"], mom["p"]),
        mdot(mom["pp"], mom["pp"]),
    ]
