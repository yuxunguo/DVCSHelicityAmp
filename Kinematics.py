"""User-frame kinematic builders and validation checks.

The repository uses a direct COM-frame parameterization specified by ``pIn``,
``pOut``, ``qOut``, ``theta_in``, ``phi_in``, and ``phiOut``. Scan scripts use
the independent user-frame set ``(s, theta_in, phi_in, qOut, phiOut)`` and
solve ``pOut`` from energy conservation.

All four-vectors are contravariant arrays in ``[E, px, py, pz]`` order. The
external electron and photon are placed on shell by construction. The electron
is massless by default and can be made massive with ``electron_mass``; the
proton mass is supplied explicitly as ``m``.
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
# q' = qOut (1, cos phiOut, sin phiOut, 0)
# k' = (sqrt(pOut^2+qOut^2+2 pOut qOut sin phiOut),
#       -qOut cos phiOut, -pOut - qOut sin phiOut, 0)
#
# p = (sqrt(pIn^2+m^2),
#      pIn sin theta_in cos phi_in, pIn sin theta_in sin phi_in, pIn cos theta_in)
# k = pIn (1, -sin theta_in cos phi_in, -sin theta_in sin phi_in, -cos theta_in)
# ============================================================

def k_user(pIn, theta_in, phi_in, electron_mass=0.0):
    """Return the incoming electron four-momentum in the user frame.

    ``pIn`` is the common incoming three-momentum magnitude in the COM frame.
    The electron points opposite to the incoming proton direction defined by
    polar angle ``theta_in`` and azimuth ``phi_in``. Its mass defaults to zero.
    """
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    return np.array([
        np.sqrt(pIn**2 + electron_mass**2),
        -pIn * np.sin(theta_in) * np.cos(phi_in),
        -pIn * np.sin(theta_in) * np.sin(phi_in),
        -pIn * np.cos(theta_in),
    ])


def p_user(pIn, theta_in, phi_in, m):
    """Return the incoming proton four-momentum in the user frame."""
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    m = _validate_positive_scalar(m, "m")
    return np.array([
        np.sqrt(pIn**2 + m**2),
        pIn * np.sin(theta_in) * np.cos(phi_in),
        pIn * np.sin(theta_in) * np.sin(phi_in),
        pIn * np.cos(theta_in),
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


def qout_user(qOut, phiOut):
    """Return the outgoing real-photon four-momentum in the user frame."""
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    return qOut * np.array([1.0, np.cos(phiOut), np.sin(phiOut), 0.0])


def kp_user(pOut, qOut, phiOut, electron_mass=0.0):
    """Return the outgoing electron four-momentum from momentum conservation.

    The spatial momentum is fixed by ``k + p = kp + pp + qout`` in the user
    frame, and its on-shell energy includes the optional electron mass.
    """
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    kp3 = np.array([
        -qOut * np.cos(phiOut),
        -pOut - qOut * np.sin(phiOut),
        0.0,
    ])
    return np.concatenate([[np.sqrt(np.dot(kp3, kp3) + electron_mass**2)], kp3])


def momenta_user(
    pIn, pOut, qOut, theta_in, phi_in, phiOut, m, electron_mass=0.0
):
    """Return all user-frame external momenta as a dictionary.

    The returned keys are ``k`` (incoming electron), ``p`` (incoming proton),
    ``kp`` (outgoing electron), ``pp`` (outgoing proton), and ``qout``
    (outgoing real photon).
    """
    return {
        "k": k_user(pIn, theta_in, phi_in, electron_mass=electron_mass),
        "p": p_user(pIn, theta_in, phi_in, m),
        "kp": kp_user(pOut, qOut, phiOut, electron_mass=electron_mass),
        "pp": pp_user(pOut, m),
        "qout": qout_user(qOut, phiOut),
    }


def _normalize_angle(angle):
    """Normalize an angle to the interval ``[0, 2*pi)``."""
    return float(angle % (2.0 * np.pi))


def p_in_from_s(s, m, electron_mass=0.0):
    """Return the incoming COM momentum magnitude from invariant ``s``."""
    s = _validate_positive_scalar(s, "s")
    m = _validate_positive_scalar(m, "m")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    if np.sqrt(s) <= m + electron_mass:
        raise ValueError("sqrt(s) must exceed the electron-plus-proton mass.")
    kallen = (s - (m + electron_mass)**2) * (s - (m - electron_mass)**2)
    return np.sqrt(max(0.0, kallen)) / (2.0 * np.sqrt(s))


def _user_energy_residual_for_pout(
    pOut, sqrt_s, qOut, phiOut, m, electron_mass=0.0
):
    """Return final energy minus ``sqrt_s`` for a user-frame ``pOut`` trial."""
    proton_energy = np.sqrt(pOut**2 + m**2)
    electron_energy = np.sqrt(
        pOut**2 + qOut**2 + 2.0 * pOut * qOut * np.sin(phiOut)
        + electron_mass**2
    )
    return proton_energy + electron_energy + qOut - sqrt_s


def solve_pout_from_user_independent(
    s, qOut, phiOut, m, tol=1.0e-12, electron_mass=0.0
):
    """Solve outgoing proton momentum from ``s``, photon energy and ``phiOut``.

    The direct user frame conserves three-momentum by construction.  Energy
    conservation then fixes ``pOut`` for the independent set
    ``(s, theta_in, phi_in, qOut, phiOut)``.
    """
    s = _validate_positive_scalar(s, "s")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    phiOut = _validate_scalar(phiOut, "phiOut")
    m = _validate_positive_scalar(m, "m")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    sqrt_s = np.sqrt(s)

    low = 0.0
    low_value = _user_energy_residual_for_pout(
        low, sqrt_s, qOut, phiOut, m, electron_mass
    )
    if low_value > tol:
        raise ValueError(
            "No physical pOut: photon energy is too large for this s and phiOut."
        )
    if abs(low_value) <= tol:
        return 0.0

    high = max(1.0, sqrt_s)
    high_value = _user_energy_residual_for_pout(
        high, sqrt_s, qOut, phiOut, m, electron_mass
    )
    for _iteration in range(80):
        if high_value > 0.0:
            break
        high *= 2.0
        high_value = _user_energy_residual_for_pout(
            high, sqrt_s, qOut, phiOut, m, electron_mass
        )
    else:
        raise ValueError("Could not bracket a physical pOut solution.")

    for _iteration in range(100):
        mid = 0.5 * (low + high)
        mid_value = _user_energy_residual_for_pout(
            mid, sqrt_s, qOut, phiOut, m, electron_mass
        )
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


def kinematics_user_from_independent(
    s, theta_in, phi_in, qOut, phiOut, m, label=None, electron_mass=0.0
):
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
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")

    pIn = p_in_from_s(s, m, electron_mass=electron_mass)
    pOut = solve_pout_from_user_independent(
        s, qOut, phiOut, m, electron_mass=electron_mass
    )
    mom = momenta_user(
        pIn, pOut, qOut, theta_in, phi_in, phiOut, m,
        electron_mass=electron_mass,
    )
    mom["q"] = mom["k"] - mom["kp"]
    derived = invariant_q2_xb_t(mom, m)
    return {
        "frame": "user_kinematics_com",
        "label": label,
        "m": m,
        "electron_mass": electron_mass,
        "momenta": mom,
        "pIn": pIn,
        "pOut": pOut,
        "s": s,
        "theta_in": _normalize_angle(theta_in),
        "phi_in": phi_in,
        "qOut": qOut,
        "phiOut": phiOut,
        **{key: value for key, value in derived.items() if key not in {"q", "s"}},
        "energy_residual": abs(
            _user_energy_residual_for_pout(
                pOut, np.sqrt(s), qOut, phiOut, m, electron_mass
            )
        ),
    }


def energy_balance(pIn, pOut, qOut, theta_in, phi_in, phiOut, m):
    """Return E_initial - E_final; energy conservation means this is zero."""
    mom = momenta_user(pIn, pOut, qOut, theta_in, phi_in, phiOut, m)
    return mom["k"][0] + mom["p"][0] - mom["kp"][0] - mom["pp"][0] - mom["qout"][0]


def momentum_conservation_check(pIn, pOut, qOut, theta_in, phi_in, phiOut, m):
    """Return the residual three-momentum balance vector."""
    mom = momenta_user(pIn, pOut, qOut, theta_in, phi_in, phiOut, m)
    return (
        mom["k"][1:4] + mom["p"][1:4]
        - mom["kp"][1:4] - mom["pp"][1:4] - mom["qout"][1:4]
    )


def onshell_check(pIn, pOut, qOut, theta_in, phi_in, phiOut, m):
    """Return mass-shell values for ``k``, ``kp``, ``qout``, ``p``, and ``pp``."""
    mom = momenta_user(pIn, pOut, qOut, theta_in, phi_in, phiOut, m)
    return [
        mdot(mom["k"], mom["k"]),
        mdot(mom["kp"], mom["kp"]),
        mdot(mom["qout"], mom["qout"]),
        mdot(mom["p"], mom["p"]),
        mdot(mom["pp"], mom["pp"]),
    ]
