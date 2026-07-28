"""Initial proton--lepton CM-frame kinematic builders and checks.

The frame fixes the incoming proton along ``+z`` and the incoming lepton
along ``-z``.  The independent variables are

``(s, qOut, theta_out, phi_p_out, phi_gamma_out)``.

The final proton and real photon share the production-plane polar angle
``theta_out`` and have separate azimuths.  The outgoing-lepton momentum
follows from three-momentum conservation, while the final-proton magnitude
``pOut`` is solved from energy conservation.

All four-vectors are contravariant arrays in ``[E, px, py, pz]`` order. The
external lepton and photon are placed on shell by construction. Both lepton
and proton masses are explicit inputs.
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
# Initial proton--lepton CM kinematics
#
# p  = (sqrt(pIn^2+m^2), 0, 0, +pIn)
# k  = (sqrt(pIn^2+me^2), 0, 0, -pIn)
# p' = (sqrt(pOut^2+m^2), pOut n(theta_out, phi_p_out))
# q' = qOut (1, n(theta_out, phi_gamma_out))
# k' = (sqrt(|-p' - q'|^2+me^2), -p' - q')
# ============================================================

def direction_from_angles(theta, phi):
    """Return the unit vector with standard polar and azimuthal angles."""
    theta = _validate_scalar(theta, "theta")
    phi = _validate_scalar(phi, "phi")
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def spatial_angles(four_vector):
    """Return standard ``(theta, phi)`` angles for a nonzero four-vector."""
    spatial = np.asarray(four_vector, dtype=float)[1:4]
    magnitude = float(np.linalg.norm(spatial))
    if magnitude <= DEFAULT_TOL:
        return np.nan, np.nan
    theta = float(np.arccos(np.clip(spatial[2] / magnitude, -1.0, 1.0)))
    phi = _normalize_angle(np.arctan2(spatial[1], spatial[0]))
    return theta, phi


def k_cm(pIn, electron_mass):
    """Return the incoming lepton four-momentum in the initial CM frame.

    The lepton points along ``-z``.
    """
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    return np.array([
        np.sqrt(pIn**2 + electron_mass**2),
        0.0,
        0.0,
        -pIn,
    ])


def p_cm(pIn, m):
    """Return the incoming proton four-momentum along ``+z``."""
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    m = _validate_positive_scalar(m, "m")
    return np.array([
        np.sqrt(pIn**2 + m**2),
        0.0,
        0.0,
        pIn,
    ])


def pp_cm(pOut, theta_p_out, phi_p_out, m):
    """Return the outgoing proton four-momentum in the initial CM frame.

    ``theta_p_out`` and ``phi_p_out`` specify its direction.
    """
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    m = _validate_positive_scalar(m, "m")
    direction = direction_from_angles(theta_p_out, phi_p_out)
    return np.concatenate(([np.sqrt(pOut**2 + m**2)], pOut * direction))


def qout_cm(qOut, theta_gamma_out, phi_gamma_out):
    """Return the outgoing real-photon four-momentum in the initial CM frame."""
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    direction = direction_from_angles(theta_gamma_out, phi_gamma_out)
    return np.concatenate(([qOut], qOut * direction))


def kp_cm(
    pOut,
    qOut,
    theta_p_out,
    phi_p_out,
    theta_gamma_out,
    phi_gamma_out,
    electron_mass,
):
    """Return the outgoing electron four-momentum from momentum conservation.

    The spatial momentum is fixed by ``k + p = kp + pp + qout`` in the CM
    frame, and its on-shell energy includes the explicit lepton mass.
    """
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    proton3 = pOut * direction_from_angles(theta_p_out, phi_p_out)
    photon3 = qOut * direction_from_angles(theta_gamma_out, phi_gamma_out)
    kp3 = -proton3 - photon3
    return np.concatenate([[np.sqrt(np.dot(kp3, kp3) + electron_mass**2)], kp3])


def momenta_cm(
    pIn,
    pOut,
    qOut,
    theta_p_out,
    phi_p_out,
    theta_gamma_out,
    phi_gamma_out,
    m,
    electron_mass,
):
    """Return all initial-CM external momenta as a dictionary.

    The returned keys are ``k`` (incoming electron), ``p`` (incoming proton),
    ``kp`` (outgoing electron), ``pp`` (outgoing proton), and ``qout``
    (outgoing real photon).
    """
    return {
        "k": k_cm(pIn, electron_mass),
        "p": p_cm(pIn, m),
        "kp": kp_cm(
            pOut,
            qOut,
            theta_p_out,
            phi_p_out,
            theta_gamma_out,
            phi_gamma_out,
            electron_mass=electron_mass,
        ),
        "pp": pp_cm(pOut, theta_p_out, phi_p_out, m),
        "qout": qout_cm(qOut, theta_gamma_out, phi_gamma_out),
    }


def _normalize_angle(angle):
    """Normalize an angle to the interval ``[0, 2*pi)``."""
    return float(angle % (2.0 * np.pi))


def p_in_from_s(s, m, electron_mass):
    """Return the incoming COM momentum magnitude from invariant ``s``."""
    s = _validate_positive_scalar(s, "s")
    m = _validate_positive_scalar(m, "m")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    if np.sqrt(s) <= m + electron_mass:
        raise ValueError("sqrt(s) must exceed the electron-plus-proton mass.")
    kallen = (s - (m + electron_mass)**2) * (s - (m - electron_mass)**2)
    return np.sqrt(max(0.0, kallen)) / (2.0 * np.sqrt(s))


def _cm_energy_residual_for_pout(
    pOut, sqrt_s, qOut, opening_cosine, m, electron_mass
):
    """Return final energy minus ``sqrt_s`` for a CM-frame ``pOut`` trial."""
    proton_energy = np.sqrt(pOut**2 + m**2)
    electron_energy = np.sqrt(
        pOut**2 + qOut**2 + 2.0 * pOut * qOut * opening_cosine
        + electron_mass**2
    )
    return proton_energy + electron_energy + qOut - sqrt_s


def solve_pout_from_cm_independent(
    s,
    qOut,
    theta_out,
    phi_p_out,
    phi_gamma_out,
    m,
    electron_mass,
    tol=1.0e-12,
):
    """Solve the outgoing proton magnitude from energy conservation.

    The angular dependence enters through the opening angle between the final
    proton and photon.
    """
    s = _validate_positive_scalar(s, "s")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    proton_direction = direction_from_angles(theta_out, phi_p_out)
    photon_direction = direction_from_angles(theta_out, phi_gamma_out)
    opening_cosine = float(np.dot(proton_direction, photon_direction))
    m = _validate_positive_scalar(m, "m")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    sqrt_s = np.sqrt(s)

    low = 0.0
    low_value = _cm_energy_residual_for_pout(
        low, sqrt_s, qOut, opening_cosine, m, electron_mass
    )
    if abs(low_value) <= tol:
        return 0.0

    # For photon emission with a negative opening cosine, the energy residual can
    # initially decrease,
    # producing two positive roots even when its value at pOut=0 is positive.
    # Locate the unique convex minimum and choose the first physical root.
    if low_value > tol:
        def derivative(momentum):
            lepton_energy = np.sqrt(
                momentum**2
                + qOut**2
                + 2.0 * momentum * qOut * opening_cosine
                + electron_mass**2
            )
            return (
                momentum / np.sqrt(momentum**2 + m**2)
                + (momentum + qOut * opening_cosine) / lepton_energy
            )

        if derivative(0.0) >= 0.0:
            raise ValueError(
                "No physical pOut: photon energy is too large for these "
                "initial-CM angles."
            )
        minimum_high = max(1.0, sqrt_s)
        while derivative(minimum_high) <= 0.0:
            minimum_high *= 2.0
        minimum_low = 0.0
        for _iteration in range(100):
            midpoint = 0.5 * (minimum_low + minimum_high)
            if derivative(midpoint) > 0.0:
                minimum_high = midpoint
            else:
                minimum_low = midpoint
        minimum = 0.5 * (minimum_low + minimum_high)
        minimum_value = _cm_energy_residual_for_pout(
            minimum, sqrt_s, qOut, opening_cosine, m, electron_mass
        )
        if minimum_value > tol:
            raise ValueError(
                "No physical pOut: photon energy is too large for these "
                "initial-CM angles."
            )
        if abs(minimum_value) <= tol:
            return minimum
        low, low_value = 0.0, low_value
        high, high_value = minimum, minimum_value
        for _iteration in range(100):
            mid = 0.5 * (low + high)
            mid_value = _cm_energy_residual_for_pout(
                mid, sqrt_s, qOut, opening_cosine, m, electron_mass
            )
            if abs(mid_value) <= tol:
                return mid
            if mid_value > 0.0:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    high = max(1.0, sqrt_s)
    high_value = _cm_energy_residual_for_pout(
        high, sqrt_s, qOut, opening_cosine, m, electron_mass
    )
    for _iteration in range(80):
        if high_value > 0.0:
            break
        high *= 2.0
        high_value = _cm_energy_residual_for_pout(
            high, sqrt_s, qOut, opening_cosine, m, electron_mass
        )
    else:
        raise ValueError("Could not bracket a physical pOut solution.")

    for _iteration in range(100):
        mid = 0.5 * (low + high)
        mid_value = _cm_energy_residual_for_pout(
            mid, sqrt_s, qOut, opening_cosine, m, electron_mass
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


def kinematics_cm_from_independent(
    s,
    qOut,
    theta_out,
    phi_p_out,
    phi_gamma_out,
    m,
    electron_mass,
    label=None,
):
    """Build initial proton--lepton CM kinematics from independent variables.

    The incoming proton/lepton axes are fixed at ``+z``/``-z``.  Both final
    proton and photon share ``theta_out`` and have independent azimuths;
    ``pOut`` is solved.
    """
    s = _validate_positive_scalar(s, "s")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    theta_out = _validate_scalar(theta_out, "theta_out")
    if not 0.0 <= theta_out <= np.pi:
        raise ValueError("theta_out must lie in [0, pi].")
    phi_p_out = _normalize_angle(_validate_scalar(phi_p_out, "phi_p_out"))
    phi_gamma_out = _normalize_angle(
        _validate_scalar(phi_gamma_out, "phi_gamma_out")
    )
    m = _validate_positive_scalar(m, "m")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")

    pIn = p_in_from_s(s, m, electron_mass=electron_mass)
    pOut = solve_pout_from_cm_independent(
        s,
        qOut,
        theta_out,
        phi_p_out,
        phi_gamma_out,
        m,
        electron_mass,
    )
    mom = momenta_cm(
        pIn,
        pOut,
        qOut,
        theta_out,
        phi_p_out,
        theta_out,
        phi_gamma_out,
        m,
        electron_mass,
    )
    mom["q"] = mom["k"] - mom["kp"]
    theta_lepton_out, phi_lepton_out = spatial_angles(mom["kp"])
    opening_cosine = float(np.dot(
        direction_from_angles(theta_out, phi_p_out),
        direction_from_angles(theta_out, phi_gamma_out),
    ))
    derived = invariant_q2_xb_t(mom, m)
    return {
        "frame": "initial_proton_lepton_com",
        "label": label,
        "m": m,
        "electron_mass": electron_mass,
        "momenta": mom,
        "pIn": pIn,
        "pOut": pOut,
        "s": s,
        "theta_p_in": 0.0,
        "phi_p_in": 0.0,
        "theta_lepton_in": float(np.pi),
        "phi_lepton_in": 0.0,
        "qOut": qOut,
        "theta_out": float(theta_out),
        "theta_p_out": float(theta_out),
        "phi_p_out": phi_p_out,
        "theta_gamma_out": float(theta_out),
        "phi_gamma_out": phi_gamma_out,
        "theta_lepton_out": theta_lepton_out,
        "phi_lepton_out": phi_lepton_out,
        "cos_p_gamma_out": opening_cosine,
        **{key: value for key, value in derived.items() if key not in {"q", "s"}},
        "energy_residual": abs(
            _cm_energy_residual_for_pout(
                pOut, np.sqrt(s), qOut, opening_cosine, m, electron_mass
            )
        ),
    }
