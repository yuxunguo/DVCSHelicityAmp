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
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    return pIn * np.array([
        1.0,
        -np.sin(th) * np.cos(ph),
        -np.sin(th) * np.sin(ph),
        -np.cos(th),
    ])


def p_user(pIn, th, ph, m):
    pIn = _validate_nonnegative_scalar(pIn, "pIn")
    m = _validate_positive_scalar(m, "m")
    return np.array([
        np.sqrt(pIn**2 + m**2),
        pIn * np.sin(th) * np.cos(ph),
        pIn * np.sin(th) * np.sin(ph),
        pIn * np.cos(th),
    ])


def pp_user(pOut, m):
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    m = _validate_positive_scalar(m, "m")
    return np.array([
        np.sqrt(pOut**2 + m**2),
        0.0,
        pOut,
        0.0,
    ])


def qout_user(qOut, phOut):
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    return qOut * np.array([1.0, np.cos(phOut), np.sin(phOut), 0.0])


def kp_user(pOut, qOut, phOut):
    pOut = _validate_nonnegative_scalar(pOut, "pOut")
    qOut = _validate_nonnegative_scalar(qOut, "qOut")
    kp3 = np.array([
        -qOut * np.cos(phOut),
        -pOut - qOut * np.sin(phOut),
        0.0,
    ])
    return np.concatenate([[np.linalg.norm(kp3)], kp3])


def momenta_user(pIn, pOut, qOut, th, ph, phOut, m):
    return {
        "k": k_user(pIn, th, ph),
        "p": p_user(pIn, th, ph, m),
        "kp": kp_user(pOut, qOut, phOut),
        "pp": pp_user(pOut, m),
        "qout": qout_user(qOut, phOut),
    }


# ============================================================
# Scalar exclusive kinematics
#
# Eb is a beam-energy scalar that fixes s = m^2 + 2 m Eb.
# The public wrapper below returns momenta in the initial e+p COM frame.
# Independent variables: Eb, Q2, xB, t, phi.
# y = Q2 / (2 m xB Eb) is derived, not independent.
# ============================================================

def _clip_physical_cosine(value, name, tol=1e-10):
    if value < -1.0 - tol or value > 1.0 + tol:
        raise ValueError(
            f"{name}={value:.16g} is outside the physical range [-1, 1]."
        )
    return float(np.clip(value, -1.0, 1.0))


def _kinematics_target_rest_from_beam_energy(Eb, Q2, xB, t, phi, m):
    """
    Build a target-rest event used as a scalar construction frame.

    This helper is intentionally private; public scalar kinematics are exposed
    in the initial e+p COM frame by kinematics_cm_from_beam_energy.
    """
    Eb = _validate_positive_scalar(Eb, "Eb")
    Q2 = _validate_positive_scalar(Q2, "Q2")
    xB = _validate_positive_scalar(xB, "xB")
    t = _validate_scalar(t, "t")
    phi = _validate_scalar(phi, "phi")
    m = _validate_positive_scalar(m, "m")

    if t >= -DEFAULT_TOL:
        raise ValueError("t must be negative for spacelike momentum transfer.")

    nu = Q2 / (2.0 * m * xB)
    y = nu / Eb
    kp_energy = Eb - nu
    if kp_energy <= DEFAULT_TOL:
        raise ValueError("Final electron energy is not positive for this kinematics.")

    cos_lepton = 1.0 - Q2 / (2.0 * Eb * kp_energy)
    cos_lepton = _clip_physical_cosine(cos_lepton, "cos(theta_e)")
    sin_lepton = np.sqrt(max(0.0, 1.0 - cos_lepton**2))

    k = np.array([Eb, 0.0, 0.0, Eb])
    p = np.array([m, 0.0, 0.0, 0.0])
    kp = np.array([
        kp_energy,
        kp_energy * sin_lepton,
        0.0,
        kp_energy * cos_lepton,
    ])
    q = k - kp
    q_vec = q[1:4]
    q_abs = np.linalg.norm(q_vec)
    if q_abs <= DEFAULT_TOL:
        raise ValueError("Virtual photon three-momentum is zero.")

    pp_energy = m - t / (2.0 * m)
    pp_abs2 = pp_energy**2 - m**2
    if pp_abs2 < -DEFAULT_TOL:
        raise ValueError("Final proton momentum is not real for this t.")
    pp_abs = np.sqrt(max(0.0, pp_abs2))
    if pp_abs <= DEFAULT_TOL:
        raise ValueError("Final proton momentum is zero; phi is undefined.")

    target_q_dot = m**2 + m * nu - 0.5 * Q2
    cos_hadron = (
        pp_energy * (m + nu) - target_q_dot
    ) / (pp_abs * q_abs)
    cos_hadron = _clip_physical_cosine(cos_hadron, "cos(theta_pq)")
    sin_hadron = np.sqrt(max(0.0, 1.0 - cos_hadron**2))

    e_zq = q_vec / q_abs
    lepton_normal = np.cross(k[1:4], kp[1:4])
    normal_abs = np.linalg.norm(lepton_normal)
    if normal_abs <= DEFAULT_TOL:
        raise ValueError("Lepton plane is undefined for this kinematics.")
    e_y = lepton_normal / normal_abs
    e_x = np.cross(e_y, e_zq)
    e_x /= np.linalg.norm(e_x)

    pp_vec = pp_abs * (
        sin_hadron * np.cos(phi) * e_x
        + sin_hadron * np.sin(phi) * e_y
        + cos_hadron * e_zq
    )
    pp = np.concatenate([[pp_energy], pp_vec])
    qout = p + q - pp
    if qout[0] <= DEFAULT_TOL:
        raise ValueError("Final photon energy is not positive for this kinematics.")

    momenta = {
        "k": k,
        "p": p,
        "kp": kp,
        "pp": pp,
        "qout": qout,
        "q": q,
    }
    return {
        "Eb": Eb,
        "Q2": Q2,
        "xB": xB,
        "t": t,
        "phi": phi,
        "m": m,
        "nu": nu,
        "y": y,
        "momenta": momenta,
    }


def _boost_z(v, beta):
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    boosted = np.array(v, dtype=float, copy=True)
    boosted[0] = gamma * (v[0] - beta * v[3])
    boosted[3] = gamma * (v[3] - beta * v[0])
    return boosted


def kinematics_cm_from_beam_energy(Eb, Q2, xB, t, phi, m):
    """
    Return exclusive electroproduction momenta in the initial e+p COM frame.

    The scalar input set is (Eb, Q2, xB, t, phi). Eb fixes the invariant
    s = m^2 + 2 m Eb; it is not returned as a target-rest-frame momentum.
    """
    kin = _kinematics_target_rest_from_beam_energy(Eb, Q2, xB, t, phi, m)
    beta_cm = kin["Eb"] / (kin["Eb"] + kin["m"])
    momenta = {
        name: _boost_z(vec, beta_cm)
        for name, vec in kin["momenta"].items()
    }
    s = kin["m"]**2 + 2.0 * kin["m"] * kin["Eb"]

    return {
        **kin,
        "frame": "initial_ep_cm",
        "s": s,
        "sqrt_s": np.sqrt(s),
        "beta_target_rest_to_cm": beta_cm,
        "momenta": momenta,
    }


def momenta_cm_from_beam_energy(Eb, Q2, xB, t, phi, m):
    return kinematics_cm_from_beam_energy(Eb, Q2, xB, t, phi, m)["momenta"]


def energy_balance(pIn, pOut, qOut, th, ph, phOut, m):
    """Return E_initial - E_final; energy conservation means this is zero."""
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return mom["k"][0] + mom["p"][0] - mom["kp"][0] - mom["pp"][0] - mom["qout"][0]


def momentum_conservation_check(pIn, pOut, qOut, th, ph, phOut, m):
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return (
        mom["k"][1:4] + mom["p"][1:4]
        - mom["kp"][1:4] - mom["pp"][1:4] - mom["qout"][1:4]
    )


def onshell_check(pIn, pOut, qOut, th, ph, phOut, m):
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return [
        mdot(mom["k"], mom["k"]),
        mdot(mom["kp"], mom["kp"]),
        mdot(mom["qout"], mom["qout"]),
        mdot(mom["p"], mom["p"]),
        mdot(mom["pp"], mom["pp"]),
    ]
