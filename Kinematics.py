import numpy as np

from Algebra import (
    DEFAULT_TOL,
    _as_four_vector,
    _check_not_singular,
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


def _normalize_angle(angle):
    return float(angle % (2.0 * np.pi))


def _mom_spatial(mom, name):
    return _as_four_vector(mom[name], name, dtype=float)[1:4]


def _rotation_to_user_frame(mom, target_phi_out=None):
    pp_vec = _mom_spatial(mom, "pp")
    qout_vec = _mom_spatial(mom, "qout")
    pp_abs = np.linalg.norm(pp_vec)
    qout_abs = np.linalg.norm(qout_vec)
    if pp_abs <= DEFAULT_TOL or qout_abs <= DEFAULT_TOL:
        raise ValueError("User-frame rotation needs nonzero pp and qout momenta.")

    ey = pp_vec / pp_abs
    qout_x_part = qout_vec - np.dot(qout_vec, ey) * ey
    qout_x_abs = np.linalg.norm(qout_x_part)
    if qout_x_abs <= DEFAULT_TOL:
        raise ValueError("User-frame rotation is undefined when qout is parallel to pp.")

    ex_sign = -1.0 if target_phi_out is not None and np.cos(target_phi_out) < 0.0 else 1.0
    ex = ex_sign * qout_x_part / qout_x_abs
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)
    return np.vstack([ex, ey, ez])


def _rotate_four_vector(v, rotation):
    rotated = _as_four_vector(v, "v", dtype=float).copy()
    rotated[1:4] = rotation @ rotated[1:4]
    return rotated


def _rotate_momenta(mom, rotation):
    return {name: _rotate_four_vector(vector, rotation) for name, vector in mom.items()}


def _pp_qout_spatial_cosine(mom):
    pp_vec = _mom_spatial(mom, "pp")
    qout_vec = _mom_spatial(mom, "qout")
    den = np.linalg.norm(pp_vec) * np.linalg.norm(qout_vec)
    _check_not_singular(den, "pp-qout spatial-angle denominator")
    return float(np.dot(pp_vec, qout_vec) / den)


def _solve_phi_hadron_for_phi_out(Eb, Q2, xB, t, target_phi_out, m, label=None):
    target_sin = np.sin(target_phi_out)

    def residual(phi_hadron):
        kin = kinematics_cm_from_beam_energy(Eb, Q2, xB, t, phi_hadron, m)
        return _pp_qout_spatial_cosine(kin["momenta"]) - target_sin

    samples = np.linspace(0.0, 2.0 * np.pi, 721)
    values = [residual(phi) for phi in samples]
    best_index = int(np.argmin(np.abs(values)))
    if abs(values[best_index]) <= 1e-11:
        return _normalize_angle(samples[best_index])

    brackets = []
    for left_index in range(len(samples) - 1):
        left_value = values[left_index]
        right_value = values[left_index + 1]
        if left_value == 0.0:
            return _normalize_angle(samples[left_index])
        if left_value * right_value < 0.0:
            brackets.append((samples[left_index], samples[left_index + 1]))

    if not brackets:
        case_name = label if label is not None else "unknown case"
        raise ValueError(
            "phi_out is outside the physical range for this "
            f"kinematics ({case_name}, target phi_out={target_phi_out:.8e} rad). "
            f"Closest residual is {values[best_index]:.3e}."
        )

    left, right = brackets[0]
    left_value = residual(left)
    for _ in range(80):
        mid = 0.5 * (left + right)
        mid_value = residual(mid)
        if abs(mid_value) <= 1e-13:
            return _normalize_angle(mid)
        if left_value * mid_value <= 0.0:
            right = mid
        else:
            left = mid
            left_value = mid_value
    return _normalize_angle(0.5 * (left + right))


def kinematics_user_from_scalar_inputs(
    Eb, Q2, xB, t, phi, m, azimuth_input="phi_hadron", label=None
):
    """
    Build scalar-input exclusive kinematics and return them in the user frame.

    The scalar input set is still (Eb, Q2, xB, t, phi).  The azimuth_input
    flag chooses whether phi is the hadron-plane azimuth used by the scalar
    COM builder, or the user-frame final-photon azimuth phOut.
    """
    if azimuth_input == "phi_hadron":
        phi_hadron = _normalize_angle(_validate_scalar(phi, "phi"))
        target_phi_out = None
    elif azimuth_input == "phi_out":
        target_phi_out = _normalize_angle(_validate_scalar(phi, "phi"))
        phi_hadron = _solve_phi_hadron_for_phi_out(
            Eb, Q2, xB, t, target_phi_out, m, label=label
        )
    else:
        raise ValueError("azimuth_input must be 'phi_hadron' or 'phi_out'.")

    kin = kinematics_cm_from_beam_energy(Eb, Q2, xB, t, phi_hadron, m)
    rotation = _rotation_to_user_frame(kin["momenta"], target_phi_out=target_phi_out)
    rotated = _rotate_momenta(kin["momenta"], rotation)

    p_vec = rotated["p"][1:4]
    pIn = np.linalg.norm(p_vec)
    _check_not_singular(pIn, "user backend pIn")
    pOut = np.linalg.norm(rotated["pp"][1:4])
    qOut = np.linalg.norm(rotated["qout"][1:4])
    th = np.arccos(np.clip(p_vec[2] / pIn, -1.0, 1.0))
    ph = np.arctan2(p_vec[1], p_vec[0])
    phOut = np.arctan2(rotated["qout"][2], rotated["qout"][1])

    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, kin["m"])
    mom["q"] = mom["k"] - mom["kp"]
    residual = max(
        np.linalg.norm(mom[name] - rotated[name])
        for name in ("k", "p", "kp", "pp", "qout", "q")
    )

    return {
        **kin,
        "frame": "user_kinematics_com",
        "momenta": mom,
        "azimuth_input": azimuth_input,
        "input_azimuth": _normalize_angle(phi),
        "phi_hadron": _normalize_angle(phi_hadron),
        "phi_out": _normalize_angle(phOut),
        "user_params": {
            "pIn": pIn,
            "pOut": pOut,
            "qOut": qOut,
            "th": _normalize_angle(th),
            "ph": _normalize_angle(ph),
            "phOut": _normalize_angle(phOut),
        },
        "user_rebuild_residual": residual,
    }


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
