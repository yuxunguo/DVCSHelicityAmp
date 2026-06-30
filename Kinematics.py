import numpy as np

from Algebra import _validate_nonnegative_scalar, _validate_positive_scalar, mdot


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
