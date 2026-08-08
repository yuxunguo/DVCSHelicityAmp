"""Cartesian final-momentum coordinates for the gradient DVCS workflow.

The normalized optimizer vector is

``(p'_x,p'_y,p'_z,q'_x,q'_y,q'_z,alpha_e,alpha_p)``.

Each final three-vector uses one common scale for all of its components, so
the Cartesian metric and covariance do not distort directions axis by axis.
Angular variables are derived only for output and plotting.
"""

import numpy as np

from AlignmentScan import _evaluate_cartesian_kinematic_sample
from config import NORMALIZE_TRACE, PROTON_MASS_GEV
from GradientObjective import objective_value
from Kinematics import (
    direction_from_angles,
    kinematics_cm_from_cartesian_final,
    kinematics_cm_from_independent,
)
import PhaseSpaceConfigScan as config_scan
import PhaseSpaceScan as phase_scan


SCAN_DIMENSION = 8
PERIODIC_UNIT_COORDINATES = (6, 7)
BOUNDED_UNIT_COORDINATES = (0, 1, 2, 3, 4, 5)
POLE_TOLERANCE = 1.0e-12
DOMAIN_TOLERANCE = 1.0e-10


def momentum_scales():
    """Return common proton and photon component scales for this species."""
    sqrt_s_max = float(phase_scan.SQRT_S_RANGE[1])
    photon_scale = float(phase_scan._qout_max(sqrt_s_max**2))
    # sqrt(s)_max is a conservative bound on every final proton component.
    return sqrt_s_max, photon_scale


def unit_to_cartesian(unit_point):
    """Map one normalized vector to final momenta and mixing angles."""
    unit_point = np.asarray(unit_point, dtype=float)
    if unit_point.shape != (SCAN_DIMENSION,):
        raise ValueError(f"unit_point must have shape ({SCAN_DIMENSION},).")
    proton_scale, photon_scale = momentum_scales()
    proton3 = (2.0 * unit_point[:3] - 1.0) * proton_scale
    photon3 = (2.0 * unit_point[3:6] - 1.0) * photon_scale
    return proton3, photon3, float(unit_point[6] * np.pi), float(
        unit_point[7] * np.pi
    )


def cartesian_to_unit(proton3, photon3, alpha_e, alpha_p):
    """Map physical Cartesian data to the normalized optimizer box."""
    proton_scale, photon_scale = momentum_scales()
    proton3 = np.asarray(proton3, dtype=float)
    photon3 = np.asarray(photon3, dtype=float)
    unit_point = np.concatenate((
        0.5 * (proton3 / proton_scale + 1.0),
        0.5 * (photon3 / photon_scale + 1.0),
        (float(alpha_e) / np.pi, float(alpha_p) / np.pi),
    ))
    return unit_point


def angular_physical_to_unit(point):
    """Convert one legacy angular physical point to Cartesian optimizer form."""
    (
        s,
        theta_p,
        theta_gamma,
        q_out,
        phi_p,
        phi_gamma,
        alpha_e,
        alpha_p,
    ) = map(float, point)
    kin = kinematics_cm_from_independent(
        s,
        q_out,
        theta_p,
        phi_p,
        theta_gamma,
        phi_gamma,
        PROTON_MASS_GEV,
        electron_mass=phase_scan.LEPTON_MASS_GEV,
    )
    return cartesian_to_unit(
        kin["momenta"]["pp"][1:],
        kin["momenta"]["qout"][1:],
        alpha_e,
        alpha_p,
    )


def angular_unit_to_cartesian_unit(angular_unit):
    """Map a valid legacy unit-box design point into Cartesian coordinates."""
    angular_unit = np.asarray(angular_unit, dtype=float)
    sqrt_s = phase_scan.SQRT_S_RANGE[0] + angular_unit[0] * (
        phase_scan.SQRT_S_RANGE[1] - phase_scan.SQRT_S_RANGE[0]
    )
    s = sqrt_s**2
    fraction = phase_scan.QOUT_FRACTION_RANGE[0] + angular_unit[3] * (
        phase_scan.QOUT_FRACTION_RANGE[1]
        - phase_scan.QOUT_FRACTION_RANGE[0]
    )
    angular_physical = np.asarray((
        s,
        phase_scan.THETA_P_OUT_RANGE[0] + angular_unit[1] * (
            phase_scan.THETA_P_OUT_RANGE[1]
            - phase_scan.THETA_P_OUT_RANGE[0]
        ),
        phase_scan.THETA_GAMMA_OUT_RANGE[0] + angular_unit[2] * (
            phase_scan.THETA_GAMMA_OUT_RANGE[1]
            - phase_scan.THETA_GAMMA_OUT_RANGE[0]
        ),
        fraction * phase_scan._qout_max(s),
        angular_unit[4] * 2.0 * np.pi,
        angular_unit[5] * 2.0 * np.pi,
        angular_unit[6] * np.pi,
        angular_unit[7] * np.pi,
    ))
    return angular_physical_to_unit(angular_physical)


def row_to_unit(row):
    """Recover a Cartesian normalized point from a saved result row."""
    if "p_out_x" not in row:
        proton3 = vector_from_angles(
            float(row["pOut"]),
            float(row["theta_p_out"]),
            float(row["phi_p_out"]),
        )
        photon3 = vector_from_angles(
            float(row["qOut"]),
            float(row["theta_gamma_out"]),
            float(row["phi_gamma_out"]),
        )
        return cartesian_to_unit(
            proton3,
            photon3,
            float(row["alpha_e"]),
            float(row["alpha_p"]),
        )
    return cartesian_to_unit(
        [float(row[f"p_out_{axis}"]) for axis in "xyz"],
        [float(row[f"q_out_{axis}"]) for axis in "xyz"],
        float(row["alpha_e"]),
        float(row["alpha_p"]),
    )


def physical_start_to_unit(start):
    """Convert a readable angular anchor dictionary into Cartesian form."""
    return angular_physical_to_unit((
        float(start["sqrt_s"]) ** 2,
        float(start["theta_p_out"]),
        float(start["theta_gamma_out"]),
        float(start["qOut"]),
        float(start["phi_p_out"]),
        float(start["phi_gamma_out"]),
        float(start["alpha_e"]),
        float(start["alpha_p"]),
    ))


def reconstructed_coordinates(unit_point, *, undefined_phi_nan=True):
    """Return Cartesian fields plus derived angular plotting coordinates."""
    proton3, photon3, alpha_e, alpha_p = unit_to_cartesian(unit_point)
    kin = kinematics_cm_from_cartesian_final(
        proton3,
        photon3,
        PROTON_MASS_GEV,
        electron_mass=phase_scan.LEPTON_MASS_GEV,
    )
    phi_p = float(kin["phi_p_out"])
    phi_gamma = float(kin["phi_gamma_out"])
    if undefined_phi_nan and not kin["phi_p_out_defined"]:
        phi_p = np.nan
    if undefined_phi_nan and not kin["phi_gamma_out_defined"]:
        phi_gamma = np.nan
    return {
        "coordinate_system": "cartesian_final_momenta",
        "sqrt_s": float(kin["sqrt_s"]),
        "theta_p_out": float(kin["theta_p_out"]),
        "theta_gamma_out": float(kin["theta_gamma_out"]),
        "qOut": float(kin["qOut"]),
        "phi_p_out": phi_p,
        "phi_gamma_out": phi_gamma,
        "phi_p_out_defined": bool(kin["phi_p_out_defined"]),
        "phi_gamma_out_defined": bool(kin["phi_gamma_out_defined"]),
        "alpha_e": alpha_e,
        "alpha_p": alpha_p,
        **{f"p_out_{axis}": float(proton3[index]) for index, axis in enumerate("xyz")},
        **{f"q_out_{axis}": float(photon3[index]) for index, axis in enumerate("xyz")},
        "cartesian_energy_residual": float(kin["energy_residual"]),
        "cartesian_momentum_residual": float(kin["momentum_residual"]),
    }


def is_physical_unit_point(unit_point):
    """Return whether a Cartesian point belongs to the configured scan domain."""
    unit_point = np.asarray(unit_point, dtype=float)
    if unit_point.shape != (SCAN_DIMENSION,) or not np.all(np.isfinite(unit_point)):
        return False
    if np.any(unit_point[:6] < 0.0) or np.any(unit_point[:6] > 1.0):
        return False
    try:
        coordinates = reconstructed_coordinates(unit_point, undefined_phi_nan=False)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return False
    sqrt_s = coordinates["sqrt_s"]
    if not (
        phase_scan.SQRT_S_RANGE[0] - DOMAIN_TOLERANCE
        <= sqrt_s
        <= phase_scan.SQRT_S_RANGE[1] + DOMAIN_TOLERANCE
    ):
        return False
    q_fraction = coordinates["qOut"] / phase_scan._qout_max(sqrt_s**2)
    return bool(
        phase_scan.QOUT_FRACTION_RANGE[0] - DOMAIN_TOLERANCE
        <= q_fraction
        <= phase_scan.QOUT_FRACTION_RANGE[1] + DOMAIN_TOLERANCE
    )


def evaluate_unit_point(
    unit_point,
    *,
    lepton_name,
    lepton_mass,
    evaluation_id,
    objective_name,
    stage="cartesian_gradient",
):
    """Evaluate one exact Cartesian point and return objective plus saved row."""
    if not is_physical_unit_point(unit_point):
        return np.inf, None
    proton3, photon3, alpha_e, alpha_p = unit_to_cartesian(unit_point)
    anchor = {
        "kinematic_point": f"{stage}_{evaluation_id:010d}",
        "s_regime": stage,
        "theta_p_gamma_regime": stage,
        "qOut_regime": stage,
    }
    settings = {
        "m": PROTON_MASS_GEV,
        "lepton_mass": lepton_mass,
        "lepton_name": lepton_name,
        "normalize_trace": NORMALIZE_TRACE,
        "angle_max_rad": np.deg2rad(10.0),
        "return_amplitudes": True,
        "skip_fixed_polarizations": True,
    }
    try:
        result = _evaluate_cartesian_kinematic_sample(
            (anchor, proton3, photon3, settings)
        )
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return np.inf, None
    if not result["ok"]:
        return np.inf, None

    result["row"].update(reconstructed_coordinates(unit_point))
    row, mixing_row = phase_scan._mixing_observables_from_kinematic_result(
        result,
        stage=stage,
        sample_id=evaluation_id,
        lepton_name=lepton_name,
        alpha_e=alpha_e,
        alpha_p=alpha_p,
    )
    mixing_row.update({
        key: value
        for key, value in reconstructed_coordinates(unit_point).items()
        if key not in mixing_row
    })
    value = objective_value(
        mixing_row,
        config_scan.mixing_prefix(lepton_name),
        objective_name,
        store=True,
    )
    if not np.isfinite(value):
        return np.inf, None
    return float(value), mixing_row


def vector_from_angles(magnitude, theta, phi):
    """Convenience helper used by round-trip tests."""
    return float(magnitude) * direction_from_angles(float(theta), float(phi))
