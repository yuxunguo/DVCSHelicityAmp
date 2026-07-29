"""Lightweight process worker for gradient configuration contours.

This module intentionally avoids importing SciPy. On Windows each spawned
worker imports its callable module independently, so keeping this dependency
chain light permits all configured CPU workers without exhausting the paging
file while loading SciPy DLLs.
"""

import numpy as np

import PhaseSpaceScan as phase_scan


INVALID_OBJECTIVE = 1.0e3
PERIODIC_UNIT_COORDINATES = (3, 4, 5, 6)


def _objective_key(lepton_name, objective_name):
    """Return the coherent mixing-angle objective column."""
    return (
        f"lepton_{lepton_name}_theta_mix_proton_theta_p_mix_"
        f"{objective_name}"
    )


def _normalized_to_point(unit_point):
    """Map one normalized seven-vector to physical phase-space coordinates."""
    unit_point = np.asarray(unit_point, dtype=float)
    sqrt_s = (
        phase_scan.SQRT_S_RANGE[0]
        + unit_point[0]
        * (phase_scan.SQRT_S_RANGE[1] - phase_scan.SQRT_S_RANGE[0])
    )
    s = sqrt_s**2
    qout_fraction = (
        phase_scan.QOUT_FRACTION_RANGE[0]
        + unit_point[2]
        * (
            phase_scan.QOUT_FRACTION_RANGE[1]
            - phase_scan.QOUT_FRACTION_RANGE[0]
        )
    )
    return np.asarray(
        (
            s,
            phase_scan.THETA_OUT_RANGE[0]
            + unit_point[1]
            * (
                phase_scan.THETA_OUT_RANGE[1]
                - phase_scan.THETA_OUT_RANGE[0]
            ),
            qout_fraction * phase_scan._qout_max(s),
            unit_point[3] * 2.0 * np.pi,
            unit_point[4] * 2.0 * np.pi,
            unit_point[5] * np.pi,
            unit_point[6] * np.pi,
        ),
        dtype=float,
    )


def _objective_evaluation(
    unit_point,
    lepton_name,
    lepton_mass,
    evaluation_id,
    objective_name,
):
    """Evaluate one objective without importing the gradient optimizer."""
    result = phase_scan._evaluate_sample(
        _normalized_to_point(unit_point),
        sample_id=evaluation_id,
        stage="gradient_contour",
        lepton_name=lepton_name,
        lepton_mass=lepton_mass,
    )
    if result is None or result[1] is None:
        return INVALID_OBJECTIVE
    value = float(
        result[1].get(_objective_key(lepton_name, objective_name), np.nan)
    )
    return value if np.isfinite(value) else INVALID_OBJECTIVE


def _move_unit_point(point, displacement):
    """Apply bounded nonperiodic and wrapped periodic displacement."""
    neighbor = np.asarray(point, dtype=float).copy()
    neighbor += np.asarray(displacement, dtype=float)
    neighbor[:3] = np.clip(neighbor[:3], 0.0, 1.0)
    neighbor[3:] %= 1.0
    return neighbor


def _maximum_contour_radius(center, direction):
    """Return the non-repeating radial extent along one direction."""
    limits = []
    for index, component in enumerate(direction):
        if abs(component) <= 1.0e-15:
            continue
        if index in PERIODIC_UNIT_COORDINATES:
            limits.append(0.5 / abs(component))
        elif component > 0.0:
            limits.append((1.0 - center[index]) / component)
        else:
            limits.append(center[index] / -component)
    return max(0.0, float(min(limits, default=0.0)))


def _trace_contour(
    evaluate,
    center,
    base_value,
    directions,
    contour_delta,
    initial_radius,
    bisection_iterations,
):
    """Trace one direction chunk of a local seven-dimensional contour."""
    target = float(base_value) + contour_delta
    boundary_points = []
    for direction in directions:
        maximum_radius = _maximum_contour_radius(center, direction)
        if maximum_radius <= 1.0e-14:
            continue
        low_radius = 0.0
        high_radius = min(initial_radius, maximum_radius)
        high_point = _move_unit_point(center, high_radius * direction)
        high_value = float(evaluate(high_point))
        while high_value < target and high_radius < maximum_radius:
            low_radius = high_radius
            high_radius = min(2.0 * high_radius, maximum_radius)
            high_point = _move_unit_point(center, high_radius * direction)
            high_value = float(evaluate(high_point))
        if high_value < target:
            continue
        for _iteration in range(bisection_iterations):
            middle_radius = 0.5 * (low_radius + high_radius)
            middle_point = _move_unit_point(
                center,
                middle_radius * direction,
            )
            if float(evaluate(middle_point)) < target:
                low_radius = middle_radius
            else:
                high_radius = middle_radius
        boundary_points.append(
            _move_unit_point(center, high_radius * direction)
        )
    return np.asarray(boundary_points, dtype=float).reshape((-1, 7))


def configuration_contour_task(task):
    """Compute one selected minimum's direction chunk in a light worker."""
    (
        row_index,
        chunk_index,
        row,
        lepton_name,
        lepton_mass,
        objective_name,
        directions,
        contour_delta,
        initial_radius,
        bisection_iterations,
    ) = task
    phase_scan._configure_lepton(lepton_name)
    center = np.asarray(
        [float(row[f"final_u{index}"]) for index in range(7)],
        dtype=float,
    )
    base_value = float(row[_objective_key(lepton_name, objective_name)])
    evaluation_id = row_index * 10_000_000 + chunk_index * 100_000
    evaluation_count = 0

    def evaluate(unit_point):
        nonlocal evaluation_count
        value = _objective_evaluation(
            unit_point,
            lepton_name,
            lepton_mass,
            evaluation_id + evaluation_count,
            objective_name,
        )
        evaluation_count += 1
        return value

    boundary_points = _trace_contour(
        evaluate,
        center,
        base_value,
        directions,
        contour_delta,
        initial_radius,
        bisection_iterations,
    )
    return row_index, chunk_index, boundary_points
