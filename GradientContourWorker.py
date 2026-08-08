"""Lightweight process worker for gradient configuration contours.

This module intentionally avoids importing SciPy. On Windows each spawned
worker imports its callable module independently, so keeping this dependency
chain light permits all configured CPU workers without exhausting the paging
file while loading SciPy DLLs.
"""

import numpy as np

import CartesianPhaseSpaceCoordinates as cartesian_coordinates
from GradientObjective import objective_value
import PhaseSpaceScan as phase_scan


INVALID_OBJECTIVE = 1.0e3
SCAN_DIMENSION = 8
PERIODIC_UNIT_COORDINATES = (4, 5, 6, 7)


def _normalized_to_point(unit_point):
    """Map one normalized eight-vector to physical phase-space coordinates."""
    unit_point = np.asarray(unit_point, dtype=float)
    sqrt_s = (
        phase_scan.SQRT_S_RANGE[0]
        + unit_point[0]
        * (phase_scan.SQRT_S_RANGE[1] - phase_scan.SQRT_S_RANGE[0])
    )
    s = sqrt_s**2
    qout_fraction = (
        phase_scan.QOUT_FRACTION_RANGE[0]
        + unit_point[3]
        * (
            phase_scan.QOUT_FRACTION_RANGE[1]
            - phase_scan.QOUT_FRACTION_RANGE[0]
        )
    )
    return np.asarray(
        (
            s,
            phase_scan.THETA_P_OUT_RANGE[0]
            + unit_point[1]
            * (
                phase_scan.THETA_P_OUT_RANGE[1]
                - phase_scan.THETA_P_OUT_RANGE[0]
            ),
            phase_scan.THETA_GAMMA_OUT_RANGE[0]
            + unit_point[2]
            * (
                phase_scan.THETA_GAMMA_OUT_RANGE[1]
                - phase_scan.THETA_GAMMA_OUT_RANGE[0]
            ),
            qout_fraction * phase_scan._qout_max(s),
            unit_point[4] * 2.0 * np.pi,
            unit_point[5] * 2.0 * np.pi,
            unit_point[6] * np.pi,
            unit_point[7] * np.pi,
        ),
        dtype=float,
    )


def _objective_evaluation(
    unit_point,
    lepton_name,
    lepton_mass,
    evaluation_id,
    objective_name,
    coordinate_system="angular",
):
    """Evaluate one objective without importing the gradient optimizer."""
    if coordinate_system == "cartesian":
        value, row = cartesian_coordinates.evaluate_unit_point(
            unit_point,
            lepton_name=lepton_name,
            lepton_mass=lepton_mass,
            evaluation_id=evaluation_id,
            objective_name=objective_name,
            stage="cartesian_gradient_contour",
        )
        return (
            value
            if row is not None and np.isfinite(value)
            else INVALID_OBJECTIVE
        )
    result = phase_scan._evaluate_sample(
        _normalized_to_point(unit_point),
        sample_id=evaluation_id,
        stage="gradient_contour",
        lepton_name=lepton_name,
        lepton_mass=lepton_mass,
    )
    if result is None or result[1] is None:
        return INVALID_OBJECTIVE
    value = objective_value(
        result[1],
        f"lepton_{lepton_name}_alpha_e_mix_proton_alpha_p_mix",
        objective_name,
    )
    return value if np.isfinite(value) else INVALID_OBJECTIVE


def _periodic_coordinates(coordinate_system):
    """Return periodic axes for the selected coordinate system."""
    return (6, 7) if coordinate_system == "cartesian" else (4, 5, 6, 7)


def _move_unit_point(point, displacement, coordinate_system="angular"):
    """Apply bounded nonperiodic and wrapped periodic displacement."""
    neighbor = np.asarray(point, dtype=float).copy()
    neighbor += np.asarray(displacement, dtype=float)
    periodic = _periodic_coordinates(coordinate_system)
    bounded = tuple(
        index for index in range(SCAN_DIMENSION) if index not in periodic
    )
    neighbor[list(bounded)] = np.clip(neighbor[list(bounded)], 0.0, 1.0)
    neighbor[list(periodic)] %= 1.0
    return neighbor


def _maximum_contour_radius(center, direction, coordinate_system="angular"):
    """Return the non-repeating radial extent along one direction."""
    limits = []
    for index, component in enumerate(direction):
        if abs(component) <= 1.0e-15:
            continue
        if index in _periodic_coordinates(coordinate_system):
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
    coordinate_system="angular",
    return_boundary_flags=False,
):
    """Trace one direction chunk of a local eight-dimensional contour."""
    target = float(base_value) + contour_delta
    boundary_points = []
    boundary_flags = []
    for direction in directions:
        maximum_radius = _maximum_contour_radius(
            center, direction, coordinate_system
        )
        if maximum_radius <= 1.0e-14:
            continue
        low_radius = 0.0
        high_radius = min(initial_radius, maximum_radius)
        high_point = _move_unit_point(
            center, high_radius * direction, coordinate_system
        )
        high_value = float(evaluate(high_point))
        while high_value < target and high_radius < maximum_radius:
            low_radius = high_radius
            high_radius = min(2.0 * high_radius, maximum_radius)
            high_point = _move_unit_point(
                center, high_radius * direction, coordinate_system
            )
            high_value = float(evaluate(high_point))
        if high_value < target:
            continue
        for _iteration in range(bisection_iterations):
            middle_radius = 0.5 * (low_radius + high_radius)
            middle_point = _move_unit_point(
                center,
                middle_radius * direction,
                coordinate_system,
            )
            if float(evaluate(middle_point)) < target:
                low_radius = middle_radius
            else:
                high_radius = middle_radius
        boundary_point = _move_unit_point(
            center, high_radius * direction, coordinate_system
        )
        domain_boundary_limited = (
            coordinate_system == "cartesian"
            and not cartesian_coordinates.is_physical_unit_point(boundary_point)
        )
        if domain_boundary_limited:
            # If the objective basin reaches the physical phase-space edge
            # before reaching the requested level, retain the last point on
            # the physical side.  Returning the invalid high bracket would
            # contaminate the Cartesian covariance with out-of-domain data.
            boundary_point = _move_unit_point(
                center, low_radius * direction, coordinate_system
            )
        boundary_points.append(boundary_point)
        boundary_flags.append(domain_boundary_limited)
    points = np.asarray(boundary_points, dtype=float).reshape(
        (-1, SCAN_DIMENSION)
    )
    if return_boundary_flags:
        return points, np.asarray(boundary_flags, dtype=bool)
    return points


def configuration_contour_task(task):
    """Compute one selected minimum's direction chunk in a light worker."""
    coordinate_system = task[11] if len(task) > 11 else "angular"
    (
        row_index,
        chunk_index,
        center,
        base_value,
        lepton_name,
        lepton_mass,
        objective_name,
        directions,
        contour_delta,
        initial_radius,
        bisection_iterations,
    ) = task[:11]
    phase_scan._configure_lepton(lepton_name)
    center = np.asarray(center, dtype=float)
    base_value = float(base_value)
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
            coordinate_system,
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
        coordinate_system,
    )
    return row_index, chunk_index, boundary_points


def configuration_complete_contour_task(task):
    """Trace replacement rays until a requested successful-point count is met."""
    coordinate_system = task[14] if len(task) > 14 else "angular"
    (
        row_index,
        center,
        base_value,
        lepton_name,
        lepton_mass,
        objective_name,
        initial_directions,
        contour_delta,
        initial_radius,
        bisection_iterations,
        target_success_count,
        maximum_attempt_count,
        replacement_seed,
        replacement_batch_size,
    ) = task[:14]
    phase_scan._configure_lepton(lepton_name)
    center = np.asarray(center, dtype=float)
    base_value = float(base_value)
    evaluation_id = row_index * 10_000_000
    evaluation_count = 0

    def evaluate(unit_point):
        nonlocal evaluation_count
        value = _objective_evaluation(
            unit_point,
            lepton_name,
            lepton_mass,
            evaluation_id + evaluation_count,
            objective_name,
            coordinate_system,
        )
        evaluation_count += 1
        return value

    initial_directions = np.asarray(initial_directions, dtype=float).reshape(
        (-1, SCAN_DIMENSION)
    )
    attempted_count = len(initial_directions)
    initial_points, initial_boundary_flags = _trace_contour(
        evaluate,
        center,
        base_value,
        initial_directions,
        contour_delta,
        initial_radius,
        bisection_iterations,
        coordinate_system,
        return_boundary_flags=True,
    )
    initial_success_count = len(initial_points)
    point_batches = [initial_points] if len(initial_points) else []
    boundary_flag_batches = (
        [initial_boundary_flags] if len(initial_boundary_flags) else []
    )
    successful_count = initial_success_count
    rng = np.random.default_rng(int(replacement_seed))
    while (
        successful_count < target_success_count
        and attempted_count < maximum_attempt_count
    ):
        remaining_attempts = maximum_attempt_count - attempted_count
        batch_count = min(
            int(replacement_batch_size),
            remaining_attempts,
            max(32, target_success_count - successful_count),
        )
        directions = rng.normal(size=(batch_count, SCAN_DIMENSION))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points, boundary_flags = _trace_contour(
            evaluate,
            center,
            base_value,
            directions,
            contour_delta,
            initial_radius,
            bisection_iterations,
            coordinate_system,
            return_boundary_flags=True,
        )
        attempted_count += batch_count
        if len(points):
            point_batches.append(points)
            boundary_flag_batches.append(boundary_flags)
            successful_count += len(points)

    all_points = (
        np.vstack(point_batches)
        if point_batches
        else np.empty((0, SCAN_DIMENSION), dtype=float)
    )
    retained_points = all_points[:target_success_count]
    all_boundary_flags = (
        np.concatenate(boundary_flag_batches)
        if boundary_flag_batches
        else np.empty(0, dtype=bool)
    )
    retained_boundary_count = int(
        np.count_nonzero(all_boundary_flags[:target_success_count])
    )
    return (
        row_index,
        retained_points,
        attempted_count,
        initial_success_count,
        len(all_points),
        evaluation_count,
        retained_boundary_count,
    )
