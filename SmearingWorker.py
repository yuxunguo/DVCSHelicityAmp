"""Lightweight process worker for K-per-ray verified interior smear sampling.

Reuses the already-computed contour rays (direction + boundary radius) from
GradientPhaseSpaceContour.py as a sampling prior: for each saved boundary
point, draw ``k`` random depths along the same direction
(``r = R_boundary * u**(1/8)``, uniform in 8D volume) and spend exactly one
fresh objective evaluation per candidate to verify it is still inside the
``contour_delta`` cut, rather than assuming the ray's monotonicity. This
avoids importing SciPy for the same Windows process-spawn reason as
GradientContourWorker.py.
"""

import numpy as np

import PhaseSpaceScan as phase_scan
from GradientContourWorker import (
    PERIODIC_UNIT_COORDINATES,
    SCAN_DIMENSION,
    _move_unit_point,
    _normalized_to_point,
    _objective_evaluation,
)


def _periodic_aware_direction(center, boundary):
    """Return the unit direction and radius from center to boundary point."""
    diff = np.asarray(boundary, dtype=float) - np.asarray(center, dtype=float)
    for index in PERIODIC_UNIT_COORDINATES:
        diff[index] = (diff[index] + 0.5) % 1.0 - 0.5
    radius = float(np.linalg.norm(diff))
    if radius <= 1.0e-15:
        return None, 0.0
    return diff / radius, radius


def smear_task(task):
    """Generate and verify up to ``k`` interior samples per boundary ray."""
    (
        minimum_id,
        center,
        boundary_points,
        lepton_name,
        lepton_mass,
        objective_name,
        contour_delta,
        k,
        seed,
    ) = task
    phase_scan._configure_lepton(lepton_name)
    center = np.asarray(center, dtype=float)
    rng = np.random.default_rng(seed)
    evaluation_id = 0

    def evaluate(unit_point):
        nonlocal evaluation_id
        value = _objective_evaluation(
            unit_point, lepton_name, lepton_mass, evaluation_id, objective_name,
        )
        evaluation_id += 1
        return value

    base_value = evaluate(center)
    target = base_value + contour_delta

    accepted_unit_points = []
    tried = 0
    for boundary in boundary_points:
        direction, radius = _periodic_aware_direction(center, boundary)
        if direction is None:
            continue
        for _sample_index in range(k):
            depth_fraction = rng.random()
            sample_radius = radius * depth_fraction ** (1.0 / SCAN_DIMENSION)
            candidate = _move_unit_point(center, sample_radius * direction)
            tried += 1
            if evaluate(candidate) <= target:
                accepted_unit_points.append(candidate)

    accepted_array = np.asarray(accepted_unit_points, dtype=float).reshape(
        (-1, SCAN_DIMENSION)
    )
    physical_rows = [
        _normalized_to_point(point).tolist() for point in accepted_array
    ]
    return minimum_id, physical_rows, tried, len(accepted_unit_points), float(base_value)
