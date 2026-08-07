"""Preconditioned hit-and-run sampler for uniform coverage of {D <= D_min}.

Unlike SmearingWorker.py (which reuses precomputed contour rays from a fixed
center and therefore only sees the part of the region visible via a straight
line from the minimum), this module runs an actual Markov chain: from the
current point, propose a random direction, find the true segment of the
level set along that line by bisection (in both directions), and move to a
uniformly-chosen point on that segment. Given enough steps this provably
converges to the uniform distribution on the sublevel set, including any
non-star-shaped parts a fixed-center ray can't reach.

Direction proposals are preconditioned once per minimum using that
minimum's own axis-aligned contour rays (already computed by
GradientPhaseSpaceContour.py) as a diagonal scale estimate, so the chain
does not waste steps on directions mismatched to the region's true
anisotropy.
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

AXIS_DIRECTIONS = np.vstack((np.eye(SCAN_DIMENSION), -np.eye(SCAN_DIMENSION)))


def _periodic_aware_diff(a, b):
    """Return a - b for each coordinate, wrapped to the shortest signed path."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    for index in PERIODIC_UNIT_COORDINATES:
        diff[index] = (diff[index] + 0.5) % 1.0 - 0.5
    return diff


def axis_scales_from_contour(center, boundary_points, directions):
    """Return a per-axis characteristic scale from the axis-aligned rays.

    ``directions`` and ``boundary_points`` are the same arrays saved by the
    contour stage; the first ``2 * SCAN_DIMENSION`` directions are always the
    ``+/- unit vector`` axis rays (see ``_contour_directions``).
    """
    center = np.asarray(center, dtype=float)
    scales = np.full(SCAN_DIMENSION, np.nan)
    for axis in range(SCAN_DIMENSION):
        radii = []
        for sign, row_index in ((+1.0, axis), (-1.0, SCAN_DIMENSION + axis)):
            if row_index >= len(boundary_points):
                continue
            direction = directions[row_index]
            if not np.allclose(direction, sign * np.eye(SCAN_DIMENSION)[axis]):
                continue
            diff = _periodic_aware_diff(boundary_points[row_index], center)
            radii.append(float(np.linalg.norm(diff)))
        if radii:
            scales[axis] = float(np.mean(radii))
    fallback = np.nanmedian(scales) if np.any(np.isfinite(scales)) else 0.05
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-12), scales, fallback)
    return scales


def _maximum_signed_radius(point, direction):
    """Return the largest |t| before ``point + t*direction`` leaves the domain."""
    limits = []
    for index, component in enumerate(direction):
        if abs(component) <= 1.0e-15:
            continue
        if index in PERIODIC_UNIT_COORDINATES:
            limits.append(0.5 / abs(component))
        elif component > 0.0:
            limits.append((1.0 - point[index]) / component)
        else:
            limits.append(point[index] / -component)
    return max(0.0, float(min(limits, default=0.0)))


def _find_one_side(evaluate, point, direction, target, initial_radius, bisection_iterations):
    """Return the signed distance to the boundary along +direction from point."""
    maximum_radius = _maximum_signed_radius(point, direction)
    if maximum_radius <= 1.0e-14:
        return 0.0
    low, high = 0.0, min(initial_radius, maximum_radius)
    high_value = evaluate(_move_unit_point(point, high * direction))
    while high_value < target and high < maximum_radius:
        low = high
        high = min(2.0 * high, maximum_radius)
        high_value = evaluate(_move_unit_point(point, high * direction))
    if high_value < target:
        return maximum_radius  # region extends to the domain edge
    for _ in range(bisection_iterations):
        mid = 0.5 * (low + high)
        if evaluate(_move_unit_point(point, mid * direction)) < target:
            low = mid
        else:
            high = mid
    return low


def find_segment(evaluate, point, direction, target, initial_radius=0.01, bisection_iterations=8):
    """Return (t_low, t_high) bracketing the level set along the line."""
    forward = _find_one_side(evaluate, point, direction, target, initial_radius, bisection_iterations)
    backward = _find_one_side(evaluate, point, -direction, target, initial_radius, bisection_iterations)
    return -backward, forward


def hit_and_run_chain(
    evaluate, start, axis_scales, target, n_steps, *, rng,
    initial_radius=0.01, bisection_iterations=8,
):
    """Run one hit-and-run chain and return every visited point (unthinned)."""
    point = np.asarray(start, dtype=float).copy()
    visited = np.empty((n_steps, SCAN_DIMENSION), dtype=float)
    step_evals = np.empty(n_steps, dtype=int)
    for step in range(n_steps):
        raw = rng.normal(size=SCAN_DIMENSION) * axis_scales
        norm = np.linalg.norm(raw)
        direction = raw / norm if norm > 1.0e-15 else raw
        eval_count = 0

        def counted_evaluate(p):
            nonlocal eval_count
            eval_count += 1
            return evaluate(p)

        t_low, t_high = find_segment(
            counted_evaluate, point, direction, target,
            initial_radius=initial_radius, bisection_iterations=bisection_iterations,
        )
        t = rng.uniform(t_low, t_high)
        point = _move_unit_point(point, t * direction)
        visited[step] = point
        step_evals[step] = eval_count
    return visited, step_evals


def hit_and_run_task(task):
    """Run a chain for one minimum and return unit-cube + physical samples."""
    (
        minimum_id, center, contour_directions, contour_boundary_points,
        lepton_name, lepton_mass, objective_name, contour_delta,
        n_steps, burn_in, thin, seed,
    ) = task
    phase_scan._configure_lepton(lepton_name)
    center = np.asarray(center, dtype=float)
    rng = np.random.default_rng(seed)
    eval_id = 0

    def evaluate(unit_point):
        nonlocal eval_id
        value = _objective_evaluation(
            unit_point, lepton_name, lepton_mass, eval_id, objective_name,
        )
        eval_id += 1
        return value

    base_value = evaluate(center)
    target = base_value + contour_delta
    axis_scales = axis_scales_from_contour(
        center, contour_boundary_points, contour_directions,
    )

    visited, step_evals = hit_and_run_chain(
        evaluate, center, axis_scales, target, n_steps, rng=rng,
    )
    kept = visited[burn_in::thin]
    physical_rows = [_normalized_to_point(point).tolist() for point in kept]
    return minimum_id, physical_rows, int(step_evals.sum()) + 1, len(kept)
