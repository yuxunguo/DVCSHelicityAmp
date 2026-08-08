"""Free (zero-evaluation) Gaussian smearing from the full contour covariance.

For each retained minimum, build the empirical covariance matrix from ALL of
its saved contour boundary points (every one of the ~1536 directions, not
just the 16 axis-aligned rays), and draw samples directly from
``N(center, covariance)``. This reuses only already-computed contour data:
no new physics evaluations are needed at all, unlike SmearingWorker.py's
verified interior sampling.

The full covariance (not a diagonal approximation) is used so that
cross-axis correlations already visible in the raw contour cloud (e.g.
sqrt_s vs phi_gamma_out) are preserved in the drawn samples.
"""

import numpy as np

from GradientContourWorker import (
    PERIODIC_UNIT_COORDINATES,
    SCAN_DIMENSION,
    _normalized_to_point,
)


def periodic_aware_diffs(points, center, periodic_coordinates=None):
    """Return points - center, wrapped to the shortest signed path per axis."""
    diffs = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    if periodic_coordinates is None:
        periodic_coordinates = PERIODIC_UNIT_COORDINATES
    for index in periodic_coordinates:
        diffs[:, index] = (diffs[:, index] + 0.5) % 1.0 - 0.5
    return diffs


def full_covariance_from_contour(
    center, boundary_points, *, periodic_coordinates=None,
):
    """Return the empirical 8x8 covariance from all saved boundary points."""
    diffs = periodic_aware_diffs(
        boundary_points,
        center,
        periodic_coordinates=periodic_coordinates,
    )
    return np.cov(diffs.T)


def rescale_covariance_for_cut(covariance, original_cut, target_cut):
    """Rescale a covariance built at ``original_cut`` to approximate ``target_cut``.

    Assumes locally quadratic behavior: boundary radius scales as
    ``sqrt(delta)`` along a fixed direction, so variance (radius squared)
    scales linearly in the cut. This is a free approximation, not a
    recomputation -- see the caveats about non-quadratic regions discussed
    alongside GenerateGaussianSamples.py's CUT_SCALE_FACTOR.
    """
    if original_cut <= 0.0:
        raise ValueError("original_cut must be positive.")
    return covariance * (target_cut / original_cut)


def draw_gaussian_samples(
    center, covariance, n_samples, rng, *, return_diagnostics=False,
):
    """Draw a boundary-aware Gaussian sample, returned in physical units.

    The first four normalized coordinates are bounded rather than periodic.
    Out-of-domain proposals on those axes are rejected and redrawn instead of
    clipped onto a boundary, which avoids artificial edge pileups.  The final
    four coordinates are periodic and are wrapped onto ``[0, 1)``.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (SCAN_DIMENSION,):
        raise ValueError(f"center must have shape ({SCAN_DIMENSION},).")
    if np.any((center[:4] < 0.0) | (center[:4] > 1.0)):
        raise ValueError("bounded center coordinates must lie in [0, 1].")
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative.")

    # Regularize in case some directions are numerically degenerate.
    covariance = np.asarray(covariance, dtype=float)
    covariance = covariance + 1.0e-14 * np.eye(SCAN_DIMENSION)
    accepted_batches = []
    accepted_count = 0
    attempted_count = 0
    rejected_count = 0
    while accepted_count < n_samples:
        remaining = n_samples - accepted_count
        proposals = rng.multivariate_normal(
            mean=center, cov=covariance, size=remaining, method="cholesky",
        )
        attempted_count += remaining
        valid = np.all(
            (proposals[:, :4] >= 0.0) & (proposals[:, :4] <= 1.0),
            axis=1,
        )
        rejected_count += int(np.count_nonzero(~valid))
        accepted = proposals[valid]
        if len(accepted):
            accepted_batches.append(accepted)
            accepted_count += len(accepted)

    if n_samples:
        draws_unit = np.concatenate(accepted_batches, axis=0)
        draws_unit[:, list(PERIODIC_UNIT_COORDINATES)] %= 1.0
    else:
        draws_unit = np.empty((0, SCAN_DIMENSION), dtype=float)
    physical = np.array([_normalized_to_point(point) for point in draws_unit])
    # _normalized_to_point's first column is s = sqrt_s**2, not sqrt_s itself.
    if len(physical):
        physical[:, 0] = np.sqrt(np.maximum(0.0, physical[:, 0]))
    diagnostics = {
        "requested_samples": n_samples,
        "attempted_samples": attempted_count,
        "accepted_samples": accepted_count,
        "rejected_samples": rejected_count,
        "rejection_fraction": (
            rejected_count / attempted_count if attempted_count else 0.0
        ),
    }
    return (physical, diagnostics) if return_diagnostics else physical
