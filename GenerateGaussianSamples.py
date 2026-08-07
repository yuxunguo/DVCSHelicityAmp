"""Generate free Gaussian smear samples (full covariance) for retained minima.

For every retained local minimum, builds the empirical covariance matrix from
ALL of its saved contour boundary points (every direction, not a diagonal
subset) and draws ``SAMPLES_PER_MINIMUM`` points directly from that Gaussian.
No physics evaluations are used at all, so this runs independently of (and
much faster than) GenerateSmearSamples.py's verified interior sampling.

Run with

    python GenerateGaussianSamples.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from GaussianSmearWorker import (
    draw_gaussian_samples,
    full_covariance_from_contour,
    rescale_covariance_for_cut,
)
from GradientContourWorker import SCAN_DIMENSION

STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTONS = ("electron",)
SAMPLES_PER_MINIMUM = 2000
RANDOM_SEED = 20260806

SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"
PHYSICAL_FIELDS = (
    "sqrt_s", "theta_p_out", "theta_gamma_out", "qOut",
    "phi_p_out", "phi_gamma_out", "alpha_e", "alpha_p",
)


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_minimum_contour(path):
    """Return (center_u, ALL boundary points, contour_delta) from one file."""
    center = None
    boundary_points = []
    contour_delta = None
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if contour_delta is None:
                contour_delta = float(row["contour_delta"])
            if row["record_type"] == "minimum":
                center = [float(row[f"center_u{i}"]) for i in range(8)]
                continue
            if row["record_type"] == "sample":
                boundary_points.append([float(row[f"u{i}"]) for i in range(8)])
    return center, boundary_points, contour_delta


def generate_state_gaussian(
    lepton_name, state_key, n_samples=SAMPLES_PER_MINIMUM, target_cut=None,
    width_budget=None,
):
    """Build the free Gaussian smear CSV for one lepton/state pair.

    If ``target_cut`` is given, every minimum's covariance is rescaled to the
    same fixed cut (see GaussianSmearWorker.rescale_covariance_for_cut).

    If ``width_budget`` is given instead, each minimum i gets its own target
    cut ``width_budget - D_i``, where ``D_i`` is that minimum's own
    ``*_above_global_minimum`` value -- so every point in the combined plot
    (minimum position + local spread) satisfies total distance from the
    global optimum <= width_budget, consistently across minima. Minima with
    ``D_i >= width_budget`` have no budget left and are skipped entirely.

    Exactly one of ``target_cut`` / ``width_budget`` should be given; passing
    neither reuses each minimum's original (unscaled) contour_delta.
    """
    if target_cut is not None and width_budget is not None:
        raise ValueError("Pass at most one of target_cut and width_budget.")

    cluster_dir = SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
    minima_path = cluster_dir / "clustered_minima.csv"
    contour_dir = SCAN_ROOT / lepton_name / "Data" / state_key / "contour" / "local_minima"
    if not minima_path.exists():
        print(f"skip {lepton_name}/{state_key}: missing {minima_path}")
        return

    rows = _read_csv_rows(minima_path)
    retained_rows = [r for r in rows if r["within_polarization_cluster_cut"] == "True"]
    retained_ids = sorted({r["local_minimum_id"] for r in retained_rows}, key=int)
    if not retained_ids:
        print(f"skip {lepton_name}/{state_key}: no retained minima")
        return

    above_col = None
    d_above_min_by_id = {}
    if width_budget is not None:
        above_col = next(name for name in rows[0] if name.endswith("_above_global_minimum"))
        d_above_min_by_id = {
            r["local_minimum_id"]: float(r[above_col]) for r in retained_rows
        }

    # Do not use Python's randomized hash(), which changes across processes.
    rng = np.random.default_rng(
        RANDOM_SEED + 100 * LEPTONS.index(lepton_name) + STATES.index(state_key)
    )
    output_rows = []
    diagnostic_rows = []
    missing = 0
    degenerate = 0
    exhausted = 0
    for minimum_id in retained_ids:
        if width_budget is not None:
            remaining = width_budget - d_above_min_by_id[minimum_id]
            if remaining <= 0.0:
                exhausted += 1
                continue
        contour_path = contour_dir / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
        if not contour_path.exists():
            missing += 1
            continue
        center, boundary_points, original_cut = _load_minimum_contour(contour_path)
        if center is None or len(boundary_points) < SCAN_DIMENSION:
            missing += 1
            continue
        center = np.asarray(center, dtype=float)
        boundary_points = np.asarray(boundary_points, dtype=float)
        covariance = full_covariance_from_contour(center, boundary_points)
        if width_budget is not None:
            covariance = rescale_covariance_for_cut(covariance, original_cut, remaining)
        elif target_cut is not None:
            covariance = rescale_covariance_for_cut(covariance, original_cut, target_cut)
        if not np.all(np.isfinite(covariance)):
            degenerate += 1
            continue
        physical, diagnostics = draw_gaussian_samples(
            center,
            covariance,
            n_samples,
            rng,
            return_diagnostics=True,
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        diagnostic_rows.append({
            "local_minimum_id": minimum_id,
            "contour_point_count": len(boundary_points),
            "original_contour_cut": original_cut,
            "target_covariance_cut": (
                remaining if width_budget is not None
                else target_cut if target_cut is not None
                else original_cut
            ),
            "covariance_rank": np.linalg.matrix_rank(covariance),
            "covariance_min_eigenvalue": float(eigenvalues[0]),
            "covariance_max_eigenvalue": float(eigenvalues[-1]),
            **diagnostics,
        })
        for row in physical:
            output_rows.append({
                "local_minimum_id": minimum_id,
                **dict(zip(PHYSICAL_FIELDS, row)),
            })

    if missing:
        print(f"{lepton_name}/{state_key}: {missing} retained minima had no usable contour data.")
    if degenerate:
        print(f"{lepton_name}/{state_key}: {degenerate} minima had a degenerate covariance and were skipped.")
    if exhausted:
        print(f"{lepton_name}/{state_key}: {exhausted} minima had no width budget left ({above_col} >= {width_budget}).")

    if width_budget is not None:
        suffix = f"_widthbudget{width_budget:g}"
    elif target_cut is not None:
        suffix = f"_cut{target_cut:g}"
    else:
        suffix = ""
    output_path = cluster_dir / f"gaussian_points{suffix}.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("local_minimum_id", *PHYSICAL_FIELDS))
        writer.writeheader()
        writer.writerows(output_rows)
    diagnostics_path = cluster_dir / f"gaussian_sampling_diagnostics{suffix}.csv"
    diagnostic_fields = (
        "local_minimum_id",
        "contour_point_count",
        "original_contour_cut",
        "target_covariance_cut",
        "covariance_rank",
        "covariance_min_eigenvalue",
        "covariance_max_eigenvalue",
        "requested_samples",
        "attempted_samples",
        "accepted_samples",
        "rejected_samples",
        "rejection_fraction",
    )
    with open(diagnostics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    total_attempted = sum(row["attempted_samples"] for row in diagnostic_rows)
    total_rejected = sum(row["rejected_samples"] for row in diagnostic_rows)
    rejection_fraction = total_rejected / total_attempted if total_attempted else 0.0
    print(
        f"{lepton_name}/{state_key}: {len(output_rows)} Gaussian points from "
        f"{len(retained_ids) - missing - degenerate} minima; rejected "
        f"{total_rejected}/{total_attempted} out-of-domain proposals "
        f"({rejection_fraction:.2%}) -> {output_path}"
    )
    print(f"wrote {diagnostics_path}")


def _parse_args(argv=None):
    """Parse optional selectors while preserving the historical default run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lepton", action="append", choices=LEPTONS,
        help="Lepton to generate; repeat for multiple leptons.",
    )
    parser.add_argument(
        "--state", action="append", choices=STATES,
        help="State to generate; repeat for multiple states.",
    )
    width_group = parser.add_mutually_exclusive_group()
    width_group.add_argument(
        "--target-cut",
        type=float,
        help="Rescale every saved contour covariance to this fixed cut.",
    )
    width_group.add_argument(
        "--width-budget",
        type=float,
        help="Use this total D-above-minimum width budget.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Generate Gaussian smear samples for every configured lepton and state."""
    args = _parse_args(argv)
    leptons = tuple(args.lepton or LEPTONS)
    states = tuple(args.state or STATES)
    for lepton_name in leptons:
        for state_key in states:
            generate_state_gaussian(
                lepton_name,
                state_key,
                target_cut=args.target_cut,
                width_budget=args.width_budget,
            )


if __name__ == "__main__":
    raise SystemExit(main())
