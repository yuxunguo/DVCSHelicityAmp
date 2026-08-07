"""Top up low-count minima's contour boundary points to a minimum threshold.

Some minima's original contour search (GradientPhaseSpaceContour.py) found
far fewer than the configured 1536 boundary points, because many random
directions never cross the objective cut before hitting the domain edge
(see GradientContourWorker._trace_contour). This finds every retained
minimum currently below TARGET_COUNT, generates fresh directions (a new
random seed, so they don't duplicate the ones already saved) for exactly
the deficit, runs the same bisection search used by the original contour
stage, and rewrites that minimum's contour file with the combined
(old + new) boundary points and a corrected point count.

Run with

    python TopUpContourSamples.py
"""

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import PhaseSpaceScan as phase_scan
from config import ELECTRON_MASS_GEV, MUON_MASS_GEV, SCAN_WORKERS
from GradientContourWorker import (
    SCAN_DIMENSION,
    _objective_evaluation,
    configuration_contour_task,
)

STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTONS = ("electron",)
TARGET_COUNT = 400
LEPTON_MASSES_GEV = {"electron": ELECTRON_MASS_GEV, "muon": MUON_MASS_GEV}
CONFIG_CONTOUR_INITIAL_RADIUS = 0.01
CONFIG_CONTOUR_BISECTION_ITERATIONS = 8
TOPUP_SEED_OFFSET = 900_000_000  # clear of the original per-minimum direction seeds

SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"


def _read_contour_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _new_random_directions(seed, n_directions):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_directions, SCAN_DIMENSION))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def _write_contour_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_up_state(lepton_name, state_key, target_count=TARGET_COUNT):
    """Top up every retained minimum below target_count for one state."""
    lepton_mass = LEPTON_MASSES_GEV[lepton_name]
    cluster_dir = SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
    minima_path = cluster_dir / "clustered_minima.csv"
    contour_dir = SCAN_ROOT / lepton_name / "Data" / state_key / "contour" / "local_minima"
    if not minima_path.exists():
        print(f"skip {lepton_name}/{state_key}: missing {minima_path}")
        return

    with open(minima_path, newline="", encoding="utf-8") as handle:
        cluster_rows = list(csv.DictReader(handle))
    retained_ids = sorted(
        {r["local_minimum_id"] for r in cluster_rows if r["within_polarization_cluster_cut"] == "True"},
        key=int,
    )

    tasks = []
    file_state = {}
    for minimum_id in retained_ids:
        path = contour_dir / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
        if not path.exists():
            continue
        saved_rows = _read_contour_rows(path)
        if not saved_rows:
            continue
        meta = saved_rows[0]
        current_count = int(meta["contour_point_count"])
        if current_count >= target_count:
            continue
        deficit = target_count - current_count
        center = np.array([float(meta[f"center_u{i}"]) for i in range(8)])
        objective_name = meta["objective_name"]
        contour_delta = float(meta["contour_delta"])
        minimum_id_int = int(minimum_id)
        seed = (TOPUP_SEED_OFFSET + minimum_id_int) % (2**32)
        directions = _new_random_directions(seed, deficit)
        file_state[minimum_id_int] = {
            "path": path,
            "saved_rows": saved_rows,
            "fieldnames": list(saved_rows[0].keys()),
        }
        tasks.append((
            minimum_id_int, 0, center, None, lepton_name, lepton_mass,
            objective_name, directions, contour_delta,
            CONFIG_CONTOUR_INITIAL_RADIUS, CONFIG_CONTOUR_BISECTION_ITERATIONS,
        ))

    if not tasks:
        print(f"{lepton_name}/{state_key}: nothing below {target_count}, skipping.")
        return

    print(f"{lepton_name}/{state_key}: {len(tasks)} minima below {target_count}, topping up ...")

    phase_scan._configure_lepton(lepton_name)

    def evaluate_base_value(center, objective_name):
        return _objective_evaluation(center, lepton_name, lepton_mass, 0, objective_name)

    resolved_tasks = []
    for task in tasks:
        (minimum_id, chunk_index, center, _base_value, lepton_name_, lepton_mass_,
         objective_name, directions, contour_delta, initial_radius, bisection_iterations) = task
        base_value = evaluate_base_value(center, objective_name)
        resolved_tasks.append((
            minimum_id, chunk_index, center, base_value, lepton_name_, lepton_mass_,
            objective_name, directions, contour_delta, initial_radius, bisection_iterations,
        ))

    started = time.perf_counter()
    workers = max(1, min(int(SCAN_WORKERS), len(resolved_tasks)))
    if workers <= 1:
        results = [configuration_contour_task(task) for task in resolved_tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(configuration_contour_task, resolved_tasks, chunksize=1))

    updated = 0
    for minimum_id, _chunk_index, new_boundary_points in results:
        state = file_state[minimum_id]
        saved_rows = state["saved_rows"]
        fieldnames = state["fieldnames"]
        base = {k: saved_rows[0][k] for k in fieldnames if not k.startswith("u") and k not in ("record_type", "contour_sample_id") and not k.startswith("center_u")}
        base.update({f"center_u{i}": saved_rows[0][f"center_u{i}"] for i in range(8)})
        new_count = len(saved_rows) - 1 + len(new_boundary_points)  # -1 for the "minimum" row
        base["contour_point_count"] = new_count
        for row in saved_rows:
            row["contour_point_count"] = new_count
        next_sample_id = max(
            (int(r["contour_sample_id"]) for r in saved_rows if r["record_type"] == "sample"),
            default=-1,
        ) + 1
        for offset, point in enumerate(new_boundary_points):
            physical = _physical_columns(point)
            row = {**base, "record_type": "sample", "contour_sample_id": next_sample_id + offset}
            for i in range(8):
                row[f"u{i}"] = float(point[i])
            row.update(physical)
            saved_rows.append(row)
        _write_contour_rows(state["path"], saved_rows, fieldnames)
        updated += 1

    elapsed = time.perf_counter() - started
    print(f"{lepton_name}/{state_key}: topped up {updated} minima in {elapsed:.1f}s")


def _physical_columns(unit_point):
    from GradientContourWorker import _normalized_to_point
    import math
    values = _normalized_to_point(unit_point)
    s_value, theta_p_out, theta_gamma_out, qout, phi_p_out, phi_gamma_out, alpha_e, alpha_p = values
    return {
        "sqrt_s": math.sqrt(max(0.0, s_value)),
        "theta_p_out": theta_p_out,
        "theta_gamma_out": theta_gamma_out,
        "qOut": qout,
        "phi_p_out": phi_p_out,
        "phi_gamma_out": phi_gamma_out,
        "alpha_e": alpha_e,
        "alpha_p": alpha_p,
    }


def main():
    """Top up every configured lepton and state to TARGET_COUNT."""
    for lepton_name in LEPTONS:
        for state_key in STATES:
            top_up_state(lepton_name, state_key)


if __name__ == "__main__":
    raise SystemExit(main())
