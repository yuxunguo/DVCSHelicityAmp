"""Generate K-per-ray verified interior smear samples for retained minima.

For every retained local minimum (``within_polarization_cluster_cut`` True in
``Data/<state>/cluster/clustered_minima.csv``), reuse its already-computed
contour rays (``Data/<state>/contour/local_minima/*.csv``) as a sampling
prior: for each saved boundary ray, draw ``K`` random depths along that same
direction and verify each with one fresh objective evaluation
(``SmearingWorker.smear_task``).

Each minimum's result is cached to its own file under
``Data/<state>/cluster/smear/local_minima/`` as soon as it is computed, so an
interrupted run can be resumed: with ``REUSE_SAVED_SMEAR_SAMPLES = True``
(the default), a rerun skips every minimum whose cached file already matches
the active objective, contour delta, and K, and only computes what is still
missing. The combined ``Data/<state>/cluster/smear_points_k<K>.csv`` is then
reassembled from the per-minimum cache files, which is cheap even when
nothing new was computed.

Run with

    python GenerateSmearSamples.py
"""

import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import ELECTRON_MASS_GEV, MUON_MASS_GEV, SCAN_WORKERS
from SmearingWorker import smear_task

LEPTON_MASSES_GEV = {
    "electron": ELECTRON_MASS_GEV,
    "muon": MUON_MASS_GEV,
}
STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
LEPTONS = ("electron",)
K = 2
RANDOM_SEED = 20260806
REUSE_SAVED_SMEAR_SAMPLES = True

SCAN_ROOT = Path("Output") / "GradientPhaseSpaceScan"
PHYSICAL_FIELDS = (
    "sqrt_s", "theta_p_out", "theta_gamma_out", "qOut",
    "phi_p_out", "phi_gamma_out", "alpha_e", "alpha_p",
)
META_FIELDS = (
    "objective_name", "contour_delta", "k", "local_minimum_id",
    "tried", "accepted", "base_value", "record_type",
)


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_minimum_contour(path):
    """Return (center_u, boundary_points, objective_name, contour_delta)."""
    center = None
    objective_name = None
    contour_delta = None
    boundary_points = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if objective_name is None:
                objective_name = row["objective_name"]
                contour_delta = float(row["contour_delta"])
            if row["record_type"] == "minimum":
                center = [float(row[f"center_u{i}"]) for i in range(8)]
                continue
            if row["record_type"] == "sample":
                boundary_points.append([float(row[f"u{i}"]) for i in range(8)])
    return center, boundary_points, objective_name, contour_delta


def _per_minimum_path(smear_dir, minimum_id, k):
    return smear_dir / f"local_minimum_{int(minimum_id):04d}_smear_k{k}.csv"


def _write_per_minimum_cache(
    path, minimum_id, objective_name, contour_delta, k, tried, accepted,
    base_value, physical_rows,
):
    """Write one minimum's cached smear result, atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=(*META_FIELDS, *PHYSICAL_FIELDS)
        )
        writer.writeheader()
        base = {
            "objective_name": objective_name,
            "contour_delta": contour_delta,
            "k": k,
            "local_minimum_id": minimum_id,
            "tried": tried,
            "accepted": accepted,
            "base_value": base_value,
        }
        writer.writerow({**base, "record_type": "meta", **{f: "" for f in PHYSICAL_FIELDS}})
        for physical in physical_rows:
            s_value, theta_p_out, theta_gamma_out, qout, phi_p_out, phi_gamma_out, alpha_e, alpha_p = physical
            writer.writerow({
                **base,
                "record_type": "sample",
                "sqrt_s": math.sqrt(max(0.0, s_value)),
                "theta_p_out": theta_p_out,
                "theta_gamma_out": theta_gamma_out,
                "qOut": qout,
                "phi_p_out": phi_p_out,
                "phi_gamma_out": phi_gamma_out,
                "alpha_e": alpha_e,
                "alpha_p": alpha_p,
            })
    tmp_path.replace(path)


def _try_load_cached(path, objective_name, contour_delta, k, minimum_id):
    """Return cached sample rows if the file matches current settings, else None."""
    if not path.exists():
        return None
    try:
        rows = _read_csv_rows(path)
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    meta = rows[0]
    try:
        matches = (
            meta["objective_name"] == objective_name
            and math.isclose(float(meta["contour_delta"]), contour_delta, rel_tol=0.0, abs_tol=1.0e-15)
            and int(meta["k"]) == int(k)
            and int(meta["local_minimum_id"]) == int(minimum_id)
        )
    except (KeyError, ValueError):
        return None
    if not matches:
        return None
    return [row for row in rows if row["record_type"] == "sample"]


def generate_state_smear(lepton_name, state_key, k=K):
    """Build (or resume) the smear-sample cache for one lepton/state pair."""
    lepton_mass = LEPTON_MASSES_GEV[lepton_name]
    cluster_dir = SCAN_ROOT / lepton_name / "Data" / state_key / "cluster"
    minima_path = cluster_dir / "clustered_minima.csv"
    contour_dir = SCAN_ROOT / lepton_name / "Data" / state_key / "contour" / "local_minima"
    smear_dir = cluster_dir / "smear" / "local_minima"
    if not minima_path.exists():
        print(f"skip {lepton_name}/{state_key}: missing {minima_path}")
        return

    rows = _read_csv_rows(minima_path)
    retained_ids = sorted(
        {row["local_minimum_id"] for row in rows if row["within_polarization_cluster_cut"] == "True"},
        key=int,
    )
    if not retained_ids:
        print(f"skip {lepton_name}/{state_key}: no retained minima")
        return

    tasks = []
    reused = 0
    missing_contours = 0
    combined_sample_rows = {}
    for minimum_id in retained_ids:
        contour_path = contour_dir / f"local_minimum_{int(minimum_id):04d}_contour_samples.csv"
        if not contour_path.exists():
            missing_contours += 1
            continue
        center, boundary_points, objective_name, contour_delta = _load_minimum_contour(
            contour_path
        )
        if center is None or not boundary_points:
            missing_contours += 1
            continue

        cache_path = _per_minimum_path(smear_dir, minimum_id, k)
        if REUSE_SAVED_SMEAR_SAMPLES:
            cached = _try_load_cached(cache_path, objective_name, contour_delta, k, minimum_id)
            if cached is not None:
                combined_sample_rows[minimum_id] = cached
                reused += 1
                continue

        seed = (RANDOM_SEED + int(minimum_id)) % (2**32)
        tasks.append((
            minimum_id, center, boundary_points, lepton_name, lepton_mass,
            objective_name, contour_delta, k, seed, cache_path,
        ))
    if missing_contours:
        print(
            f"{lepton_name}/{state_key}: {missing_contours} retained minima "
            "have no saved contour and were skipped."
        )
    if reused:
        print(
            f"{lepton_name}/{state_key}: reusing {reused} already-cached "
            "minima; computing the rest."
        )
    if not tasks and not combined_sample_rows:
        print(f"skip {lepton_name}/{state_key}: no usable contour data")
        return

    if tasks:
        print(
            f"{lepton_name}/{state_key}: {len(tasks)} minima to compute, "
            f"K={k}, {SCAN_WORKERS} workers ..."
        )
        started = time.perf_counter()
        workers = max(1, min(int(SCAN_WORKERS), len(tasks)))
        total_tried = 0
        total_accepted = 0
        completed = 0
        task_by_id = {task[0]: task for task in tasks}

        def _handle_result(minimum_id, physical_rows, tried, accepted, base_value):
            nonlocal total_tried, total_accepted, completed
            task = task_by_id[minimum_id]
            _, _center, _boundary, _lepton, _mass, objective_name, contour_delta, _k, _seed, cache_path = task
            _write_per_minimum_cache(
                cache_path, minimum_id, objective_name, contour_delta, k,
                tried, accepted, base_value, physical_rows,
            )
            total_tried += tried
            total_accepted += accepted
            combined_sample_rows[minimum_id] = _try_load_cached(
                cache_path, objective_name, contour_delta, k, minimum_id,
            )
            completed += 1
            report_every = max(1, len(tasks) // 20)
            if completed % report_every == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"  {lepton_name}/{state_key}: {completed}/{len(tasks)} "
                    f"minima cached, {elapsed:.0f}s elapsed",
                    flush=True,
                )

        if workers <= 1:
            for task in tasks:
                _handle_result(task[0], *smear_task(task[:-1])[1:])
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(smear_task, task[:-1]): task[0]
                    for task in tasks
                }
                for future in as_completed(futures):
                    minimum_id, physical_rows, tried, accepted, base_value = future.result()
                    _handle_result(minimum_id, physical_rows, tried, accepted, base_value)

        elapsed = time.perf_counter() - started
        rate = total_tried / elapsed if elapsed > 0 else float("nan")
        acceptance = total_accepted / total_tried if total_tried else float("nan")
        print(
            f"{lepton_name}/{state_key}: computed {len(tasks)} minima, "
            f"tried={total_tried} accepted={total_accepted} "
            f"(acceptance={acceptance:.1%}) elapsed={elapsed:.1f}s "
            f"({rate:.0f} eval/s)"
        )

    output_rows = []
    for minimum_id, sample_rows in combined_sample_rows.items():
        for row in sample_rows:
            output_rows.append({
                "local_minimum_id": minimum_id,
                **{field: row[field] for field in PHYSICAL_FIELDS},
            })
    output_path = cluster_dir / f"smear_points_k{k}.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("local_minimum_id", *PHYSICAL_FIELDS)
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"{lepton_name}/{state_key}: {len(output_rows)} smear points from "
        f"{len(combined_sample_rows)} minima -> {output_path}"
    )


def main():
    """Generate smear samples for every configured lepton and state."""
    for lepton_name in LEPTONS:
        for state_key in STATES:
            generate_state_smear(lepton_name, state_key)


if __name__ == "__main__":
    raise SystemExit(main())
