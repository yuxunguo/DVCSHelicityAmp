"""Cross-check angular minima before using them as Cartesian starts.

Every saved angular minimum is reconstructed into exact final three-vectors,
re-evaluated without the angular ``pOut`` root solver, and compared with its
stored objective and kinematics.  The detailed table is the gate for the full
Cartesian production run.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import CartesianGradientPhaseSpaceDefinitions as definitions
import CartesianPhaseSpaceCoordinates as coordinates
import PhaseSpaceConfigScan as config_scan
import PhaseSpaceScan as phase_scan
from AlignmentScan import LEPTON_SPECS


STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")
ANGULAR_ROOT = Path("Output") / "GradientPhaseSpaceScan"
CARTESIAN_ROOT = Path("Output") / "CartesianGradientPhaseSpaceScan"
DEFAULT_OBJECTIVE_TOLERANCE = 1.0e-5
DEFAULT_KINEMATIC_TOLERANCE = 1.0e-10
DEFAULT_THRESHOLD = 0.01


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No cross-check rows were produced for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _circular_difference(first, second):
    return abs((float(first) - float(second) + np.pi) % (2.0 * np.pi) - np.pi)


def _direction(theta, phi):
    return np.asarray((
        np.sin(float(theta)) * np.cos(float(phi)),
        np.sin(float(theta)) * np.sin(float(phi)),
        np.cos(float(theta)),
    ))


def _crosscheck_task(task):
    state, objective_name, row, objective_tolerance, kinematic_tolerance = task
    lepton_name = "electron"
    phase_scan._configure_lepton(lepton_name)
    unit_point = coordinates.row_to_unit(row)
    original_proton3 = coordinates.vector_from_angles(
        float(row["pOut"]), row["theta_p_out"], row["phi_p_out"]
    )
    original_photon3 = coordinates.vector_from_angles(
        float(row["qOut"]), row["theta_gamma_out"], row["phi_gamma_out"]
    )
    reconstructed_proton3, reconstructed_photon3, _alpha_e, _alpha_p = (
        coordinates.unit_to_cartesian(unit_point)
    )
    value, cartesian_row = coordinates.evaluate_unit_point(
        unit_point,
        lepton_name=lepton_name,
        lepton_mass=LEPTON_SPECS[lepton_name]["mass"],
        evaluation_id=int(row["local_minimum_id"]),
        objective_name=objective_name,
        stage="angular_cartesian_crosscheck",
    )
    objective_key = f"{config_scan.mixing_prefix(lepton_name)}_{objective_name}"
    stored_value = float(row[objective_key])
    if cartesian_row is None:
        return {
            "state": state,
            "local_minimum_id": row["local_minimum_id"],
            "status": "evaluation_failed",
            "stored_objective": stored_value,
            "cartesian_objective": np.nan,
            "objective_abs_difference": np.nan,
            "sqrt_s_abs_difference": np.nan,
            "pOut_abs_difference": np.nan,
            "qOut_abs_difference": np.nan,
            "theta_p_abs_difference": np.nan,
            "theta_gamma_abs_difference": np.nan,
            "phi_p_abs_difference": np.nan,
            "phi_gamma_abs_difference": np.nan,
            "proton_direction_distance": np.nan,
            "photon_direction_distance": np.nan,
            "proton_vector_roundtrip_distance": np.nan,
            "photon_vector_roundtrip_distance": np.nan,
            "energy_residual": np.nan,
            "momentum_residual": np.nan,
            "within_objective_tolerance": False,
            "within_kinematic_tolerance": False,
        }
    differences = {
        "sqrt_s_abs_difference": abs(
            float(row["sqrt_s"]) - float(cartesian_row["sqrt_s"])
        ),
        "pOut_abs_difference": abs(
            float(row["pOut"]) - float(cartesian_row["pOut"])
        ),
        "qOut_abs_difference": abs(
            float(row["qOut"]) - float(cartesian_row["qOut"])
        ),
        "theta_p_abs_difference": abs(
            float(row["theta_p_out"])
            - float(cartesian_row["theta_p_out"])
        ),
        "theta_gamma_abs_difference": abs(
            float(row["theta_gamma_out"])
            - float(cartesian_row["theta_gamma_out"])
        ),
        "phi_p_abs_difference": (
            _circular_difference(row["phi_p_out"], cartesian_row["phi_p_out"])
            if cartesian_row["phi_p_out_defined"] else np.nan
        ),
        "phi_gamma_abs_difference": (
            _circular_difference(
                row["phi_gamma_out"], cartesian_row["phi_gamma_out"]
            )
            if cartesian_row["phi_gamma_out_defined"] else np.nan
        ),
        "proton_direction_distance": float(np.linalg.norm(
            _direction(row["theta_p_out"], row["phi_p_out"])
            - _direction(
                cartesian_row["theta_p_out"],
                0.0 if not cartesian_row["phi_p_out_defined"] else cartesian_row["phi_p_out"],
            )
        )),
        "photon_direction_distance": float(np.linalg.norm(
            _direction(row["theta_gamma_out"], row["phi_gamma_out"])
            - _direction(
                cartesian_row["theta_gamma_out"],
                0.0 if not cartesian_row["phi_gamma_out_defined"] else cartesian_row["phi_gamma_out"],
            )
        )),
        "proton_vector_roundtrip_distance": float(np.linalg.norm(
            original_proton3 - reconstructed_proton3
        )),
        "photon_vector_roundtrip_distance": float(np.linalg.norm(
            original_photon3 - reconstructed_photon3
        )),
    }
    finite_kinematic_differences = [
        float(differences[key])
        for key in (
            "sqrt_s_abs_difference",
            "pOut_abs_difference",
            "qOut_abs_difference",
            "proton_vector_roundtrip_distance",
            "photon_vector_roundtrip_distance",
        )
        if np.isfinite(differences[key])
    ]
    objective_difference = abs(stored_value - value)
    kinematic_ok = max(finite_kinematic_differences, default=0.0) <= (
        kinematic_tolerance
    )
    objective_ok = objective_difference <= objective_tolerance
    return {
        "state": state,
        "local_minimum_id": row["local_minimum_id"],
        "status": "pass" if objective_ok and kinematic_ok else "mismatch",
        "stored_objective": stored_value,
        "cartesian_objective": value,
        "objective_abs_difference": objective_difference,
        **differences,
        "energy_residual": cartesian_row["cartesian_energy_residual"],
        "momentum_residual": cartesian_row["cartesian_momentum_residual"],
        "within_objective_tolerance": objective_ok,
        "within_kinematic_tolerance": kinematic_ok,
    }


def crosscheck_state(
    state,
    *,
    workers,
    objective_tolerance=DEFAULT_OBJECTIVE_TOLERANCE,
    kinematic_tolerance=DEFAULT_KINEMATIC_TOLERANCE,
    threshold=DEFAULT_THRESHOLD,
    maximum_rejections=10,
    limit=None,
):
    definition = definitions.SCAN_DEFINITIONS[state]
    source = (
        ANGULAR_ROOT / "electron" / "Data" / state / "scan"
        / "local_minima.csv"
    )
    rows = _read_csv(source)
    objective_key = (
        f"{config_scan.mixing_prefix('electron')}_{definition.objective_name}"
    )
    rows = [row for row in rows if float(row[objective_key]) <= threshold]
    if limit is not None:
        rows = rows[:limit]
    tasks = [
        (
            state,
            definition.objective_name,
            row,
            objective_tolerance,
            kinematic_tolerance,
        )
        for row in rows
    ]
    if workers == 1:
        checked = [_crosscheck_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            checked = list(executor.map(_crosscheck_task, tasks, chunksize=1))
    output = (
        CARTESIAN_ROOT / "electron" / "Data" / state / "scan"
        / "angular_to_cartesian_crosscheck.csv"
    )
    _write_csv(output, checked)
    passed = sum(row["status"] == "pass" for row in checked)
    maximum_objective_difference = max(
        (
            float(row["objective_abs_difference"])
            for row in checked
            if np.isfinite(float(row["objective_abs_difference"]))
        ),
        default=np.nan,
    )
    print(
        f"{state}: {passed}/{len(checked)} passed; maximum objective "
        f"difference={maximum_objective_difference:.3e}; {output}",
        flush=True,
    )
    rejected = len(checked) - passed
    if rejected > maximum_rejections:
        raise RuntimeError(
            f"{state} angular-to-Cartesian cross-check failed for "
            f"{rejected} minima, exceeding the allowed "
            f"{maximum_rejections}."
        )
    return output, passed, rejected


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", choices=STATES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--maximum-rejections", type=int, default=10)
    parser.add_argument(
        "--objective-tolerance", type=float, default=DEFAULT_OBJECTIVE_TOLERANCE
    )
    parser.add_argument(
        "--kinematic-tolerance", type=float, default=DEFAULT_KINEMATIC_TOLERANCE
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    for state in tuple(args.state or STATES):
        crosscheck_state(
            state,
            workers=args.workers,
            objective_tolerance=args.objective_tolerance,
            kinematic_tolerance=args.kinematic_tolerance,
            threshold=args.threshold,
            maximum_rejections=args.maximum_rejections,
            limit=args.limit,
        )


if __name__ == "__main__":
    raise SystemExit(main())
