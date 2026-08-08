"""Generate direct fixed-width covariances for Cartesian electron minima.

The production defaults retain every minimum satisfying the strict absolute
objective cut ``D < 0.01``, trace the boundary at ``D_i + 0.02``, replace
failed rays until 1,536 contour points are obtained, and save the periodic-
aware full 8x8 covariance in Cartesian optimizer coordinates.

Use ``--smoke-test`` first.  Its 64-direction result is written under a
separate variant directory and is never treated as production output.

Commands from the repository root::

    python GenerateCartesianDirectThresholdCovariances.py --smoke-test
    python GenerateCartesianDirectThresholdCovariances.py --preflight
    python GenerateCartesianDirectThresholdCovariances.py --workers 24

The cumulative production stages are::

    python GenerateCartesianDirectThresholdCovariances.py --incremental-stage 512 --workers 24
    python GenerateCartesianDirectThresholdCovariances.py --incremental-stage 1024 --workers 24
    python GenerateCartesianDirectThresholdCovariances.py --incremental-stage 1536 --workers 24
"""

import argparse
import os
from pathlib import Path

import CartesianGradientPhaseSpaceDefinitions as cartesian_definitions
from config import SCAN_WORKERS
import GenerateDirectThresholdCovariances as driver


STATES = driver.STATES
CARTESIAN_ROOT = Path("Output") / "CartesianGradientPhaseSpaceScan"
PRODUCTION_DIRECTIONS = 1536
SMOKE_DIRECTIONS = 64
CARTESIAN_BISECTION_ITERATIONS = 16
BENCHMARK24_STATE_COUNTS = {
    "W": 4,
    "GHZ": 3,
    "CEP": 5,
    "CEGAMMA": 5,
    "CPGAMMA": 7,
}
INCREMENTAL_STAGES = (512, 1024, 1536)


def _incremental_variant(direction_count):
    return (
        f"delta0.02_incremental_n{int(direction_count)}_threshold0.01_"
        f"b{CARTESIAN_BISECTION_ITERATIONS}_boundaryaware"
    )


def _configure_driver(
    direction_count,
    smoke_test,
    benchmark10=False,
    benchmark24=False,
    benchmark_round=1,
    incremental_stage=None,
):
    """Point the resumable direct-contour driver at Cartesian artifacts."""
    driver.definitions = cartesian_definitions
    driver.SCAN_ROOT = CARTESIAN_ROOT
    driver.TARGET_SUCCESS_COUNT = int(direction_count)
    previous_count = 0
    if incremental_stage is not None:
        stage_index = INCREMENTAL_STAGES.index(int(incremental_stage))
        if stage_index:
            previous_count = INCREMENTAL_STAGES[stage_index - 1]
    driver.PREVIOUS_SUCCESS_COUNT = previous_count
    driver.INCREMENTAL_MODE = incremental_stage is not None
    driver.PREVIOUS_OUTPUT_VARIANT = (
        _incremental_variant(previous_count) if previous_count else None
    )
    driver.MAXIMUM_ATTEMPT_COUNT = 20 * (
        driver.TARGET_SUCCESS_COUNT - previous_count
    )
    driver.DIRECT_BISECTION_ITERATIONS = CARTESIAN_BISECTION_ITERATIONS
    driver.REPLACEMENT_BATCH_SIZE = min(
        256, max(32, driver.TARGET_SUCCESS_COUNT)
    )
    if incremental_stage is not None:
        variant = _incremental_variant(incremental_stage)
    elif benchmark24:
        variant = (
            f"delta0.02_benchmark_k24_round{int(benchmark_round)}_"
            f"n{driver.TARGET_SUCCESS_COUNT}_"
            f"threshold0.01_b{driver.DIRECT_BISECTION_ITERATIONS}_"
            "boundaryaware"
        )
    elif benchmark10:
        variant = (
            f"delta0.02_benchmark_k10_n{driver.TARGET_SUCCESS_COUNT}_"
            f"threshold0.01_b{driver.DIRECT_BISECTION_ITERATIONS}_"
            "boundaryaware"
        )
    elif smoke_test:
        variant = (
            f"delta0.02_smoke_n{driver.TARGET_SUCCESS_COUNT}_"
            f"threshold0.01_b{driver.DIRECT_BISECTION_ITERATIONS}_"
            "boundaryaware"
        )
    elif driver.TARGET_SUCCESS_COUNT == PRODUCTION_DIRECTIONS:
        variant = (
            "delta0.02_all_threshold0.01_"
            f"b{driver.DIRECT_BISECTION_ITERATIONS}_boundaryaware"
        )
    else:
        variant = (
            f"delta0.02_n{driver.TARGET_SUCCESS_COUNT}_threshold0.01_"
            f"b{driver.DIRECT_BISECTION_ITERATIONS}_boundaryaware"
        )
    for field in (
        "domain_boundary_point_count",
        "domain_boundary_point_fraction",
        "previous_success_count",
        "added_success_count",
        "previous_output_variant",
    ):
        if field not in driver.FIELDNAMES:
            driver.FIELDNAMES = (*driver.FIELDNAMES, field)
    driver.OUTPUT_VARIANT = variant
    driver.OUTPUT_ROOT = (
        CARTESIAN_ROOT / driver.LEPTON_NAME / "Data"
        / f"contour_width_comparison_{variant}"
    )
    driver.SUMMARY_PATH = driver.OUTPUT_ROOT / "generation_summary.csv"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", choices=STATES)
    parser.add_argument("--workers", type=int, default=SCAN_WORKERS)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--directions", type=int,
        help="Successful contour points per minimum (production: 1536).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run one GHZ minimum with 64 directions in isolated output.",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Audit production selections/checkpoints without tracing rays.",
    )
    parser.add_argument(
        "--benchmark10", action="store_true",
        help=(
            "Trace two photon-energy-stratified minima per state with 1024 "
            "directions in isolated output."
        ),
    )
    parser.add_argument(
        "--benchmark24", action="store_true",
        help=(
            "Trace 24 workload-weighted, photon-energy-stratified minima "
            "with 1024 directions in isolated output."
        ),
    )
    parser.add_argument(
        "--benchmark-round", type=int, default=1,
        help="Deterministic benchmark selection round (default: 1).",
    )
    parser.add_argument(
        "--incremental-stage",
        type=int,
        choices=INCREMENTAL_STAGES,
        help="Cumulative production stage: 512, 1024, or 1536 points.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive.")
    states = tuple(args.state or STATES)
    limit = args.limit
    directions = (
        args.directions
        if args.directions is not None
        else PRODUCTION_DIRECTIONS
    )
    if args.incremental_stage is not None:
        if args.directions is not None:
            raise ValueError(
                "--incremental-stage cannot be combined with --directions."
            )
        directions = args.incremental_stage
    if directions < 2 * driver.SCAN_DIMENSION:
        raise ValueError("directions must be at least 16.")
    benchmark_per_state = None
    selected_modes = sum(
        (
            args.smoke_test,
            args.benchmark10,
            args.benchmark24,
            args.incremental_stage is not None,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "Choose only one of --smoke-test, --benchmark10, --benchmark24, "
            "or --incremental-stage."
        )
    if args.benchmark10:
        if args.state is not None:
            raise ValueError("--benchmark10 always benchmarks all five states.")
        if args.limit is not None:
            raise ValueError("--benchmark10 cannot be combined with --limit.")
        if args.directions is None:
            directions = 1024
        benchmark_per_state = 2
    if args.benchmark24:
        if args.state is not None:
            raise ValueError("--benchmark24 always benchmarks all five states.")
        if args.limit is not None:
            raise ValueError("--benchmark24 cannot be combined with --limit.")
        if args.directions is None:
            directions = 1024
        benchmark_per_state = BENCHMARK24_STATE_COUNTS
    if args.benchmark_round < 1:
        raise ValueError("--benchmark-round must be positive.")
    if args.smoke_test:
        if args.state is None:
            states = ("GHZ",)
        if args.directions is None:
            directions = SMOKE_DIRECTIONS
        if limit is None:
            limit = 1
    _configure_driver(
        directions,
        args.smoke_test,
        args.benchmark10,
        args.benchmark24,
        args.benchmark_round,
        args.incremental_stage,
    )
    if args.preflight:
        descriptors = driver._selected_descriptors(
            states, driver.DEFAULT_THRESHOLD, args.workers,
        )
        requested_descriptors = (
            driver._stratified_descriptor_subset(
                descriptors,
                benchmark_per_state,
                round_index=args.benchmark_round,
            )
            if benchmark_per_state is not None
            else descriptors
        )
        completed = driver._completed_rows(
            descriptors, driver.DEFAULT_THRESHOLD,
        )
        pending_descriptors = [
            item for item in requested_descriptors
            if (item["state"], item["minimum_id"]) not in completed
        ]
        if limit is not None:
            pending_descriptors = pending_descriptors[:limit]
        predecessor_count = driver.audit_incremental_predecessors(
            pending_descriptors
        )
        requested_ids = ",".join(
            f"{item['state']}:{item['minimum_id']}"
            for item in pending_descriptors
        )
        print(
            f"preflight ready: selected={len(descriptors)}, "
            f"requested={len(requested_descriptors)}, "
            f"compatible_completed={len(completed)}, "
            f"pending_checked={len(pending_descriptors)}, "
            f"directions={driver.TARGET_SUCCESS_COUNT}, "
            f"bisection_iterations={driver.DIRECT_BISECTION_ITERATIONS}, "
            f"output_variant={driver.OUTPUT_VARIANT}",
            flush=True,
        )
        if predecessor_count:
            print(
                f"incremental predecessors validated: {predecessor_count} "
                f"at {driver.PREVIOUS_SUCCESS_COUNT} points from "
                f"{driver.PREVIOUS_OUTPUT_VARIANT}",
                flush=True,
            )
        print(f"pending minima checked: {requested_ids}", flush=True)
        return 0
    driver.run(
        states=states,
        threshold=driver.DEFAULT_THRESHOLD,
        workers=min(args.workers, os.cpu_count() or args.workers),
        limit=limit,
        benchmark_per_state=benchmark_per_state,
        benchmark_round=args.benchmark_round,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
