"""Stage 1: regenerate electron minima in Cartesian final momenta.

The production defaults are all five entanglement objectives and exactly
2,048 starts per objective.  Validated ``D <= 0.01`` angular minima are used
first; deterministic physical Sobol points fill the remaining start slots.
"""

import argparse

from config import SCAN_WORKERS
import CartesianGradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


STATES = ("W", "GHZ", "CEP", "CEGAMMA", "CPGAMMA")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", choices=STATES)
    parser.add_argument("--workers", type=int, default=SCAN_WORKERS)
    parser.add_argument("--remake-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    workers = definitions.validated_workers(args.workers)
    selected = definitions.selected_definitions(tuple(args.state or STATES))
    for index, definition in enumerate(selected, start=1):
        print("=" * 72, flush=True)
        print(
            f"Cartesian electron minimum scan {index}/{len(selected)}: "
            f"{definition.key} ({definition.objective_name}); "
            f"starts={gradient_tool.CARTESIAN_TOTAL_STARTS}",
            flush=True,
        )
        print("=" * 72, flush=True)
        gradient_tool.run_minima_scan(
            definition,
            leptons_to_scan=("electron",),
            gradient_workers=workers,
            remake_plot_from_csv=args.remake_plots,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
