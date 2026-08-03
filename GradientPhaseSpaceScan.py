"""Stage 1: find and save every verified entanglement-objective minimum.

Edit the explicit globals below, then run

    python GradientPhaseSpaceScan.py
"""

import sys

from config import SCAN_WORKERS
import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


# Valid keys include W, GHZ, CEP, CEGAMMA, and CPGAMMA.  The concurrence
# searches maximize C by minimizing the saved one_minus_C objective.
SCANS_TO_RUN = ("CEP", "CPGAMMA", "CEGAMMA")
LEPTONS_TO_SCAN = ("electron", "muon")
GRADIENT_WORKERS = SCAN_WORKERS
# True reads scan/local_minima.csv and only rebuilds all_local_minima.pdf.
REMAKE_MINIMA_PLOT_FROM_CSV = False


def run_selected_scans():
    """Run the selected local-minimum searches sequentially."""
    if not isinstance(REMAKE_MINIMA_PLOT_FROM_CSV, bool):
        raise TypeError("REMAKE_MINIMA_PLOT_FROM_CSV must be a bool.")
    scan_definitions = definitions.selected_definitions(SCANS_TO_RUN)
    leptons = definitions.validated_leptons(LEPTONS_TO_SCAN)
    workers = definitions.validated_workers(GRADIENT_WORKERS)
    reports = {}
    for index, definition in enumerate(scan_definitions, start=1):
        print("=" * 72, flush=True)
        print(
            f"Minimum scan {index}/{len(scan_definitions)}: "
            f"{definition.key} ({definition.objective_name})",
            flush=True,
        )
        print("=" * 72, flush=True)
        reports[definition.key] = gradient_tool.run_minima_scan(
            definition,
            leptons_to_scan=leptons,
            gradient_workers=workers,
            remake_plot_from_csv=REMAKE_MINIMA_PLOT_FROM_CSV,
        )
    return reports


def main():
    """Run stage 1 using only the explicit global controls."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "GradientPhaseSpaceScan.py accepts no command-line arguments; "
            "edit its explicit global controls instead."
        )
    run_selected_scans()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
