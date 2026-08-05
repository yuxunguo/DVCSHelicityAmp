"""Repair stale muon pairwise contours using current species coordinates.

This recovery stage validates every saved muon contour against its raw
minimum, reuses matching files, and regenerates only missing or stale files.
"""

import sys

from config import SCAN_WORKERS
import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


SCANS_TO_REPAIR = ("CEP", "CPGAMMA", "CEGAMMA")
CONTOUR_WORKERS = SCAN_WORKERS


def run_repairs():
    """Validate and repair the selected muon contour datasets."""
    reports = {}
    for index, definition in enumerate(
        definitions.selected_definitions(SCANS_TO_REPAIR),
        start=1,
    ):
        print("=" * 72, flush=True)
        print(
            f"Muon contour repair {index}/{len(SCANS_TO_REPAIR)}: "
            f"{definition.key} ({definition.objective_name})",
            flush=True,
        )
        print("=" * 72, flush=True)
        reports[definition.key] = gradient_tool.run_minimum_contours(
            definition,
            leptons_to_contour=("muon",),
            contour_workers=CONTOUR_WORKERS,
            reuse_saved_minima=True,
        )
    return reports


def main():
    """Run the explicit repair selection without command-line arguments."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "RepairPairwiseMuonContours.py accepts no command-line "
            "arguments; edit its explicit controls instead."
        )
    run_repairs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
