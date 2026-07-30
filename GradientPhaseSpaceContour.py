"""Stage 2: generate contours for raw minima before clustering.

Edit the explicit globals below, then run

    python GradientPhaseSpaceContour.py

The output is keyed by ``local_minimum_id`` from ``scan/local_minima.csv``.
Changing the later polarization-cluster criteria therefore does not change
which minimum owns a contour.
"""

import sys

from config import SCAN_WORKERS
import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


# The active muon GHZ search must finish first. The queued contour run then
# generates GHZ contours followed by a clean W contour regeneration.
SCANS_TO_RUN = ("GHZ", "W")
LEPTONS_TO_CONTOUR = ("muon",)
CONTOUR_WORKERS = SCAN_WORKERS
# Reuse and validate completed per-minimum files after an interrupted run.
REUSE_SAVED_MINIMUM_CONTOURS = True


def run_selected_contours():
    """Generate raw-minimum contour packages sequentially."""
    scan_definitions = definitions.selected_definitions(SCANS_TO_RUN)
    leptons = definitions.validated_leptons(LEPTONS_TO_CONTOUR)
    workers = definitions.validated_workers(CONTOUR_WORKERS)
    if not isinstance(REUSE_SAVED_MINIMUM_CONTOURS, bool):
        raise TypeError("REUSE_SAVED_MINIMUM_CONTOURS must be a bool.")
    reports = {}
    for index, definition in enumerate(scan_definitions, start=1):
        print("=" * 72, flush=True)
        print(
            f"Raw-minimum contours {index}/{len(scan_definitions)}: "
            f"{definition.key} ({definition.objective_name})",
            flush=True,
        )
        print("=" * 72, flush=True)
        reports[definition.key] = gradient_tool.run_minimum_contours(
            definition,
            leptons_to_contour=leptons,
            contour_workers=workers,
            reuse_saved_minima=REUSE_SAVED_MINIMUM_CONTOURS,
        )
    return reports


def main():
    """Run stage 2 using only the explicit global controls."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "GradientPhaseSpaceContour.py accepts no command-line arguments; "
            "edit its explicit global controls instead."
        )
    run_selected_contours()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
