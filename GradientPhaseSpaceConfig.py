"""Stage 3: configure every minimum in each polarization cluster.

Edit the explicit globals below, then run

    python GradientPhaseSpaceConfig.py
"""

import sys

from config import SCAN_WORKERS


#SCANS_TO_RUN = ("W", "GHZ")
#LEPTONS_TO_CONFIGURE = ("electron", "muon", "heavy", "massless")
SCANS_TO_RUN = ("W", )
LEPTONS_TO_CONFIGURE = ("electron", )
# Contour evaluation uses lightweight workers that do not import SciPy, so all
# configured CPU workers can run without exhausting the Windows paging file.
CONFIG_WORKERS = SCAN_WORKERS
# One-based polarization-cluster numbers. Use None to configure every cluster.
POLARIZATION_CLUSTERS_TO_CONFIGURE = None
SAVE_CONTOUR_DATA = True
USE_SAVED_CONTOUR_DATA = False


def run_selected_configs():
    """Generate all-point configuration PDFs per polarization cluster."""
    import GradientPhaseSpaceDefinitions as definitions
    import GradientPhaseSpaceScanTool as gradient_tool

    scan_definitions = definitions.selected_definitions(SCANS_TO_RUN)
    leptons = definitions.validated_leptons(LEPTONS_TO_CONFIGURE)
    workers = definitions.validated_workers(CONFIG_WORKERS)
    if not isinstance(SAVE_CONTOUR_DATA, bool):
        raise TypeError("SAVE_CONTOUR_DATA must be a bool.")
    if not isinstance(USE_SAVED_CONTOUR_DATA, bool):
        raise TypeError("USE_SAVED_CONTOUR_DATA must be a bool.")
    if POLARIZATION_CLUSTERS_TO_CONFIGURE is not None:
        if (
            not POLARIZATION_CLUSTERS_TO_CONFIGURE
            or any(
                not isinstance(cluster_number, int)
                or isinstance(cluster_number, bool)
                or cluster_number < 1
                for cluster_number in POLARIZATION_CLUSTERS_TO_CONFIGURE
            )
        ):
            raise ValueError(
                "POLARIZATION_CLUSTERS_TO_CONFIGURE must be None or a "
                "nonempty tuple of positive one-based cluster numbers."
            )
    reports = {}
    for index, definition in enumerate(scan_definitions, start=1):
        print("=" * 72, flush=True)
        print(
            f"Cluster ConfigGen {index}/{len(scan_definitions)}: "
            f"{definition.key} ({definition.objective_name})",
            flush=True,
        )
        print("=" * 72, flush=True)
        reports[definition.key] = gradient_tool.run_cluster_configgen(
            definition,
            leptons_to_configure=leptons,
            config_workers=workers,
            polarization_clusters_to_configure=(
                POLARIZATION_CLUSTERS_TO_CONFIGURE
            ),
            save_contour_data=SAVE_CONTOUR_DATA,
            use_saved_contour_data=USE_SAVED_CONTOUR_DATA,
        )
    return reports


def main():
    """Run stage 3 using only the explicit global controls."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "GradientPhaseSpaceConfig.py accepts no command-line arguments; "
            "edit its explicit global controls instead."
        )
    run_selected_configs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
