"""Stage 2: cluster saved minima by incoming polarization.

Edit the explicit globals below, then run

    python GradientPhaseSpaceCluster.py
"""

import sys

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


#SCANS_TO_RUN = ("W", "GHZ")
#LEPTONS_TO_CLUSTER = ("electron", "muon", "heavy", "massless")
SCANS_TO_RUN = ("W",)
LEPTONS_TO_CLUSTER = ("electron",)
# The objective cut is measured above the global minimum. Polarization is
# clustered first with alpha_e/alpha_p treated as pi-periodic coordinates.
# The cluster assignment is the complete selection used by the configuration
# stage; every retained minimum is configured.
POLARIZATION_CLUSTER_CUT = 0.05
POLARIZATION_CLUSTER_COUNT = 6
POLARIZATION_CLUSTER_SEED = 314159


def run_selected_clusters():
    """Cluster the selected states and leptons sequentially."""
    scan_definitions = definitions.selected_definitions(SCANS_TO_RUN)
    leptons = definitions.validated_leptons(LEPTONS_TO_CLUSTER)
    reports = {}
    for index, definition in enumerate(scan_definitions, start=1):
        print("=" * 72, flush=True)
        print(
            f"Phase-space clustering {index}/{len(scan_definitions)}: "
            f"{definition.key} ({definition.objective_name})",
            flush=True,
        )
        print("=" * 72, flush=True)
        reports[definition.key] = gradient_tool.run_phase_space_clustering(
            definition,
            leptons_to_cluster=leptons,
            polarization_objective_cut=POLARIZATION_CLUSTER_CUT,
            polarization_cluster_count=POLARIZATION_CLUSTER_COUNT,
            polarization_cluster_seed=POLARIZATION_CLUSTER_SEED,
        )
    return reports


def main():
    """Run stage 2 using only the explicit global controls."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "GradientPhaseSpaceCluster.py accepts no command-line arguments; "
            "edit its explicit global controls instead."
        )
    run_selected_clusters()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
