"""Stage 2: cluster saved minima by incoming polarization.

Edit the explicit globals below, then run

    python GradientPhaseSpaceCluster.py
"""

import sys

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


SCANS_TO_RUN = ("W", "GHZ")
LEPTONS_TO_CLUSTER = ("electron", "muon")
# The objective cut is measured above the global minimum. Narrow capture bands
# retain the alpha_e = pi/4 and 3pi/4 line clusters. Points outside those bands
# fall into the periodic 0/pi endpoint or pi/2 stratum.
# The cluster assignment is the complete selection used by the configuration
# stage; every retained minimum is configured.
POLARIZATION_CLUSTER_CUT = 0.05
POLARIZATION_CLUSTER_COUNT = 6
POLARIZATION_CLUSTER_SEED = 314159
POLARIZATION_ALPHA_E_LINE_HALF_WIDTH = np.pi / 24.0


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
            polarization_alpha_e_line_half_width=(
                POLARIZATION_ALPHA_E_LINE_HALF_WIDTH
            ),
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
