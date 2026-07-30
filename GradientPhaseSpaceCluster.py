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
# The objective cut is measured above the global minimum. W uses narrow
# alpha_e capture bands around pi/4 and 3pi/4. GHZ instead uses the two
# alpha_e regions bounded by 0, pi/2, and pi, with two periodic alpha_p
# clusters in each region.
# The cluster assignment is the complete selection used by the configuration
# stage; every retained minimum is configured.
POLARIZATION_CLUSTER_CUT = 0.05
POLARIZATION_CLUSTER_SEED = 314159
W_POLARIZATION_CLUSTER_COUNT = 6
W_ALPHA_E_LINE_HALF_WIDTH = np.pi / 24.0
GHZ_POLARIZATION_CLUSTER_COUNT = 4
GHZ_ALPHA_E_BOUNDARIES = (0.0, np.pi / 2.0, np.pi)


def run_selected_clusters():
    """Cluster the selected states and leptons sequentially."""
    scan_definitions = definitions.selected_definitions(SCANS_TO_RUN)
    leptons = definitions.validated_leptons(LEPTONS_TO_CLUSTER)
    reports = {}
    for index, definition in enumerate(scan_definitions, start=1):
        if definition.key == "W":
            cluster_count = W_POLARIZATION_CLUSTER_COUNT
            alpha_e_line_half_width = W_ALPHA_E_LINE_HALF_WIDTH
            alpha_e_boundaries = None
        elif definition.key == "GHZ":
            cluster_count = GHZ_POLARIZATION_CLUSTER_COUNT
            alpha_e_line_half_width = None
            alpha_e_boundaries = GHZ_ALPHA_E_BOUNDARIES
        else:
            raise ValueError(
                f"No polarization clustering setup for {definition.key!r}."
            )
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
            polarization_cluster_count=cluster_count,
            polarization_cluster_seed=POLARIZATION_CLUSTER_SEED,
            polarization_alpha_e_line_half_width=alpha_e_line_half_width,
            polarization_alpha_e_boundaries=alpha_e_boundaries,
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
