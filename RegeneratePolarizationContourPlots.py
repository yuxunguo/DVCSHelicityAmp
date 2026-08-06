"""Regenerate one-page polarization phase-space contour PDFs.

This plot-only stage reads the existing clustered minima and validated,
minimum-owned contour CSVs.  It does not rerun minimization, contour
generation, clustering, or ConfigGen.
"""

import sys

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


SCANS_TO_PLOT = ("CEP", "CPGAMMA", "CEGAMMA")
LEPTONS_TO_PLOT = ("electron", "muon")
POLARIZATION_CLUSTER_CUT = 0.05
GHZ_ALPHA_E_BOUNDARIES = (0.0, np.pi / 2.0, np.pi)


def _cluster_guides(scan_key):
    if scan_key == "CEP":
        return None, GHZ_ALPHA_E_BOUNDARIES
    if scan_key in ("CPGAMMA", "CEGAMMA"):
        return None, None
    raise ValueError(f"No contour-plot setup for {scan_key!r}.")


def regenerate_selected_plots():
    """Write the full and per-cluster PDFs for every selected data set."""
    scan_definitions = definitions.selected_definitions(SCANS_TO_PLOT)
    leptons = definitions.validated_leptons(LEPTONS_TO_PLOT)
    written = []
    for definition in scan_definitions:
        gradient_tool.configure_scan(
            definition,
            leptons_to_process=leptons,
            gradient_workers=1,
        )
        alpha_e_line_half_width, alpha_e_boundaries = _cluster_guides(
            definition.key
        )
        for lepton_name in leptons:
            output_dirs = gradient_tool.species_output_dirs(lepton_name)
            clustered_rows = gradient_tool._read_csv(
                output_dirs["cluster_data"] / "clustered_minima.csv"
            )
            polarization_clusters = gradient_tool._read_csv(
                output_dirs["cluster_data"] / "polarization_clusters.csv"
            )
            objective_key = gradient_tool._objective_key(lepton_name)
            optimum = min(
                float(row[objective_key]) for row in clustered_rows
            )
            output_path = (
                output_dirs["plots"]
                / "polarization_cluster_phase_space.pdf"
            )
            print("=" * 72, flush=True)
            print(
                f"Regenerating {definition.key} {lepton_name} contour "
                f"phase-space PDFs",
                flush=True,
            )
            gradient_tool._polarization_cluster_plot(
                clustered_rows,
                polarization_clusters,
                lepton_name,
                POLARIZATION_CLUSTER_CUT,
                optimum,
                alpha_e_line_half_width,
                alpha_e_boundaries,
                output_path,
            )
            cluster_paths = [
                output_path.with_name(
                    f"{output_path.stem}_P{int(cluster['polarization_cluster_id']) + 1:02d}"
                    f"{output_path.suffix}"
                )
                for cluster in polarization_clusters
            ]
            written.extend((output_path, *cluster_paths))
            print(
                f"Wrote {1 + len(cluster_paths)} one-page PDFs under "
                f"{output_dirs['plots']}",
                flush=True,
            )
    return tuple(written)


def main():
    if len(sys.argv) != 1:
        raise SystemExit(
            "RegeneratePolarizationContourPlots.py accepts no command-line "
            "arguments; edit its explicit controls instead."
        )
    regenerate_selected_plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
