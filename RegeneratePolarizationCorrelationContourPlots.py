"""Regenerate clustered and unclustered correlation PDFs with contours.

The script covers W, GHZ, and all three pairwise concurrences for electron
and muon.  It consumes existing clustered minima and validated contour CSVs.
W/GHZ muon contours are intentionally omitted because their species-coordinate
repair was explicitly aborted.
"""

import sys

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


SCANS_TO_PLOT = ("W", "GHZ", "CEP", "CPGAMMA", "CEGAMMA")
LEPTONS_TO_PLOT = ("electron", "muon")
POLARIZATION_CLUSTER_CUT = 0.05
W_ALPHA_E_LINE_HALF_WIDTH = np.pi / 24.0
GHZ_ALPHA_E_BOUNDARIES = (0.0, np.pi / 2.0, np.pi)


def _cluster_guides(scan_key):
    if scan_key == "W":
        return W_ALPHA_E_LINE_HALF_WIDTH, None
    if scan_key in ("GHZ", "CEP"):
        return None, GHZ_ALPHA_E_BOUNDARIES
    if scan_key in ("CPGAMMA", "CEGAMMA"):
        return None, None
    raise ValueError(f"No correlation contour setup for {scan_key!r}.")


def _include_contours(scan_key, lepton_name):
    return not (lepton_name == "muon" and scan_key in ("W", "GHZ"))


def regenerate_selected_plots():
    """Write 20 correlation PDFs for every species and measurement."""
    scan_definitions = definitions.selected_definitions(SCANS_TO_PLOT)
    leptons = definitions.validated_leptons(LEPTONS_TO_PLOT)
    written_rows = []
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
            include_contours = _include_contours(
                definition.key, lepton_name
            )
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
            print("=" * 72, flush=True)
            print(
                f"Regenerating {definition.key} {lepton_name} clustered and "
                "unclustered correlation PDFs "
                f"(contours={'yes' if include_contours else 'no'})",
                flush=True,
            )
            rows = gradient_tool._write_polarization_correlation_pdfs(
                clustered_rows,
                polarization_clusters,
                lepton_name,
                POLARIZATION_CLUSTER_CUT,
                optimum,
                alpha_e_line_half_width,
                alpha_e_boundaries,
                output_dirs["plots"] / "polarization_correlations",
                include_contours=include_contours,
            )
            written_rows.extend(rows)
            print(
                f"Wrote {len(rows)} correlation contour PDFs under "
                f"{output_dirs['plots'] / 'polarization_correlations'}",
                flush=True,
            )
    return tuple(written_rows)


def main():
    if len(sys.argv) != 1:
        raise SystemExit(
            "RegeneratePolarizationCorrelationContourPlots.py accepts no "
            "command-line arguments; edit its explicit controls instead."
        )
    regenerate_selected_plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
