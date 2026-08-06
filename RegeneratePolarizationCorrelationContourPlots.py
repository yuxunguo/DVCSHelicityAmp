"""Regenerate correlation PDFs both with and without projected contours.

The script covers W, GHZ, and all three pairwise concurrences for electron
and muon.  Each version has separate clustered and unclustered subfolders.
It consumes existing clustered minima and validated contour CSVs.  W/GHZ muon
contour versions are intentionally skipped because their species-coordinate
repair was explicitly aborted; their contour-free versions are still written.
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


def _example_configuration_path(output_dirs):
    """Return the existing shared example configuration for index rows."""
    index_path = (
        output_dirs["plots"]
        / "polarization_correlations"
        / "unclustered"
        / "representative_configuration_index.csv"
    )
    rows = gradient_tool._read_csv(index_path)
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one example configuration row in {index_path}, "
            f"found {len(rows)}."
        )
    return rows[0]["configuration_path"]


def _existing_variant_index_rows(
    clustered_rows,
    polarization_clusters,
    objective_cut,
    optimum,
    variant_name,
    variant_dir,
):
    """Build index rows for an already-rendered correlation variant."""
    selected_rows = [
        row for row in clustered_rows
        if gradient_tool._as_bool(row["within_polarization_cluster_cut"])
    ]
    representative_rows = [
        row for row in selected_rows
        if gradient_tool._as_bool(
            row["polarization_cluster_representative"]
        )
    ]
    example_cluster_id = gradient_tool.EXAMPLE_POLARIZATION_CLUSTER_IDS[
        gradient_tool.SCAN_KEY
    ]
    rows = []
    for mode in ("clustered", "unclustered"):
        mode_dir = variant_dir / mode
        representative_count = (
            len(representative_rows) if mode == "clustered" else 1
        )
        panel_definitions = [
            (0, "", "", "", "", f"00_summary_{gradient_tool.SCAN_KEY}.pdf")
        ]
        panel_definitions.extend(
            (
                panel_index,
                x_name,
                y_name,
                x_label,
                y_label,
                (
                    f"{panel_index:02d}_{y_name}_vs_{x_name}_{mode}_"
                    f"{gradient_tool.SCAN_KEY}.pdf"
                ),
            )
            for panel_index, (x_name, y_name, x_label, y_label)
            in enumerate(
                gradient_tool.POLARIZATION_CORRELATION_PANELS,
                start=1,
            )
        )
        for (
            panel_index,
            x_name,
            y_name,
            x_label,
            y_label,
            filename,
        ) in panel_definitions:
            path = mode_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing correlation plot required by index: {path}"
                )
            rows.append(
                {
                    "panel_index": panel_index,
                    "mode": mode,
                    "x_name": x_name,
                    "y_name": y_name,
                    "x_label": x_label,
                    "y_label": y_label,
                    "retained_minima": len(selected_rows),
                    "polarization_clusters": len(polarization_clusters),
                    "representative_minima": representative_count,
                    "example_polarization_cluster": (
                        f"P{example_cluster_id + 1}"
                        if mode == "unclustered" else ""
                    ),
                    "objective_name": gradient_tool.OBJECTIVE_NAME,
                    "objective_cut_above_global_minimum": objective_cut,
                    "global_minimum": optimum,
                    "plot_path": str(path),
                    "contour_version": variant_name,
                }
            )
    return rows


def regenerate_selected_plots(*, index_only=False):
    """Write separate contour and contour-free correlation PDF trees."""
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
            dataset_rows = []
            contours_available = _include_contours(
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
            example_configuration_path = _example_configuration_path(
                output_dirs
            )
            variants = [("without_contours", False)]
            if contours_available:
                variants.insert(0, ("with_contours", True))
            else:
                print("=" * 72, flush=True)
                print(
                    f"Skipping {definition.key} {lepton_name} "
                    "with_contours: validated species-coordinate contours "
                    "are unavailable after the aborted repair.",
                    flush=True,
                )
            for variant_name, include_contours in variants:
                variant_dir = (
                    output_dirs["plots"]
                    / "polarization_correlations"
                    / variant_name
                )
                if index_only:
                    rows = _existing_variant_index_rows(
                        clustered_rows,
                        polarization_clusters,
                        POLARIZATION_CLUSTER_CUT,
                        optimum,
                        variant_name,
                        variant_dir,
                    )
                else:
                    print("=" * 72, flush=True)
                    print(
                        f"Regenerating {definition.key} {lepton_name} "
                        "clustered and unclustered correlation PDFs "
                        f"({variant_name})",
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
                        variant_dir,
                        include_contours=include_contours,
                    )
                for row in rows:
                    row["contour_version"] = variant_name
                    row["example_configuration_path"] = (
                        example_configuration_path
                        if row["mode"] == "unclustered" else ""
                    )
                dataset_rows.extend(rows)
                if not index_only:
                    print(
                        f"Wrote {len(rows)} {variant_name} correlation PDFs "
                        f"under {variant_dir}",
                        flush=True,
                    )
            index_path = gradient_tool._write_csv(
                output_dirs["cluster_data"]
                / "polarization_correlation_plot_index.csv",
                dataset_rows,
            )
            written_rows.extend(dataset_rows)
            print(
                f"Wrote {len(dataset_rows)} correlation index rows to "
                f"{index_path}",
                flush=True,
            )
    return tuple(written_rows)


def main():
    arguments = sys.argv[1:]
    if arguments not in ([], ["--index-only"]):
        raise SystemExit(
            "usage: RegeneratePolarizationCorrelationContourPlots.py "
            "[--index-only]"
        )
    regenerate_selected_plots(index_only=arguments == ["--index-only"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
