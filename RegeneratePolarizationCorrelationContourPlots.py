"""Regenerate correlation PDFs both with and without projected contours.

The script covers W, GHZ, and all three pairwise concurrences for electron
and muon.  Each version has separate clustered and unclustered subfolders.
It consumes existing clustered minima and validated contour CSVs, but applies
an absolute objective cut to the rows used by the plots.  W/GHZ muon contour
versions are intentionally skipped because their species-coordinate repair was
explicitly aborted; their contour-free versions are still written.
"""

import argparse

import numpy as np

import GradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


SCANS_TO_PLOT = ("W", "GHZ", "CEP", "CPGAMMA", "CEGAMMA")
LEPTONS_TO_PLOT = ("electron", "muon")
POLARIZATION_ABSOLUTE_CUT = 0.01
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


def _apply_absolute_objective_cut(
    clustered_rows,
    polarization_clusters,
    lepton_name,
    objective_cut,
):
    """Select objective <= cut while retaining established cluster labels."""
    objective_key = gradient_tool._objective_key(lepton_name)
    rows = [dict(row) for row in clustered_rows]
    selected_rows = [
        row
        for row in rows
        if float(row[objective_key]) <= objective_cut + 1.0e-12
    ]
    if not selected_rows:
        raise RuntimeError(
            f"No {gradient_tool.OBJECTIVE_NAME} minima satisfy the absolute "
            f"cut <= {objective_cut:g}."
        )
    unassigned_rows = [
        row
        for row in selected_rows
        if str(row.get("polarization_cluster_id", "")).strip() == ""
    ]
    if unassigned_rows:
        raise RuntimeError(
            f"{len(unassigned_rows)} minima satisfying the absolute "
            "objective cut have no saved polarization-cluster assignment."
        )

    for row in rows:
        row["within_polarization_cluster_cut"] = (
            float(row[objective_key]) <= objective_cut + 1.0e-12
        )
        row["polarization_cluster_representative"] = False

    retained_clusters = []
    for source_cluster in sorted(
        polarization_clusters,
        key=lambda item: int(item["polarization_cluster_id"]),
    ):
        cluster = dict(source_cluster)
        cluster_id = int(cluster["polarization_cluster_id"])
        members = [
            row
            for row in selected_rows
            if int(row["polarization_cluster_id"]) == cluster_id
        ]
        if not members:
            continue
        representative = min(
            members,
            key=lambda row: float(row[objective_key]),
        )
        representative["polarization_cluster_representative"] = True
        values = np.asarray(
            [float(row[objective_key]) for row in members],
            dtype=float,
        )
        cluster.update(
            {
                "member_count": len(members),
                "representative_local_minimum_id": representative[
                    "local_minimum_id"
                ],
                "objective_absolute_cut": objective_cut,
                "best_objective": float(values.min()),
                "mean_objective": float(values.mean()),
            }
        )
        cluster.pop("objective_cut_above_global_minimum", None)
        retained_clusters.append(cluster)
    return rows, retained_clusters


def _mark_absolute_cut_metadata(rows, objective_cut):
    """Replace legacy above-minimum cut metadata in plot-index rows."""
    for row in rows:
        row.pop("objective_cut_above_global_minimum", None)
        row["objective_absolute_cut"] = objective_cut
        row["objective_cut_basis"] = "absolute"
    return rows


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
    show_example = gradient_tool._show_unclustered_example()
    rows = []
    for mode in ("clustered", "unclustered"):
        mode_dir = variant_dir / mode
        representative_count = (
            len(representative_rows)
            if mode == "clustered" else int(show_example)
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
                        if mode == "unclustered" and show_example else ""
                    ),
                    "objective_name": gradient_tool.OBJECTIVE_NAME,
                    "objective_absolute_cut": objective_cut,
                    "objective_cut_basis": "absolute",
                    "global_minimum": optimum,
                    "plot_path": str(path),
                    "contour_version": variant_name,
                }
            )
    return rows


def regenerate_selected_plots(
    *,
    index_only=False,
    scan_keys=SCANS_TO_PLOT,
    leptons_to_plot=LEPTONS_TO_PLOT,
    output_folder="polarization_correlations",
    include_ghz_example=False,
):
    """Write separate contour and contour-free correlation PDF trees."""
    if include_ghz_example:
        gradient_tool.UNCLUSTERED_EXAMPLE_COLORS["GHZ"] = "#29B6F6"
    scan_definitions = definitions.selected_definitions(tuple(scan_keys))
    leptons = definitions.validated_leptons(tuple(leptons_to_plot))
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
            clustered_rows, polarization_clusters = (
                _apply_absolute_objective_cut(
                    clustered_rows,
                    polarization_clusters,
                    lepton_name,
                    POLARIZATION_ABSOLUTE_CUT,
                )
            )
            retained = sum(
                gradient_tool._as_bool(
                    row["within_polarization_cluster_cut"]
                )
                for row in clustered_rows
            )
            print(
                f"{definition.key} {lepton_name}: retained {retained}/"
                f"{len(clustered_rows)} minima at absolute "
                f"{gradient_tool.OBJECTIVE_NAME} <= "
                f"{POLARIZATION_ABSOLUTE_CUT:g} in "
                f"{len(polarization_clusters)} nonempty clusters",
                flush=True,
            )
            example_configuration_path = (
                _example_configuration_path(output_dirs)
                if gradient_tool._show_unclustered_example()
                else ""
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
                    / output_folder
                    / variant_name
                )
                if index_only:
                    rows = _existing_variant_index_rows(
                        clustered_rows,
                        polarization_clusters,
                        POLARIZATION_ABSOLUTE_CUT,
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
                        POLARIZATION_ABSOLUTE_CUT,
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
                        if row["mode"] == "unclustered"
                        and gradient_tool._show_unclustered_example()
                        else ""
                    )
                _mark_absolute_cut_metadata(
                    rows,
                    POLARIZATION_ABSOLUTE_CUT,
                )
                dataset_rows.extend(rows)
                if not index_only:
                    print(
                        f"Wrote {len(rows)} {variant_name} correlation PDFs "
                        f"under {variant_dir}",
                        flush=True,
                    )
            if output_folder == "polarization_correlations":
                index_path = (
                    output_dirs["cluster_data"]
                    / "polarization_correlation_plot_index.csv"
                )
            else:
                index_path = (
                    output_dirs["plots"]
                    / output_folder
                    / "polarization_correlation_plot_index.csv"
                )
            index_path = gradient_tool._write_csv(index_path, dataset_rows)
            written_rows.extend(dataset_rows)
            print(
                f"Wrote {len(dataset_rows)} correlation index rows to "
                f"{index_path}",
                flush=True,
            )
    return tuple(written_rows)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument(
        "--state", action="append", choices=SCANS_TO_PLOT,
        help="State to regenerate; repeat for multiple states.",
    )
    parser.add_argument(
        "--lepton", action="append", choices=LEPTONS_TO_PLOT,
        help="Lepton to regenerate; repeat for multiple species.",
    )
    parser.add_argument(
        "--output-folder",
        default="polarization_correlations",
        help="Plot subfolder under the selected state/species plot root.",
    )
    parser.add_argument(
        "--include-ghz-example",
        action="store_true",
        help=(
            "Restore the bright-blue exact-example star in unclustered GHZ "
            "plots; the star remains absent from the legend."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    regenerate_selected_plots(
        index_only=args.index_only,
        scan_keys=tuple(args.state or SCANS_TO_PLOT),
        leptons_to_plot=tuple(args.lepton or LEPTONS_TO_PLOT),
        output_folder=args.output_folder,
        include_ghz_example=args.include_ghz_example,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
