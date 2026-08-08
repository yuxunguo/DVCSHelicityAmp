"""Regenerate Cartesian-W correlation scatters with the prior W example."""

from pathlib import Path

import numpy as np

import CartesianGradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


LEPTON_NAME = "electron"
OBJECTIVE_CUT = 0.01
CLUSTER_COUNT = 6
CLUSTER_SEED = 314159
ALPHA_E_LINE_HALF_WIDTH = np.pi / 24.0
ANGULAR_ROOT = Path("Output") / "GradientPhaseSpaceScan"


def main():
    definition = definitions.SCAN_DEFINITIONS["W"]
    gradient_tool.configure_scan(
        definition,
        leptons_to_process=(LEPTON_NAME,),
        gradient_workers=1,
    )
    output_dirs = gradient_tool.species_output_dirs(LEPTON_NAME)
    minima_path = output_dirs["scan_data"] / "local_minima.csv"
    minimum_rows = gradient_tool._read_csv(minima_path)
    clustered_rows, clusters, optimum = (
        gradient_tool._cluster_polarization_minima(
            minimum_rows,
            LEPTON_NAME,
            objective_cut=OBJECTIVE_CUT,
            cluster_count=CLUSTER_COUNT,
            random_seed=CLUSTER_SEED,
            alpha_e_line_half_width=ALPHA_E_LINE_HALF_WIDTH,
            alpha_e_boundaries=None,
        )
    )

    old_clustered_path = (
        ANGULAR_ROOT / LEPTON_NAME / "Data" / "W" / "cluster"
        / "clustered_minima.csv"
    )
    old_rows = gradient_tool._read_csv(old_clustered_path)
    prior_example = next(
        row for row in old_rows
        if int(row["polarization_cluster_id"])
        == gradient_tool.EXAMPLE_POLARIZATION_CLUSTER_IDS["W"]
        and gradient_tool._as_bool(row["polarization_cluster_representative"])
    )

    cluster_dir = output_dirs["cluster_data"]
    gradient_tool._write_csv(cluster_dir / "clustered_minima.csv", clustered_rows)
    gradient_tool._write_csv(cluster_dir / "polarization_clusters.csv", clusters)
    output_dir = (
        output_dirs["plots"] / "polarization_correlations"
        / "without_contours"
    )
    index_rows = gradient_tool._write_polarization_correlation_pdfs(
        clustered_rows,
        clusters,
        LEPTON_NAME,
        OBJECTIVE_CUT,
        optimum,
        ALPHA_E_LINE_HALF_WIDTH,
        None,
        output_dir,
        include_contours=False,
        example_row_override=prior_example,
    )
    for row in index_rows:
        row["example_source"] = "prior angular W P4 representative"
        row["example_local_minimum_id"] = prior_example["local_minimum_id"]
    index_path = gradient_tool._write_csv(
        cluster_dir / "polarization_correlation_plot_index.csv",
        index_rows,
    )
    print(f"Cartesian W minima: {len(minimum_rows)}")
    print(
        "Retained scatter points: "
        f"{sum(gradient_tool._as_bool(row['within_polarization_cluster_cut']) for row in clustered_rows)}"
    )
    print(
        "Preserved example: angular W local_minimum_id="
        f"{prior_example['local_minimum_id']} (P4)"
    )
    print(f"Correlation plots: {output_dir}")
    print(f"Plot index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
