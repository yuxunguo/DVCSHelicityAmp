"""Regenerate Cartesian correlation scatter plots with absolute D < 0.01."""

import argparse
from pathlib import Path

import numpy as np

import CartesianGradientPhaseSpaceDefinitions as definitions
import GradientPhaseSpaceScanTool as gradient_tool


LEPTON_NAME = "electron"
ABSOLUTE_OBJECTIVE_CUT = 0.01
CLUSTER_SEED = 314159
ANGULAR_ROOT = Path("Output") / "GradientPhaseSpaceScan"
STATE_CLUSTER_SETTINGS = {
    "W": (6, np.pi / 24.0, None),
    "GHZ": (4, None, (0.0, np.pi / 2.0, np.pi)),
    "CEP": (4, None, (0.0, np.pi / 2.0, np.pi)),
    "CEGAMMA": (5, None, None),
    "CPGAMMA": (3, None, None),
}
GHZ_EXAMPLE_COLOR = "#56B4E9"


def _prior_example(state_key):
    path = (
        ANGULAR_ROOT / LEPTON_NAME / "Data" / state_key / "cluster"
        / "clustered_minima.csv"
    )
    rows = gradient_tool._read_csv(path)
    example = dict(next(
        row for row in rows
        if int(row["polarization_cluster_id"])
        == gradient_tool.EXAMPLE_POLARIZATION_CLUSTER_IDS[state_key]
        and gradient_tool._as_bool(row["polarization_cluster_representative"])
    ))
    for angle_name, flag_name in (
        ("theta_p_out", "phi_p_out_defined"),
        ("theta_gamma_out", "phi_gamma_out_defined"),
    ):
        theta = float(example[angle_name])
        example[flag_name] = not (
            np.isclose(theta, 0.0, rtol=0.0, atol=1.0e-12)
            or np.isclose(theta, np.pi, rtol=0.0, atol=1.0e-12)
        )
    return example


def remake_state(state_key):
    definition = definitions.SCAN_DEFINITIONS[state_key]
    cluster_count, alpha_e_half_width, alpha_e_boundaries = (
        STATE_CLUSTER_SETTINGS[state_key]
    )
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
            objective_cut=ABSOLUTE_OBJECTIVE_CUT,
            cluster_count=cluster_count,
            random_seed=CLUSTER_SEED,
            alpha_e_line_half_width=alpha_e_half_width,
            alpha_e_boundaries=alpha_e_boundaries,
        )
    )
    cluster_dir = output_dirs["cluster_data"]
    gradient_tool._write_csv(cluster_dir / "clustered_minima.csv", clustered_rows)
    gradient_tool._write_csv(cluster_dir / "polarization_clusters.csv", clusters)
    output_dir = (
        output_dirs["plots"] / "polarization_correlations"
        / "without_contours"
    )
    example_override = _prior_example("W") if state_key == "W" else None
    index_rows = gradient_tool._write_polarization_correlation_pdfs(
        clustered_rows,
        clusters,
        LEPTON_NAME,
        ABSOLUTE_OBJECTIVE_CUT,
        optimum,
        alpha_e_half_width,
        alpha_e_boundaries,
        output_dir,
        include_contours=False,
        example_row_override=example_override,
    )
    for row in index_rows:
        row["example_source"] = (
            "prior angular W P4 representative"
            if example_override is not None else ""
        )
        row["example_local_minimum_id"] = (
            example_override["local_minimum_id"]
            if example_override is not None else ""
        )
    index_path = gradient_tool._write_csv(
        cluster_dir / "polarization_correlation_plot_index.csv",
        index_rows,
    )
    extra_output = None
    if state_key == "GHZ":
        ghz_example = _prior_example("GHZ")
        extra_output = (
            output_dirs["plots"] / "polarization_correlations"
            / "without_contours_with_example"
        )
        example_index_rows = (
            gradient_tool._write_polarization_correlation_pdfs(
                clustered_rows,
                clusters,
                LEPTON_NAME,
                ABSOLUTE_OBJECTIVE_CUT,
                optimum,
                alpha_e_half_width,
                alpha_e_boundaries,
                extra_output,
                include_contours=False,
                example_row_override=ghz_example,
                show_unclustered_example_override=True,
                unclustered_example_color_override=GHZ_EXAMPLE_COLOR,
            )
        )
        for row in example_index_rows:
            row["example_source"] = "prior angular GHZ P2 representative"
            row["example_local_minimum_id"] = ghz_example["local_minimum_id"]
        gradient_tool._write_csv(
            cluster_dir / "polarization_correlation_plot_index_with_example.csv",
            example_index_rows,
        )
    retained = sum(
        gradient_tool._as_bool(row["within_polarization_cluster_cut"])
        for row in clustered_rows
    )
    print(f"Cartesian {state_key} minima: {len(minimum_rows)}")
    print(
        f"Retained at absolute objective < {ABSOLUTE_OBJECTIVE_CUT:g}: "
        f"{retained}"
    )
    print(f"Correlation plots: {output_dir}")
    if extra_output is not None:
        print(f"Correlation plots with prior example: {extra_output}")
    print(f"Plot index: {index_path}")
    return output_dir


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state", action="append", choices=tuple(STATE_CLUSTER_SETTINGS)
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    for state_key in tuple(args.state or STATE_CLUSTER_SETTINGS):
        remake_state(state_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
