"""Collect canonical-component gradient configurations into state PDFs.

The source configuration and amplitude CSVs are produced by
``GradientPhaseSpaceConfig.py``.  A canonical display configuration is defined
here by the number of retained final-state components: three for W and two for
GHZ, using ConfigGen's existing per-component fraction cutoff.
"""

from collections import Counter
import csv
from pathlib import Path
import textwrap

import ComptonHelicityRatioVsT as helicity_analysis
import ConfigGen as config_gen
import GradientPhaseSpaceDefinitions as definitions
import PhaseSpaceConfigScan as config_scan


STATES_TO_COLLECT = ("GHZ", "W")
LEPTONS_TO_COLLECT = ("electron",)
CANONICAL_COMPONENT_COUNTS = {
    "GHZ": 2,
    "W": 3,
}
PROGRESS_INTERVAL = 25
GHZ_HELICITY_MECHANISMS = (
    "helicity_flipping",
    "helicity_conserving",
)
HELICITY_FLIP_DOMINANCE_THRESHOLD = 1.0


def _isolated_power_fractions(first, second):
    """Return two normalized isolated powers, excluding their interference."""
    first = float(first)
    second = float(second)
    total = first + second
    if total <= 0.0:
        return float("nan"), float("nan")
    return first / total, second / total


def _annotate_ghz_helicity(row, cluster_number):
    """Attach complete-chain helicity and virtual-photon diagnostics."""
    annotated = dict(row)
    diagnostic = helicity_analysis.compton_ratio_row(
        {"polarization_cluster_number": cluster_number},
        row,
    )
    annotated.update(diagnostic)
    flip_fraction, conserving_fraction = _isolated_power_fractions(
        diagnostic["full_chain_flip_norm2"],
        diagnostic["full_chain_conserving_norm2"],
    )
    transverse_fraction, longitudinal_fraction = _isolated_power_fractions(
        diagnostic["full_chain_transverse_norm2"],
        diagnostic["full_chain_longitudinal_norm2"],
    )
    flip_ratio = float(
        diagnostic["full_chain_flip_to_conserving_ratio"]
    )
    annotated.update({
        "helicity_mechanism": (
            "helicity_flipping"
            if flip_ratio > HELICITY_FLIP_DOMINANCE_THRESHOLD
            else "helicity_conserving"
        ),
        "helicity_flip_fraction": flip_fraction,
        "helicity_conserving_fraction": conserving_fraction,
        "virtual_photon_transverse_fraction": transverse_fraction,
        "virtual_photon_longitudinal_fraction": longitudinal_fraction,
        "helicity_interference_relative": (
            float(
                diagnostic[
                    "full_chain_flip_conserving_interference"
                ]
            )
            / (
                float(diagnostic["full_chain_flip_norm2"])
                + float(diagnostic["full_chain_conserving_norm2"])
            )
        ),
        "virtual_photon_interference_relative": (
            float(
                diagnostic[
                    "full_chain_longitudinal_transverse_interference"
                ]
            )
            / (
                float(diagnostic["full_chain_transverse_norm2"])
                + float(diagnostic["full_chain_longitudinal_norm2"])
            )
        ),
    })
    return annotated


def _state_roots(definition, lepton_name):
    """Return the state/species data and plot roots."""
    species_root = Path(definition.output_root) / lepton_name
    return (
        species_root / "Data" / definition.key / definition.file_tag,
        species_root / "Plots" / definition.key,
    )


def _cluster_package_paths(definition, lepton_name):
    """Return paired configuration/amplitude CSV paths in cluster order."""
    data_root, _plot_root = _state_roots(definition, lepton_name)
    paths = []
    for cluster_dir in sorted(data_root.glob("polarization_cluster_*")):
        combined_dir = cluster_dir / "combined"
        prefix = f"min_{definition.file_tag}_{cluster_dir.name}"
        configuration_path = (
            combined_dir / f"{prefix}_configuration_examples.csv"
        )
        amplitude_path = (
            combined_dir
            / f"{prefix}_final_state_amplitude_decomposition.csv"
        )
        if not configuration_path.exists() or not amplitude_path.exists():
            raise FileNotFoundError(
                "Missing canonical-collection input pair: "
                f"{configuration_path}, {amplitude_path}"
            )
        paths.append((configuration_path, amplitude_path))
    if not paths:
        raise FileNotFoundError(
            f"No polarization-cluster configuration packages under {data_root}."
        )
    return paths


def _canonical_cluster_packages(definition, lepton_name, component_count):
    """Load canonical rows independently for every polarization cluster."""
    cluster_packages = {}
    seen_detail_ids = set()
    for configuration_path, amplitude_path in _cluster_package_paths(
        definition,
        lepton_name,
    ):
        configurations = config_gen.read_csv_rows(configuration_path)
        amplitudes = config_gen.read_csv_rows(amplitude_path)
        counts = Counter(row["detail_id"] for row in amplitudes)
        configuration_ids = {row["detail_id"] for row in configurations}
        unknown_ids = set(counts) - configuration_ids
        if unknown_ids:
            raise ValueError(
                f"{amplitude_path} contains unknown detail IDs: "
                f"{sorted(unknown_ids)[:5]}"
            )
        cluster_name = configuration_path.parent.parent.name
        cluster_number = int(cluster_name.rsplit("_", 1)[1])
        row_cluster_numbers = {
            _cluster_number(row) for row in configurations
        }
        if row_cluster_numbers != {cluster_number}:
            raise ValueError(
                f"{configuration_path} has cluster IDs "
                f"{sorted(row_cluster_numbers)}, expected P{cluster_number}."
            )
        canonical_rows = []
        component_counts = []
        for row in configurations:
            detail_id = row["detail_id"]
            if detail_id in seen_detail_ids:
                raise ValueError(f"Duplicate configuration detail_id {detail_id}.")
            seen_detail_ids.add(detail_id)
            retained_count = counts.get(detail_id, 0)
            component_counts.append(retained_count)
            if retained_count == component_count:
                canonical_rows.append(row)
        if definition.key == "GHZ":
            canonical_rows = [
                _annotate_ghz_helicity(row, cluster_number)
                for row in canonical_rows
            ]
        cluster_packages[cluster_number] = {
            "rows": canonical_rows,
            "total_configurations": len(configurations),
            "component_count_distribution": dict(
                sorted(Counter(component_counts).items())
            ),
            "configuration_path": configuration_path,
            "amplitude_path": amplitude_path,
        }
    return cluster_packages


def _cluster_number(row):
    """Return a one-based polarization-cluster number."""
    return int(float(row["polarization_cluster_id"])) + 1


def _output_paths(definition, lepton_name, cluster_number):
    """Return colocated canonical PDF and CSV-index paths for one cluster."""
    data_root, plot_root = _state_roots(definition, lepton_name)
    species_label = lepton_name.title()
    state_label = definition.state_file_label
    cluster_name = f"polarization_cluster_{cluster_number:02d}"
    cluster_label = f"Polarization_Cluster_{cluster_number:02d}"
    return (
        plot_root
        / cluster_name
        / (
            f"Canonical_{state_label}_Configurations_{species_label}_"
            f"{cluster_label}.pdf"
        ),
        data_root
        / cluster_name
        / "combined"
        / (
            f"canonical_{definition.file_tag}_{cluster_name}_"
            f"configurations_{lepton_name}.csv"
        ),
    )


def _helicity_output_paths(
    definition,
    lepton_name,
    cluster_number,
    mechanism,
):
    """Return PDF/manifest paths for one GHZ helicity-mechanism subset."""
    if mechanism not in GHZ_HELICITY_MECHANISMS:
        raise ValueError(f"Unknown GHZ helicity mechanism {mechanism!r}.")
    data_root, plot_root = _state_roots(definition, lepton_name)
    species_label = lepton_name.title()
    state_label = definition.state_file_label
    cluster_name = f"polarization_cluster_{cluster_number:02d}"
    cluster_label = f"Polarization_Cluster_{cluster_number:02d}"
    mechanism_label = "_".join(
        word.title() for word in mechanism.split("_")
    )
    return (
        plot_root
        / cluster_name
        / mechanism
        / (
            f"Canonical_{state_label}_Configurations_{species_label}_"
            f"{cluster_label}_{mechanism_label}.pdf"
        ),
        data_root
        / cluster_name
        / "combined"
        / mechanism
        / (
            f"canonical_{definition.file_tag}_{cluster_name}_"
            f"{mechanism}_configurations_{lepton_name}.csv"
        ),
    )


def _write_manifest(path, definition, lepton_name, rows, component_count):
    """Write an auditable index of every configuration included in the PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "canonical_index",
        "state",
        "lepton",
        "retained_component_count",
        "polarization_cluster_number",
        "detail_id",
        "selected_observable",
        "selected_concurrence",
        "selected_purity",
        "alpha_e",
        "alpha_p",
        "sqrt_s",
        "qOut",
        "theta_p_out",
        "theta_gamma_out",
        "phi_p_out",
        "phi_gamma_out",
        "helicity_mechanism",
        "full_chain_flip_to_conserving_ratio",
        "helicity_flip_fraction",
        "helicity_conserving_fraction",
        "full_chain_longitudinal_to_transverse_ratio",
        "virtual_photon_transverse_fraction",
        "virtual_photon_longitudinal_fraction",
        "helicity_interference_relative",
        "virtual_photon_interference_relative",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({
                "canonical_index": index,
                "state": definition.key,
                "lepton": lepton_name,
                "retained_component_count": component_count,
                "polarization_cluster_number": _cluster_number(row),
                "detail_id": row["detail_id"],
                "selected_observable": row["selected_observable"],
                "selected_concurrence": row["selected_concurrence"],
                "selected_purity": row["selected_purity"],
                "alpha_e": row["alpha_e"],
                "alpha_p": row["alpha_p"],
                "sqrt_s": row["sqrt_s"],
                "qOut": row["qOut"],
                "theta_p_out": row["theta_p_out"],
                "theta_gamma_out": row["theta_gamma_out"],
                "phi_p_out": row["phi_p_out"],
                "phi_gamma_out": row["phi_gamma_out"],
                "helicity_mechanism": row.get("helicity_mechanism", ""),
                "full_chain_flip_to_conserving_ratio": row.get(
                    "full_chain_flip_to_conserving_ratio",
                    "",
                ),
                "helicity_flip_fraction": row.get(
                    "helicity_flip_fraction",
                    "",
                ),
                "helicity_conserving_fraction": row.get(
                    "helicity_conserving_fraction",
                    "",
                ),
                "full_chain_longitudinal_to_transverse_ratio": row.get(
                    "full_chain_longitudinal_to_transverse_ratio",
                    "",
                ),
                "virtual_photon_transverse_fraction": row.get(
                    "virtual_photon_transverse_fraction",
                    "",
                ),
                "virtual_photon_longitudinal_fraction": row.get(
                    "virtual_photon_longitudinal_fraction",
                    "",
                ),
                "helicity_interference_relative": row.get(
                    "helicity_interference_relative",
                    "",
                ),
                "virtual_photon_interference_relative": row.get(
                    "virtual_photon_interference_relative",
                    "",
                ),
            })
    return path


def _write_cover_page(
    pdf,
    plt,
    definition,
    lepton_name,
    cluster_number,
    package,
    component_count,
    manifest_path,
    mechanism=None,
):
    """Write one summary page describing the canonical selection."""
    rows = package["rows"]
    lines = [
        f"species: {lepton_name}",
        f"polarization cluster: P{cluster_number}",
        f"source cluster configurations: {package['total_configurations']}",
        f"canonical configurations collected: {len(rows)}",
        "",
        (
            "canonical display criterion: exactly "
            f"{component_count} retained final-state components"
        ),
        (
            "retained-component cutoff: fraction >= "
            f"{config_gen.AMPLITUDE_MIN_FRACTION:.1%}"
        ),
        (
            "normalized amplitude display: bar length = |A_tilde|, "
            "bar color/text = phase"
        ),
    ]
    if mechanism is not None:
        lines.extend([
            "",
            (
                "helicity-mechanism subset: "
                f"{mechanism.replace('_', ' ')}"
            ),
            (
                "classification: complete-chain "
                "R_flip/conserve "
                f"{'>' if mechanism == 'helicity_flipping' else '<='} "
                f"{HELICITY_FLIP_DOMINANCE_THRESHOLD:g}"
            ),
            (
                "per-page labels: isolated flip/conserve and T/L powers; "
                "coherent cross terms are reported separately"
            ),
        ])
    lines.extend(["", "cluster retained-component count distribution:"])
    lines.extend(
        f"  {count} components: {frequency} configurations"
        for count, frequency in package[
            "component_count_distribution"
        ].items()
    )
    lines.extend(["", "CSV index:"])
    lines.extend(
        textwrap.wrap(
            str(manifest_path),
            width=88,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    lines.append("")
    if rows:
        lines.append(
            "Each following page is one complete momentum, kinematic-summary, "
            "and normalized-amplitude configuration."
        )
    else:
        lines.append(
            "No configuration in this polarization cluster satisfies the "
            "canonical retained-component criterion."
        )
    fig, ax = plt.subplots(figsize=(15.5, 11.0), constrained_layout=True)
    ax.axis("off")
    ax.set_title(
        (
            f"Canonical {definition.state_file_label} configurations - "
            f"cluster P{cluster_number}"
            + (
                f" - {mechanism.replace('_', ' ')}"
                if mechanism is not None
                else ""
            )
        ),
        fontsize=24,
        pad=18,
    )
    ax.text(
        0.05,
        0.94,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=15,
        linespacing=1.35,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _render_collection(
    definition,
    lepton_name,
    cluster_number,
    package,
    component_count,
    plt,
    PdfPages,
    mechanism=None,
):
    """Render one combined or helicity-split canonical collection."""
    rows = package["rows"]
    if mechanism is None:
        pdf_path, manifest_path = _output_paths(
            definition,
            lepton_name,
            cluster_number,
        )
    else:
        pdf_path, manifest_path = _helicity_output_paths(
            definition,
            lepton_name,
            cluster_number,
            mechanism,
        )
    _write_manifest(
        manifest_path,
        definition,
        lepton_name,
        rows,
        component_count,
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    mechanism_text = (
        f" [{mechanism.replace('_', ' ')}]"
        if mechanism is not None
        else ""
    )
    print(
        f"Collecting {len(rows)} canonical {definition.key} "
        f"configurations for {lepton_name} P{cluster_number}"
        f"{mechanism_text} -> {pdf_path}",
        flush=True,
    )
    with PdfPages(pdf_path) as pdf:
        _write_cover_page(
            pdf,
            plt,
            definition,
            lepton_name,
            cluster_number,
            package,
            component_count,
            manifest_path,
            mechanism=mechanism,
        )
        for index, row in enumerate(rows, start=1):
            config_scan._save_mixing_detail_pages(pdf, plt, [row])
            if index % PROGRESS_INTERVAL == 0 or index == len(rows):
                print(f"  rendered {index}/{len(rows)}", flush=True)
    if not rows:
        print("  no matching configurations; wrote cover only", flush=True)
    return {
        "state": definition.key,
        "lepton": lepton_name,
        "cluster_number": cluster_number,
        "canonical_count": len(rows),
        "component_count": component_count,
        "helicity_mechanism": mechanism or "combined",
        "pdf_path": pdf_path,
        "manifest_path": manifest_path,
    }


def collect_state(definition, lepton_name):
    """Create combined and GHZ-helicity-split PDFs per polarization cluster."""
    component_count = CANONICAL_COMPONENT_COUNTS[definition.key]
    packages = _canonical_cluster_packages(
        definition,
        lepton_name,
        component_count,
    )
    plt, PdfPages = config_gen._require_matplotlib()
    reports = []
    for cluster_number, package in sorted(packages.items()):
        reports.append(_render_collection(
            definition,
            lepton_name,
            cluster_number,
            package,
            component_count,
            plt,
            PdfPages,
        ))
        if definition.key != "GHZ":
            continue
        for mechanism in GHZ_HELICITY_MECHANISMS:
            split_package = dict(package)
            split_package["rows"] = [
                row for row in package["rows"]
                if row["helicity_mechanism"] == mechanism
            ]
            reports.append(_render_collection(
                definition,
                lepton_name,
                cluster_number,
                split_package,
                component_count,
                plt,
                PdfPages,
                mechanism=mechanism,
            ))
    return reports


def run_collections():
    """Collect every explicitly selected state/species pair."""
    scan_definitions = definitions.selected_definitions(STATES_TO_COLLECT)
    leptons = definitions.validated_leptons(LEPTONS_TO_COLLECT)
    reports = []
    for definition in scan_definitions:
        for lepton_name in leptons:
            reports.extend(collect_state(definition, lepton_name))
    return reports


def main():
    """Run the explicit canonical-configuration collection controls."""
    reports = run_collections()
    print("Canonical configuration collection complete.", flush=True)
    for report in reports:
        print(
            f"  {report['state']} ({report['lepton']}) "
            f"P{report['cluster_number']}: "
            f"{report['helicity_mechanism']}, "
            f"{report['canonical_count']} configs, "
            f"{report['component_count']} components -> "
            f"{report['pdf_path']}",
            flush=True,
        )
        print(f"    index: {report['manifest_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
