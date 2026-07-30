"""Collect canonical-component gradient configurations into state PDFs.

The source configuration and amplitude CSVs are produced by
``GradientPhaseSpaceConfig.py``.  A canonical display configuration is defined
here by the number of retained final-state components: three for W and two for
GHZ, using ConfigGen's existing per-component fraction cutoff.
"""

from collections import Counter
import csv
from pathlib import Path

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


def _canonical_rows(definition, lepton_name, component_count):
    """Load all configuration rows having exactly ``component_count`` records."""
    canonical_rows = []
    all_component_counts = Counter()
    total_configurations = 0
    seen_detail_ids = set()
    source_paths = []
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
        total_configurations += len(configurations)
        all_component_counts.update(counts.values())
        for row in configurations:
            detail_id = row["detail_id"]
            if detail_id in seen_detail_ids:
                raise ValueError(f"Duplicate configuration detail_id {detail_id}.")
            seen_detail_ids.add(detail_id)
            if counts.get(detail_id, 0) == component_count:
                canonical_rows.append(row)
        source_paths.extend((configuration_path, amplitude_path))
    if not canonical_rows:
        raise ValueError(
            f"No {definition.key} configurations have exactly "
            f"{component_count} retained components."
        )
    return {
        "rows": canonical_rows,
        "total_configurations": total_configurations,
        "component_count_distribution": dict(sorted(all_component_counts.items())),
        "source_paths": source_paths,
    }


def _cluster_number(row):
    """Return a one-based polarization-cluster number."""
    return int(float(row["polarization_cluster_id"])) + 1


def _output_paths(definition, lepton_name):
    """Return the canonical PDF and CSV-index paths."""
    data_root, plot_root = _state_roots(definition, lepton_name)
    species_label = lepton_name.title()
    state_label = definition.state_file_label
    return (
        plot_root
        / f"Canonical_{state_label}_Configurations_{species_label}.pdf",
        data_root
        / "canonical"
        / f"canonical_{definition.file_tag}_configurations_{lepton_name}.csv",
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
            })
    return path


def _write_cover_page(
    pdf,
    plt,
    definition,
    lepton_name,
    package,
    component_count,
    manifest_path,
):
    """Write one summary page describing the canonical selection."""
    rows = package["rows"]
    cluster_counts = Counter(_cluster_number(row) for row in rows)
    lines = [
        f"species: {lepton_name}",
        f"source configurations: {package['total_configurations']}",
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
        "",
        "canonical configurations by polarization cluster:",
    ]
    lines.extend(
        f"  P{cluster_number}: {count}"
        for cluster_number, count in sorted(cluster_counts.items())
    )
    lines.extend([
        "",
        "all retained-component count distribution:",
    ])
    lines.extend(
        f"  {count} components: {frequency} configurations"
        for count, frequency in package[
            "component_count_distribution"
        ].items()
    )
    lines.extend([
        "",
        f"CSV index: {manifest_path}",
        "",
        (
            "Each following page is one complete momentum, kinematic-summary, "
            "and normalized-amplitude configuration."
        ),
    ])
    fig, ax = plt.subplots(figsize=(15.5, 11.0), constrained_layout=True)
    ax.axis("off")
    ax.set_title(
        f"Canonical {definition.state_file_label} configurations",
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


def collect_state(definition, lepton_name):
    """Create one canonical state PDF and companion CSV index."""
    component_count = CANONICAL_COMPONENT_COUNTS[definition.key]
    package = _canonical_rows(definition, lepton_name, component_count)
    rows = package["rows"]
    pdf_path, manifest_path = _output_paths(definition, lepton_name)
    _write_manifest(
        manifest_path,
        definition,
        lepton_name,
        rows,
        component_count,
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt, PdfPages = config_gen._require_matplotlib()
    print(
        f"Collecting {len(rows)} canonical {definition.key} configurations "
        f"for {lepton_name} -> {pdf_path}",
        flush=True,
    )
    with PdfPages(pdf_path) as pdf:
        _write_cover_page(
            pdf,
            plt,
            definition,
            lepton_name,
            package,
            component_count,
            manifest_path,
        )
        for index, row in enumerate(rows, start=1):
            config_scan._save_mixing_detail_pages(pdf, plt, [row])
            if index % PROGRESS_INTERVAL == 0 or index == len(rows):
                print(f"  rendered {index}/{len(rows)}", flush=True)
    return {
        "state": definition.key,
        "lepton": lepton_name,
        "canonical_count": len(rows),
        "component_count": component_count,
        "pdf_path": pdf_path,
        "manifest_path": manifest_path,
    }


def run_collections():
    """Collect every explicitly selected state/species pair."""
    scan_definitions = definitions.selected_definitions(STATES_TO_COLLECT)
    leptons = definitions.validated_leptons(LEPTONS_TO_COLLECT)
    return [
        collect_state(definition, lepton_name)
        for definition in scan_definitions
        for lepton_name in leptons
    ]


def main():
    """Run the explicit canonical-configuration collection controls."""
    reports = run_collections()
    print("Canonical configuration collection complete.", flush=True)
    for report in reports:
        print(
            f"  {report['state']} ({report['lepton']}): "
            f"{report['canonical_count']} configs, "
            f"{report['component_count']} components -> "
            f"{report['pdf_path']}",
            flush=True,
        )
        print(f"    index: {report['manifest_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
