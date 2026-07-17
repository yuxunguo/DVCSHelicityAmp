"""Generate ConfigGen-style packages from PhaseSpaceScan results.

The continuous PhaseSpaceScan photon-energy coordinate is divided into three
balanced low/mid/high ``E_gamma`` bands before reusing ConfigGen's clustering,
amplitude decomposition, per-polarization CSV, and parallel PDF machinery.
The inherited targets include minimum-distance ``D_W`` and maximum-magic
``M2_magic`` configuration regions.
"""

from pathlib import Path

import numpy as np

import ConfigGen as config
from AlignmentScan import LEPTON_SPECS
from config import SCAN_WORKERS
from PlotUtils import print_console_text


# Script controls. Edit these values before running PhaseSpaceConfigScan.py.
PHASE_SPACE_CONFIG_LEPTONS = ("electron", "muon", "heavy", "massless")
PHASE_SPACE_CONFIG_WORKERS = SCAN_WORKERS
PHASE_SPACE_CONFIG_PLOT_WORKERS = max(1, min(SCAN_WORKERS, 24))
ENERGY_BAND_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
ENERGY_BAND_LABELS = ("low_Egamma", "mid_Egamma", "high_Egamma")
PHASE_SPACE_CONFIG_TARGETS = config.CONFIG_TARGETS

PHASE_SPACE_OUTPUT_ROOT = Path("Output") / "PhaseSpaceScan"
OUTPUT_ROOT = Path("Output") / "PhaseSpaceConfigScan"
LOG_PATH = Path("Output") / "PhaseSpaceConfigScan.log"


def phase_space_input_path(lepton_name):
    """Return the full PhaseSpaceScan CSV for one lepton species."""
    try:
        stem = LEPTON_SPECS[lepton_name]["file_stem"]
    except KeyError as exc:
        raise ValueError(
            f"Unknown lepton {lepton_name!r}; choose from {tuple(LEPTON_SPECS)}."
        ) from exc
    path = (
        PHASE_SPACE_OUTPUT_ROOT
        / lepton_name
        / f"{stem}_entanglement_phase_space.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PhaseSpaceScan input {path}. Run PhaseSpaceScan.py first."
        )
    return path


def assign_energy_bands(rows):
    """Assign balanced low/mid/high E_gamma bands to phase-space rows."""
    qout_values = np.asarray(
        [config.parse_float(row.get("qOut")) for row in rows],
        dtype=float,
    )
    finite = np.isfinite(qout_values)
    if not np.any(finite):
        raise ValueError("PhaseSpaceScan rows contain no finite qOut values.")

    finite_qout = qout_values[finite]
    boundaries = np.quantile(finite_qout, ENERGY_BAND_QUANTILES)
    if boundaries[0] >= boundaries[1]:
        raise ValueError(
            "Cannot form distinct photon-energy bands from the saved qOut values."
        )

    counts = {label: 0 for label in ENERGY_BAND_LABELS}
    for row, qout in zip(rows, qout_values):
        row["phase_space_stage"] = row.get("qOut_regime", "")
        if not np.isfinite(qout):
            row["qOut_regime"] = "invalid_Egamma"
            continue
        if qout <= boundaries[0]:
            label = ENERGY_BAND_LABELS[0]
        elif qout <= boundaries[1]:
            label = ENERGY_BAND_LABELS[1]
        else:
            label = ENERGY_BAND_LABELS[2]
        row["qOut_regime"] = label
        counts[label] += 1

    return {
        "minimum": float(np.min(finite_qout)),
        "lower_boundary": float(boundaries[0]),
        "upper_boundary": float(boundaries[1]),
        "maximum": float(np.max(finite_qout)),
        "counts": counts,
    }


def prepared_input_path(lepton_name):
    """Return the worker-readable CSV with PhaseSpaceScan energy bands."""
    return OUTPUT_ROOT / lepton_name / "phase_space_config_input.csv"


def energy_band_report(bands, prepared_path):
    """Return report lines describing the PhaseSpaceScan energy partition."""
    return [
        "PhaseSpaceConfigScan photon-energy preparation",
        f"  prepared worker csv: {prepared_path}",
        "  energy bands use qOut terciles over valid PhaseSpaceScan rows",
        (
            f"  {ENERGY_BAND_LABELS[0]}: "
            f"{bands['minimum']:.8g} <= E_gamma <= "
            f"{bands['lower_boundary']:.8g} GeV "
            f"({bands['counts'][ENERGY_BAND_LABELS[0]]} rows)"
        ),
        (
            f"  {ENERGY_BAND_LABELS[1]}: "
            f"{bands['lower_boundary']:.8g} < E_gamma <= "
            f"{bands['upper_boundary']:.8g} GeV "
            f"({bands['counts'][ENERGY_BAND_LABELS[1]]} rows)"
        ),
        (
            f"  {ENERGY_BAND_LABELS[2]}: "
            f"{bands['upper_boundary']:.8g} < E_gamma <= "
            f"{bands['maximum']:.8g} GeV "
            f"({bands['counts'][ENERGY_BAND_LABELS[2]]} rows)"
        ),
    ]


def run_species(lepton_name):
    """Build one species' configuration packages from PhaseSpaceScan rows."""
    input_path = phase_space_input_path(lepton_name)
    config.configure_lepton(
        lepton_name,
        input_path=input_path,
        output_root=OUTPUT_ROOT,
    )
    config.CONFIGGEN_KINEMATIC_WORKERS = PHASE_SPACE_CONFIG_WORKERS
    config.CONFIGGEN_PLOT_WORKERS = PHASE_SPACE_CONFIG_PLOT_WORKERS

    rows = config.read_csv_rows(input_path)
    config.validate_config_target_columns(rows)
    bands = assign_energy_bands(rows)
    prepared_path = prepared_input_path(lepton_name)
    config.write_dict_csv(prepared_path, rows)

    config.clean_egamma_config_outputs()
    config.clean_data_outputs()
    egamma_outputs, egamma_detail_rows = (
        config.save_all_egamma_target_region_pdfs(
            rows,
            input_path=prepared_path,
        )
    )
    packages = [
        config.build_target_package(target, rows, egamma_detail_rows)
        for target in PHASE_SPACE_CONFIG_TARGETS
    ]
    report = config.build_report(
        input_path,
        len(rows),
        packages,
        egamma_outputs,
        source_label="PhaseSpaceScan",
    )
    return "\n".join(energy_band_report(bands, prepared_path)) + "\n" + report


def validate_settings():
    """Validate explicit script controls before changing output trees."""
    unknown = set(PHASE_SPACE_CONFIG_LEPTONS) - set(LEPTON_SPECS)
    if unknown:
        raise ValueError(f"Unknown PhaseSpaceConfigScan leptons: {sorted(unknown)}")
    if not PHASE_SPACE_CONFIG_LEPTONS:
        raise ValueError("PHASE_SPACE_CONFIG_LEPTONS must not be empty.")
    if PHASE_SPACE_CONFIG_WORKERS < 1:
        raise ValueError("PHASE_SPACE_CONFIG_WORKERS must be positive.")
    if PHASE_SPACE_CONFIG_PLOT_WORKERS < 1:
        raise ValueError("PHASE_SPACE_CONFIG_PLOT_WORKERS must be positive.")
    if len(ENERGY_BAND_QUANTILES) != 2:
        raise ValueError("ENERGY_BAND_QUANTILES must contain two boundaries.")
    if not 0.0 < ENERGY_BAND_QUANTILES[0] < ENERGY_BAND_QUANTILES[1] < 1.0:
        raise ValueError("ENERGY_BAND_QUANTILES must be ordered inside (0, 1).")
    if "D_W" not in {observable for observable, _tag in PHASE_SPACE_CONFIG_TARGETS}:
        raise ValueError("PhaseSpaceConfigScan requires the D_W ConfigGen target.")
    if "M2_magic" not in {
        observable for observable, _tag in PHASE_SPACE_CONFIG_TARGETS
    }:
        raise ValueError(
            "PhaseSpaceConfigScan requires the M2_magic ConfigGen target."
        )


def main():
    """Generate PhaseSpaceScan-driven configurations for selected species."""
    validate_settings()
    reports = [run_species(name) for name in PHASE_SPACE_CONFIG_LEPTONS]
    report_text = "\n\n".join(report.rstrip() for report in reports) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report_text, encoding="utf-8")
    print_console_text(report_text)


if __name__ == "__main__":
    main()
