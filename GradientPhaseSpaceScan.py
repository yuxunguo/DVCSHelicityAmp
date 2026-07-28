"""Run the gradient scans selected by the explicit global controls below.

Edit :data:`SCANS_TO_RUN`, :data:`LEPTONS_TO_OPTIMIZE`,
:data:`GRADIENT_WORKERS`, :data:`REGENERATE_PLOTS_FROM_CSV`, and
:data:`SAVE_CONTOUR_DATA`, and :data:`GRADIENT_OUTPUT_ROOT`, then run

    python GradientPhaseSpaceScan.py
"""

from pathlib import Path
import sys

import numpy as np

import GradientPhaseSpaceScanTool as gradient_tool
from config import SCAN_WORKERS


# Explicit interface controls. These are the only run-selection settings.
SCANS_TO_RUN = ("W", "GHZ")
LEPTONS_TO_OPTIMIZE = ("electron", "muon", "heavy", "massless")
GRADIENT_WORKERS = SCAN_WORKERS
REGENERATE_PLOTS_FROM_CSV = False
SAVE_CONTOUR_DATA = True
GRADIENT_OUTPUT_ROOT = Path("Output") / "GradientPhaseSpaceScan"


W_PHYSICS_ANCHORS = {
    "electron": (
        {
            "name": "epcm_standard_W",
            "sqrt_s": 1.1518524360498226,
            "theta_out": 0.5 * np.pi,
            "qOut": 0.1771320126293574,
            "phi_p_out": 0.0,
            "phi_gamma_out": 3.032,
            "theta_e": 0.834,
            "theta_p": (-0.036) % np.pi,
        },
    ),
}

GHZ_PHYSICS_ANCHORS = {
    "electron": (
        {
            "name": "canonical_GHZ_hard_photon_endpoint",
            "sqrt_s": 4.2441101238978085,
            "theta_out": 1.57041484647354,
            "qOut": 2.0182672022904082,
            "phi_p_out": 1.5708006597909723,
            "phi_gamma_out": 4.712388736566098,
            "theta_e": 2.35626799,
            "theta_p": 2.35589509,
        },
    ),
}


SCAN_DEFINITIONS = {
    "W": gradient_tool.GradientScanDefinition(
        key="W",
        objective_name="D_W",
        file_tag="dw",
        latex=r"D_W",
        state_file_label="W",
        output_root=GRADIENT_OUTPUT_ROOT,
        physics_anchor_starts=W_PHYSICS_ANCHORS,
    ),
    "GHZ": gradient_tool.GradientScanDefinition(
        key="GHZ",
        objective_name="dGHZ",
        file_tag="dghz",
        latex=r"d_{\mathrm{GHZ}}",
        state_file_label="GHZ",
        output_root=GRADIENT_OUTPUT_ROOT,
        physics_anchor_starts=GHZ_PHYSICS_ANCHORS,
    ),
}


def selected_scan_keys():
    """Return validated, de-duplicated global scan keys in execution order."""
    if isinstance(SCANS_TO_RUN, str):
        raise TypeError("SCANS_TO_RUN must be a tuple or list of scan names.")
    selected = []
    for raw_key in SCANS_TO_RUN:
        key = str(raw_key).upper()
        if key not in SCAN_DEFINITIONS:
            raise ValueError(
                f"Unknown gradient scan {raw_key!r}; "
                f"choose from {tuple(SCAN_DEFINITIONS)}."
            )
        if key not in selected:
            selected.append(key)
    if not selected:
        raise ValueError("At least one gradient scan must be selected.")
    return tuple(selected)


def validate_interface_settings():
    """Validate global controls and return the selected scan keys."""
    keys = selected_scan_keys()
    if isinstance(LEPTONS_TO_OPTIMIZE, str):
        raise TypeError(
            "LEPTONS_TO_OPTIMIZE must be a tuple or list of species names."
        )
    unknown_leptons = set(LEPTONS_TO_OPTIMIZE) - set(
        gradient_tool.LEPTON_SPECS
    )
    if unknown_leptons:
        raise ValueError(f"Unknown lepton species: {sorted(unknown_leptons)}")
    if not LEPTONS_TO_OPTIMIZE:
        raise ValueError("LEPTONS_TO_OPTIMIZE must not be empty.")
    if not isinstance(GRADIENT_WORKERS, (int, np.integer)) or isinstance(
        GRADIENT_WORKERS, (bool, np.bool_)
    ):
        raise TypeError("GRADIENT_WORKERS must be an integer.")
    if GRADIENT_WORKERS < 1:
        raise ValueError("GRADIENT_WORKERS must be positive.")
    if not isinstance(REGENERATE_PLOTS_FROM_CSV, bool):
        raise TypeError("REGENERATE_PLOTS_FROM_CSV must be a bool.")
    if not isinstance(SAVE_CONTOUR_DATA, bool):
        raise TypeError("SAVE_CONTOUR_DATA must be a bool.")
    return keys


def run_selected_scans():
    """Run the globally selected definitions sequentially."""
    keys = validate_interface_settings()
    reports = {}
    for index, key in enumerate(keys, start=1):
        definition = SCAN_DEFINITIONS[key]
        print("=" * 72, flush=True)
        print(
            f"Gradient scan {index}/{len(keys)}: {key} "
            f"({definition.objective_name})",
            flush=True,
        )
        print("=" * 72, flush=True)
        reports[key] = gradient_tool.run_scan(
            definition,
            leptons_to_optimize=tuple(LEPTONS_TO_OPTIMIZE),
            gradient_workers=GRADIENT_WORKERS,
            regenerate_plots_from_csv=REGENERATE_PLOTS_FROM_CSV,
            save_contour_data=SAVE_CONTOUR_DATA,
        )
    return reports


def main():
    """Run the scans selected by the explicit global controls."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "GradientPhaseSpaceScan.py does not accept command-line "
            "arguments; edit the explicit global controls instead."
        )
    run_selected_scans()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
