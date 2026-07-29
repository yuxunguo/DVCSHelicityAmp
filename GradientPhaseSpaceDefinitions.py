"""Shared W/GHZ definitions for the three gradient workflow stages."""

from numbers import Integral
from pathlib import Path

import numpy as np

import GradientPhaseSpaceScanTool as gradient_tool


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


def selected_definitions(scan_keys):
    """Return validated, de-duplicated definitions in execution order."""
    if isinstance(scan_keys, str):
        raise TypeError("The scan selection must be a tuple or list.")
    selected = []
    for raw_key in scan_keys:
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
    return tuple(SCAN_DEFINITIONS[key] for key in selected)


def validated_leptons(leptons):
    """Return validated lepton names for one independent stage."""
    if isinstance(leptons, str):
        raise TypeError("The lepton selection must be a tuple or list.")
    selected = tuple(leptons)
    unknown = set(selected) - set(gradient_tool.LEPTON_SPECS)
    if unknown:
        raise ValueError(f"Unknown lepton species: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one lepton species must be selected.")
    return selected


def validated_workers(workers):
    """Return one validated worker count."""
    if isinstance(workers, bool) or not isinstance(workers, Integral):
        raise TypeError("The worker count must be an integer.")
    if workers < 1:
        raise ValueError("The worker count must be positive.")
    return int(workers)
