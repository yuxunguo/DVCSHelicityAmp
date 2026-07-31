"""Shared entanglement definitions for the gradient workflow stages."""

from numbers import Integral
from pathlib import Path

import numpy as np

import GradientPhaseSpaceScanTool as gradient_tool
from GradientObjective import PAIRWISE_CONCURRENCE_NAMES


GRADIENT_OUTPUT_ROOT = Path("Output") / "GradientPhaseSpaceScan"

PAIRWISE_CONCURRENCE_SCAN_KEYS = (
    "CEP",
    "CEGAMMA",
    "CPGAMMA",
)

W_PHYSICS_ANCHORS = {
    "electron": (
        {
            "name": "epcm_low_energy_W",
            "sqrt_s": 1.0769666847798625,
            "theta_p_out": 2.0 * np.pi - 3.429,
            "theta_gamma_out": 1.298,
            "qOut": 0.0744376368622376,
            "phi_p_out": np.pi,
            "phi_gamma_out": 0.0,
            "alpha_e": 5.503 % np.pi,
            "alpha_p": 3.056 % np.pi,
        },
        {
            "name": "epcm_standard_W",
            "sqrt_s": 1.1518524360498226,
            "theta_p_out": 0.0,
            "theta_gamma_out": 3.032,
            "qOut": 0.17713201262935663,
            "phi_p_out": 0.0,
            "phi_gamma_out": 0.0,
            "alpha_e": 0.834,
            "alpha_p": (-0.036) % np.pi,
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
        physics_anchor_starts={},
    ),
    "CEP": gradient_tool.GradientScanDefinition(
        key="CEP",
        objective_name="one_minus_C_e_p",
        file_tag="max_c_ep",
        latex=r"1-C_{ep}",
        state_file_label="Cep",
        output_root=GRADIENT_OUTPUT_ROOT,
        physics_anchor_starts={},
    ),
    "CEGAMMA": gradient_tool.GradientScanDefinition(
        key="CEGAMMA",
        objective_name="one_minus_C_e_gamma",
        file_tag="max_c_e_gamma",
        latex=r"1-C_{e\gamma}",
        state_file_label="Cegamma",
        output_root=GRADIENT_OUTPUT_ROOT,
        physics_anchor_starts={},
    ),
    "CPGAMMA": gradient_tool.GradientScanDefinition(
        key="CPGAMMA",
        objective_name="one_minus_C_p_gamma",
        file_tag="max_c_p_gamma",
        latex=r"1-C_{p\gamma}",
        state_file_label="Cpgamma",
        output_root=GRADIENT_OUTPUT_ROOT,
        physics_anchor_starts={},
    ),
}

if {
    SCAN_DEFINITIONS[key].objective_name.removeprefix("one_minus_")
    for key in PAIRWISE_CONCURRENCE_SCAN_KEYS
} != set(PAIRWISE_CONCURRENCE_NAMES):
    raise RuntimeError("Pairwise concurrence gradient definitions are incomplete.")


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
    unknown = set(selected) - set(gradient_tool.GRADIENT_LEPTON_SPECS)
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
