"""Entanglement definitions for the Cartesian final-momentum rerun."""

from dataclasses import replace
from pathlib import Path

from GradientPhaseSpaceDefinitions import (
    PAIRWISE_CONCURRENCE_SCAN_KEYS,
    SCAN_DEFINITIONS as ANGULAR_SCAN_DEFINITIONS,
    selected_definitions as _unused_selected_definitions,
    validated_leptons,
    validated_workers,
)


CARTESIAN_OUTPUT_ROOT = Path("Output") / "CartesianGradientPhaseSpaceScan"

SCAN_DEFINITIONS = {
    key: replace(
        definition,
        output_root=CARTESIAN_OUTPUT_ROOT,
        coordinate_system="cartesian",
    )
    for key, definition in ANGULAR_SCAN_DEFINITIONS.items()
}


def selected_definitions(scan_keys):
    """Return validated Cartesian definitions in requested order."""
    if isinstance(scan_keys, str):
        raise TypeError("The scan selection must be a tuple or list.")
    selected = []
    for raw_key in scan_keys:
        key = str(raw_key).upper()
        if key not in SCAN_DEFINITIONS:
            raise ValueError(
                f"Unknown Cartesian gradient scan {raw_key!r}; "
                f"choose from {tuple(SCAN_DEFINITIONS)}."
            )
        if key not in selected:
            selected.append(key)
    if not selected:
        raise ValueError("At least one Cartesian gradient scan is required.")
    return tuple(SCAN_DEFINITIONS[key] for key in selected)
