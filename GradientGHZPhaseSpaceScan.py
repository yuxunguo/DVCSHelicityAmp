"""Find local dGHZ minima and generate ConfigGen-style configurations.

The minimized invariant distance is

    dGHZ = sqrt(C_e_p^2 + C_e_gamma^2 + C_p_gamma^2 + (F3 - 1)^2).

This script reuses :mod:`GradientPhaseSpaceScan`'s Sobol screening, bounded
L-BFGS-B optimization, periodic multiscale local search, basin
deduplication, contour sampling, and configuration-generation workflow.
"""

from pathlib import Path

import GradientPhaseSpaceScan as gradient_scan


# Script controls.
LEPTONS_TO_OPTIMIZE = ("electron", "muon", "heavy", "massless")
GRADIENT_WORKERS = gradient_scan.GRADIENT_WORKERS
# Set True to rebuild PDFs from each species' existing local_minima.csv.
REGENERATE_PLOTS_FROM_CSV = False

OUTPUT_ROOT = Path("Output") / "GradientGHZPhaseSpaceScan"
CONFIG_OUTPUT_ROOT = Path("Output") / "GradientGHZPhaseSpaceConfig"
LOG_PATH = OUTPUT_ROOT / "GradientGHZPhaseSpaceScan.log"

# Hard-photon endpoint seed migrated to the common-theta initial-CM frame.
# It is a deterministic search start; the new-frame optimum is re-established
# by the gradient and multiscale local searches.
CANONICAL_GHZ_ENDPOINT_STARTS = {
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


def main():
    """Minimize dGHZ and generate local-minimum configurations."""
    gradient_scan.LEPTONS_TO_OPTIMIZE = LEPTONS_TO_OPTIMIZE
    gradient_scan.GRADIENT_WORKERS = GRADIENT_WORKERS
    gradient_scan.REGENERATE_PLOTS_FROM_CSV = REGENERATE_PLOTS_FROM_CSV
    gradient_scan.configure_objective(
        name="dGHZ",
        file_tag="dghz",
        latex=r"d_{\mathrm{GHZ}}",
        output_root=OUTPUT_ROOT,
        config_output_root=CONFIG_OUTPUT_ROOT,
        log_path=LOG_PATH,
        state_file_label="GHZ",
        physics_anchor_starts=CANONICAL_GHZ_ENDPOINT_STARTS,
    )
    gradient_scan.main()


if __name__ == "__main__":
    main()
