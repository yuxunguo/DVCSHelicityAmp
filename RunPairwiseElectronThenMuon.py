"""Run pairwise ConfigGen in electron-first recovery order.

The electron configuration packages are completed first from their validated
contours.  The interrupted muon contour repair then resumes, reusing every
already validated file, before the muon configuration packages are generated.
"""

from datetime import datetime
import sys

import GradientPhaseSpaceConfig as config_stage
from RepairPairwiseMuonContours import run_repairs


def _timestamp():
    """Return a readable local timestamp for the production log."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _run_configs(lepton_name):
    """Run every configured pairwise ConfigGen package for one species."""
    config_stage.LEPTONS_TO_CONFIGURE = (lepton_name,)
    config_stage.run_selected_configs()


def main():
    """Run electron configs, muon contour repair, then muon configs."""
    if len(sys.argv) != 1:
        raise SystemExit(
            "RunPairwiseElectronThenMuon.py accepts no command-line "
            "arguments."
        )
    print(f"[{_timestamp()}] Starting all electron ConfigGen packages.", flush=True)
    _run_configs("electron")
    print(
        f"[{_timestamp()}] Electron ConfigGen complete; resuming muon "
        "contour repair.",
        flush=True,
    )
    run_repairs()
    print(
        f"[{_timestamp()}] Muon contour repair complete; starting all "
        "muon ConfigGen packages.",
        flush=True,
    )
    _run_configs("muon")
    print(
        f"[{_timestamp()}] Electron-first pairwise production completed "
        "successfully.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
