"""Central physics and scan settings for the DVCS helicity calculations."""

import os


# Masses are in GeV.
PROTON_MASS_GEV = 0.938
ELECTRON_MASS_GEV = 0.00051099895

# Shared numerical scan behavior.
NORMALIZE_TRACE = True
SCAN_WORKERS = max(1, min(os.cpu_count() or 1, 8))
