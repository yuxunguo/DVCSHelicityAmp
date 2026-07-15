"""Central physics and scan settings for the DVCS helicity calculations."""

import os


# Masses are in GeV.
PROTON_MASS_GEV = 0.938
ELECTRON_MASS_GEV = 0.00051099895
MUON_MASS_GEV = 0.1056583755
HEAVY_LEPTON_MASS_GEV = 1.0
MASSLESS_LEPTON_MASS_GEV = 0.0

# Shared numerical scan behavior.
NORMALIZE_TRACE = True
SCAN_WORKERS = max(1, min(os.cpu_count() or 1, 24))
