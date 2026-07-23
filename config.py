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

# PhaseSpaceScan sampling budgets. The first value controls the global
# stratified scan; the second controls the local refinement stage.
PHASE_SPACE_SAMPLES = 8192 * 2
REFINEMENT_SAMPLES = 4096 * 2

# Randomized stratified starts followed by gradient and multiscale local search.
DW_GRADIENT_RANDOM_STARTS = 32
DW_GRADIENT_MAX_ITERATIONS = 80
DW_GRADIENT_TOLERANCE = 1.0e-7
# Resolution in normalized scan coordinates. This controls both the numerical
# gradient displacement and the final local-minimum neighbor check.
DW_GRADIENT_SCAN_PRECISION = 1.0e-5
DW_GRADIENT_MINIMUM_SEPARATION = 0.002
DW_GRADIENT_RANDOM_SEED = 314159
DW_LOCAL_SEARCH_INITIAL_STEP = 0.05
DW_LOCAL_SEARCH_STEP_REDUCTION = 0.5
DW_LOCAL_SEARCH_MAX_POLLS = 512
DW_LOCAL_SEARCH_RANDOM_DIRECTIONS = 4
DW_LOCAL_SEARCH_OBJECTIVE_TOLERANCE = 1.0e-10

# Mutually exclusive incoming-polarization scan mode.
#
# True:  scan only the coherent
#        cos(theta_e)|+> + sin(theta_e)|-> tensor
#        cos(theta_p)|+> + sin(theta_p)|-> preparation.
# False: scan only the established fixed polarization cases.
SCAN_INITIAL_MIXING_ANGLES = True

# PhaseSpaceConfigScan keeps and displays points whose observable is no more
# than this absolute distance from that observable's scanned minimum/maximum.
PHASE_SPACE_CONFIG_STEP = 0.1
