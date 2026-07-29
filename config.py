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
SCAN_WORKERS = max(1, os.cpu_count() or 1)

# PhaseSpaceScan sampling budgets. The first value controls the global
# stratified scan; the second controls the local refinement stage.
PHASE_SPACE_SAMPLES = 8192 * 2
REFINEMENT_SAMPLES = 4096 * 2

# Shared W/GHZ randomized starts, gradient optimization, and local search.
ENTANGLEMENT_GRADIENT_RANDOM_STARTS = 2 ** 10
ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES = 8192 * 4
ENTANGLEMENT_GRADIENT_SCREENED_STARTS = 2 ** 10
ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION = 0.06
ENTANGLEMENT_GRADIENT_MAX_ITERATIONS = 80
ENTANGLEMENT_GRADIENT_TOLERANCE = 1.0e-7
# Resolution in normalized scan coordinates. This controls both the numerical
# gradient displacement and the final local-minimum neighbor check.
ENTANGLEMENT_GRADIENT_SCAN_PRECISION = 1.0e-5
ENTANGLEMENT_GRADIENT_MINIMUM_SEPARATION = 0.002
ENTANGLEMENT_GRADIENT_RANDOM_SEED = 314159
ENTANGLEMENT_LOCAL_SEARCH_INITIAL_STEP = 0.05
ENTANGLEMENT_LOCAL_SEARCH_STEP_REDUCTION = 0.5
ENTANGLEMENT_LOCAL_SEARCH_MAX_POLLS = 512
ENTANGLEMENT_LOCAL_SEARCH_RANDOM_DIRECTIONS = 4
ENTANGLEMENT_LOCAL_SEARCH_OBJECTIVE_TOLERANCE = 1.0e-10

# Mutually exclusive incoming-polarization scan mode.
#
# True:  scan only the coherent
#        cos(alpha_e)|+> + sin(alpha_e)|-> tensor
#        cos(alpha_p)|+> + sin(alpha_p)|-> preparation.
SCAN_INITIAL_MIXING_ANGLES = True

# PhaseSpaceConfigScan keeps and displays points whose observable is no more
# than this absolute threshold from that observable's scanned minimum/maximum.
PHASE_SPACE_CONFIG_THRESHOLD = 0.05

# The generic gradient tool samples this objective-level contour around every
# selected local minimum in the full eight-dimensional phase space.
PHASE_SPACE_CONFIG_CONTOUR_DELTA = 0.05
# Total eight-dimensional radial directions sampled for each contour.
PHASE_SPACE_CONFIG_CONTOUR_SAMPLES = 2 ** 8
