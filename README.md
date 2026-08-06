# DVCSHelicityAmp

Numerical Bethe–Heitler helicity amplitudes, spin-density matrices, and
three-particle entanglement scans for exclusive electroproduction.

The Python workflow builds COM-frame kinematics, evaluates the complete
Bethe–Heitler helicity-amplitude table, contracts prepared incoming spin
states, and studies the outgoing electron–proton–photon density matrix.

## Quick start

Requirements:

```text
numpy
matplotlib
scipy
```

Run the main workflows from the repository root:

```sh
python3 BHHelicityAmp.py     # amplitude benchmark
python3 SpinDensityMat.py    # spin-density scans
python3 AlignmentScan.py     # angular alignment and entanglement scan
python3 ConfigGen.py         # selected configurations from AlignmentScan
python3 PhaseSpaceScan.py    # adaptive all-observable/all-lepton phase-space scan
python3 PhaseSpaceConfigScan.py  # ConfigGen packages from PhaseSpaceScan results
python3 GradientPhaseSpaceScan.py # stage 1: entanglement gradient searches
python3 GradientPhaseSpaceContour.py # stage 2: contours of raw minima
python3 GradientPhaseSpaceCluster.py # stage 3: cluster saved minima
python3 GradientPhaseSpaceConfig.py # stage 4: configs from clusters/contours
python3 GradientLocalUnitaryFidelity.py # post-process W/GHZ LU fidelities
python3 GradientCanonicalConfigCollect.py # canonical W/GHZ PDFs by cluster
python3 EpCMEntanglementScan.py   # reference-centered electron ep-CM scan
python3 EpCMConfigGen.py          # config packages from the focused ep-CM scan
python3 ProtonVirtualPhotonAmp.py # proton-current virtual-photon decomposition
python3 QuasiRealComptonHelicity.py  # gamma* lepton CM helicity components
```

Versioned gradient-search artifacts are written under
`Output/GradientPhaseSpaceScan/`. All other generated data, plots, and logs
are written under the ignored `Output_local/` tree.

## Main files

```text
config.py             Shared masses, normalization, and worker settings
Algebra.py            Dirac algebra, spinors, and photon polarizations
Kinematics.py         Initial proton-lepton CM momenta and kinematic checks
BHHelicityAmp.py      Bethe–Heitler amplitudes and benchmarks
SpinDensityMat.py     Density matrices and entanglement observables
AlignmentScan.py      Fine angular scan at characteristic kinematics
ConfigGen.py          Ranked-region configuration and plot generator
PhaseSpaceScan.py      Adaptive eight-dimensional kinematic/polarization scan
PhaseSpaceConfigScan.py ConfigGen-style packages from PhaseSpaceScan results
GradientPhaseSpaceDefinitions.py Shared gradient objectives, anchors, and root
GradientPhaseSpaceScan.py Stage 1 local-minimum search interface
GradientPhaseSpaceContour.py Stage 2 raw-minimum contour interface
GradientPhaseSpaceCluster.py Stage 3 phase-space clustering interface
GradientPhaseSpaceConfig.py Stage 4 cluster ConfigGen/contour-plot interface
GradientLocalUnitaryFidelity.py Canonical W/GHZ fidelity after local rotations
GradientCanonicalConfigCollect.py Canonical-component PDFs and indexes by cluster
GradientPhaseSpaceScanTool.py Shared implementation for all four stages
EpCMEntanglementScan.py Exact ep-CM scan with a slow final proton
EpCMConfigGen.py      ConfigGen packages for the focused ep-CM scan
ProtonVirtualPhotonAmp.py Proton helicity/current decomposition into T-/T+/L virtual photons

`ProtonVirtualPhotonAmp.py` normalizes each projected helicity amplitude by
`A_unpol = sqrt((1/2) sum_{sIn,sOut,lambda=T-/T+/L} |A|^2)`, where the factor
`1/2` averages the incoming proton helicity. Its CSV retains both raw and
normalized complex amplitudes. The editable `theta_p` and `phi_p` values in
`main()` tilt the final proton away from the incoming `+z` direction while
keeping both proton momenta on shell; `theta_p=0` recovers collinear recoil.
The editable `theta_p_values` and `z_values` grids generate
normalized-magnitude and phase curves as functions of `theta_p` and `z`,
together with separate scan CSVs. Both scans also store and plot
`R_L/T = |A_L|^2 / (|A_T-|^2 + |A_T+|^2)` for each proton-helicity transition.
Its default point is the ep-CM W-state benchmark
`|p|=0.194 GeV`, `|p'|=0.165 GeV`, and `theta_gamma=3.032`, with exact
energy conservation fixing `E_gamma`. It additionally writes the raw
`6 x 2` proton-emission matrix and the coherently prepared
`(p', gamma*)` amplitudes to `epcm_w_proton_emission_matrix.csv` and
`epcm_w_prepared_intermediate_state.csv`.
QuasiRealComptonHelicity.py Off-shell gamma* lepton Compton helicity analysis
FixedHelicityTest.py  Small editable fixed-helicity example
Mathematica/          Wolfram Language implementation and benchmarks
```

## Conventions and configuration

Four-vectors are contravariant arrays ordered as `[E, px, py, pz]`, with
metric `diag(1, -1, -1, -1)`. Helicity labels are doubled helicities:
`-1` and `+1`.

The amplitude table is ordered as

```text
incoming: (hIn, sIn)
outgoing: (hOut, sOut, lambda)
```

where `h` labels the electron, `s` the proton, and `lambda` the real photon.
The outgoing basis has eight states and its density matrix is `8 x 8`.

Shared settings are in `config.py`:

```python
PROTON_MASS_GEV = 0.938
ELECTRON_MASS_GEV = 0.00051099895
NORMALIZE_TRACE = True
SCAN_WORKERS = ...
```

Every reusable amplitude and kinematic API requires an explicit lepton mass.
Use `ELECTRON_MASS_GEV` for electrons or pass `0.0` deliberately for the
massless approximation.

## Kinematics

`Kinematics.py` uses the initial proton--lepton CM frame. The incoming proton
is fixed along `+z`, the incoming lepton along `-z`, and the independent
variables are

```text
s, qOut, theta_p_out, phi_p_out, theta_gamma_out, phi_gamma_out
```

The final proton and real photon have independent polar angles
`theta_p_out`, `theta_gamma_out` and independent azimuths `phi_p_out`,
`phi_gamma_out`.
The code computes the incoming COM momentum `pIn`, solves the outgoing proton
momentum `pOut` from energy conservation, and derives the outgoing-lepton
angles `theta_lepton_out` and `phi_lepton_out` from momentum conservation.

The returned kinematic record includes the momenta `k`, `p`, `kp`, `pp`, and
`qout`, together with `Q2`, `xB`, `t`, `W2`, and `y`.

Example with the physical electron mass:

```python
from config import ELECTRON_MASS_GEV, PROTON_MASS_GEV
from Kinematics import kinematics_cm_from_independent

kin = kinematics_cm_from_independent(
    s, qOut, theta_p_out, phi_p_out, theta_gamma_out, phi_gamma_out,
    PROTON_MASS_GEV,
    electron_mass=ELECTRON_MASS_GEV,
)
```

Proton form factors are obtained from the YAHL 2018 lookup table in
`YAHL 2018/proton_lookup.dat` and converted from Sachs to Dirac/Pauli form.

## Bethe–Heitler amplitudes

The main numerical entry points in `BHHelicityAmp.py` are:

```text
bh_amplitude_core                         one fixed-helicity amplitude
bh_unpolarized_squared_amplitude_core     helicity-summed |M|^2
proton_current_helicity_decomposition     proton F1/F2 and GE/GM helicity tensors
electron_current_helicity_decomposition   pointlike electron helicity current
bh_amplitude_table                        complete 4 x 8 amplitude table
```

Pass the same electron mass to both kinematics and amplitudes:

```python
from config import ELECTRON_MASS_GEV, PROTON_MASS_GEV
from BHHelicityAmp import bh_amplitude_table

amplitudes = bh_amplitude_table(
    kin["momenta"], PROTON_MASS_GEV, F1, F2,
    electron_mass=ELECTRON_MASS_GEV,
)
```

Running `BHHelicityAmp.py` writes the analytic comparison to
`Output_local/BHHelicityAmp.log`. The analytic benchmark remains a massless-electron
check.

## Density matrices and entanglement

`SpinDensityMat.py` forms the process matrix from the `4 x 8` amplitude table
and contracts the selected incoming electron–proton state. Supported prepared
states are:

```text
unpolarized
L_proton, L_lepton
Tx_proton, Ty_proton
Tx_lepton, Ty_lepton
LL    = L electron + L proton
Lanti = L+ electron + L- proton (opposite helicities)
LTx   = L electron + Tx proton
LTy   = L electron + Ty proton
TxTx  = Tx electron + Tx proton
TxTy  = Tx electron + Ty proton
```

In the compact double-polarization keys, the electron state is listed first.
Plots, reports, and display-label columns name both particles explicitly.
`L` denotes the direct positive-helicity state, not a helicity asymmetry.
`Lanti` is the pure incoming state `(h_lepton, h_proton) = (+1, -1)`.
Unnamed particles are averaged incoherently with `I/2`.

The stored observables are:

```text
C_e_p, C_e_gamma, C_p_gamma       pairwise Wootters concurrences
C_e_rest, C_p_rest, C_gamma_rest one-to-rest concurrences
D_W                               distance from ideal W pair concurrences
dGHZ                              distance from ideal GHZ concurrence invariants
F3                                concurrence-triangle observable
M_e, M_p, M_gamma                 CKW monogamy residuals
M2_magic                          second stabilizer Renyi entropy (magic)
purity                            Tr(rho^2)
```

Pairwise concurrence is evaluated for pure and mixed outgoing states. The
implemented one-to-rest, `F3`, and CKW formulas are pure-state formulas; those
columns are set to zero for mixed states and should be interpreted together
with `purity`.

For the outgoing three-qubit density matrix, magic/nonstabilizerness is
computed from all 64 Pauli strings:

```text
M2_magic = -ln[(1/8) sum_P Tr(P rho)^4 / Tr(rho^2)^2]
P = (I, X, Y, Z) tensor (I, X, Y, Z) tensor (I, X, Y, Z)
```

For a pure state, `Tr(rho^2)=1` and the purity denominator drops out.
The purity-normalized expression is also stored for incoherently averaged
mixed outgoing states; for those mixed ensembles it can be negative and should
not be interpreted as the pure-state nonstabilizerness monotone.
AlignmentScan and PhaseSpaceScan rank and refine magic by maximizing the signed
`M2_magic` value. The selected points therefore have the largest measured
second stabilizer Renyi entropy in each polarization.

When `NORMALIZE_TRACE` is enabled, stored density matrices have unit trace.
The unnormalized prepared-state signal remains available as `spin_signal_M2`,
and the fully unpolarized result as `squared_amplitude_M2`.

## AlignmentScan and ConfigGen

`AlignmentScan.py` scans `phi_p_out` and `phi_gamma_out` at characteristic
values of `s`, `theta_p_out`, `theta_gamma_out`, and `qOut`. It records the outgoing
lepton–photon opening angle and writes full, aligned-only, and ranked tables
directly in each species directory:

```text
Output_local/AlignmentScan/<lepton>/
Output_local/AlignmentScan/<lepton>/concurrence_scan_lepton_<species>_<polarization>_proton_<polarization>.pdf
```

The physical mass of each configured lepton regulates exactly collinear
lepton propagators, while the massless species retains the singular limit.

`ConfigGen.py` reads the full concurrence phase-space CSV, locates strong
regions for the species-labelled lepton–proton and lepton–photon concurrence,
proton–photon concurrence, `F3`, GHZ purity, and W purity. It writes:

ConfigGen also selects the maximum signed `M2_magic` configuration for every
polarization. Magic heatmaps use the pure-state theoretical maximum `ln(9/2)`
as their upper color limit, and outputs are written under `Data/m2_magic/`.

```text
Output_local/ConfigGen/<lepton>/Data/<target>/lepton_<species>_<polarization>_proton_<polarization>/...
Output_local/ConfigGen/<lepton>/lepton_<species>_<polarization>_proton_<polarization>/<E_gamma>_<target>_regions.pdf
```

Every polarization folder names both incoming states explicitly, for example
`lepton_muon_L_proton_unpolarized` or `lepton_muon_L_proton_Tx`. Polarization
tokens preserve the conventional capitalization `L`, `Tx`, and `Ty`.

Each configuration package includes reconstructed momenta and an outgoing
helicity-amplitude decomposition. Incoherent incoming ensembles remain
separate; they are never replaced by a coherent amplitude sum.

The entanglement scans evaluate the W-concurrence distance

```text
D_W = sqrt((C_e_p - 2/3)^2 + (C_p_gamma - 2/3)^2
           + (C_e_gamma - 2/3)^2)
```

for every AlignmentScan and PhaseSpaceScan point and polarization. Smaller
values are more W-like, so ranked CSVs, refinement seeds, and ConfigGen select
the minima. The per-polarization PDFs include a reversed-color `D_W` heatmap,
and ConfigGen writes the low-distance configuration package under `Data/dw/`.

The GHZ-invariant distance is

```text
dGHZ = sqrt(C_e_p^2 + C_e_gamma^2 + C_p_gamma^2 + (F3 - 1)^2)
```

and is also minimized. For a pure three-qubit state, `dGHZ = 0` means all
pairwise concurrences vanish and `F3 = 1`, identifying the maximally entangled
GHZ local-unitary orbit.

`PhaseSpaceScan.py` performs a stratified eight-dimensional scan over six
kinematic coordinates plus the coherent incoming angles `alpha_e` and
`alpha_p`, followed by local kinematic refinement. The coherent state is
`|e> = cos(alpha_e)|+> + sin(alpha_e)|->` and
`|p> = cos(alpha_p)|+> + sin(alpha_p)|->`, with each angle covering `[0, pi)`.
The original fixed-polarization results are preserved without duplication.
The electron scan also contains an exact deterministic seed equivalent by
spatial rotation to the reference point `pIn=0.130 GeV`, `pOut=0.028 GeV`,
`theta_p'=3.429`, and `theta_gamma=1.298`, together with mixing angles
`5.503 mod pi` and `3.056`.

The script runs electron, muon, heavy-lepton, and massless-lepton species by
default, and writes independent full, aligned,
ranked, and plotted fixed-polarization results under
`Output_local/PhaseSpaceScan/<lepton>/`. Each species additionally receives
`<stem>_mixing_angle_phase_space.csv`, `<stem>_mixing_angle_top.csv`, and
`phase_space_scan_lepton_<species>_alpha_e_mix_proton_alpha_p_mix.pdf`. The
mixed-angle PDF retains all three original kinematic projections and the
observable histogram, then adds projections involving `alpha_e` and
`alpha_p`.

The central `SCAN_INITIAL_MIXING_ANGLES` option in `config.py` selects one
mode required by the eight-dimensional scan. Every sampled row contains six
kinematic variables plus `alpha_e` and `alpha_p`, and only the coherent
mixed-angle polarization is evaluated; fixed-polarization mode is not part of
this scan.
Its plot filenames use the same explicit convention:
`phase_space_scan_lepton_<species>_<polarization>_proton_<polarization>.pdf`.
Point evaluations run in parallel. Edit `LEPTONS_TO_SCAN`,
`PHASE_SPACE_SCAN_WORKERS`, ranges, and output settings at the top of
`PhaseSpaceScan.py`. Set the global and refinement point budgets with
`PHASE_SPACE_SAMPLES` and `REFINEMENT_SAMPLES` in `config.py`.

`PhaseSpaceConfigScan.py` consumes the PhaseSpaceScan CSV selected by the same
central mode switch. In fixed mode it retains the ConfigGen per-polarization
workflow. In mixing-angle mode it clusters in all eight variables, writes
`alpha_e` and `alpha_p` into the configuration, momentum, and amplitude CSVs,
and contracts each amplitude with the exact coherent incoming state. Its PDFs
retain the kinematic projections and add the two mixing-angle dimensions.
Each PDF then appends the momentum,
four-vector, kinematic-summary, and final-state-amplitude page for every
selected minimum/maximum. The central `PHASE_SPACE_CONFIG_THRESHOLD` setting
in `config.py` limits both displayed and selectable points to an absolute
distance no larger than the threshold from each observable's scanned maximum
(or its minimum for `D_W`); its default is `0.05`. Because `PhaseSpaceScan` uses a
continuous photon energy, valid rows are divided into balanced low, middle,
and high `E_gamma` bands before configuration selection. Outputs are written under
`Output_local/PhaseSpaceConfigScan/<lepton>/`, with a combined report at
`Output_local/PhaseSpaceConfigScan.log`.
Because it inherits the same ConfigGen targets, it also writes maximum-magic
packages under `Data/m2_magic/`.

The gradient workflow has four independent, sequential stages. Every stage
uses explicit globals and accepts no command-line arguments.

1. `GradientPhaseSpaceScan.py` performs the expensive search. Randomized
   Latin-hypercube starts, screened Sobol starts, and physics anchors are
   optimized with bounded L-BFGS-B and multiscale local polling. It writes
   `optimization_runs.csv`, raw `local_minima.csv`, and
   `all_local_minima.pdf`; it does not cluster or call ConfigGen.
2. `GradientPhaseSpaceContour.py` reads the raw `local_minima.csv` and
   generates the full eight-dimensional contour for every minimum before any
   cluster assignment exists. It saves one authoritative, resumable file per
   minimum under `contour/local_minima/`, keyed by raw `local_minimum_id`, so
   changing a later polarization cut, seed, or cluster partition does not
   change contour ownership.
3. `GradientPhaseSpaceCluster.py` reads `local_minima.csv`, applies the
   objective cut, and applies a state-specific polarization partition. W
   retains narrow tunable `alpha_e` bands around `pi/4` and `3pi/4` and finds
   six `alpha_p`-periodic configurations. GHZ uses the `alpha_e` boundaries
   `(0, pi/2, pi)` and finds two periodic `alpha_p` groups inside each of the
   two resulting regions, for four configurations. Pairwise-concurrence scans
   cluster the full periodic
   `(alpha_e, alpha_p)` plane into six configurations without imposing the W
   or GHZ partitions. The minimum with the best
   objective in each cluster is marked as that polarization cluster's
   configuration representative. It writes the assignments and polarization
   summary without performing optimization or ConfigGen work. Every assigned
   minimum is passed to the configuration stage.
4. `GradientPhaseSpaceConfig.py` reads `clustered_minima.csv`, joins each row
   to the pre-cluster contour data by `local_minimum_id`, and generates a
   separate data package and configuration PDF for every parent polarization
   cluster. Each PDF starts with a summary plot containing every member's
   contour, followed by a reconstructed configuration page and an 8D contour
   page for every minimum in that polarization cluster.

The shared definitions, anchors, labels, and output root live in
`GradientPhaseSpaceDefinitions.py`. Valid scan keys are `W`, `GHZ`, `CEP`,
`CEGAMMA`, and `CPGAMMA`. The last three maximize `C_e_p`, `C_e_gamma`, and
`C_p_gamma` by minimizing the bounded loss `1-C`; both the physical
concurrence and derived loss are retained in the saved rows. Select objectives
and leptons independently at the top of each stage script. For example, use
`SCANS_TO_RUN = ("CEP", "CEGAMMA", "CPGAMMA")`, then run:

```sh
python3 GradientPhaseSpaceScan.py
python3 GradientPhaseSpaceContour.py
python3 GradientPhaseSpaceCluster.py
python3 GradientPhaseSpaceConfig.py
```

`GradientLocalUnitaryFidelity.py` is an independent post-processing check over
the saved stage-1 minima. For every selected scan and lepton, it reconstructs
the normalized pure final state in `(e_out, p_out, gamma_out)` order and
maximizes the canonical W and GHZ fidelities over
`U_e tensor U_p tensor U_gamma`. Each SU(2) acts in the `(-,+)` helicity basis.
The per-minimum CSV saves both fidelities, the unrotated fixed-basis
fidelities, optimizer diagnostics, and every rotation as Z-Y-Z Euler angles
and complex 2-by-2 matrix entries.

By default, `CANDIDATES_PER_TARGET_PER_SCAN = None` checks every saved
minimum. Set it to a positive integer for a faster screened run: each target
then selects that many minima with the smallest matching LU-invariant distance
(`D_W` or `dGHZ`), and their union is checked against both targets. Missing
selected scan inputs are recorded explicitly in the combined summary and can
instead be made fatal with
`SKIP_MISSING_SCAN_INPUTS = False`. Run:

```sh
python3 GradientLocalUnitaryFidelity.py
```

Per-scan outputs are written under
`Data/<state>/fidelity/local_unitary_fidelity_by_minimum.csv` and
`Data/<state>/fidelity/local_unitary_fidelity_summary.csv`. The combined CSV,
comparison PDF, and log are written under
`Output/GradientPhaseSpaceScan/Fidelity/`. Set
`REMAKE_SUMMARY_PLOT_FROM_CSV = True` to rebuild only the combined PDF.
Each available species/scan also receives
`Plots/<state>/fidelity/local_unitary_fidelity_distribution.pdf`. These
distribution PDFs show histograms and empirical CDFs of the infidelity
`1-F_LU` on logarithmic axes; exact numerical zeros are displayed at a
documented `1e-15` plotting floor without changing the saved CSV values.
Beside each all-minima PDF, the workflow writes
`local_unitary_fidelity_distribution_after_objective_cut.pdf`, applying the
same stage-3 rule `objective - objective_min <= POLARIZATION_CLUSTER_CUT`
before plotting. For W and GHZ this is respectively the `D_W` and `dGHZ` cut.

For W and GHZ only, after stage 4, `GradientCanonicalConfigCollect.py` reads
the saved configuration and amplitude-decomposition CSVs without rerunning
the search or contours. It collects configurations with exactly three retained components
for W and exactly two retained components for GHZ. "Retained" uses
`AMPLITUDE_MIN_FRACTION` (2% by default), so the PDF cover states the cutoff
explicitly. Each per-cluster PDF contains one cover followed by one compact
momentum, kinematic-summary, and normalized-amplitude page per matching
configuration. One PDF is written inside every polarization cluster's plot
folder, directly beside the full configuration PDF, and its companion CSV
index is written in that cluster's `combined/` data folder. A cluster with no
canonical match receives a cover-only PDF and an empty CSV index so the
per-cluster package remains complete.

Stage 1 exposes `REMAKE_MINIMA_PLOT_FROM_CSV`. Leave it `False` for a new
search. Set it to `True` to read the existing
`Data/<state>/scan/local_minima.csv` and rebuild
`Plots/<state>/all_local_minima.pdf` without rerunning Sobol screening,
L-BFGS-B, or multiscale local search.

Each stage stops with a missing-file error if its required predecessor output
does not exist. Separate logs are written as
`<state>_gradient_phase_space_scan.log`,
`<state>_gradient_phase_space_contour.log`,
`<state>_gradient_phase_space_cluster.log`, and
`<state>_gradient_phase_space_config.log`.

The screening pool size, optimized screened-start count, and separation are
set by `ENTANGLEMENT_GRADIENT_SCREENING_SAMPLES`,
`ENTANGLEMENT_GRADIENT_SCREENED_STARTS`, and
`ENTANGLEMENT_GRADIENT_SCREENING_SEPARATION` in `config.py`;
`optimization_runs.csv` records the source and screening value of every start.
The normalized gradient and local-verification resolution is controlled by
`ENTANGLEMENT_GRADIENT_SCAN_PRECISION`; the other
`ENTANGLEMENT_GRADIENT_*` and `ENTANGLEMENT_LOCAL_SEARCH_*` settings control
starts, convergence, basin separation, and multiscale polishing.
Stage 3 is polarization-first. The explicit `POLARIZATION_CLUSTER_CUT` and
`POLARIZATION_CLUSTER_SEED` globals are shared. W uses
`W_POLARIZATION_CLUSTER_COUNT = 6` and
`W_ALPHA_E_LINE_HALF_WIDTH`, while GHZ uses
`GHZ_POLARIZATION_CLUSTER_COUNT = 4` and
`GHZ_ALPHA_E_BOUNDARIES = (0, pi/2, pi)`. The default objective cut retains
minima with `objective - objective_min <= 0.05`; `alpha_p` is clustered with
period `pi` inside each state-specific `alpha_e` region. Pairwise concurrence
uses `PAIRWISE_CONCURRENCE_POLARIZATION_CLUSTER_COUNT = 6` and periodic
k-means in both mixing angles; its same `0.05` loss cut means
`C_max - C <= 0.05`. The best-objective
member is still identified in the summary, but every cluster member is
configured. The assignments are saved in `clustered_minima.csv`, while parent
regions, centers, sizes, and representative IDs are saved in
`polarization_clusters.csv`. `polarization_cluster_phase_space.pdf` is the
one-page full retained-minimum view. Each cluster is written as a separate
one-page `polarization_cluster_phase_space_PXX.pdf`, colored by objective
value. `RegeneratePolarizationContourPlots.py` adds validated projected 8D
contours without rerunning the scan, clustering, or contour calculation.
The nine correlation panels from the overview are also exported as separate
one-page PDFs. A normal scan writes the contour-free set under
`Plots/<state>/polarization_correlations/without_contours/`, while the
plot-only regeneration command described below also writes a parallel
`with_contours/` set when validated contours are available. Both version trees
contain `clustered/` and `unclustered/` folders. Each begins with a matching
3-by-3 overview named `00_summary_<state>.pdf`, followed by the nine individual
panels in the same order. The
`clustered/` version uses the same P1...Pn colors and markers, while the
`unclustered/` version plots the identical retained minima with cluster labels
hidden. GHZ unclustered panels use exactly the P2 marker/color and W
unclustered panels use exactly the P4 marker/color. Clustered panels mark each
cluster representative with a gold star. Each unclustered panel instead marks
only the P2 (GHZ) or P4 (W) representative and labels it `Example`; objective-cut
text is omitted. That example's two configuration pages are extracted to
the same `polarization_correlations/unclustered/` folder, with an index CSV in
the common `polarization_correlations/unclustered/` configuration folder. Each
unclustered row in
`Data/<state>/cluster/polarization_correlation_plot_index.csv` records the
matching `example_configuration_path` in addition to the axes, mode, cut,
retained count, and plot path.
`RegeneratePolarizationCorrelationContourPlots.py` regenerates both modes and
their summary pages in parallel `polarization_correlations/with_contours/` and
`polarization_correlations/without_contours/` trees. Each tree contains its own
`clustered/` and `unclustered/` folders. Every correlation PDF ends with its
state key, for example `00_summary_W.pdf` or
`01_qOut_vs_sqrt_s_clustered_GHZ.pdf`. W/GHZ muon `with_contours` trees are
intentionally skipped because their species-coordinate repair was aborted;
their `without_contours` plots are still written with the same shared axis
settings. Pass `--index-only` to rebuild the combined plot index CSVs from
already-rendered version trees without rerendering the PDFs.

The eight coordinates are `sqrt(s)`, `theta_p_out`, `theta_gamma_out`, the
physical `E_gamma` fraction, the final-proton and photon azimuths, `alpha_e`,
and `alpha_p`.
This workflow requires `SCAN_INITIAL_MIXING_ANGLES = True`. Every minimum
assigned to a polarization cluster receives reconstructed configuration,
momentum, coherent final-state amplitude, contour data, and PDF pages.
The contour level and number of sampled radial directions are controlled by
`PHASE_SPACE_CONFIG_CONTOUR_DELTA` and
`PHASE_SPACE_CONFIG_CONTOUR_SAMPLES`; the production default is 1536
directions per minimum. Stage 2 exposes
`REUSE_SAVED_MINIMUM_CONTOURS`. When enabled, valid completed per-minimum
files are reused after an interrupted run, and the lightweight contour index
is refreshed after each completed minimum. Saved objective, contour settings,
minimum IDs, and centers are validated before reuse.
Stage 2 uses all `CONTOUR_WORKERS` through `GradientContourWorker.py`. That
lightweight process entrypoint intentionally avoids SciPy imports, preventing
Windows process-spawn duplication of the optimizer DLLs while allowing
`CONTOUR_WORKERS = SCAN_WORKERS`.
Set `POLARIZATION_CLUSTERS_TO_CONFIGURE` to `None` for every cluster or to a
tuple of one-based cluster numbers, such as `(1, 4)`, for isolated runs.
All angular scan and configuration plots use the shared `PlotUtils.py`
formatting: polar axes span `0` to `pi` with quarter-pi ticks, azimuthal axes
span `0` to `2*pi` with half-pi ticks, and major grid lines coincide with
those tick positions. Every phase-space projection adds a small unlabeled
margin outside its physical kinematic bounds, keeping endpoint markers and
contours fully visible.

All gradient objectives share one main root and use a lepton-first,
artifact-type-first hierarchy:

```text
Output/GradientPhaseSpaceScan/
  electron/
    Data/
      W/
        scan/
        contour/
        cluster/
        dw/
          polarization_cluster_01/combined/
            canonical_dw_polarization_cluster_01_configurations_electron.csv
          ...
          polarization_cluster_06/combined/
      GHZ/
        scan/
        contour/
        cluster/
        dghz/
          polarization_cluster_01/combined/
            canonical_dghz_polarization_cluster_01_configurations_electron.csv
          ...
      CEP/
        scan/
        contour/
        cluster/
        max_c_ep/polarization_cluster_01/combined/
      CEGAMMA/
        scan/
        contour/
        cluster/
        max_c_e_gamma/polarization_cluster_01/combined/
      CPGAMMA/
        scan/
        contour/
        cluster/
        max_c_p_gamma/polarization_cluster_01/combined/
    Plots/
      W/polarization_cluster_01/
        Canonical_W_Configurations_Electron_Polarization_Cluster_01.pdf
      ...
      W/polarization_cluster_06/
      GHZ/polarization_cluster_01/
        Canonical_GHZ_Configurations_Electron_Polarization_Cluster_01.pdf
      CEP/polarization_cluster_01/
      CEGAMMA/polarization_cluster_01/
      CPGAMMA/polarization_cluster_01/
  muon/
    Data/W/
    Data/GHZ/
    Data/CEP/
    Data/CEGAMMA/
    Data/CPGAMMA/
    Plots/W/
    Plots/GHZ/
    Plots/CEP/
    Plots/CEGAMMA/
    Plots/CPGAMMA/
  Logs/
```

Only electron and muon are accepted by the gradient workflow. Raw scan CSVs
are under `Data/<state>/scan/`, stage-2 contours under
`Data/<state>/contour/`, stage-3 outputs under `Data/<state>/cluster/`, and
each parent polarization cluster receives its own
objective data folder and PDF folder. Every polarization-cluster member
receives a configuration page and pairwise projections of
`objective = objective(local minimum) + PHASE_SPACE_CONFIG_CONTOUR_DELTA`.
Periodic contours are split at the physical plot boundary.

The W configuration pages additionally recalculate the three CKW residuals
`M_l`, `M_p`, and `M_gamma`. When both `D_W` and all residuals pass the
`W_DW_SMALL_THRESHOLD` and `W_MONOGAMY_SMALL_THRESHOLD` controls in
`PhaseSpaceConfigScan.py`, a deterministic multistart search maximizes the
fidelity with canonical W over three local SU(2) rotations. The GHZ definition
uses its deterministic electron hard-photon endpoint anchor, and the shared
photon-energy fraction reaches `0.999999` while excluding the singular exact
endpoint.

## Reference-centered electron ep-CM scan

`EpCMEntanglementScan.py` is centered on the electron W-state reference point
with incoming ep-CM momentum `p = 0.130 GeV` and final-proton momentum
`p' = 0.028 GeV`. It constructs the final state by an exact two-body boost,
rather than by using approximate massless four-vectors. The default
focused polarization set additionally includes direct proton preparations
along `-Tx` and `-Ty`, with the incoming lepton averaged incoherently.
It also includes the direct product preparations `L lepton + -Tx proton` and
`L lepton + -Ty proton`.
An additional `mixing_angles` case prepares both incoming particles
coherently as
`|p> = cos(alpha_p)|+> + sin(alpha_p)|->` and
`|l> = cos(alpha_e)|+> + sin(alpha_e)|->`. The independent
`ALPHA_E_VALUES_RAD` and `ALPHA_P_VALUES_RAD` axes cover one physical
period `[0, pi)` and include the benchmark angles `5.503 mod pi` and `3.056`
radians exactly. The scan stores them as `alpha_e_rad` and
`alpha_p_rad`, so they cannot be confused with the final-proton recoil
coordinate `theta_p_rad`.
`EpCMConfigGen.py` generates the corresponding configuration CSVs and PDFs.
The default grid scans final-proton momentum from `0.036` down to `0.020 GeV`,
including `0.028 GeV` exactly. The internal two-body angle scans
`theta_cm = 1.30--1.55` and includes `1.4276943335`, which maps to the
reference ep-CM photon angle `theta_gamma = 1.298`. The oriented recoil angle
scans `3.30--3.56` and includes `theta_p' = 3.429` at input `phi_p' = 0`.
The CSV records the virtuality, final-proton energy, absolute energy loss, and
energy-loss fraction. It writes full CSVs, per-observable rankings, heatmaps,
and an anchor-momentum report
under:

```text
Output_local/EpCMEntanglementScan/
```

The full CSV contains the same 13 entanglement/projection quantities and the
same explicit electron polarization/observable labels as `AlignmentScan`,
alongside `z`, `theta_cm`, `mu`, and slow-proton diagnostics.
The fixed-polarization results are written to
`ep_cm_entanglement_scan.csv`. The coherent two-angle scan is written
separately to `ep_cm_mixing_angle_scan.csv`, with its own ranked
`ep_cm_mixing_angle_top.csv`, so fixed-polarization rows are not duplicated
for every `(alpha_e, alpha_p)` pair.
The per-polarization PDFs plot the absolute value of all 13 quantities; purity
is retained only in the CSV as the mixed-state diagnostic used by
`AlignmentScan`, not as an entanglement heatmap.

Like `AlignmentScan`, point evaluations use a balanced process pool and each
polarization PDF is rendered independently in a bounded plot-process pool.
Edit `SCAN_WORKER_COUNT` and `SCAN_PLOT_WORKER_COUNT` at the top of the script.

After the scan completes, run `python3 EpCMConfigGen.py`. It selects separated
local absolute maxima (and local minimum-`D_W` regions) for all 13 AlignmentScan
observables and every incoming polarization. Each polarization/observable PDF
contains the absolute-value scan map followed by detail pages with exact
kinematics, momentum vectors, and ensemble-aware outgoing helicity-amplitude
components. CSV versions of the configuration, momentum, and amplitude records
are written alongside the marked region PDFs under
`Output_local/EpCMConfigGen/`.
The decomposition labels final helicities explicitly as `h_l`, `h_p`, and
`h_gamma`; it also writes `final_state_ket` in the user's `|p gamma l>`
ordering. Momentum figures follow the AlignmentScan ConfigGen convention:
incoming trajectories terminate at the interaction origin, lepton lines are
dashed, the photon line is wavy, and particles are labelled
`P`, `P'`, `l`, `l'`, and `q_gamma`.
Configuration amplitudes and independent target PDFs are also process-parallel;
their controls are `CONFIGGEN_KINEMATIC_WORKERS` and `CONFIGGEN_PLOT_WORKERS`.
EpCMConfigGen amplitude-decomposition CSVs store the complex phase in radians,
degrees, and units of pi. Each detailed PDF labels every retained amplitude
bar with its phase and includes a phase colorbar from `-pi` to `+pi`.

## Quasi-real Compton helicity components

`QuasiRealComptonHelicity.py` evaluates the reduced tree-level amplitude for
`gamma* + lepton -> gamma + lepton` directly in the incoming
virtual-photon--lepton CM frame. The incoming spacelike photon is decomposed
into `-1`, `+1`, the normalized coherent state
`(|+> + |->)/sqrt(2)`, and longitudinal polarizations, while the
incoming/outgoing lepton and final real photon retain explicit helicity labels. The editable
settings at the top control the lepton mass, CM invariant mass, virtuality, and
angular range. The script writes the complete component table, helicity-summed
polarization responses, Ward-identity diagnostics, and angular plots under:

```text
Output_local/QuasiRealComptonHelicity/
```

The amplitudes omit the overall QED factor `e^2`. Longitudinal results are
reported as spacelike polarization-basis responses, not as probabilities for
an asymptotic photon.

The script also evaluates the exact ep-CM W-state event shared with
`ProtonVirtualPhotonAmp.py`. It writes the `4 x 6` Compton matrix, the
factorized `8 x 4` Bethe--Heitler matrix, and the normalized three-particle
state to `epcm_w_compton_matrix.csv`, `epcm_w_full_scattering_matrix.csv`, and
`epcm_w_final_state.csv`. The factorization
`M = (I_p x C)(I_p x G x I_e)(H x I_e)` is checked entry by entry against the
direct Bethe--Heitler amplitude table. The final-state CSV contains both the
complete result and the two-branch approximation that retains only
`|p'(+1) L>` and `|p'(-1) T->`.

The default Compton setup uses `sqrt(s_gamma_l) = 80 GeV`, corresponding to a
nominal 40 GeV per incoming particle, and evaluates `Q2 = 1, 10, 40 GeV^2`.
Exact CM kinematics give slightly unequal photon and massive-lepton energies.

`FIXED_INCOMING_LEPTON_HELICITY` defaults to `+1`. The comparison PDF uses
columns `T-`, `T+`, and `L`; its first row uses the fixed `+1` lepton and its
second row uses the coherent transverse lepton
`(|+> + |->)/sqrt(2)`. The coherent transverse-photon sum remains available
in the component and response CSVs but is excluded from both plot products.
The transverse-lepton amplitudes are written separately to
`compton_transverse_initial_lepton_components.csv`.

## Prepared-spin example

For fixed-helicity or transversely polarized incoming particles, edit
`ELECTRON_STATE` and `PROTON_STATE` at the top of `FixedHelicityTest.py`.
Each accepts `-1`, `+1`, `"L"`, `"Tx"`, `"-Tx"`, `"Ty"`, or `"-Ty"`:

```sh
python3 FixedHelicityTest.py
```

It writes:

```text
Output_local/FixedHelicityTest/momentum_configuration.csv
Output_local/FixedHelicityTest/outgoing_amplitudes.csv
Output_local/FixedHelicityTest/entanglement_measurements.csv
Output_local/FixedHelicityTest/configuration_summary.pdf
```

The selected pure incoming product state is combined coherently in the
helicity basis. Its outgoing state is formed from all eight
`(hOut, sOut, lambda)` amplitudes.

## Wolfram Language

The `Mathematica/` directory mirrors the Python kinematics, amplitude-table,
density-matrix, and concurrence conventions. Its public kinematic and
amplitude functions take an explicit charged-lepton mass `Ml`. Run the
physical-electron-mass numerical benchmark with:

```sh
wolframscript -file Mathematica/BenchmarkNumeric.wl
```

On macOS, the kernel can be invoked directly:

```sh
/Applications/Wolfram.app/Contents/MacOS/WolframKernel \
  -script Mathematica/BenchmarkNumeric.wl
```

For a symbolic channel, edit `helicityInputs` in
`Mathematica/AnalyticAmplitude.wl`.
