# DVCSHelicityAmp

Numerical Bethe-Heitler helicity amplitudes and spin-density matrix scans for
exclusive electroproduction.

The code builds the external kinematics, evaluates helicity amplitudes for
the Bethe-Heitler process, checks benchmark squared amplitudes, and scans the
outgoing three-particle spin-density matrix. The current spin-density workflow
also computes concurrence observables and the multipartite quantity `F3`
following Eq. (3)-(6) of arXiv:2310.01477v2.

## Repository Layout

```text
Algebra.py         Dirac algebra, spinors, photon polarization vectors.
Kinematics.py     Four-momentum builders and kinematic validation checks.
BHHelicityAmp.py  Bethe-Heitler amplitudes and benchmark log generation.
SpinDensityMat.py Spin-density matrix scans and entanglement observables.
AlignmentScan.py  C_e_p/C_e_gamma/C_p_gamma, M_e/M_p/M_gamma, and F3 scan at characteristic kinematics.
ConfigGen.py      Max C_e_p/C_p_gamma/C_e_gamma/F3 configuration scans from AlignmentScan CSVs.
Mathematica/      Analytic Wolfram Language kinematics, amplitudes, density matrices, and concurrence.
Output/           Generated logs, scan data, CSV files, and plots.
```

All source modules use contravariant four-vectors in `[E, px, py, pz]` order.
The metric convention is implemented in `Algebra.mdot`.

## Dependencies

The scripts are plain Python modules. They require:

```text
numpy
matplotlib
```

`matplotlib` is only needed when saving plots. The scan script forces the
non-interactive `Agg` backend internally before plotting, so it can run without
opening GUI windows.

## Running The Code

Run the Bethe-Heitler benchmark:

```sh
python BHHelicityAmp.py
```

Run the spin-density matrix scans:

```sh
python SpinDensityMat.py
```

Run the C_e_p/C_e_gamma/C_p_gamma, M_e/M_p/M_gamma, and F3 scan at characteristic kinematics:

```sh
python AlignmentScan.py
```

Generate max C_e_p, C_p_gamma, and C_e_gamma configuration scans from the alignment scan:

```sh
python ConfigGen.py
```

Syntax-check all source files:

```sh
python -m py_compile Algebra.py Kinematics.py BHHelicityAmp.py SpinDensityMat.py AlignmentScan.py ConfigGen.py
```

## Physics And Index Conventions

Helicity labels are doubled helicities:

```text
-1, +1
```

The incoming spin labels are:

```text
hIn  incoming electron helicity
sIn  incoming proton spin/helicity
```

The outgoing spin basis used by `SpinDensityMat.py` is ordered as:

```text
(hOut, sOut, lambda)
```

The particle numbering used in the concurrence observables is:

```text
particle 1  outgoing electron helicity hOut
particle 2  outgoing proton spin/helicity sOut
particle 3  outgoing real-photon helicity lambda
```

The final-state density matrix is therefore an `8 x 8` matrix over the three
two-state outgoing degrees of freedom.

## Kinematics

`Kinematics.py` uses one user-frame COM parameterization.

The kinematics result exposes the direct momentum variables:

```text
pIn    incoming COM three-momentum magnitude
pOut   outgoing proton three-momentum magnitude
qOut   outgoing real-photon momentum magnitude
m      proton mass
```

The scan scripts use the independent user-frame variables:

```text
s         total incoming e+p invariant mass squared
theta_in  incoming proton polar angle
phi_in    incoming proton azimuth used internally
phi_in_electron  incoming electron azimuth, phi_in + pi mod 2pi, used in scan plots
qOut      outgoing real-photon energy/momentum magnitude
phiOut    outgoing real-photon azimuth
m         proton mass
```

`pIn` is fixed by `s`, and `pOut` is solved from energy conservation. Derived
invariants such as `Q2`, `xB`, `t`, `W2`, and `y` are written to CSV/logs as
diagnostics, not used as independent scan variables.

The current spin-density scan uses the proton mass

```text
m = 0.938
```

At each kinematic point, the proton form factors are computed from the YAHL
2018 proton lookup table in `YAHL 2018/proton_lookup.dat`. The table gives
Sachs central values as `GEp/GD` and `GMp/(mu_p GD)`. The scan reconstructs
`GE(t)` and `GM(t)` with `Q2_transfer = -t`, then converts them to Dirac and
Pauli form factors with

```text
tau = Q2_transfer / (4 m^2)
F1 = (GE + tau GM) / (1 + tau)
F2 = (GM - GE) / (1 + tau)
```

The generated CSV/NPZ outputs store the resulting per-point `F1` and `F2`
columns alongside the kinematic invariants.

The active scan grids are:

```text
coarse alignment anchors  mid/high s, high theta_in, low/mid/high qOut
coarse two-angle scan     phi_in_electron and phiOut at each anchor
s scan values             user-frame COM energy grid in SpinDensityMat.py
qOut scan values          outgoing photon energy grid in SpinDensityMat.py
theta_in scan values      incoming proton polar-angle grid in SpinDensityMat.py
phiOut scan values        outgoing photon azimuth grid in SpinDensityMat.py
```

Two scans are generated:

```text
user_s_qOut           scan over s and qOut
user_theta_in_phiOut  scan over theta_in and phiOut
```

## Bethe-Heitler Amplitude Workflow

`BHHelicityAmp.py` exposes low-level and convenience functions.

Important entry points:

```text
bh_amplitude_core
    Evaluate one fixed-helicity Bethe-Heitler amplitude from explicit
    four-momenta and a supplied photon polarization vector.

bh_unpolarized_squared_amplitude_core
    Sum |M|^2 over all incoming and outgoing helicity labels for explicit
    four-momenta.

bh_amplitude_user
    Evaluate a fixed-helicity amplitude using direct user-frame momentum
    parameters.

bh_unpolarized_squared_amplitude_user
    Evaluate the helicity-summed squared amplitude using the direct user-frame
    parameters.
```

Running `BHHelicityAmp.py` writes:

```text
Output/BHHelicityAmp.log
```

That log contains benchmark tables comparing the numerical helicity-summed
result against the analytic benchmark path used in the script.

## Mathematica / Wolfram Language Workflow

`Mathematica/BHHelicityAmp.wl` mirrors the Python conventions for general
`thetaIn`. Its independent inputs are
`(s, thetaIn, phiIn, EGamma, phiGamma, Mp)`. It obtains `pIn` and the physical
analytic root for `pOut`, constructs all external four-momenta, evaluates the
complete `4 x 8` Bethe-Heitler helicity-amplitude table, contracts an arbitrary
incoming `4 x 4` spin density matrix, and computes reduced density matrices
and concurrence.

Run the example from the repository root with:

```sh
wolframscript -file Mathematica/BenchmarkNumeric.wl
```

On macOS, if `wolframscript` has not been configured with a kernel path, use
the application kernel directly:

```sh
/Applications/Wolfram.app/Contents/MacOS/WolframKernel -script Mathematica/BenchmarkNumeric.wl
```

The main workflow is:

```text
kin = UserKinematics[s, thetaIn, phiIn, EGamma, phiGamma, Mp];
amps = BHAmplitudeTable[kin, Mp, F1, F2];
rhoIn = InitialSpinDensity["L", "L"];
rhoOut = OutgoingDensityMatrix[amps, rhoIn];
observables = EntanglementObservables[rhoOut];
```

To print one complete symbolic helicity amplitude, edit `helicityInputs` in
`Mathematica/AnalyticAmplitude.wl` and run:

```sh
/Applications/Wolfram.app/Contents/MacOS/WolframKernel \
  -script Mathematica/AnalyticAmplitude.wl
```

The input ordering is
`{hIn,hOut,sIn,sOut,lambda}`. Add `--summary` to validate that the result is a
fully contracted scalar without printing the very large expression.

For one fully symbolic channel, use real kinematic assumptions and simplify
only after selecting the helicities, for example:

```text
kinSymbolic = UserKinematics[s, thetaIn, phiIn, EGamma, phiGamma, Mp];
ampSymbolic = BHAmplitude[
  kinSymbolic, -1, -1, -1, 1, 1, Mp, F1t, F2t
];
ampExplicit = ComplexExpand[ampSymbolic,
  TargetFunctions -> {Re, Im}] // Together;
```

`OutgoingDensityMatrix[amps,rhoIn,False]` returns the unnormalized density
numerator; its trace is the squared-amplitude signal for that incoming state.
The default third argument is `True`, which returns a unit-trace matrix.

The helicity basis is ordered as `{-1,+1}`. The amplitude axes are
`(hIn,sIn)` and `(hOut,sOut,lambda)`. Pairwise Wootters concurrence is
calculated for both pure and mixed outgoing states. One-to-rest concurrence,
`F3`, and CKW residuals are returned only when the outgoing state is pure.

## Spin-Density Matrix Workflow

For each valid kinematic point, `SpinDensityMat.py` builds the amplitude table:

```text
A[in_state, out_state]
```

where:

```text
in_state   (hIn, sIn), 4 possibilities
out_state  (hOut, sOut, lambda), 8 possibilities
```

At every kinematic point the code first constructs the full five-qubit
process density matrix in the basis
`(hIn,sIn,hOut,sOut,lambda)`:

```text
R_(a,f;b,g) = A_(a,f) * conj(A_(b,g)),  shape(R) = 32 x 32
```

The incoming state is `rho_e (x) rho_p`. An unpolarized particle uses `I/2`;
`L`, `Tx`, and `Ty` use pure-state projectors. Contracting and tracing the two
incoming qubits gives the full outgoing matrix:

```text
rho_out = Tr_in[(rho_in^T (x) I_out) R],  shape(rho_out) = 8 x 8
```

When `NORMALIZE_TRACE = True`, each stored outgoing density matrix is
normalized by its own trace, so valid points satisfy `Tr(rho_out)=1`.
`squared_amplitude_M2` remains the fully unpolarized cross section and
`spin_signal_M2/squared_amplitude_M2` is stored as `cross_section_ratio`.

```text
Tr(rho) = 1
```

The script runs a trace benchmark at several kinematic points before saving
the scans. The benchmark verifies the trace condition after normalization.

## Entanglement Observables

All reductions are taken from this contracted `8 x 8` matrix. Pairwise
Wootters concurrence is valid for both mixed and pure outgoing states. The
one-to-rest concurrence, `F3`, and CKW residual formulas used here require a
pure three-qubit state. `SpinDensityMat.py` stores zero for the one-to-rest and
`F3` columns when the contracted outgoing state is mixed; the accompanying
purity column identifies those rows as outside the pure-state formula's
domain.

The output columns are:

```text
C_e_p       two-body concurrence between outgoing electron and proton
C_e_gamma   two-body concurrence between outgoing electron and real photon
C_p_gamma   two-body concurrence between outgoing proton and real photon
C_e_rest    one-to-rest concurrence for electron against proton plus photon
C_p_rest    one-to-rest concurrence for proton against electron plus photon
C_gamma_rest one-to-rest concurrence for photon against electron plus proton
F3          multipartite observable built from the three one-to-rest concurrences
M_e         CKW monogamy residual C_e_rest^2 - C_e_p^2 - C_e_gamma^2
M_p         CKW monogamy residual C_p_rest^2 - C_e_p^2 - C_p_gamma^2
M_gamma     CKW monogamy residual C_gamma_rest^2 - C_e_gamma^2 - C_p_gamma^2
```

## Generated Output

Running `SpinDensityMat.py` cleans and regenerates the spin-density scan
outputs. Spin-case folders follow the direct prepared-state names:

```text
Output/SpinDensityMat.log
Output/SpinDensityMat/unpolarized/user_s_qOut/
Output/SpinDensityMat/unpolarized/user_theta_in_phiOut/
Output/SpinDensityMat/L_proton/...
Output/SpinDensityMat/L_electron/...
Output/SpinDensityMat/Tx_proton/...
Output/SpinDensityMat/Ty_proton/...
Output/SpinDensityMat/Tx_electron/...
Output/SpinDensityMat/Ty_electron/...
Output/SpinDensityMat/LL|LTx|LTy|TxTx|TxTy/...
```

Each scan folder contains:

```text
spin_density_scan_<spin-label>_<scan>.npz
    Numpy archive with the full scan arrays, kinematic grids, density
    matrices, squared amplitudes, traces, validity masks, and entanglement
    measures.

spin_entanglement_scan_<spin-label>_<scan>.csv
    Summary CSV containing the contracted-state purity, cross-section ratio,
    pairwise mixed-state concurrences, and pure-only observables where valid.

spin_entanglement_scan_<spin-label>_<scan>.pdf
    Multi-page PDF heatmaps for the defined concurrence observables and F3.

SpinDensityScan/
    Per-kinematic-point CSV files and two matrix plots per valid point:
    one for the amplitude-normalized density-matrix norm and one for phase.
    Filenames begin with spin_density_<spin-label>_<scan-axis>_...

```

The spin labels used in filenames are the same prepared-state names.

Running `AlignmentScan.py` cleans and regenerates:

```text
Output/AlignmentScan.log
Output/AlignmentScan/electron_photon_spin_correlation_phase_space.csv
Output/AlignmentScan/electron_photon_spin_correlation_aligned.csv
Output/AlignmentScan/ConcurrenceScan/
```

The main spin-correlation CSVs and C_e_p/C_e_gamma/C_p_gamma plus
M_e/M_p/M_gamma and F3 locator outputs are still generated.

The alignment scan records the opening angle theta(e', gamma) over 6
characteristic user-frame anchors. Each anchor fixes `s`, `theta_in`, and
`qOut`, then scans the two remaining angular variables `phi_in_electron` and
`phi_gamma` on a 48 by 48 grid. The stored outgoing-photon azimuth column is
still named `phiOut`, and the internal proton azimuth is still written as
`phi_in`.
The `ConcurrenceScan` folder stores all twelve prepared-state categories:
`unpolarized`; `L_proton`, `L_electron`; `Tx_proton`, `Ty_proton`;
`Tx_electron`, `Ty_electron`; and `LL`, `LTx`, `LTy`, `TxTx`, `TxTy`.
In a double label the electron state is listed first and the proton state
second. `L` means the direct positive-helicity state, not a helicity
asymmetry.
Set `HEATMAP_PLOT_STYLE` in `AlignmentScan.py` to `"grid"` for binned cell
plots or `"contour"` for filled contour plots. AlignmentScan heatmaps plot
the incoming proton azimuth `phi_in` on the x axis and `phi_gamma` on the y
axis, with guide lines at `phi_in = pi/2` and `phi_gamma = pi/2`.
`HEATMAP_MAX_BINS` controls the plotted bin count per angular axis, and
`HEATMAP_CONTOUR_LEVELS` controls the number of filled contour bands.
Heatmap color scales are `0..1` for concurrence and `-1..1` only for
quantities such as numerical monogamy residuals that can be signed.

The top-level spin-density log records the scan settings, particle map, trace
benchmark, normalization convention, saved paths, and invalid kinematic
points if any occur.

Every category uses the same `32 x 32 -> 8 x 8` contraction. A particle not
named as polarized is traced with `I/2`; named states are contracted with
their direct projectors. No polarized category is formed as an asymmetry.

The final electron-photon alignment scan uses `ALIGNMENT_ANGLE_MAX_DEG`
in `AlignmentScan.py` as its small-angle cut. Its main spin-correlation observable is
`<hOut * lambda>`, where `hOut` is the outgoing electron helicity label and
`lambda` is the final real-photon helicity label. The full phase-space CSV
contains all valid angle points; the aligned-only CSV contains only points
passing the configured 3D `theta(e', gamma)` cut and may contain only a header
when the active grid has no such points.

The alignment concurrence/F3/monogamy PDFs contain only the per-anchor
`phi_in_electron` by `phi_gamma` correlation maps for each configured
characteristic kinematic point.

## CSV Structure

The entanglement summary CSV files include kinematic metadata and observable
columns:

```text
spin_case,entanglement_mode,Q2,t,phi,squared_amplitude_M2,spin_signal_M2,
cross_section_ratio,purity,trace,normalized_to_unit_trace,
C_e_p,C_e_gamma,C_p_gamma,
C_e_rest,C_p_rest,C_gamma_rest,F3,M_e,M_p,M_gamma
```

The per-point density-matrix CSV files include:

```text
spin_case,entanglement_mode,Q2,t,phi,squared_amplitude_M2,spin_signal_M2,
cross_section_ratio,purity,trace,normalized_to_unit_trace,
C_e_p,C_e_gamma,C_p_gamma,
C_e_rest,C_p_rest,C_gamma_rest,F3,M_e,M_p,M_gamma,
row_index,row_h_out,row_s_out,row_lambda,col_index,
col_h_out,col_s_out,col_lambda,rho_real,rho_imag,rho_abs,rho_phase
```

`spin_signal_M2` is the cross section for the requested prepared state.
`rho_abs` is the matrix-entry norm after unit-trace normalization.
`rho_phase` is the complex phase in radians.

The alignment-scan CSV files include:

```text
initial_spin_averaging_version,Q2,xB,t,phi,theta_e_gamma_rad,theta_e_gamma_deg,aligned,
squared_amplitude_M2,
<spin_case>_trace,<spin_case>_spin_signal_M2,
<spin_case>_cross_section_ratio,<spin_case>_purity,
<spin_case>_h_out_mean,<spin_case>_lambda_mean,
<spin_case>_h_lambda,<spin_case>_h_lambda_connected
```

`initial_spin_averaging_version=prepared_spin_ensemble_v4` identifies CSVs
created through the full five-qubit process-density contraction.

The `AlignmentScan.py` `ConcurrenceScan` CSV files focus on the C_e_p,
C_e_gamma, C_p_gamma, M_e, M_p, M_gamma, and F3 locator observables for each
spin case:

```text
<spin_case>_C_e_p
<spin_case>_C_e_gamma
<spin_case>_C_p_gamma
<spin_case>_M_e
<spin_case>_M_p
<spin_case>_M_gamma
<spin_case>_F3
```

The `<spin_case>` prefixes are `unpolarized`, `L_proton`, `L_electron`,
`Tx_proton`, `Ty_proton`, `Tx_electron`, `Ty_electron`, `LL`, `LTx`, `LTy`,
`TxTx`, and `TxTy`. The same folder writes
`electron_photon_concurrence_top.csv`, a ranked locator table used to inspect
the best C_e_p, C_e_gamma, C_p_gamma, M_e, M_p, M_gamma, and F3 points. It
also writes `electron_photon_e_gamma_top.csv` for the electron-photon
downstream configuration generator. The concurrence/F3/monogamy PDFs include only two-angle
`phi_in_electron` by `phi_gamma` maps for every configured characteristic
anchor.

`ConfigGen.py` reads
`Output/AlignmentScan/ConcurrenceScan/electron_photon_concurrence_phase_space.csv`
when available, then falls back to ranked locator CSVs. It finds the strongest
regions for each target observable `C_e_p`, `C_p_gamma`, `C_e_gamma`, and `F3`,
clusters them in each fixed-`E_gamma` `phi_in` by
`phi_gamma` scan for each polarization config, and writes the numerical
configuration data under `Output/ConfigGen/Data`:

ConfigGen scans `F3` for all configured polarization cases, including
unpolarized, single-particle polarized, and double-polarized inputs. The
generated tables include `selected_purity`. For mixed outgoing states,
`SpinDensityMat.py` stores `F3 = 0` because the implemented `F3` formula is a
pure-three-qubit-state observable; those zero-valued scans are retained so the
full polarization set is represented.

`ConfigGen.py` rejects older AlignmentScan CSVs without
`prepared_spin_ensemble_v4`; rerun `AlignmentScan.py` before configuration
generation when the convention is missing or stale.

```text
Output/ConfigGen.log
Output/ConfigGen/Data/<target>/combined/*.csv
Output/ConfigGen/Data/<target>/<polarization>/*.csv
Output/ConfigGen/Config_Plot_By_Egamma/<polarization>/<E_gamma>_<target>_regions.pdf
```

Each PDF fixes one `E_gamma` value, one target entanglement observable, and one
polarization config. No PDF combines different `E_gamma` values or different
polarization configs. The first page is the fixed-energy observable scan map
for that target and polarization, with the located maximum regions marked;
the map uses the incoming proton azimuth `phi_in` as the x coordinate, draws
guide lines at `phi_in = pi/2` and `phi_gamma = pi/2`, and uses a fixed
`0..1` observable color scale;
the following pages show the reconstructed momentum configuration, kinematics,
and final-state helicity-amplitude decomposition for each selected region.
For incoherent initial-spin ensembles, the decomposition keeps each initial
component separate and records `initial_component` and `ensemble_weight`. It
does not replace an unpolarized ensemble by a coherent amplitude sum. Amplitude
decompositions retain only components contributing at least `2%` of the full
ensemble-weighted norm and at most the leading eight components; each table
also records the retained total fraction.
The target CSVs include the corresponding per-`E_gamma` region rows in the
momentum and amplitude tables, and per-spin CSV files are also written for the
spin cases represented in those selected configuration rows.

Each characteristic page shows the rebuilt user-frame momentum configuration
as a 3D vector plot and as a transverse `p_x-p_y` projection. The plotted
momenta are displayed as `\ell`, `P`, `\ell'`, `P'`, and `q_gamma`; the
virtual photon `q` is omitted from these configuration plots. The momentum
drawings use dashed lines for electrons, solid lines for protons, and wavy
transverse lines for photons; the 3D panel uses a dotted photon line. The
pages also list kinematic variables and four-momenta with math labels and
include the final-state helicity-amplitude decomposition using `h_e`, `h_p`,
and `h_gamma`, with positive helicities written as `+1`.
