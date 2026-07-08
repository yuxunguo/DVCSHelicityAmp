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
AlignmentScan.py  C12/C13/C23, M1/M2/M3, and F3 scan at characteristic kinematics.
ConfigGen.py      Representative high-C13 configs from AlignmentScan CSVs.
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

On this Windows checkout, the Python launcher may vary by environment. The
working interpreter used for the current output regeneration was:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe
```

## Running The Code

Run the Bethe-Heitler benchmark:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe BHHelicityAmp.py
```

Run the spin-density matrix scans:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe SpinDensityMat.py
```

Run the C12/C13/C23, M1/M2/M3, and F3 scan at characteristic kinematics:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe AlignmentScan.py
```

Generate representative high-C13 configurations from the alignment scan:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe ConfigGen.py
```

Syntax-check all source files:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe -m py_compile Algebra.py Kinematics.py BHHelicityAmp.py SpinDensityMat.py AlignmentScan.py ConfigGen.py
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

The direct backend variables are:

```text
pIn    incoming COM three-momentum magnitude
pOut   outgoing proton three-momentum magnitude
qOut   outgoing real-photon momentum magnitude
th     incoming proton polar angle
ph     incoming proton azimuth
phOut  outgoing real-photon azimuth in the user frame
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

The current spin-density scan settings in `SpinDensityMat.py` are:

```text
m = 0.938
F1 = 1.0
F2 = 0.0
```

The active scan grids are:

```text
coarse alignment anchors  low/medium/high s, low/high theta_in, low/medium/high qOut
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

The outgoing density matrix is constructed as:

```text
rho_ij = sum_initial A_initial,i * conj(A_initial,j)
```

The squared amplitude used for normalization is:

```text
M^2 = sum_initial,outgoing |A_initial,outgoing|^2
```

When `NORMALIZE_TRACE = True`, the stored density matrix is normalized by
this `M^2`, so valid scan points should satisfy:

```text
Tr(rho) = 1
```

The script runs a trace benchmark at several kinematic points before saving
the scans. The benchmark verifies the trace condition after normalization.

## Entanglement Observables

The concurrence observables in `SpinDensityMat.py` are evaluated from one
fixed incoming pure amplitude row:

```text
ENTANGLEMENT_INITIAL_STATE = (+1, +1)
```

This is separate from the helicity-summed density matrix saved for the scan.
The reason is that Eq. (3)-(6) of arXiv:2310.01477v2 are pure-state
three-qubit formulas. Summing over incoming helicities produces a mixed
outgoing state, which is not directly compatible with those pure-state
concurrence definitions.

The output columns are:

```text
C12     two-body concurrence between outgoing particles 1 and 2
C13     two-body concurrence between outgoing particles 1 and 3
C23     two-body concurrence between outgoing particles 2 and 3
C1_23   one-to-rest concurrence for particle 1 against particles 2 and 3
C2_13   one-to-rest concurrence for particle 2 against particles 1 and 3
C3_12   one-to-rest concurrence for particle 3 against particles 1 and 2
F3      multipartite observable built from C1_23, C2_13, C3_12
M1      CKW monogamy residual C1_23^2 - C12^2 - C13^2
M2      CKW monogamy residual C2_13^2 - C12^2 - C23^2
M3      CKW monogamy residual C3_12^2 - C13^2 - C23^2
```

With the particle map above, `C1_23` measures entanglement of the outgoing
electron with the outgoing proton plus real photon, `C2_13` measures the
outgoing proton against the other two, and `C3_12` measures the outgoing
photon against the other two.

## Generated Output

Running `SpinDensityMat.py` cleans and regenerates the spin-density scan
outputs. The current unpolarized density-matrix scans are written under
`unpolarized`, the incoming-electron polarized helicity-difference scans are
written under `polarized`, and the coherent transverse incoming-electron scans
are written under `transverse_Tx` and `transverse_Ty`:

```text
Output/SpinDensityMat.log
Output/SpinDensityMat/unpolarized/user_s_qOut/
Output/SpinDensityMat/unpolarized/user_theta_in_phiOut/
Output/SpinDensityMat/polarized/user_s_qOut/
Output/SpinDensityMat/polarized/user_theta_in_phiOut/
Output/SpinDensityMat/transverse_Tx/user_s_qOut/
Output/SpinDensityMat/transverse_Tx/user_theta_in_phiOut/
Output/SpinDensityMat/transverse_Ty/user_s_qOut/
Output/SpinDensityMat/transverse_Ty/user_theta_in_phiOut/
```

Each scan folder contains:

```text
spin_density_scan_<spin-label>_<scan>.npz
    Numpy archive with the full scan arrays, kinematic grids, density
    matrices, squared amplitudes, traces, validity masks, and entanglement
    measures.

spin_entanglement_scan_<spin-label>_<scan>.csv
    Summary CSV containing one row per valid kinematic point. The unpolarized
    folders contain pure-initial-state observables; the polarized folders
    contain hIn=+1 minus hIn=-1 entanglement differences at the configured
    incoming proton spin; the transverse Tx and Ty folders contain observables
    for (hIn=+1 + hIn=-1)/sqrt(2) and
    (hIn=+1 + i hIn=-1)/sqrt(2), respectively, at the configured incoming
    proton spin.

spin_entanglement_scan_<spin-label>_<scan>.pdf
    Multi-page PDF heatmaps for the concurrence observables and F3. Polarized
    plots use a signed color scale for the helicity-difference observables.

SpinDensityScan/
    Per-kinematic-point CSV files and two matrix plots per valid point:
    one for the amplitude-normalized density-matrix norm and one for phase.
    Filenames begin with spin_density_<spin-label>_<scan-axis>_...

```

The spin labels used in filenames are `unpolarized`,
`longitudinal_polarized`, `transverse_Tx`, and `transverse_Ty`.

Running `AlignmentScan.py` cleans and regenerates:

```text
Output/AlignmentScan.log
Output/AlignmentScan/electron_photon_spin_correlation_phase_space.csv
Output/AlignmentScan/electron_photon_spin_correlation_aligned.csv
Output/AlignmentScan/DensityMatScan/
Output/AlignmentScan/AmplitudeScan/
Output/AlignmentScan/ConcurrenceScan/
```

Set `RUN_ALIGNMENT_DENSITY_MATRIX_SCAN` or `RUN_ALIGNMENT_AMPLITUDE_SCAN` to
`False` in `AlignmentScan.py` to skip those optional CSV/PDF output families.
The main spin-correlation CSVs and C12/C13/C23 plus M1/M2/M3 and F3 locator
outputs are still generated.

The alignment scan records the opening angle theta(e', gamma) over 18
characteristic user-frame anchors. Each anchor fixes `s`, `theta_in`, and
`qOut`, then scans the two remaining angular variables `phi_in_electron` and
`phi_gamma` on a 72 by 96 grid. The stored outgoing-photon azimuth column is
still named `phiOut`, and the internal proton azimuth is still written as
`phi_in`.
The `DensityMatScan` folder stores reduced 4 by 4
electron-photon density-matrix CSVs and magnitude/phase PDFs. The
`AmplitudeScan` folder stores 2 by 2 complex electron-photon amplitude CSVs
and magnitude/phase PDFs. The `ConcurrenceScan` folder stores concurrence CSVs
and PDFs. The density-matrix and concurrence folders cover unpolarized,
longitudinal polarized, transverse Tx polarized, and transverse Ty polarized
incoming-electron spin cases, plus a double-transverse case where the incoming
electron and proton are both polarized along the same transverse Tx direction.

The top-level spin-density log records the scan settings, particle map, trace
benchmark, normalization convention, saved paths, and invalid kinematic
points if any occur.

The polarized scan matrix is
`sum_sIn rho(hIn=+1,sIn) - sum_sIn rho(hIn=-1,sIn)`. When trace
normalization is enabled, this helicity-difference matrix is divided by the
unpolarized squared amplitude `M^2`, so the matrix output remains available
even when the helicity-difference trace is zero.

The polarized entanglement scan is
`E(hIn=+1,sIn) - E(hIn=-1,sIn)` for each concurrence/F3 observable, using
the configured `ENTANGLEMENT_INITIAL_STATE` proton spin.

The transverse Tx scan matrix is
`sum_sIn rho((hIn=+1 + hIn=-1)/sqrt(2),sIn)`, including the coherent
interference between incoming electron helicities. The transverse Ty scan
matrix uses
`sum_sIn rho((hIn=+1 + i hIn=-1)/sqrt(2),sIn)`. When trace normalization is
enabled, each transverse matrix is divided by the unpolarized squared
amplitude `M^2`.

The transverse entanglement scans use the same Tx and Ty coherent incoming
electron states at the configured `ENTANGLEMENT_INITIAL_STATE` proton spin.

The alignment-only double-transverse category uses the coherent incoming state
`(|hIn=+1> + |hIn=-1>) (|sIn=+1> + |sIn=-1>) / 2`, so the initial electron
and proton are both polarized along the same transverse Tx direction.

The final electron-photon alignment scan uses `ALIGNMENT_ANGLE_MAX_DEG`
in `AlignmentScan.py` as its small-angle cut. Its main spin-correlation observable is
`<hOut * lambda>`, where `hOut` is the outgoing electron helicity label and
`lambda` is the final real-photon helicity label. The full phase-space CSV
contains all valid angle points; correlation columns are filled for aligned
points where the amplitude table is evaluated.

The alignment concurrence/F3/monogamy PDFs contain only the per-anchor
`phi_in_electron` by `phi_gamma` correlation maps for each characteristic
kinematic point.

## CSV Structure

The entanglement summary CSV files include kinematic metadata and observable
columns:

```text
spin_case,entanglement_mode,Q2,t,phi,squared_amplitude_M2,spin_signal_M2,
trace,normalized_by_squared_amplitude,
entanglement_h_in,entanglement_s_in,C12,C13,C23,C1_23,C2_13,C3_12,
F3,M1,M2,M3
```

The per-point density-matrix CSV files include:

```text
spin_case,entanglement_mode,Q2,t,phi,squared_amplitude_M2,spin_signal_M2,
trace,normalized_by_squared_amplitude,
entanglement_h_in,entanglement_s_in,C12,C13,C23,C1_23,C2_13,C3_12,
F3,M1,M2,M3,row_index,row_h_out,row_s_out,row_lambda,col_index,
col_h_out,col_s_out,col_lambda,rho_real,rho_imag,rho_abs,rho_phase
```

`spin_signal_M2` is the same as `squared_amplitude_M2` for unpolarized scans,
the signed helicity-difference trace numerator for polarized scans, and the
transverse trace numerator for Tx and Ty transverse scans.
`rho_abs` is the matrix-entry norm after the configured `M^2` normalization.
`rho_phase` is the complex phase in radians.

The alignment-scan CSV files include:

```text
Q2,xB,t,phi,theta_e_gamma_rad,theta_e_gamma_deg,aligned,
squared_amplitude_M2,
<spin_case>_trace,<spin_case>_spin_signal_M2,
<spin_case>_h_out_mean,<spin_case>_lambda_mean,
<spin_case>_h_lambda,<spin_case>_h_lambda_connected
```

The `DensityMatScan` CSV files add:

```text
<spin_case>_rho_ep_r0_c0_real,<spin_case>_rho_ep_r0_c0_imag,
...
<spin_case>_rho_ep_r3_c3_real,<spin_case>_rho_ep_r3_c3_imag
```

The `AmplitudeScan` CSV files add:

```text
amplitude_normalization_sqrt_M2,
<spin_case>_amp_ep_norm_r0_c0_real,<spin_case>_amp_ep_norm_r0_c0_imag,
...
<spin_case>_amp_ep_norm_r1_c1_real,<spin_case>_amp_ep_norm_r1_c1_imag
```

The `AlignmentScan.py` `ConcurrenceScan` CSV files focus on the C12, C13, C23,
M1, M2, M3, and F3 locator observables for each spin case:

```text
<spin_case>_C12
<spin_case>_C13
<spin_case>_C23
<spin_case>_M1
<spin_case>_M2
<spin_case>_M3
<spin_case>_F3
```

The `<spin_case>` prefixes are `unpolarized`, `longitudinal_polarized`, `Tx`,
`Ty`, and `double_transverse`. The same folder also writes
`electron_photon_concurrence_top.csv`, a ranked locator table used to inspect
the best C12, C13, C23, M1, M2, M3, and F3 points. It also keeps
`electron_photon_c13_top.csv` for the C13-only downstream configuration
generator. The concurrence/F3/monogamy PDFs include only two-angle
`phi_in_electron` by `phi_gamma` maps for every characteristic anchor. The
`rho_ep_r*_c*` columns are the proton-traced
4 by 4 electron-photon reduced density matrix entries, stored as real and
imaginary parts. The reduced basis is ordered as
`(hOut, lambda) = (-1,-1), (-1,+1), (+1,-1), (+1,+1)`. The reduced-density
PDFs show all 16 matrix entries as 4 by 4 grids across the full valid
kinematic scan, with separate magnitude and phase files.

The `AmplitudeScan` matrices are ordered by outgoing electron helicity rows
`hOut = -1,+1` and photon helicity columns `lambda = -1,+1`, and coherently
sum over the outgoing proton spin. The unpolarized amplitude uses an equal
incoming-spin superposition, the longitudinal-polarized amplitude uses the
`hIn=+1` minus `hIn=-1` combination at the configured proton spin, and the
transverse Tx amplitude uses the `hIn=+1` plus `hIn=-1` combination at the
configured proton spin, and the transverse Ty amplitude uses
`hIn=+1` plus `i hIn=-1`. The double-transverse amplitude uses the product of
the same Tx coherent superposition for both incoming electron and incoming
proton. The stored and plotted entries are normalized as
`M / sqrt(M^2_unpol)`, where `M^2_unpol` is the `squared_amplitude_M2` value in
the same row.

`ConfigGen.py` reads `Output/AlignmentScan/ConcurrenceScan/electron_photon_c13_top.csv`
directly, or falls back to the full concurrence phase-space CSV, and writes:

```text
Output/ConfigGen.log
Output/ConfigGen/high_c13_configuration_examples.csv
Output/ConfigGen/high_c13_cluster_summary.csv
Output/ConfigGen/high_c13_momentum_configurations.csv
Output/ConfigGen/high_c13_final_state_amplitude_decomposition.csv
Output/ConfigGen/high_c13_user_frame_configurations.pdf
```

The ConfigGen PDF focuses on the polarized `Tx` and `Ty` high-C13 clusters.
It starts with angular cluster maps, then adds one characteristic page per
selected cluster. Each characteristic page shows the rebuilt user-frame
momentum configuration as a 3D vector plot and as a transverse `px-py`
projection, with incoming `k` and `p` arrows ending at the origin. The plotted
momenta are `k`, `p`, `kp`, `pp`, and `qout`; the virtual photon `q` is omitted
from these configuration plots. The page also lists the corresponding
kinematic variables and four-momenta and plots the final-state helicity-
amplitude decomposition.
