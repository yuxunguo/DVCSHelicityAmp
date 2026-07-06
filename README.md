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
AlignmentScan.py  Final electron-photon alignment phase-space scan.
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

Run the final electron-photon alignment scan:

```powershell
C:\Users\sFerm\AppData\Local\Python\bin\python.exe AlignmentScan.py
```

Syntax-check all source files:

```powershell
py -m py_compile Algebra.py Kinematics.py BHHelicityAmp.py SpinDensityMat.py AlignmentScan.py
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

`Kinematics.py` supports two related parameterizations.

The direct user-frame backend uses:

```text
pIn    incoming COM three-momentum magnitude
pOut   outgoing proton three-momentum magnitude
qOut   outgoing real-photon momentum magnitude
th     incoming proton polar angle
ph     incoming proton azimuth
phOut  outgoing real-photon azimuth in the user frame
m      proton mass
```

The scalar exclusive input path uses:

```text
Eb   beam energy
Q2   photon virtuality, positive Q^2
xB   Bjorken x
t    momentum transfer, usually negative
phi  hadronic azimuthal angle when AZIMUTH_INPUT = "phi_hadron"
m    proton mass
```

The scalar path is built in the target rest frame, boosted to the initial
electron-proton COM frame, and rotated into the user-frame convention.

The current spin-density scan settings in `SpinDensityMat.py` are:

```text
Eb = 11.0
xB = 0.36
m = 0.938
F1 = 1.0
F2 = 0.0
AZIMUTH_INPUT = "phi_hadron"
```

The active scan grids are:

```text
Q2 scan values   1.0 to 6.0, 11 points
t scan values    -1.2 to -0.2, 11 points
phi scan values  0 to 2*pi, 12 points, endpoint excluded
```

Two scans are generated:

```text
Q2_t    scan over Q2 and t at fixed phi = 0.7
Q2_phi  scan over Q2 and phi at fixed t = -0.4
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
    Evaluate a fixed-helicity amplitude using the direct user-frame scalar
    momentum parameters.

bh_unpolarized_squared_amplitude_user
    Evaluate the helicity-summed squared amplitude using the direct user-frame
    parameters.

bh_amplitude_cm_from_beam_energy
    Evaluate a fixed-helicity amplitude using `(Eb, Q2, xB, t, phi)`.

bh_unpolarized_squared_amplitude_cm_from_beam_energy
    Evaluate the helicity-summed squared amplitude using
    `(Eb, Q2, xB, t, phi)`.
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
M1      one-particle reduced determinant term for particle 1
M2      one-particle reduced determinant term for particle 2
M3      one-particle reduced determinant term for particle 3
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
are written under `transverse`:

```text
Output/SpinDensityMat.log
Output/SpinDensityMat/unpolarized/Q2_t/
Output/SpinDensityMat/unpolarized/Q2_phi/
Output/SpinDensityMat/polarized/Q2_t/
Output/SpinDensityMat/polarized/Q2_phi/
Output/SpinDensityMat/transverse/Q2_t/
Output/SpinDensityMat/transverse/Q2_phi/
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
    incoming proton spin; the transverse folders contain observables for
    (hIn=-1 + hIn=+1)/sqrt(2) at the configured incoming proton spin.

spin_entanglement_scan_<spin-label>_<scan>.pdf
    Multi-page PDF heatmaps for the concurrence observables and F3. Polarized
    plots use a signed color scale for the helicity-difference observables.

SpinDensityScan/
    Per-kinematic-point CSV files and two matrix plots per valid point:
    one for the amplitude-normalized density-matrix norm and one for phase.
    Filenames begin with spin_density_<spin-label>_Q2_...

```

The spin labels used in filenames are `unpolarized`,
`longitudinal_polarized`, and `transverse`.

Running `AlignmentScan.py` cleans and regenerates:

```text
Output/SpinDensityMat/AlignmentScan/AlignmentScan.log
Output/SpinDensityMat/AlignmentScan/electron_photon_spin_correlation_phase_space.csv
Output/SpinDensityMat/AlignmentScan/electron_photon_spin_correlation_aligned.csv
Output/SpinDensityMat/AlignmentScan/electron_photon_spin_correlation_unpolarized.pdf
Output/SpinDensityMat/AlignmentScan/electron_photon_spin_correlation_polarized.pdf
```

The alignment scan records the opening angle theta(e', gamma) over Q2, xB,
t, and phi, and computes final electron-photon spin correlations for points
inside the configured small-angle cut. The unpolarized and polarized PDFs are
separate binned theta-vs-kinematics temperature-map documents.

For the current grids, each spin case generates these per-point artifact
counts:

```text
Q2_t    121 CSV files, 242 matrix PDFs
Q2_phi  132 CSV files, 264 matrix PDFs
```

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

The transverse scan matrix is
`sum_sIn rho((hIn=-1 + hIn=+1)/sqrt(2),sIn)`, including the coherent
interference between incoming electron helicities. When trace normalization is
enabled, it is divided by the unpolarized squared amplitude `M^2`.

The transverse entanglement scan uses
`E((hIn=-1 + hIn=+1)/sqrt(2),sIn)` at the configured
`ENTANGLEMENT_INITIAL_STATE` proton spin.

The final electron-photon alignment scan uses `ALIGNMENT_ANGLE_MAX_DEG`
in `AlignmentScan.py` as its small-angle cut. Its main spin-correlation observable is
`<hOut * lambda>`, where `hOut` is the outgoing electron helicity label and
`lambda` is the final real-photon helicity label. The full phase-space CSV
contains all valid angle points; correlation columns are filled for aligned
points where the amplitude table is evaluated.

The alignment PDFs plot observables as binned heatmaps with `theta(e', gamma)`
on the horizontal axis and `Q2`, `xB`, `t`, or `phi` on the vertical axis.
Marker rings on the heatmap pages show occupied bins.

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
transverse trace numerator for transverse scans.
`rho_abs` is the matrix-entry norm after the configured `M^2` normalization.
`rho_phase` is the complex phase in radians.

The alignment-scan CSV files include:

```text
Q2,xB,t,phi,theta_e_gamma_rad,theta_e_gamma_deg,aligned,
squared_amplitude_M2,unpolarized_trace,unpolarized_h_out_mean,
unpolarized_lambda_mean,unpolarized_h_lambda,
unpolarized_h_lambda_connected,unpolarized_C13,polarized_trace,
polarized_spin_signal_M2,polarized_h_lambda,polarized_delta_C13
```
