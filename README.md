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
```

Run the main workflows from the repository root:

```sh
python3 BHHelicityAmp.py     # amplitude benchmark
python3 SpinDensityMat.py    # spin-density scans
python3 AlignmentScan.py     # angular alignment and entanglement scan
python3 ConfigGen.py         # selected configurations from AlignmentScan
python3 PhaseSpaceScan.py    # adaptive all-observable/all-lepton phase-space scan
python3 PhaseSpaceConfigScan.py  # ConfigGen packages from PhaseSpaceScan results
python3 EpCMEntanglementScan.py   # slow-recoil-proton heavy-lepton scan
python3 EpCMConfigGen.py          # config packages from the focused ep-CM scan
python3 ProtonVirtualPhotonAmp.py # proton-current virtual-photon decomposition
python3 QuasiRealComptonHelicity.py  # gamma* lepton CM helicity components
```

Generated data, plots, and logs are written under `Output/`.

## Main files

```text
config.py             Shared masses, normalization, and worker settings
Algebra.py            Dirac algebra, spinors, and photon polarizations
Kinematics.py         User-frame momenta and kinematic checks
BHHelicityAmp.py      Bethe–Heitler amplitudes and benchmarks
SpinDensityMat.py     Density matrices and entanglement observables
AlignmentScan.py      Fine angular scan at characteristic kinematics
ConfigGen.py          Ranked-region configuration and plot generator
PhaseSpaceScan.py      Adaptive five-dimensional entanglement phase-space scan
PhaseSpaceConfigScan.py ConfigGen-style packages from PhaseSpaceScan results
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

AlignmentScan and ConfigGen use the physical electron mass. The reusable
low-level amplitude and kinematic APIs retain `electron_mass=0.0` as their
backward-compatible default.

## Kinematics

`Kinematics.py` uses a user-frame COM parameterization with independent
variables

```text
s, theta_in, phi_in, qOut, phiOut
```

Here `theta_in` and `phi_in` define the incoming proton direction, while the
incoming electron points oppositely. `qOut` and `phiOut` specify the outgoing
real photon. The code computes the incoming COM momentum `pIn` and solves the
outgoing proton momentum `pOut` from energy conservation.

The returned kinematic record includes the momenta `k`, `p`, `kp`, `pp`, and
`qout`, together with `Q2`, `xB`, `t`, `W2`, and `y`.

Example with the physical electron mass:

```python
from config import ELECTRON_MASS_GEV, PROTON_MASS_GEV
from Kinematics import kinematics_user_from_independent

kin = kinematics_user_from_independent(
    s, theta_in, phi_in, qOut, phiOut,
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
bh_amplitude_user                         user-frame convenience wrapper
bh_unpolarized_squared_amplitude_user     user-frame unpolarized wrapper
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
`Output/BHHelicityAmp.log`. The analytic benchmark remains a massless-electron
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

`AlignmentScan.py` scans `phi_in_lepton` and `phiOut` at characteristic
values of `s`, `theta_in`, and `qOut`. It records the outgoing
lepton–photon opening angle and writes full, aligned-only, and ranked tables
directly in each species directory:

```text
Output/AlignmentScan/<lepton>/
Output/AlignmentScan/<lepton>/concurrence_scan_lepton_<species>_<polarization>_proton_<polarization>.pdf
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
Output/ConfigGen/<lepton>/Data/<target>/lepton_<species>_<polarization>_proton_<polarization>/...
Output/ConfigGen/<lepton>/lepton_<species>_<polarization>_proton_<polarization>/<E_gamma>_<target>_regions.pdf
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

`PhaseSpaceScan.py` performs a stratified five-dimensional scan followed by local
refinement around the best candidate for every AlignmentScan observable and
polarization. It runs electron, muon, heavy-lepton, and massless-lepton
species by default, and writes independent AlignmentScan-compatible full,
aligned, ranked, and plotted results under `Output/PhaseSpaceScan/<lepton>/`.
Its plot filenames use the same explicit convention:
`phase_space_scan_lepton_<species>_<polarization>_proton_<polarization>.pdf`.
Point evaluations run in parallel. Edit `LEPTONS_TO_SCAN`,
`PHASE_SPACE_SCAN_WORKERS`, sample counts, ranges, and output settings at the
top of `PhaseSpaceScan.py`.

`PhaseSpaceConfigScan.py` consumes those full phase-space CSVs and applies the
same clustering, reconstructed-momentum, helicity-amplitude decomposition, and
per-polarization PDF workflow as `ConfigGen.py`. Because `PhaseSpaceScan` uses
a continuous photon energy, its valid rows are divided into balanced low,
middle, and high `E_gamma` bands before configuration selection. Outputs are
written under `Output/PhaseSpaceConfigScan/<lepton>/`, with a combined report
at `Output/PhaseSpaceConfigScan.log`.
Because it inherits the same ConfigGen targets, it also writes maximum-magic
packages under `Data/m2_magic/`.

## Focused ep-CM slow-recoil-proton scan

`EpCMEntanglementScan.py` targets the heavy-lepton region with incoming ep-CM
momentum `p = 50 GeV` and a slow final proton after transferring most of its
energy to the virtual photon. It constructs the final
state by an exact two-body boost, rather than by using approximate massless
four-vectors. The default
focused polarization set additionally includes direct proton preparations
along `-Tx` and `-Ty`, with the incoming lepton averaged incoherently.
It also includes the direct product preparations `L lepton + -Tx proton` and
`L lepton + -Ty proton`.
`EpCMConfigGen.py` generates the corresponding configuration CSVs and PDFs.
The default grid samples the final-proton momentum logarithmically from
`5 GeV` down to `0.05 GeV` at 13 points, equivalent to
`z = 0.90--0.999`, and covers the complete final-state polar-angle range
`theta_cm = 0--pi` in three-degree steps. The proton azimuth is fixed to zero
to define the reference scattering plane, while its recoil polar angle spans
`theta_p = 0--pi` in ten-degree steps. This
high-transfer region is not labelled
quasi-real: the CSV records the virtuality, final-proton energy, absolute
energy loss, and energy-loss fraction. It writes a full CSV, per-observable
rankings, heatmaps versus final-proton momentum reduced over `theta_p`, and an
anchor-momentum report
under:

```text
Output/EpCMEntanglementScan/
```

The full CSV contains the same 13 entanglement/projection quantities and the
same explicit heavy-lepton polarization/observable labels as `AlignmentScan`,
alongside `z`, `theta_cm`, `mu`, and slow-proton diagnostics.
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
`Output/EpCMConfigGen/`.
The decomposition labels final helicities explicitly as `h_l`, `h_p`, and
`h_gamma`. Momentum figures follow the AlignmentScan ConfigGen convention:
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
Output/QuasiRealComptonHelicity/
```

The amplitudes omit the overall QED factor `e^2`. Longitudinal results are
reported as spacelike polarization-basis responses, not as probabilities for
an asymptotic photon.

The default Compton setup uses `sqrt(s_gamma_l) = 80 GeV`, corresponding to a
nominal 40 GeV per incoming particle, and evaluates `Q2 = 1, 10, 100 GeV^2`.
Exact CM kinematics give slightly unequal photon and massive-lepton energies.

`FIXED_INCOMING_LEPTON_HELICITY` defaults to `+1`. A dedicated filtered CSV
and PDF compare the four final `(h_l_out, h_gamma_out)` channels separately
for each incoming virtual-photon polarization across `theta_cm`. This includes
the coherent transverse sum at amplitude level, so its plots retain the
interference between the `+1` and `-1` components. The CSV also ranks the four
final channels and records their fractions within each initial state.

## Prepared-spin example

For fixed-helicity or transversely polarized incoming particles, edit
`ELECTRON_STATE` and `PROTON_STATE` at the top of `FixedHelicityTest.py`.
Each accepts `-1`, `+1`, `"L"`, `"Tx"`, `"-Tx"`, `"Ty"`, or `"-Ty"`:

```sh
python3 FixedHelicityTest.py
```

It writes:

```text
Output/FixedHelicityTest/momentum_configuration.csv
Output/FixedHelicityTest/outgoing_amplitudes.csv
Output/FixedHelicityTest/entanglement_measurements.csv
Output/FixedHelicityTest/configuration_summary.pdf
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
