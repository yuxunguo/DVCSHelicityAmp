(* ::Package:: *)

BeginPackage["BHHelicityAmp`"];

POutAnalytic::usage = "POutAnalytic[s,EGamma,phiGamma,Mp] gives the physical outgoing-proton momentum.";
UserKinematics::usage = "UserKinematics[s,thetaIn,phiIn,EGamma,phiGamma,Mp] returns external four-momenta and invariants.";
KinematicChecks::usage = "KinematicChecks[kin,Mp] returns conservation and on-shell residuals.";
BHAmplitude::usage = "BHAmplitude[kin,hIn,hOut,sIn,sOut,lambda,Mp,F1,F2] gives one Bethe-Heitler helicity amplitude.";
BHAmplitudeTable::usage = "BHAmplitudeTable[kin,Mp,F1,F2] gives the 4x8 amplitude table in the bases (hIn,sIn) and (hOut,sOut,lambda), ordered with labels {-1,+1}.";
SingleSpinDensity::usage = "SingleSpinDensity[axis] gives an incoming one-qubit density matrix for Unpolarized, L, Tx, Ty, or an explicit state.";
InitialSpinDensity::usage = "InitialSpinDensity[eAxis,pAxis] gives the incoming electron-proton 4x4 density matrix.";
OutgoingDensityMatrix::usage = "OutgoingDensityMatrix[amplitudes,rhoIn,normalize] returns the outgoing 8x8 density matrix. normalize defaults to True.";
ReducedDensityMatrix::usage = "ReducedDensityMatrix[rho,keep] traces unwanted qubits. Qubits {1,2,3} are electron, proton, photon.";
TwoQubitConcurrence::usage = "TwoQubitConcurrence[rho2] evaluates Wootters concurrence for a numerical 4x4 density matrix.";
EntanglementObservables::usage = "EntanglementObservables[rho] returns pairwise concurrence and pure-state multipartite observables.";

Begin["`Private`"];

$Helicities = {-1, 1};
$Eta = {1, -1, -1, -1};
$Sigma1 = {{0, 1}, {1, 0}};
$Sigma2 = {{0, -I}, {I, 0}};
$Sigma3 = {{1, 0}, {0, -1}};
$Identity2 = IdentityMatrix[2];
$Zero2 = ConstantArray[0, {2, 2}];
$Gamma = {
  ArrayFlatten[{{$Identity2, $Zero2}, {$Zero2, -$Identity2}}],
  ArrayFlatten[{{$Zero2, $Sigma1}, {-$Sigma1, $Zero2}}],
  ArrayFlatten[{{$Zero2, $Sigma2}, {-$Sigma2, $Zero2}}],
  ArrayFlatten[{{$Zero2, $Sigma3}, {-$Sigma3, $Zero2}}]
};

MinkowskiDot[a_, b_] := Expand[a . ($Eta b)];
Covariant[v_] := $Eta v;
Slash[v_] := Sum[$Eta[[mu]] v[[mu]] $Gamma[[mu]], {mu, 1, 4}];
SpinorBar[u_] := Conjugate[u] . $Gamma[[1]];

ChiNorth[p3_, h_] := Module[{px, py, pz, pAbs, den},
  {px, py, pz} = p3; pAbs = Sqrt[p3 . p3];
  den = Sqrt[2 pAbs (pAbs + pz)];
  If[h == 1, {pAbs + pz, px + I py}/den,
    {-px + I py, pAbs + pz}/den]
];

ChiSouth[p3_, h_] := Module[{px, py, pz, pAbs, den},
  {px, py, pz} = p3; pAbs = Sqrt[p3 . p3];
  den = Sqrt[2 pAbs (pAbs - pz)];
  If[h == 1, {px - I py, pAbs - pz}/den,
    {-(pAbs - pz), px + I py}/den]
];

ChiHelicity[p3_, h_] := Module[{pAbs, useSouth},
  pAbs = Sqrt[p3 . p3];
  useSouth = VectorQ[p3, NumericQ] && Abs[N[pAbs + p3[[3]]]] <= 10^-12;
  If[useSouth, ChiSouth[p3, h], ChiNorth[p3, h]]
];

ElectronSpinor[k_, h_] := Module[{chi = ChiHelicity[k[[2 ;; 4]], h]},
  Join[Sqrt[k[[1]]] chi, h Sqrt[k[[1]]] chi]
];

ProtonSpinor[p_, h_, Mp_] := Module[{pAbs, chi},
  pAbs = Sqrt[p[[2 ;; 4]] . p[[2 ;; 4]]];
  chi = ChiHelicity[p[[2 ;; 4]], h];
  Join[Sqrt[p[[1]] + Mp] chi, h pAbs/Sqrt[p[[1]] + Mp] chi]
];

PhotonPolarization[q_, lambda_] := Module[
  {qx, qy, qz, qAbs, rho, cosTheta, sinTheta, cosPhi, sinPhi, eTheta, ePhi},
  {qx, qy, qz} = q[[2 ;; 4]];
  qAbs = Sqrt[qx^2 + qy^2 + qz^2]; rho = Sqrt[qx^2 + qy^2];
  cosTheta = qz/qAbs; sinTheta = rho/qAbs;
  cosPhi = qx/rho; sinPhi = qy/rho;
  eTheta = {cosTheta cosPhi, cosTheta sinPhi, -sinTheta};
  ePhi = {-sinPhi, cosPhi, 0};
  Join[{0}, (eTheta + I lambda ePhi)/Sqrt[2]]
];

POutAnalytic[s_, EGamma_, phiGamma_, Mp_] := Module[
  {rootS, available, c, denominator, discriminant},
  rootS = Sqrt[s]; available = rootS - EGamma;
  c = available^2 + Mp^2 - EGamma^2;
  denominator = available^2 - EGamma^2 Sin[phiGamma]^2;
  discriminant = c^2 - 4 Mp^2 denominator;
  (-c EGamma Sin[phiGamma] + available Sqrt[discriminant])/(2 denominator)
];

UserKinematics[s_, thetaIn_, phiIn_, EGamma_, phiGamma_, Mp_] := Module[
  {pIn, pOut, eInProton, eOutProton, eOutElectron, k, p, kp, pp,
   qout, qVirtual, delta, pDotQ, q2, xb, t},
  pIn = (s - Mp^2)/(2 Sqrt[s]);
  pOut = POutAnalytic[s, EGamma, phiGamma, Mp];
  eInProton = Sqrt[pIn^2 + Mp^2]; eOutProton = Sqrt[pOut^2 + Mp^2];
  eOutElectron = Sqrt[pOut^2 + EGamma^2 + 2 pOut EGamma Sin[phiGamma]];
  k = pIn {1, -Sin[thetaIn] Cos[phiIn], -Sin[thetaIn] Sin[phiIn], -Cos[thetaIn]};
  p = {eInProton, pIn Sin[thetaIn] Cos[phiIn],
    pIn Sin[thetaIn] Sin[phiIn], pIn Cos[thetaIn]};
  pp = {eOutProton, 0, pOut, 0};
  qout = EGamma {1, Cos[phiGamma], Sin[phiGamma], 0};
  kp = {eOutElectron, -EGamma Cos[phiGamma], -pOut - EGamma Sin[phiGamma], 0};
  qVirtual = k - kp; delta = pp - p;
  q2 = -MinkowskiDot[qVirtual, qVirtual]; pDotQ = MinkowskiDot[p, qVirtual];
  xb = q2/(2 pDotQ); t = MinkowskiDot[delta, delta];
  <|"s" -> s, "thetaIn" -> thetaIn, "phiIn" -> phiIn,
    "EGamma" -> EGamma, "phiGamma" -> phiGamma, "Mp" -> Mp,
    "pIn" -> pIn, "pOut" -> pOut, "k" -> k, "p" -> p,
    "kp" -> kp, "pp" -> pp, "qout" -> qout, "q" -> qVirtual,
    "Q2" -> q2, "xB" -> xb, "t" -> t,
    "W2" -> MinkowskiDot[p + qVirtual, p + qVirtual],
    "y" -> pDotQ/MinkowskiDot[p, k]|>
];

KinematicChecks[kin_Association, Mp_] := Module[
  {k = kin["k"], p = kin["p"], kp = kin["kp"], pp = kin["pp"], qout = kin["qout"]},
  <|"fourMomentumResidual" -> Simplify[k + p - kp - pp - qout],
    "energyResidual" -> Simplify[k[[1]] + p[[1]] - kp[[1]] - pp[[1]] - qout[[1]]],
    "massShell" -> Simplify[{MinkowskiDot[k, k], MinkowskiDot[kp, kp],
      MinkowskiDot[qout, qout], MinkowskiDot[p, p], MinkowskiDot[pp, pp]} -
      {0, 0, 0, Mp^2, Mp^2}]|>
];

LeptonKernel[mu_, nu_, k_, kp_, qout_] := Module[{denPlus, denMinus, qSlash},
  denPlus = 2 MinkowskiDot[kp, qout]; denMinus = -2 MinkowskiDot[k, qout];
  qSlash = Slash[qout];
  (2 kp[[mu]] $Gamma[[nu]] + $Gamma[[mu]] . qSlash . $Gamma[[nu]])/denPlus +
  (2 k[[mu]] $Gamma[[nu]] - $Gamma[[nu]] . qSlash . $Gamma[[mu]])/denMinus
];

ProtonVertexLower[nu_, p_, pp_, Mp_, f1_, f2_] :=
  (f1 + f2) $Eta[[nu]] $Gamma[[nu]] -
  $Eta[[nu]] (p + pp)[[nu]] f2 IdentityMatrix[4]/(2 Mp);

BHAmplitude[kin_Association, hIn_, hOut_, sIn_, sOut_, lambda_, Mp_, f1_, f2_] := Module[
  {k, p, kp, pp, qout, electronIn, electronOut, protonIn, protonOut,
   electronBar, protonBar, epsilonCovStar, t, hadronic},
  {k, p, kp, pp, qout} = Lookup[kin, {"k", "p", "kp", "pp", "qout"}];
  electronIn = ElectronSpinor[k, hIn]; electronOut = ElectronSpinor[kp, hOut];
  protonIn = ProtonSpinor[p, sIn, Mp]; protonOut = ProtonSpinor[pp, sOut, Mp];
  electronBar = SpinorBar[electronOut]; protonBar = SpinorBar[protonOut];
  epsilonCovStar = Covariant[Conjugate[PhotonPolarization[qout, lambda]]];
  t = MinkowskiDot[pp - p, pp - p];
  hadronic = Table[protonBar . ProtonVertexLower[nu, p, pp, Mp, f1, f2] . protonIn,
    {nu, 1, 4}];
  Sum[epsilonCovStar[[mu]]
    (electronBar . LeptonKernel[mu, nu, k, kp, qout] . electronIn)
    hadronic[[nu]], {mu, 1, 4}, {nu, 1, 4}]/t
];

BHAmplitudeTable[kin_Association, Mp_, f1_, f2_] := ArrayReshape[
  Table[BHAmplitude[kin, hIn, hOut, sIn, sOut, lambda, Mp, f1, f2],
    {hIn, $Helicities}, {sIn, $Helicities}, {hOut, $Helicities},
    {sOut, $Helicities}, {lambda, $Helicities}], {4, 8}];

SingleSpinDensity[axis_] := Module[{state}, Which[
  axis === "Unpolarized" || axis === None, IdentityMatrix[2]/2,
  axis === "L", {{0, 0}, {0, 1}},
  axis === "Tx", {{1, 1}, {1, 1}}/2,
  axis === "Ty", {{1, I}, {-I, 1}}/2,
  VectorQ[axis] && Length[axis] == 2,
    state = axis/Sqrt[Conjugate[axis] . axis]; Outer[Times, state, Conjugate[state]],
  True, Message[SingleSpinDensity::axis, axis]; $Failed]];
SingleSpinDensity::axis = "Unknown spin preparation `1`.";

InitialSpinDensity[electronAxis_: "Unpolarized", protonAxis_: "Unpolarized"] :=
  KroneckerProduct[SingleSpinDensity[electronAxis], SingleSpinDensity[protonAxis]];

NormalizeDensity[rho_] := Module[{hermitian = (rho + ConjugateTranspose[rho])/2},
  hermitian/Tr[hermitian]];

OutgoingDensityMatrix[amplitudes_, rhoIn_, normalize_: True] := Module[{rho},
  rho = Transpose[amplitudes] . rhoIn . Conjugate[amplitudes];
  If[TrueQ[normalize], NormalizeDensity[rho], rho]
];

FlatQubitIndex[values_List] := 1 + 4 (values[[1]] - 1) + 2 (values[[2]] - 1) + values[[3]] - 1;

ReducedDensityMatrix[rho_, keep_List] := Module[
  {traceOut, keptStates, tracedStates, makeValues, reduced},
  traceOut = Complement[Range[3], keep]; keptStates = Tuples[Range[2], Length[keep]];
  tracedStates = Tuples[Range[2], Length[traceOut]];
  makeValues[kept_, traced_] := Module[{values = ConstantArray[1, 3]},
    Do[values[[keep[[j]]]] = kept[[j]], {j, Length[keep]}];
    Do[values[[traceOut[[j]]]] = traced[[j]], {j, Length[traceOut]}]; values];
  reduced = Table[Total@Table[rho[[
      FlatQubitIndex[makeValues[keptStates[[row]], traced]],
      FlatQubitIndex[makeValues[keptStates[[column]], traced]]]],
    {traced, tracedStates}], {row, Length[keptStates]}, {column, Length[keptStates]}];
  NormalizeDensity[reduced]
];

TwoQubitConcurrence[rho2_] := Module[{rho, spinFlip, eigenvalues, lambdas},
  rho = N[NormalizeDensity[rho2]]; spinFlip = KroneckerProduct[$Sigma2, $Sigma2];
  eigenvalues = Eigenvalues[rho . spinFlip . Conjugate[rho] . spinFlip];
  lambdas = Reverse@Sort[Sqrt[Map[Max[0., Re[#]] &, Chop[eigenvalues]]]];
  Max[0., lambdas[[1]] - Total[lambdas[[2 ;; 4]]]]
];

OneToRestConcurrence[rho_, subsystem_] := Module[{single, purity},
  single = ReducedDensityMatrix[rho, {subsystem}]; purity = Re[Tr[single . single]];
  Sqrt[Max[0., 2 (1 - purity)]]
];

EntanglementObservables[rhoInput_] := Module[
  {rho, cEP, cEG, cPG, purity, cERest, cPRest, cGRest, q, f3},
  rho = N[NormalizeDensity[rhoInput]];
  cEP = TwoQubitConcurrence[ReducedDensityMatrix[rho, {1, 2}]];
  cEG = TwoQubitConcurrence[ReducedDensityMatrix[rho, {1, 3}]];
  cPG = TwoQubitConcurrence[ReducedDensityMatrix[rho, {2, 3}]];
  purity = Re[Tr[rho . rho]];
  If[Abs[purity - 1] > 10^-9, Return[<|"purity" -> purity,
    "C_e_p" -> cEP, "C_e_gamma" -> cEG, "C_p_gamma" -> cPG,
    "C_e_rest" -> Missing["RequiresPureState"],
    "C_p_rest" -> Missing["RequiresPureState"],
    "C_gamma_rest" -> Missing["RequiresPureState"],
    "F3" -> Missing["RequiresPureState"]|>]];
  cERest = OneToRestConcurrence[rho, 1]; cPRest = OneToRestConcurrence[rho, 2];
  cGRest = OneToRestConcurrence[rho, 3]; q = (cERest + cPRest + cGRest)/2;
  f3 = Sqrt[Max[0., (16/3) q (q - cERest) (q - cPRest) (q - cGRest)]];
  <|"purity" -> purity, "C_e_p" -> cEP, "C_e_gamma" -> cEG,
    "C_p_gamma" -> cPG, "C_e_rest" -> cERest, "C_p_rest" -> cPRest,
    "C_gamma_rest" -> cGRest, "F3" -> f3,
    "M_e" -> cERest^2 - cEP^2 - cEG^2,
    "M_p" -> cPRest^2 - cEP^2 - cPG^2,
    "M_gamma" -> cGRest^2 - cEG^2 - cPG^2|>
];

End[];
EndPackage[];
