(* ::Package:: *)

(* Print one fully contracted analytic Bethe-Heitler helicity amplitude.

   All kinematic quantities remain symbolic. By default the independent
   variables are

     {s, thetaIn, phiIn, EGamma, phiGamma, Mp, F1t, F2t}.

   Kinematic definitions and units:

     s          total incoming electron-proton invariant mass squared [GeV^2]
     thetaIn    polar angle of the incoming proton momentum [rad]
     phiIn      azimuthal angle of the incoming proton momentum [rad]
     EGamma     outgoing real-photon energy |q'| in the COM frame [GeV]
     phiGamma   azimuthal angle of the outgoing real photon [rad]
     Mp         proton mass [GeV]
     F1t        Dirac form factor F1(t), dimensionless
     F2t        Pauli form factor F2(t), dimensionless

   Coordinate convention:

     All momenta are in the incoming electron-proton COM frame and use
     contravariant ordering {energy, px, py, pz}. The outgoing proton is fixed
     along +y. The outgoing photon lies in the xy plane at azimuth phiGamma.
     The incoming proton points along {thetaIn,phiIn}; the massless incoming
     electron points in the opposite direction.

   Derived variables constructed by UserKinematics:

     pIn        incoming electron/proton three-momentum magnitude [GeV]
     pOut       outgoing proton three-momentum magnitude [GeV]
     k, p       incoming electron and proton four-momenta [GeV]
     kp, pp     outgoing electron and proton four-momenta [GeV]
     qout       outgoing real-photon four-momentum [GeV]
     q          virtual-photon momentum k-kp [GeV]
     Q2         virtuality -q.q [GeV^2]
     xB         Bjorken variable Q2/(2 p.q)
     t          proton momentum transfer squared (pp-p)^2 [GeV^2]
     W2         invariant mass squared (p+q)^2 [GeV^2]
     y          inelasticity (p.q)/(p.k)

   Edit helicityInputs below to select

     {hIn, hOut, sIn, sOut, lambda}.
*)
SetDirectory[NotebookDirectory[]];
<<"BHHelicityAmp.wl";

(* Doubled-helicity convention: every entry must be -1 or +1. *)
helicityInputs = {-1, -1, -1, 1, 1};
{hIn, hOut, sIn, sOut, lambda} = helicityInputs;

If[!AllTrue[helicityInputs, MemberQ[{-1, 1}, #] &],
  Print["Invalid helicity input. Use only -1 or +1."];
  Exit[1];];

(* User-frame kinematics with pIn and the physical pOut root expressed
   analytically in terms of the independent variables. *)
kinematics = UserKinematics[s, thetaIn, phiIn, EGamma, phiGamma, Mp];

amplitude = BHAmplitude[
  kinematics,
  hIn, hOut, sIn, sOut, lambda,
  Mp, F1t, F2t
];

commandLineStrings = ToString /@ Join[$ScriptCommandLine, $CommandLine];
summaryOnly = AnyTrue[commandLineStrings, StringContainsQ[#, "--summary"] &];

Print["Helicities {hIn,hOut,sIn,sOut,lambda} = ", helicityInputs];
Print["Independent variables = ",
  {s, thetaIn, phiIn, EGamma, phiGamma, Mp, F1t, F2t}];
Print["Expression head = ", Head[amplitude]];
Print["Expression leaf count = ", LeafCount[amplitude]];
Print["Uncontracted matrix products remaining = ",
  Count[amplitude, _Dot, Infinity]];

If[!summaryOnly,
  Print["\nM[hIn,hOut,sIn,sOut,lambda] ="];
  Print[InputForm[amplitude]];
];



