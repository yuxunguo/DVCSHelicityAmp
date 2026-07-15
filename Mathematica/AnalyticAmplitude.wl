(* ::Package:: *)

(* Print one fully contracted analytic Bethe-Heitler helicity amplitude.

   All kinematic quantities remain symbolic. By default the independent
   variables are

     {s, thetaIn, phiIn, EGamma, phiGamma, Mp, Ml, F1t, F2t}.

   Kinematic definitions and units:

     s          total incoming lepton-proton invariant mass squared [GeV^2]
     thetaIn    polar angle of the incoming proton momentum [rad]
     phiIn      azimuthal angle of the incoming proton momentum [rad]
     EGamma     outgoing real-photon energy |q'| in the COM frame [GeV]
     phiGamma   azimuthal angle of the outgoing real photon [rad]
     Mp         proton mass [GeV]
     Ml         charged-lepton mass [GeV]
     F1t        Dirac form factor F1(t), dimensionless
     F2t        Pauli form factor F2(t), dimensionless

   Coordinate convention:

     All momenta are in the incoming lepton-proton COM frame and use
     contravariant ordering {energy, px, py, pz}. The outgoing proton is fixed
     along +y. The outgoing photon lies in the xy plane at azimuth phiGamma.
     The incoming proton points along {thetaIn,phiIn}; the massive incoming
     lepton points in the opposite direction.

   Derived variables constructed by UserKinematics:

     pIn        incoming lepton/proton three-momentum magnitude [GeV]
     pOut       outgoing proton three-momentum magnitude [GeV]
     k, p       incoming lepton and proton four-momenta [GeV]
     kp, pp     outgoing lepton and proton four-momenta [GeV]
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
SetDirectory[DirectoryName[ExpandFileName[$InputFileName]]];
<<"BHHelicityAmp.wl";

(* Doubled-helicity convention: every entry must be -1 or +1. *)
helicityInputs = {-1, -1, -1, 1, 1};
{hIn, hOut, sIn, sOut, lambda} = helicityInputs;

If[!AllTrue[helicityInputs, MemberQ[{-1, 1}, #] &],
  Print["Invalid helicity input. Use only -1 or +1."];
  Exit[1];];

(* User-frame kinematics with pIn and the physical pOut root expressed
   analytically in terms of the independent variables. *)
kinematics = UserKinematics[s, thetaIn, phiIn, EGamma, phiGamma, Mp, Ml];

amplitude = BHAmplitude[
  kinematics,
  hIn, hOut, sIn, sOut, lambda,
  Mp, Ml, F1t, F2t
];

commandLineStrings = ToString /@ Join[$ScriptCommandLine, $CommandLine];
summaryOnly = AnyTrue[commandLineStrings, StringContainsQ[#, "--summary"] &];

Print["Helicities {hIn,hOut,sIn,sOut,lambda} = ", helicityInputs];
Print["Independent variables = ",
  {s, thetaIn, phiIn, EGamma, phiGamma, Mp, Ml, F1t, F2t}];
Print["Expression head = ", Head[amplitude]];
Print["Expression leaf count = ", LeafCount[amplitude]];
Print["Uncontracted matrix products remaining = ",
  Count[amplitude, _Dot, Infinity]];

If[!summaryOnly,
  Print["\nM[hIn,hOut,sIn,sOut,lambda] ="];
  Print[InputForm[amplitude]];
];

